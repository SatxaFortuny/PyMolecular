"""
EGNN Triton kernel benchmark
Compares Satorras E_GCL vs. EGNN_Triton_Layer across multiple datasets and configurations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM9, MD17
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RadiusGraph, Compose, Distance

from egnn_clean import E_GCL
from layer import EGNN_Triton_Layer

# ── Global model hyperparameters ──────────────────────────────────────────────
F_NODE     = 32
F_EDGE     = 16   # must be >= 16 for tl.dot in the Triton kernel
HIDDEN_DIM = 64
MSG_OUT    = F_NODE

WARMUP = 30
ITERS  = 200

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Model wrappers ────────────────────────────────────────────────────────────

class SatorrasWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, pos, edge_index, edge_attr):
        h, new_pos, _ = self.layer(x, edge_index, pos, edge_attr=edge_attr)
        return h, new_pos


class TritonWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, pos, edge_index, edge_attr):
        pos4 = F.pad(pos, (0, 1))
        new_feat, new_coord4 = self.layer(x, pos4, edge_index, edge_attr)
        return new_feat, new_coord4[:, :3]


def make_models():
    """Create a fresh matched pair of (SatorrasWrapper, TritonWrapper)."""
    egcl = E_GCL(
        input_nf=F_NODE,
        output_nf=F_NODE,
        hidden_nf=HIDDEN_DIM,
        edges_in_d=F_EDGE,
    ).to(DEVICE)

    triton_layer = EGNN_Triton_Layer(
        f_node=F_NODE,
        f_edge=F_EDGE,
        msg_hidden_dim=HIDDEN_DIM,
        msg_out_feat=MSG_OUT,
        mov_hidden_dim=HIDDEN_DIM,
        node_hidden_dim=HIDDEN_DIM,
        rbf_dim=1,
        rbf_gamma=10.0,
    ).to(DEVICE)

    return SatorrasWrapper(egcl), TritonWrapper(triton_layer)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_edge_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Build F_EDGE-dimensional edge features from node positions.
    Features: [distance (1), unit direction (3), random padding (F_EDGE-4)]
    """
    src, dst   = edge_index
    diff       = pos[src] - pos[dst]                          # (E, 3)
    dist       = diff.norm(dim=-1, keepdim=True)              # (E, 1)
    direction  = diff / (dist + 1e-8)                         # (E, 3)  unit vector
    base       = torch.cat([dist, direction], dim=-1)         # (E, 4)

    if F_EDGE > 4:
        # Small random noise — physically meaningless but exercises the full MLP
        padding = torch.randn(base.shape[0], F_EDGE - 4,
                              device=pos.device, dtype=pos.dtype) * 0.01
        base = torch.cat([base, padding], dim=-1)

    return base[:, :F_EDGE]


def zero_edge_features(num_edges: int) -> torch.Tensor:
    """Zero-valued edge features — models still see the same architecture."""
    return torch.zeros(num_edges, F_EDGE, device=DEVICE)


def pad_node_features(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < F_NODE:
        x = F.pad(x, (0, F_NODE - x.shape[1]))
    return x[:, :F_NODE].float()


# ── Core benchmark runner ─────────────────────────────────────────────────────

def _run(model, x, pos, edge_index, edge_attr) -> float:
    """Return mean ms/iter for one model on one data configuration."""
    model.eval()
    with torch.no_grad():
        # Warmup (also triggers Triton autotune on first call)
        for _ in range(WARMUP):
            model(x, pos, edge_index, edge_attr)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            model(x, pos, edge_index, edge_attr)
        end.record()
        torch.cuda.synchronize()

    return start.elapsed_time(end) / ITERS


def _bench_pair(tag: str, x, pos, edge_index, edge_attr) -> dict:
    """Run both models and print + return results."""
    n_nodes  = x.shape[0]
    n_edges  = edge_index.shape[1]
    avg_deg  = n_edges / max(n_nodes, 1)

    print(f"\n{'='*62}")
    print(f"  {tag}")
    print(f"  nodes={n_nodes:,}  edges={n_edges:,}  avg_degree={avg_deg:.1f}")
    print(f"{'='*62}")

    satorras, triton = make_models()

    print("  [Satorras E_GCL]  running...")
    t_sat = _run(satorras, x, pos, edge_index, edge_attr)

    print("  [Triton EGNN]     running...")
    t_tri = _run(triton, x, pos, edge_index, edge_attr)

    speedup = t_sat / t_tri
    print(f"  Satorras : {t_sat:8.3f} ms/iter")
    print(f"  Triton   : {t_tri:8.3f} ms/iter")
    print(f"  Speedup  : {speedup:8.2f}x")

    return {"satorras_ms": t_sat, "triton_ms": t_tri, "speedup": speedup,
            "nodes": n_nodes, "edges": n_edges}


# ── Dataset-specific benchmark functions ──────────────────────────────────────

def benchmark_qm9(batch_size: int = 256, radius: float = 5.0) -> dict:
    """
    QM9 with a dense RadiusGraph (r=5Å) for maximum edge count.
    Edge features are always derived from atomic positions.
    batch_size=256 → ~200-300k edges per batch.
    """
    print("\n>>> Loading QM9 ...")
    dataset = QM9(root='./data/QM9', transform=RadiusGraph(r=radius))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch   = next(iter(loader)).to(DEVICE)

    x          = pad_node_features(batch.x)
    pos        = batch.pos.float()
    edge_index = batch.edge_index
    edge_attr  = make_edge_features(pos, edge_index)  # always real for QM9

    return _bench_pair(f"QM9  |  batch={batch_size}, r={radius}Å",
                       x, pos, edge_index, edge_attr)


def benchmark_md17(use_edge_features: bool,
                   molecule: str = 'aspirin',
                   batch_size: int = 512,
                   radius: float = 5.0) -> dict:
    """
    MD17 molecular dynamics dataset.
    Graphs are built with RadiusGraph; edge features are optionally computed
    from pairwise positions (distance + unit direction).
   
    Args:
        use_edge_features: if True, compute geometrical edge features from pos.
                           if False, pass zero edge features (same architecture).
        molecule: any MD17 molecule name ('aspirin', 'malonaldehyde', etc.)
        batch_size: number of MD snapshots per batch.
        radius: cutoff radius in Ångström.
    """
    print(f"\n>>> Loading MD17 ({molecule}) ...")
    dataset = MD17(root='./data/MD17', name=molecule,
                   transform=RadiusGraph(r=radius))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch   = next(iter(loader)).to(DEVICE)

    pos        = batch.pos.float()
    edge_index = batch.edge_index
    num_nodes  = pos.shape[0]
    num_edges  = edge_index.shape[1]

    # Node features: atomic number as a single normalised feature, padded to F_NODE
    x = torch.zeros(num_nodes, F_NODE, device=DEVICE)
    x[:, 0] = batch.z.float() / 10.0

    edge_attr = (make_edge_features(pos, edge_index)
                 if use_edge_features
                 else zero_edge_features(num_edges))

    feat_tag = "w/ edge feats" if use_edge_features else "no edge feats"
    tag = f"MD17 {molecule}  |  batch={batch_size}, r={radius}Å, {feat_tag}"
    return _bench_pair(tag, x, pos, edge_index, edge_attr)


def benchmark_oc20_synthetic(use_edge_features: bool,
                              num_structures: int = 32,
                              atoms_per_structure: int = 150,
                              radius: float = 6.0) -> dict:
    """
    Synthetic dataset at OC20 scale (catalyst slabs, ~50-300 atoms each).
    OC20 requires registration+manual download, so we generate realistic
    synthetic graphs with matching statistics: random atom positions in a
    20Å³ periodic-like box, dense radius graph (r=6Å), ~30-50 neighbours.

    Args:
        use_edge_features: if True, compute geometrical edge features from pos.
                           if False, pass zero edge features.
        num_structures: number of catalyst structures in the batch.
        atoms_per_structure: mean atoms per structure (±20 random jitter).
        radius: cutoff radius in Ångström.
    """
    print(f"\n>>> Building OC20-scale synthetic data ...")

    all_pos, all_x   = [], []
    all_src, all_dst = [], []
    offset = 0

    for _ in range(num_structures):
        n   = atoms_per_structure + torch.randint(-20, 20, (1,)).item()
        pos = torch.rand(n, 3, device=DEVICE) * 20.0           # 20Å box

        # Pairwise distances — O(n²) but n≈150 so trivially fast
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)             # (n, n, 3)
        dist = diff.norm(dim=-1)                               # (n, n)
        mask = (dist < radius) & (dist > 0)
        src, dst = mask.nonzero(as_tuple=True)

        all_pos.append(pos)
        all_x.append(torch.randn(n, F_NODE, device=DEVICE))
        all_src.append(src + offset)
        all_dst.append(dst + offset)
        offset += n

    pos        = torch.cat(all_pos, dim=0)
    x          = torch.cat(all_x,   dim=0)
    edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0)
    num_edges  = edge_index.shape[1]

    edge_attr = (make_edge_features(pos, edge_index)
                 if use_edge_features
                 else zero_edge_features(num_edges))

    feat_tag = "w/ edge feats" if use_edge_features else "no edge feats"
    tag = (f"OC20-scale synthetic  |  {num_structures} structs × "
           f"~{atoms_per_structure} atoms, r={radius}Å, {feat_tag}")
    return _bench_pair(tag, x, pos, edge_index, edge_attr)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"\nDevice : {DEVICE}")
    print(f"F_NODE={F_NODE}  F_EDGE={F_EDGE}  HIDDEN={HIDDEN_DIM}")
    print(f"Warmup={WARMUP} iters  |  Measure={ITERS} iters")

    results = {}

    # ── QM9 ───────────────────────────────────────────────────────────────────
    # Large batch + radius graph → dense, ~200-300k edges
    results['QM9 (batch=256, r=5.0Å)'] = \
        benchmark_qm9(batch_size=256, radius=5.0)

    # ── MD17 — all combinations ───────────────────────────────────────────────
    # Aspirin (21 atoms) — batch=512 → ~10k nodes, ~75k edges @ r=5Å
    results['MD17 aspirin   | w/ edge feats'] = \
        benchmark_md17(use_edge_features=True,  molecule='aspirin',      batch_size=512)
    results['MD17 aspirin   | no edge feats'] = \
        benchmark_md17(use_edge_features=False, molecule='aspirin',      batch_size=512)

    # Malonaldehyde (9 atoms, simpler molecule, more snapshots in batch)
    results['MD17 malonalde | w/ edge feats'] = \
        benchmark_md17(use_edge_features=True,  molecule='malonaldehyde', batch_size=512)
    results['MD17 malonalde | no edge feats'] = \
        benchmark_md17(use_edge_features=False, molecule='malonaldehyde', batch_size=512)

    # ── OC20-scale synthetic — all combinations ───────────────────────────────
    # 32 structures × ~150 atoms × ~30 neighbours ≈ 140k edges per batch
    results['OC20-scale     | w/ edge feats'] = \
        benchmark_oc20_synthetic(use_edge_features=True)
    results['OC20-scale     | no edge feats'] = \
        benchmark_oc20_synthetic(use_edge_features=False)

    # Denser variant: fewer structures but tighter radius → more edges/node
    results['OC20-scale dense r=8 | w/ edge feats'] = \
        benchmark_oc20_synthetic(use_edge_features=True,
                                 num_structures=16, atoms_per_structure=200,
                                 radius=8.0)
    results['OC20-scale dense r=8 | no edge feats'] = \
        benchmark_oc20_synthetic(use_edge_features=False,
                                 num_structures=16, atoms_per_structure=200,
                                 radius=8.0)

    # ── Summary table ─────────────────────────────────────────────────────────
    col_name = 42
    print(f"\n\n{'='*72}")
    print(f"{'BENCHMARK SUMMARY':^72}")
    print(f"{'='*72}")
    print(f"{'Dataset / Config':<{col_name}} {'Satorras':>9} {'Triton':>9} {'Speedup':>8}  {'Edges':>8}")
    print(f"{'-'*72}")
    for name, r in results.items():
        print(f"{name:<{col_name}} "
              f"{r['satorras_ms']:>8.2f}ms "
              f"{r['triton_ms']:>8.2f}ms "
              f"{r['speedup']:>7.2f}x  "
              f"{r['edges']:>8,}")
    print(f"{'='*72}")