"""
dataset_loaders.py  —  Real dataset loaders for the EGNN backward kernel benchmark.

Covers three scales of increasing graph size:

  Scale 1 — QM9  (PyG)
      Small organic molecules. 3–29 heavy atoms, avg ~9.
      A single molecule is too tiny to stress the kernel; we batch
      many molecules into one disconnected graph (exactly what PyG's
      DataLoader does internally). This tests the kernel under the
      conditions it would actually see during QM9 training.

  Scale 2 — Proteins: CA-trace graphs from the PDB  (via BioPython)
      One node per residue (Cα atom), edges from a distance cutoff.
      Single chains: ~100–500 residues.
      Assemblies / complexes: 1k–20k residues.

  Scale 3 — Open Catalyst OC20  (via PyG / LMDB)
      Adsorbate + surface slab. ~50–300 atoms per structure but
      millions of structures available. Also has rich edge features
      (distance, angle).  Good for stressing large-degree behavior.

Each loader returns a unified GraphBundle that the benchmark can
consume directly.

Dependencies
------------
    pip install torch-geometric biopython requests lmdb
    # For OC20:  pip install ocpmodels   (optional — see Scale 3 notes)
"""

from __future__ import annotations

import os
import io
import gzip
import math
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# 0.  Shared data container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GraphBundle:
    """
    Everything the benchmark kernel needs, already on `device`.

    node_feat   : (N, F_NODE)   float32
    coords      : (N, 3)        float32
    edge_feat   : (E, F_EDGE)   float32   — at minimum, [dist, dist^2]
    row_ptrs    : (N+1,)        int32     — CSR row pointers
    col_indices : (E,)          int32     — CSR column indices (src node)
    edge_id     : (E,)          int32     — edge index for edge_feat lookup
    name        : str           — human-readable label
    N, E        : int
    """
    node_feat:   torch.Tensor
    coords:      torch.Tensor
    edge_feat:   torch.Tensor
    row_ptrs:    torch.Tensor
    col_indices: torch.Tensor
    edge_id:     torch.Tensor
    name:        str
    N:           int
    E:           int

    def summary(self) -> str:
        deg = self.E / max(self.N, 1)
        return (f"{self.name:40s}  N={self.N:>8,}  E={self.E:>10,}  "
                f"avg_deg={deg:5.1f}  "
                f"F_node={self.node_feat.shape[1]}  "
                f"F_edge={self.edge_feat.shape[1]}")


# ═══════════════════════════════════════════════════════════════════════════
# 1.  CSR construction utilities
# ═══════════════════════════════════════════════════════════════════════════

def coords_to_knn_edges(coords: torch.Tensor, k: int,
                        chunk: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build directed k-NN edges from 3-D coordinates.
    Returns (src, dst) each of shape (E,).
    Chunked to stay within GPU memory for large N.
    """
    N = coords.shape[0]
    k = min(k, N - 1)
    src_list, dst_list = [], []
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        d2 = torch.cdist(coords[start:end], coords)          # (C, N)
        d2[:, start:end].fill_diagonal_(float("inf"))
        _, top = d2.topk(k, dim=1, largest=False)            # (C, k)
        row = torch.arange(start, end, device=coords.device)
        src_list.append(row.unsqueeze(1).expand_as(top).reshape(-1))
        dst_list.append(top.reshape(-1))
    return torch.cat(src_list), torch.cat(dst_list)


def coords_to_radius_edges(coords: torch.Tensor, r: float,
                           chunk: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build directed edges for all pairs within radius r.
    Returns (src, dst).
    """
    N = coords.shape[0]
    src_list, dst_list = [], []
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        d2 = torch.cdist(coords[start:end], coords)
        mask = (d2 < r) & (d2 > 0)
        rows, cols = mask.nonzero(as_tuple=True)
        src_list.append(rows + start)
        dst_list.append(cols)
    return torch.cat(src_list), torch.cat(dst_list)


def build_csr(src: torch.Tensor, dst: torch.Tensor,
              N: int, coords: torch.Tensor,
              node_feat: torch.Tensor,
              extra_edge_feat: Optional[torch.Tensor] = None,
              device: str = "cuda") -> GraphBundle:
    """
    Given directed edges (src→dst), build the CSR representation
    that the node-parallel kernel expects (dst is the "main node",
    src is the neighbour).

    edge_feat at minimum: [dist, dist^2]  (F_EDGE=2)
    extra_edge_feat: (E, K) additional features appended to the above.
    """
    src = src.to(device)
    dst = dst.to(device)
    coords = coords.to(device)
    node_feat = node_feat.to(device)

    E = src.shape[0]

    # Sort by dst so we can build CSR on dst (main node = receiver)
    order = dst.argsort()
    src_s = src[order]
    dst_s = dst[order]

    # Row pointers: for each main node i, where do its neighbours start?
    counts = torch.zeros(N, dtype=torch.int32, device=device)
    counts.scatter_add_(0, dst_s, torch.ones(E, dtype=torch.int32, device=device))
    row_ptrs = torch.zeros(N + 1, dtype=torch.int32, device=device)
    row_ptrs[1:] = counts.cumsum(0)

    col_indices = src_s.to(torch.int32)
    edge_id     = torch.arange(E, dtype=torch.int32, device=device)

    # Edge features: distance and distance^2
    delta  = coords[dst_s] - coords[src_s]         # (E, 3)
    dist2  = (delta * delta).sum(-1, keepdim=True)  # (E, 1)
    dist   = dist2.sqrt()                           # (E, 1)
    ef     = torch.cat([dist, dist2], dim=-1)       # (E, 2)

    if extra_edge_feat is not None:
        ef = torch.cat([ef, extra_edge_feat[order].to(device)], dim=-1)

    return GraphBundle(
        node_feat=node_feat, coords=coords,
        edge_feat=ef,
        row_ptrs=row_ptrs, col_indices=col_indices, edge_id=edge_id,
        name="", N=N, E=E,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Scale 1 — QM9  (via PyTorch Geometric)
# ═══════════════════════════════════════════════════════════════════════════

def load_qm9(
    root: str = "data/qm9",
    num_graphs: int = 1000,
    k_neighbors: int = 5,
    device: str = "cuda",
) -> GraphBundle:
    """
    Load `num_graphs` molecules from QM9 and concatenate them into a
    single disconnected graph — this is exactly the batching that happens
    during real training with PyG's DataLoader.

    Node features: one-hot atomic type (H C N O F → 5 dims) + charge (1)
    Edge features: distance, distance^2
    Connectivity: k-NN within each molecule (k=5 matches typical EGNN setup)

    Requires: torch-geometric
        pip install torch-geometric
    """
    try:
        from torch_geometric.datasets import QM9
        import torch_geometric.transforms as T
    except ImportError:
        raise ImportError("pip install torch-geometric")

    print(f"Loading QM9 ({num_graphs} molecules) …")
    ds = QM9(root=root)

    all_coords, all_feats = [], []
    node_offset = 0
    all_src, all_dst = [], []

    for i in range(min(num_graphs, len(ds))):
        data = ds[i]
        pos  = data.pos                           # (n, 3)
        n    = pos.shape[0]

        # k-NN within this molecule
        if n > 1:
            src, dst = coords_to_knn_edges(pos, k=min(k_neighbors, n - 1))
            all_src.append(src + node_offset)
            all_dst.append(dst + node_offset)

        # Node features: atomic_number one-hot (H=1,C=6,N=7,O=8,F=9) + formal charge
        z = data.z                                # (n,) atomic numbers
        one_hot = torch.zeros(n, 5)
        for j, elem in enumerate([1, 6, 7, 8, 9]):
            one_hot[:, j] = (z == elem).float()
        charge = data.x[:, 4:5] if data.x.shape[1] > 4 else torch.zeros(n, 1)
        feat = torch.cat([one_hot, charge], dim=-1)   # (n, 6)

        all_coords.append(pos)
        all_feats.append(feat)
        node_offset += n

    coords    = torch.cat(all_coords, dim=0).float()
    node_feat = torch.cat(all_feats,  dim=0).float()
    src       = torch.cat(all_src,    dim=0)
    dst       = torch.cat(all_dst,    dim=0)
    N         = coords.shape[0]

    bundle = build_csr(src, dst, N, coords, node_feat, device=device)
    bundle.name = f"QM9 ({num_graphs} molecules, k={k_neighbors})"
    return bundle


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Scale 2 — Protein structures from the PDB  (via BioPython)
# ═══════════════════════════════════════════════════════════════════════════

# Representative PDB IDs at increasing size:
#   Small chain    : 1UBQ  (ubiquitin,  76 residues)
#   Medium chain   : 1TIM  (TIM barrel, 249 residues)
#   Large chain    : 3J3Q  (ribosome subunit, ~1500 residues — large!)
#   Complex        : 7A4M  (antibody complex, ~900 residues)

PROTEIN_PDB_IDS = {
    "small_chain":    ["1UBQ", "1VII", "1L2Y"],     # <100 res
    "medium_chain":   ["1TIM", "2HHB", "1CRN"],     # 100–300 res
    "large_chain":    ["1AON", "3HDP", "2Y69"],     # 300–1000 res
    "complex":        ["7A4M", "6XRZ", "3J3Q"],     # 1000+ res (assemblies)
}


def _fetch_pdb(pdb_id: str, cache_dir: Path) -> Optional[str]:
    """Download a PDB file (gzipped) and return local path."""
    path = cache_dir / f"{pdb_id.lower()}.pdb"
    if path.exists():
        return str(path)
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb.gz"
    try:
        print(f"  Downloading {pdb_id} from RCSB …", end=" ", flush=True)
        with urllib.request.urlopen(url, timeout=30) as r:
            data = gzip.decompress(r.read()).decode("utf-8", errors="ignore")
        path.write_text(data)
        print("ok")
        return str(path)
    except Exception as e:
        print(f"failed ({e})")
        return None


def _pdb_to_ca_graph(
    pdb_path: str,
    radius: float = 10.0,         # Å — typical protein contact cutoff
    device: str = "cuda",
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Parse a PDB file and extract:
      - Cα coordinates       (N_res, 3)
      - Residue features     (N_res, F)  [one-hot AA type + secondary structure]
    Returns (coords, node_feat, pdb_name) or None on parse failure.
    """
    try:
        from Bio import PDB as bpdb
        from Bio.PDB.Polypeptide import three_to_one, is_aa
    except ImportError:
        raise ImportError("pip install biopython")

    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
    AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

    parser = bpdb.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("mol", pdb_path)
    except Exception:
        return None

    coords_list, feat_list = [], []
    for model in structure.get_models():
        for chain in model.get_chains():
            for residue in chain.get_residues():
                if not is_aa(residue, standard=True):
                    continue
                if "CA" not in residue:
                    continue
                ca = residue["CA"].get_vector().get_array()
                coords_list.append(ca)

                try:
                    aa = three_to_one(residue.get_resname())
                except Exception:
                    aa = "X"
                oh = torch.zeros(len(AA_LIST) + 1)  # +1 for unknown
                oh[AA_TO_IDX.get(aa, len(AA_LIST))] = 1.0
                feat_list.append(oh)
        break  # first model only

    if len(coords_list) < 4:
        return None

    coords    = torch.tensor(np.stack(coords_list), dtype=torch.float32)
    node_feat = torch.stack(feat_list)   # (N, 21)
    return coords, node_feat


def load_protein(
    pdb_ids: List[str],
    cache_dir: str = "data/pdb",
    radius: float = 10.0,
    device: str = "cuda",
    name: str = "protein",
) -> Optional[GraphBundle]:
    """
    Load one or more PDB structures, concatenate residue graphs, return bundle.

    radius : Å cutoff for edges (10 Å is a standard contact map threshold)
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    all_coords, all_feats = [], []
    all_src, all_dst = [], []
    node_offset = 0
    loaded = 0

    for pdb_id in pdb_ids:
        path = _fetch_pdb(pdb_id, cache)
        if path is None:
            continue
        result = _pdb_to_ca_graph(path, radius=radius, device="cpu")
        if result is None:
            continue
        coords, feat = result
        n = coords.shape[0]

        src, dst = coords_to_radius_edges(coords, r=radius)
        if src.shape[0] == 0:
            continue

        all_src.append(src + node_offset)
        all_dst.append(dst + node_offset)
        all_coords.append(coords)
        all_feats.append(feat)
        node_offset += n
        loaded += 1
        print(f"  {pdb_id}: {n} residues, {src.shape[0]} edges")

    if loaded == 0:
        return None

    coords    = torch.cat(all_coords).float()
    node_feat = torch.cat(all_feats).float()
    src       = torch.cat(all_src)
    dst       = torch.cat(all_dst)
    N         = coords.shape[0]

    bundle = build_csr(src, dst, N, coords, node_feat, device=device)
    bundle.name = f"{name} ({loaded} structures, r={radius}Å)"
    return bundle


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Scale 3 — Open Catalyst OC20  (via PyG / ASE)
# ═══════════════════════════════════════════════════════════════════════════
#
# OC20 is the most realistic large-scale test: adsorbate + surface slab,
# ~50–300 atoms, but millions of structures. The atomic numbers, positions,
# and tags (surface/adsorbate/subsurface) are directly relevant to EGNN.
#
# Option A (recommended): use the official OCP data loader
#     pip install ocp-models    # from https://github.com/Open-Catalyst-Project/ocp
#     from ocpmodels.datasets import LmdbDataset
#
# Option B: download a small split directly and parse with ASE
#     pip install ase
#
# Below we implement Option B for self-containedness.
# The OC20 200k val split (~3 GB) is the smallest meaningful sample.
# Download from: https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef/200k/val_20.tar.gz
#
# For the benchmark we don't need to load all 200k — we take the first
# `num_structures` and concatenate them.

OC20_ELEMENTS = list(range(1, 84))   # H to Bi

def _oc20_onehot(z: int) -> torch.Tensor:
    oh = torch.zeros(len(OC20_ELEMENTS) + 1)
    try:
        oh[OC20_ELEMENTS.index(z)] = 1.0
    except ValueError:
        oh[-1] = 1.0
    return oh


def load_oc20_lmdb(
    lmdb_path: str,
    num_structures: int = 500,
    k_neighbors: int = 12,
    device: str = "cuda",
) -> Optional[GraphBundle]:
    """
    Load from an OC20 LMDB file (produced by the OCP repo).
    Each entry is a PyG Data object stored as pickled bytes.

    lmdb_path: path to the directory containing data.lmdb
    """
    try:
        import lmdb, pickle
    except ImportError:
        raise ImportError("pip install lmdb")

    print(f"Loading OC20 from {lmdb_path} …")
    env = lmdb.open(lmdb_path, readonly=True, lock=False,
                    readahead=False, meminit=False)

    all_coords, all_feats = [], []
    all_src, all_dst = [], []
    node_offset = 0

    with env.begin() as txn:
        cursor = txn.cursor()
        for i, (_, val) in enumerate(cursor.iternext(keys=True, values=True)):
            if i >= num_structures:
                break
            data = pickle.loads(val)
            pos  = torch.tensor(data.pos, dtype=torch.float32)
            n    = pos.shape[0]
            z    = data.atomic_numbers if hasattr(data, "atomic_numbers") else data.z
            feat = torch.stack([_oc20_onehot(int(zi)) for zi in z])

            src, dst = coords_to_knn_edges(pos, k=min(k_neighbors, n - 1))
            all_src.append(src + node_offset)
            all_dst.append(dst + node_offset)
            all_coords.append(pos)
            all_feats.append(feat)
            node_offset += n

    if not all_coords:
        return None

    coords    = torch.cat(all_coords).float()
    node_feat = torch.cat(all_feats).float()
    src       = torch.cat(all_src)
    dst       = torch.cat(all_dst)
    N         = coords.shape[0]

    bundle = build_csr(src, dst, N, coords, node_feat, device=device)
    bundle.name = f"OC20 ({num_structures} structures, k={k_neighbors})"
    return bundle


def load_oc20_ase(
    traj_path: str,
    num_structures: int = 500,
    k_neighbors: int = 12,
    device: str = "cuda",
) -> Optional[GraphBundle]:
    """
    Alternative: load from an ASE trajectory file (.traj or .db).
    Useful if you have a local OC20 / OC22 trajectory.

    pip install ase
    """
    try:
        from ase.io import read as ase_read
    except ImportError:
        raise ImportError("pip install ase")

    print(f"Loading OC20 trajectory from {traj_path} …")
    frames = ase_read(traj_path, index=f":{num_structures}")

    all_coords, all_feats = [], []
    all_src, all_dst = [], []
    node_offset = 0

    for atoms in frames:
        pos  = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        n    = pos.shape[0]
        nums = atoms.get_atomic_numbers()
        feat = torch.stack([_oc20_onehot(int(z)) for z in nums])

        src, dst = coords_to_knn_edges(pos, k=min(k_neighbors, n - 1))
        all_src.append(src + node_offset)
        all_dst.append(dst + node_offset)
        all_coords.append(pos)
        all_feats.append(feat)
        node_offset += n

    if not all_coords:
        return None

    coords    = torch.cat(all_coords).float()
    node_feat = torch.cat(all_feats).float()
    src       = torch.cat(all_src)
    dst       = torch.cat(all_dst)
    N         = coords.shape[0]

    bundle = build_csr(src, dst, N, coords, node_feat, device=device)
    bundle.name = f"OC20-ASE ({num_structures} structures, k={k_neighbors})"
    return bundle


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Standard benchmark suite
# ═══════════════════════════════════════════════════════════════════════════

def load_benchmark_suite(
    device: str = "cuda",
    qm9_root: str = "data/qm9",
    pdb_cache: str = "data/pdb",
    oc20_lmdb: Optional[str] = None,   # path to OC20 LMDB dir, if you have it
    oc20_traj:  Optional[str] = None,  # path to ASE .traj, alternative
) -> List[GraphBundle]:
    """
    Load the full benchmark suite in order of increasing graph size.
    Skip any dataset whose data isn't available.
    """
    bundles = []

    # ── QM9: small batch ────────────────────────────────────────────────────
    print("\n[1/4] QM9 — small batch (100 molecules)")
    try:
        b = load_qm9(qm9_root, num_graphs=100, k_neighbors=5, device=device)
        bundles.append(b)
        print(f"  {b.summary()}")
    except Exception as e:
        print(f"  Skipped: {e}")

    # ── QM9: large batch — simulates a real training step ───────────────────
    print("\n[2/4] QM9 — large batch (5000 molecules)")
    try:
        b = load_qm9(qm9_root, num_graphs=5000, k_neighbors=5, device=device)
        bundles.append(b)
        print(f"  {b.summary()}")
    except Exception as e:
        print(f"  Skipped: {e}")

    # ── Proteins: small chains ───────────────────────────────────────────────
    print("\n[3a/4] Proteins — small chains")
    try:
        b = load_protein(PROTEIN_PDB_IDS["small_chain"], pdb_cache,
                         radius=10.0, device=device, name="protein_small")
        if b:
            bundles.append(b)
            print(f"  {b.summary()}")
    except Exception as e:
        print(f"  Skipped: {e}")

    # ── Proteins: medium chains ──────────────────────────────────────────────
    print("\n[3b/4] Proteins — medium chains")
    try:
        b = load_protein(PROTEIN_PDB_IDS["medium_chain"], pdb_cache,
                         radius=10.0, device=device, name="protein_medium")
        if b:
            bundles.append(b)
            print(f"  {b.summary()}")
    except Exception as e:
        print(f"  Skipped: {e}")

    # ── Proteins: large chains + complexes ───────────────────────────────────
    print("\n[3c/4] Proteins — large chains")
    try:
        b = load_protein(PROTEIN_PDB_IDS["large_chain"] +
                         PROTEIN_PDB_IDS["complex"], pdb_cache,
                         radius=10.0, device=device, name="protein_large")
        if b:
            bundles.append(b)
            print(f"  {b.summary()}")
    except Exception as e:
        print(f"  Skipped: {e}")

    # ── OC20 ─────────────────────────────────────────────────────────────────
    print("\n[4/4] OC20")
    if oc20_lmdb:
        try:
            b = load_oc20_lmdb(oc20_lmdb, num_structures=1000,
                               k_neighbors=12, device=device)
            if b:
                bundles.append(b)
                print(f"  {b.summary()}")
        except Exception as e:
            print(f"  Skipped (LMDB): {e}")
    elif oc20_traj:
        try:
            b = load_oc20_ase(oc20_traj, num_structures=1000,
                              k_neighbors=12, device=device)
            if b:
                bundles.append(b)
                print(f"  {b.summary()}")
        except Exception as e:
            print(f"  Skipped (ASE): {e}")
    else:
        print("  Skipped — set oc20_lmdb or oc20_traj to enable")

    return bundles


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Quick smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qm9-root", default="data/qm9")
    parser.add_argument("--pdb-cache", default="data/pdb")
    parser.add_argument("--oc20-lmdb", default=None)
    parser.add_argument("--oc20-traj", default=None)
    args = parser.parse_args()

    bundles = load_benchmark_suite(
        device=args.device,
        qm9_root=args.qm9_root,
        pdb_cache=args.pdb_cache,
        oc20_lmdb=args.oc20_lmdb,
        oc20_traj=args.oc20_traj,
    )

    print(f"\n{'='*80}")
    print(f"Loaded {len(bundles)} dataset(s):")
    print(f"{'='*80}")
    for b in bundles:
        print(f"  {b.summary()}")