import torch
import torch.nn as nn
from torch_geometric.nn import EGNNConv
from .layer.py import EGNN_Triton_Layer

NUM_NODES = 5000
NUM_EDGES = 40000

F_NODE = 32
F_EDGE = 8
HIDDEN_DIM = 64

print(f"Creating graph with {NUM_NODES} nodes and {NUM_EDGES} edges...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.randn((NUM_NODES, F_NODE), device=device)
pos = torch.randn((NUM_NODES, 3), device=device)
edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES), device=device)
edge_attr = torch.randn((NUM_EDGES, F_EDGE), device=device)

pyg_egnn = EGNNConv(
    in_channels=F_NODE,
    out_channels=F_NODE,
    edge_dim=F_EDGE,
    hidden_channels=HIDDEN_DIM
).to(device)

def run_benchmark(model, name, iters=100, warmup=20):
    model.eval()
    
    print(f"\n[{name}] Warming up GPU with {warmup} iterations...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
        torch.cuda.synchronize()
        
        print(f"[{name}] Measuring {iters} iterations...")
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iters):
            _ = model(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
        end_event.record()
        
        torch.cuda.synchronize()
        
        total_ms = start_event.elapsed_time(end_event) 
        ms_per_iter = total_ms / iters
        
        print(f"[{name}]: {ms_per_iter:.3f} ms / iteration")
        return ms_per_iter

time_pyg = run_benchmark(pyg_egnn, "PyG EGNN Oficial", iters=200)

print("\n" + "="*50)
print(f"PyG time: {time_pyg:.3f} ms")
print("="*50)
"""
my_egnn = EGNN_Triton_Layer(
    f_node=F_NODE, f_edge=F_EDGE, 
    msg_hidden_dim=HIDDEN_DIM, msg_out_feat=F_NODE, 
    mov_hidden_dim=HIDDEN_DIM, node_hidden_dim=HIDDEN_DIM,
    rbf_dim=1, rbf_gamma=10.0
).to(device)

class WrapperEGNN(nn.Module):
    def __init__(self, egnn):
        super().__init__()
        self.egnn = egnn
    def forward(self, x, pos, edge_index, edge_attr):
        return self.egnn(x, pos, edge_index, edge_attr)

time_triton = run_benchmark(WrapperEGNN(my_egnn), "EGNN Triton Custom", iters=200)

print(f"Triton time: {time_triton:.3f} ms")
speedup = time_pyg / time_triton
print(f"\nSpeedup: {speedup:.2f}")
"""