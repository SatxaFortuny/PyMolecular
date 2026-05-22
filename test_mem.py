import torch
import time
import matplotlib.pyplot as plt
from pytorch_baseline import pytorch_egnn_baseline
from backward_kernel import egnn_backward_kernel_node_parallel

def launch_triton(data_tuple, kwargs_dict):
    (h, coord, edge_feat, row_ptrs, col_indices, edge_ids, 
     w1_msg, w2_msg_src, w_d, w_e, w_hl, w1_mov, w2_mov, packed_w1_msg,
     grad_out_msg, grad_out_mov) = data_tuple
    
    num_nodes = h.size(0)
    
    # Output gradient accumulators
    grad_h = torch.zeros_like(h)
    grad_coord = torch.zeros_like(coord)
    grad_w1_msg_ptr = torch.zeros_like(packed_w1_msg) # Dummy
    grad_w2_msg_ptr = torch.zeros_like(w_hl) # Dummy
    grad_w1_mov_ptr = torch.zeros_like(w1_mov) # Dummy
    grad_w2_mov_ptr = torch.zeros_like(w2_mov) # Dummy

    grid = (num_nodes,)
    
    egnn_backward_kernel_node_parallel[grid](
        h, coord, edge_feat,
        row_ptrs, col_indices, edge_ids,
        grad_out_msg, grad_out_mov,
        grad_h, grad_coord, grad_w1_msg_ptr, grad_w2_msg_ptr, grad_w1_mov_ptr, grad_w2_mov_ptr,
        packed_w1_msg, w_hl, w1_mov, w2_mov,
        h.stride(0), coord.stride(0), edge_feat.stride(0),
        grad_out_msg.stride(0), grad_out_mov.stride(0),
        num_nodes,
        **kwargs_dict
    )
    return grad_h

def run_memory_benchmark():
    print("\n--- RUNNING MEMORY & OOM BENCHMARK ---")
    print(f"{'Nodes':>10} | {'Edges':>12} | {'PyTorch Peak (MB)':>20} | {'Triton Peak (MB)':>20}")
    print("-" * 70)
    
    node_sizes = [1000, 5000, 10000, 25000, 50000, 100000, 250000]
    avg_degree = 32

    # In BOTH run_memory_benchmark() and benchmark_speed():
    kwargs = {
	    'F_NODE': 32, 'F_EDGE': 16,
	    'HIDDEN_FEATURES': 64, 'OUT_FEATURES': 32, 'HID_FEAT_MOV_MLP': 32,
	    'MSG_BETA': 1.0, 'MOV_BETA': 1.0, 'MOV_ACT_TYPE': 0, 'MSG_ACT_TYPE': 0
    }

    for n in node_sizes:
        data = generate_graph_data(n, avg_degree, kwargs['F_NODE'], kwargs['F_EDGE'], 
                                   kwargs['HIDDEN_FEATURES'], kwargs['OUT_FEATURES'], kwargs['HID_FEAT_MOV_MLP'])
        
        # 1. PyTorch Memory Test
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pt_mem = "OOM"
        try:
            # We must recreate explicit row indices from CSR for standard PyTorch
            row_indices = torch.repeat_interleave(
                torch.arange(n, device='cuda'),
                data[3][1:] - data[3][:-1]
            )
            
            pytorch_egnn_baseline(
                data[0], data[1], data[2], row_indices, data[4],
                data[6], data[7], data[11], data[12], data[9], data[8], data[10],
                data[14], data[15]
            )
            pt_mem = f"{torch.cuda.max_memory_allocated() / (1024**2):.2f}"
        except RuntimeError as e:
            if "Out of memory" in str(e) or "alloc" in str(e).lower():
                pt_mem = "OOM 💥"
            else:
                raise e

        # 2. Triton Memory Test
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tr_mem = "OOM"
        try:
            launch_triton(data, kwargs)
            tr_mem = f"{torch.cuda.max_memory_allocated() / (1024**2):.2f}"
        except RuntimeError as e:
            if "Out of memory" in str(e):
                tr_mem = "OOM 💥"
            else:
                raise e
                
        print(f"{n:>10} | {n * avg_degree:>12} | {pt_mem:>20} | {tr_mem:>20}")



if __name__ == "__main__":
    run_memory_benchmark()

