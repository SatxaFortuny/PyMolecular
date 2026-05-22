import torch
import time
import matplotlib.pyplot as plt
from pytorch_baseline import pytorch_egnn_baseline
from backward_kernel import egnn_backward_kernel_node_parallel

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['num_nodes'], 
        x_vals=[1000, 5000, 10000, 20000, 40000, 80000], 
        line_arg='provider', 
        line_vals=['pytorch', 'triton'], 
        line_names=['PyTorch Native', 'Triton (Remat)'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='Execution Time (ms)', 
        plot_name='egnn-backward-performance', 
        args={'avg_degree': 32}
    )
)
def benchmark_speed(num_nodes, avg_degree, provider):
    # Hyperparameters mapping to your tl.constexpr
    # In BOTH run_memory_benchmark() and benchmark_speed():
    kwargs = {
	    'F_NODE': 32, 'F_EDGE': 16,
	    'HIDDEN_FEATURES': 64, 'OUT_FEATURES': 32, 'HID_FEAT_MOV_MLP': 32,
	    'MSG_BETA': 1.0, 'MOV_BETA': 1.0, 'MOV_ACT_TYPE': 0, 'MSG_ACT_TYPE': 0
	}
    data = generate_graph_data(num_nodes, avg_degree, kwargs['F_NODE'], kwargs['F_EDGE'], 
                               kwargs['HIDDEN_FEATURES'], kwargs['OUT_FEATURES'], kwargs['HID_FEAT_MOV_MLP'])
    
    row_indices = torch.repeat_interleave(torch.arange(num_nodes, device='cuda'), data[3][1:] - data[3][:-1])

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'pytorch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: pytorch_egnn_baseline(
                data[0], data[1], data[2], row_indices, data[4],
                data[6], data[7], data[11], data[12], data[9], data[8], data[10],
                data[14], data[15]
            ), quantiles=quantiles
        )
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: launch_triton(data, kwargs), quantiles=quantiles
        )
        
    return ms, min_ms, max_ms
    
if __name__ == "__main__":
    print("\n--- RUNNING SPEED BENCHMARK ---")
    benchmark_speed.run(print_data=True, show_plots=True, save_path='.')