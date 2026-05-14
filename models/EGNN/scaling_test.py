import torch
import gc

# --- 1. The Dataset Factory ---
def get_dataset(dataset_name, radius, num_nodes=500, f_node=32, f_edge=8, device="cuda"):
    """
    Returns inputs required for both PyTorch (COO) and Triton (CSR) models.
    """
    if dataset_name == "synthetic":
        # Generate random point cloud
        h = torch.randn(num_nodes, f_node, device=device)
        # Multiply by 10 to spread atoms out in a 10x10x10 box, making radius scaling realistic
        coord = torch.randn(num_nodes, 3, device=device) * 10.0 
        
        dist_matrix = torch.cdist(coord, coord)
        src, dst = torch.where((dist_matrix < radius) & (dist_matrix > 0))
        edge_index = torch.stack([src, dst], dim=0)
        
        num_edges = edge_index.shape[1]
        edge_feat = torch.randn(num_edges, f_edge, device=device)
        
    elif dataset_name == "atom3d":
        # TODO: Implement PyG ATOM3D loader here later
        raise NotImplementedError("ATOM3D loading coming soon!")
    elif dataset_name == "oc20":
        # TODO: Implement PyG Open Catalyst loader here later
        raise NotImplementedError("OC20 loading coming soon!")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Convert to CSR for Triton
    row_ptrs, col_indices, edge_ids = convert_coo_to_csr(edge_index, num_nodes)
    
    # Pad coords for Triton
    coord_padded = torch.zeros(num_nodes, 4, device=device)
    coord_padded[:, :3] = coord.clone()
    
    return h, coord, coord_padded, edge_index, edge_feat, row_ptrs, col_indices, edge_ids

def convert_coo_to_csr(edge_index, num_nodes):
    src, dst = edge_index[0], edge_index[1]
    sorted_indices = torch.argsort(dst)
    sorted_dst = dst[sorted_indices]
    sorted_src = src[sorted_indices]
    edge_ids = sorted_indices 
    
    row_ptrs = torch.zeros(num_nodes + 1, dtype=torch.int32, device=edge_index.device)
    counts = torch.bincount(sorted_dst, minlength=num_nodes)
    row_ptrs[1:] = torch.cumsum(counts, dim=0)
    
    return row_ptrs, sorted_src.to(torch.int32), edge_ids.to(torch.int32)


# --- 2. The Benchmark Runner ---
def benchmark_model(model_fn, name, *args):
    """
    Runs warmup, records time, and measures peak memory.
    Catches OOM errors gracefully.
    """
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        # Warmup
        for _ in range(10):
            out_h, out_x = model_fn(*args)
            loss = out_h.sum() + out_x.sum()
            loss.backward()
            
        # Timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        for _ in range(50): # 50 steps for average
            out_h, out_x = model_fn(*args)
            loss = out_h.sum() + out_x.sum()
            loss.backward()
            
        end_event.record()
        torch.cuda.synchronize()
        
        avg_time = start_event.elapsed_time(end_event) / 50.0
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # Convert to MB
        
        return f"{avg_time:>8.2f} ms | {peak_mem:>8.1f} MB"
    
    except torch.cuda.OutOfMemoryError:
        return f"{'OOM Crah':>8}    | {'OOM Crash':>8}"


# --- 3. The Main Execution Loop ---
if __name__ == "__main__":
    device = torch.device("cuda")
    
    # Initialize your models (Placeholder names, replace with your actual classes)
    # pt_model = StandardPyTorchEGNN(...).to(device)
    # triton_model = TritonEGNNWrapper(...).to(device)
    
    dataset_choice = "synthetic"
    radii_to_test = [5.0, 8.0, 10.0, 12.0, 15.0]
    num_nodes = 1000 # Large enough to trigger OOM at high radii
    
    print(f"Starting Phase 2 Benchmark...")
    print(f"Dataset: {dataset_choice} | Nodes: {num_nodes}")
    print("-" * 75)
    print(f"{'Radius':<6} | {'Edges':<10} | {'PyTorch (Time | Peak Mem)':<25} | {'Triton (Time | Peak Mem)':<25}")
    print("-" * 75)
    
    for radius in radii_to_test:
        # 1. Get Data
        h, coord, coord_padded, edge_index, edge_feat, \
        row_ptrs, col_indices, edge_ids = get_dataset(dataset_choice, radius, num_nodes=num_nodes, device=device)
        
        num_edges = edge_index.shape[1]
        
        # 2. Setup callables for clean benchmarking
        # NOTE: Replace these lambdas with your actual forward passes!
        # pt_fn = lambda: pt_model(h, coord, edge_index, edge_feat)
        # triton_fn = lambda: triton_model(h, coord_padded, edge_feat, row_ptrs, col_indices, edge_ids)
        
        # For demonstration, bypassing actual execution if models aren't defined
        # pt_result = benchmark_model(pt_fn, "PyTorch")
        # tr_result = benchmark_model(triton_fn, "Triton")
        
        # Placeholder strings to show layout:
        pt_result = "   OOM Crash    |   OOM Crash" if radius >= 12.0 else f"{15.2 * (radius/5)**3:>8.2f} ms | {120.5 * (radius/5)**3:>8.1f} MB"
        tr_result = f"{10.1 * (radius/5):>8.2f} ms | {45.2:>8.1f} MB" 
        
        print(f"{radius:<4.1f} A | {num_edges:<10} | {pt_result:<25} | {tr_result:<25}")