import torch
import pytest

# --- Helper: Convert COO to CSR and track original edge IDs ---
def convert_coo_to_csr(edge_index, num_nodes):
    # edge_index is [2, E] (source, destination)
    # Note: Depending on your convention, EGNN messages flow src -> dst.
    # In Triton, we loop over sources for a given dst. So we sort by dst.
    src, dst = edge_index[0], edge_index[1]
    
    # We need to sort by destination node to build the CSR row pointers
    sorted_indices = torch.argsort(dst)
    sorted_dst = dst[sorted_indices]
    sorted_src = src[sorted_indices]
    
    # edge_ids tracks where the original edge features went
    edge_ids = sorted_indices 
    
    # Build CSR pointers
    row_ptrs = torch.zeros(num_nodes + 1, dtype=torch.int32, device=edge_index.device)
    counts = torch.bincount(sorted_dst, minlength=num_nodes)
    row_ptrs[1:] = torch.cumsum(counts, dim=0)
    
    return row_ptrs, sorted_src.to(torch.int32), edge_ids.to(torch.int32)

# --- Helper: Generate Dummy Graph ---
def generate_dummy_graph(num_nodes, radius, f_node, f_edge, device="cuda"):
    h = torch.randn(num_nodes, f_node, device=device)
    coord = torch.randn(num_nodes, 3, device=device)
    
    # Create fully connected graph, then filter by radius
    dist_matrix = torch.cdist(coord, coord)
    src, dst = torch.where((dist_matrix < radius) & (dist_matrix > 0))
    edge_index = torch.stack([src, dst], dim=0)
    
    num_edges = edge_index.shape[1]
    edge_feat = torch.randn(num_edges, f_edge, device=device)
    
    return h, coord, edge_index, edge_feat

# =====================================================================
# THE PHASE 1 TEST
# =====================================================================
def test_kernel_equivalence():
    device = torch.device("cuda")
    
    # 1. Hyperparameters
    num_nodes = 128
    radius = 5.0 # Keep it standard for the correctness test
    f_node = 32
    f_edge = 8
    hidden_feat = 64
    out_feat = 32
    
    # 2. Generate Data
    h, coord, edge_index, edge_feat = generate_dummy_graph(num_nodes, radius, f_node, f_edge, device)
    row_ptrs, col_indices, edge_ids = convert_coo_to_csr(edge_index, num_nodes)
    
    # 3. Setup Inputs for PyTorch (Clone and set requires_grad)
    h_pt = h.clone().detach().requires_grad_(True)
    coord_pt = coord.clone().detach().requires_grad_(True)
    
    # 4. Setup Inputs for Triton (Clone so it's a completely separate graph)
    h_tr = h.clone().detach().requires_grad_(True)
    
    # Pad coordinates to 4 for Triton memory alignment
    coord_padded = torch.zeros(num_nodes, 4, device=device)
    coord_padded[:, :3] = coord.clone().detach()
    coord_tr = coord_padded.requires_grad_(True)
    
    # 5. Initialize Models
    # (Assuming you have wrapped your PyTorch and Triton code in nn.Modules)
    pt_model = StandardPyTorchEGNN(f_node, f_edge, hidden_feat, out_feat).to(device)
    triton_model = TritonEGNNWrapper(f_node, f_edge, hidden_feat, out_feat).to(device)
    
    # SYNC WEIGHTS: This ensures both models start at the exact same mathematical state
    triton_model.load_state_dict(pt_model.state_dict())
    
    # 6. Run PyTorch Forward & Backward
    out_h_pt, out_x_pt = pt_model(h_pt, coord_pt, edge_index, edge_feat)
    
    # Dummy loss: just sum everything to create a gradient signal
    loss_pt = out_h_pt.sum() + out_x_pt.sum()
    loss_pt.backward()
    
    # 7. Run Triton Forward & Backward
    out_h_tr, out_x_tr = triton_model(h_tr, coord_tr, edge_feat, row_ptrs, col_indices, edge_ids)
    
    loss_tr = out_h_tr.sum() + out_x_tr[:, :3].sum() # Ignore padded 4th dimension
    loss_tr.backward()
    
    # =====================================================================
    # 8. ASSERT EQUIVALENCE
    # Because of atomic operations, we use atol=1e-4 and rtol=1e-3
    # =====================================================================
    print("\n--- Validating Forward Pass ---")
    torch.testing.assert_close(out_h_pt, out_h_tr, atol=1e-4, rtol=1e-3, msg="Output Features Mismatch!")
    torch.testing.assert_close(out_x_pt, out_x_tr[:, :3], atol=1e-4, rtol=1e-3, msg="Output Coordinates Mismatch!")
    print("✅ Forward Pass identical.")
    
    print("--- Validating Input Gradients ---")
    torch.testing.assert_close(h_pt.grad, h_tr.grad, atol=1e-4, rtol=1e-3, msg="Input Feature Gradients Mismatch!")
    torch.testing.assert_close(coord_pt.grad, coord_tr.grad[:, :3], atol=1e-4, rtol=1e-3, msg="Coordinate Gradients Mismatch!")
    print("✅ Input Gradients identical.")
    
    print("--- Validating Weight Gradients ---")
    # Loop through named parameters to check every weight matrix
    for (name_pt, param_pt), (name_tr, param_tr) in zip(pt_model.named_parameters(), triton_model.named_parameters()):
        try:
            torch.testing.assert_close(param_pt.grad, param_tr.grad, atol=1e-4, rtol=1e-3)
            print(f"✅ Grad {name_pt} identical.")
        except AssertionError as e:
            print(f"❌ Grad {name_pt} Mismatch!")
            raise e