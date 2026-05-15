import torch

def generate_graph_data(num_nodes, avg_degree, F_NODE, F_EDGE, HIDDEN, OUT_FEAT, HID_MOV):
    num_edges = num_nodes * avg_degree
    
    # Create random CSR graph
    row_ptrs = torch.arange(0, num_edges + 1, avg_degree, dtype=torch.int32, device='cuda')
    row_ptrs = torch.cat([row_ptrs, torch.full((num_nodes + 1 - len(row_ptrs),), num_edges, dtype=torch.int32, device='cuda')])
    
    col_indices = torch.randint(0, num_nodes, (num_edges,), dtype=torch.int32, device='cuda')
    edge_ids = torch.arange(0, num_edges, dtype=torch.int32, device='cuda')

    # Node & Edge Features
    h = torch.randn((num_nodes, F_NODE), device='cuda', dtype=torch.float32)
    coord = torch.randn((num_nodes, 4), device='cuda', dtype=torch.float32) # Padded to 4
    coord[:, 3] = 0.0 
    edge_feat = torch.randn((num_edges, F_EDGE), device='cuda', dtype=torch.float32)

    # Weights
    w1_msg = torch.randn((F_NODE, HIDDEN), device='cuda', dtype=torch.float32)
    w2_msg_src = torch.randn((F_NODE, HIDDEN), device='cuda', dtype=torch.float32)
    w_d = torch.randn((HIDDEN,), device='cuda', dtype=torch.float32)
    w_e = torch.randn((F_EDGE, HIDDEN), device='cuda', dtype=torch.float32)
    w_hl = torch.randn((HIDDEN, OUT_FEAT), device='cuda', dtype=torch.float32)
    
    w1_mov = torch.randn((OUT_FEAT, HID_MOV), device='cuda', dtype=torch.float32)
    w2_mov = torch.randn((HID_MOV,), device='cuda', dtype=torch.float32)

    # Packed Weight Buffer for Triton (as expected by your pointer arithmetic)
    node_size = F_NODE * HIDDEN
    packed_w1_msg = torch.zeros(node_size * 2 + HIDDEN + F_EDGE * HIDDEN, device='cuda', dtype=torch.float32)
    
    # Incoming Gradients
    grad_out_msg = torch.randn((num_nodes, OUT_FEAT), device='cuda', dtype=torch.float32)
    grad_out_mov = torch.randn((num_nodes, 3), device='cuda', dtype=torch.float32)

    return (h, coord, edge_feat, row_ptrs, col_indices, edge_ids, 
            w1_msg, w2_msg_src, w_d, w_e, w_hl, w1_mov, w2_mov, packed_w1_msg,
            grad_out_msg, grad_out_mov)