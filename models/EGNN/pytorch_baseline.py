import torch

def swish_torch(x, beta=1.0):
    return x * torch.sigmoid(beta * x)

def pytorch_egnn_baseline(
    h, coord, edge_feat, row_indices, col_indices,
    w1_msg, w2_msg, w1_mov, w2_mov, w_e, w_d, w_hl,
    grad_out_msg, grad_out_mov,
    msg_beta=1.0, mov_beta=1.0
):
    # Enable gradients
    h.requires_grad_(True)
    coord.requires_grad_(True)
    w1_msg.requires_grad_(True)
    
    # --- FORWARD PASS (PyTorch saves all this in memory for backward) ---
    h_i = h[row_indices]
    h_j = h[col_indices]
    coord_i = coord[row_indices]
    coord_j = coord[col_indices]

    delta = coord_i - coord_j
    sq_dist = torch.sum(delta * delta, dim=1, keepdim=True)

    # Message MLP
    pre_act1 = (h_i @ w1_msg) + (h_j @ w2_msg) + (sq_dist * w_d) + (edge_feat @ w_e)
    post_act1 = swish_torch(pre_act1, msg_beta)
    pre_act2 = post_act1 @ w_hl
    msg_out = swish_torch(pre_act2, msg_beta)

    # Movement MLP
    mov_pre_act = msg_out @ w1_mov
    mov_post_act = swish_torch(mov_pre_act, mov_beta)
    force = torch.sum(mov_post_act * w2_mov, dim=1, keepdim=True)

    # Dummy Aggregation to Nodes to match kernel outputs
    num_nodes = h.size(0)
    out_msg_aggr = torch.zeros(num_nodes, msg_out.size(1), device=h.device)
    out_msg_aggr.scatter_add_(0, row_indices.unsqueeze(-1).expand_as(msg_out), msg_out)

    out_mov_aggr = torch.zeros(num_nodes, 3, device=h.device)
    delta_force = delta[:, :3] * force
    out_mov_aggr.scatter_add_(0, row_indices.unsqueeze(-1).expand_as(delta_force), delta_force)

    # --- BACKWARD PASS ---
    loss = (out_msg_aggr * grad_out_msg).sum() + (out_mov_aggr * grad_out_mov).sum()
    loss.backward()

    return h.grad, coord.grad