import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_E': 32}, num_warps=2),
        triton.Config({'BLOCK_E': 64}, num_warps=4),
        triton.Config({'BLOCK_E': 128}, num_warps=4),
        triton.Config({'BLOCK_E': 128}, num_warps=8),
        triton.Config({'BLOCK_E': 256}, num_warps=8),
    ],
    key=['num_edges'],
)
@triton.jit
def mpnn_forward(
    node_feat_ptr, stride_nf_n,
    coord_ptr, stride_coord_n,
    src_idx_ptr,
    dst_idx_ptr,
    edge_feat_ptr, stride_edge_n,

    w1_msg_ptr,
    w2_msg_ptr,
    w1_mov_ptr,
    w2_mov_ptr,

    rbf_centers_ptr,
    rbf_gamma,

    out_msg1_ptr,
    out_msg2_ptr,                  # ignored during inference; pass same ptr as out_msg1
    out_mov1_ptr,
    out_mov2_ptr,                  # ignored during inference; pass same ptr as out_mov1

    num_edges,

    BLOCK_E: tl.constexpr,
    F_NODE: tl.constexpr,
    F_EDGE: tl.constexpr,
    RBF_DIM: tl.constexpr,
    HIDDEN_FEATURES: tl.constexpr,
    OUT_FEATURES: tl.constexpr,
    HID_FEAT_MOV_MLP: tl.constexpr,
    MSG_BETA: tl.constexpr,
    MOV_BETA: tl.constexpr,
    MOV_ACT_TYPE: tl.constexpr,
    MSG_ACT_TYPE: tl.constexpr,
    IS_TRAINING: tl.constexpr,     # <-- new: True = keep split buffers for backward
):
    process_index = tl.program_id(axis=0)
    edge_index = process_index * BLOCK_E + tl.arange(0, BLOCK_E)

    node1_addr = src_idx_ptr + edge_index
    node2_addr = dst_idx_ptr + edge_index
    edges_mask = edge_index < num_edges

    node1 = tl.load(node1_addr, mask=edges_mask, other=0)
    node2 = tl.load(node2_addr, mask=edges_mask, other=0)

    n_feat_sequence = tl.arange(0, F_NODE)
    edge_sequence   = tl.arange(0, F_EDGE)
    coord_sequence  = tl.arange(0, 4)

    n1_feat_addr  = node_feat_ptr  + (node1[:, None] * stride_nf_n)    + n_feat_sequence[None, :]
    n2_feat_addr  = node_feat_ptr  + (node2[:, None] * stride_nf_n)    + n_feat_sequence[None, :]
    e_feat_addr   = edge_feat_ptr  + (edge_index[:, None] * stride_edge_n) + edge_sequence[None, :]
    n1_coord_addr = coord_ptr      + (node1[:, None] * stride_coord_n) + coord_sequence[None, :]
    n2_coord_addr = coord_ptr      + (node2[:, None] * stride_coord_n) + coord_sequence[None, :]

    node_mask   = edges_mask[:, None] & (n_feat_sequence[None, :] < F_NODE)
    e_feat_mask = edges_mask[:, None] & (edge_sequence[None, :]   < F_EDGE)
    coord_mask  = edges_mask[:, None] & (coord_sequence[None, :]  < 3)

    n1_feat  = tl.load(n1_feat_addr,  mask=node_mask,   other=0.0)
    n2_feat  = tl.load(n2_feat_addr,  mask=node_mask,   other=0.0)
    e_feat   = tl.load(e_feat_addr,   mask=e_feat_mask, other=0.0)
    n1_coord = tl.load(n1_coord_addr, mask=coord_mask,  other=0.0)
    n2_coord = tl.load(n2_coord_addr, mask=coord_mask,  other=0.0)

    x_inc   = n1_coord - n2_coord
    sq_dist = tl.sum(x_inc * x_inc, axis=1)

    Y1, Y2 = message_mlp(F_NODE, F_EDGE, OUT_FEATURES, HIDDEN_FEATURES,
                          n1_feat, n2_feat, sq_dist, e_feat,
                          w1_msg_ptr, w2_msg_ptr, MSG_BETA, MSG_ACT_TYPE)

    mov1, mov2 = movement_mlp(Y1, Y2, w1_mov_ptr, w2_mov_ptr,
                               OUT_FEATURES, HID_FEAT_MOV_MLP, MOV_BETA, MOV_ACT_TYPE)
    new_pos1 =  x_inc * mov1
    new_pos2 = -x_inc * mov2

    # ── Scatter ──────────────────────────────────────────────────────────────
    horizontal_sequence = tl.arange(0, OUT_FEATURES)
    mov_mask = edges_mask[:, None] & (coord_sequence[None, :] < 3)

    out_msg1_addr = out_msg1_ptr + (node1[:, None] * OUT_FEATURES)    + horizontal_sequence[None, :]
    out_mov1_addr = out_mov1_ptr + (node1[:, None] * stride_coord_n)  + coord_sequence[None, :]

    tl.atomic_add(out_msg1_addr, Y1,       mask=edges_mask[:, None])
    tl.atomic_add(out_mov1_addr, new_pos1, mask=mov_mask)

    if IS_TRAINING:
        # Training: keep dst contributions in a separate buffer so the
        # Python-side autograd graph can treat them independently.
        out_msg2_addr = out_msg2_ptr + (node2[:, None] * OUT_FEATURES)   + horizontal_sequence[None, :]
        out_mov2_addr = out_mov2_ptr + (node2[:, None] * stride_coord_n) + coord_sequence[None, :]
        tl.atomic_add(out_msg2_addr, Y2,       mask=edges_mask[:, None])
        tl.atomic_add(out_mov2_addr, new_pos2, mask=mov_mask)
    else:
        # Inference: merge dst contributions directly into the same buffer.
        # One fewer allocation; Triton dead-code-eliminates the training branch.
        out_msg1_node2_addr = out_msg1_ptr + (node2[:, None] * OUT_FEATURES)   + horizontal_sequence[None, :]
        out_mov1_node2_addr = out_mov1_ptr + (node2[:, None] * stride_coord_n) + coord_sequence[None, :]
        tl.atomic_add(out_msg1_node2_addr, Y2,       mask=edges_mask[:, None])
        tl.atomic_add(out_mov1_node2_addr, new_pos2, mask=mov_mask)

    
@triton.jit
def message_mlp(
    FEAT_N: tl.constexpr, FEAT_E: tl.constexpr, 
    OUT_FEAT: tl.constexpr, HIDDEN_FEAT: tl.constexpr, 
    n1, n2, sq_dist, edge_feat,      # sq_dist is now (BLOCK_E,) scalar per edge
    w1_msg_ptr, w2_msg_ptr,
    MSG_BETA: tl.constexpr,
    MSG_ACT_TYPE: tl.constexpr
    ):
    vertical_n_sequence = tl.arange(0, FEAT_N)
    vertical_e_sequence = tl.arange(0, FEAT_E)
    horizontal_sequence = tl.arange(0, HIDDEN_FEAT)
    
    # w1_msg layout: [W_n1 | W_n2 | W_dist_col | W_edge]
    # W_dist is stored as a single column (HIDDEN_FEAT,) — one scalar input
    node_size = FEAT_N * HIDDEN_FEAT
    node2_offset = w1_msg_ptr + node_size
    dist_col_offset = node2_offset + node_size        # (HIDDEN_FEAT,) vector
    edge_offset = dist_col_offset + HIDDEN_FEAT       # (FEAT_E, HIDDEN_FEAT)

    n1_w_addr   = w1_msg_ptr   + (vertical_n_sequence[:, None] * HIDDEN_FEAT) + horizontal_sequence[None, :]
    n2_w_addr   = node2_offset + (vertical_n_sequence[:, None] * HIDDEN_FEAT) + horizontal_sequence[None, :]
    edge_w_addr = edge_offset  + (vertical_e_sequence[:, None] * HIDDEN_FEAT) + horizontal_sequence[None, :]
    dist_w_addr = dist_col_offset + horizontal_sequence   # (HIDDEN_FEAT,)

    # Masks are all-true (weights fully in-bounds), but Triton requires mask when other= is used
    true_mask_2d_n    = (vertical_n_sequence[:, None] >= 0) & (horizontal_sequence[None, :] >= 0)
    true_mask_2d_e    = (vertical_e_sequence[:, None] >= 0) & (horizontal_sequence[None, :] >= 0)
    true_mask_1d      = horizontal_sequence >= 0

    W_n1 = tl.load(n1_w_addr,   mask=true_mask_2d_n, other=0.0)
    W_n2 = tl.load(n2_w_addr,   mask=true_mask_2d_n, other=0.0)
    W_e  = tl.load(edge_w_addr, mask=true_mask_2d_e, other=0.0)
    W_d  = tl.load(dist_w_addr, mask=true_mask_1d,   other=0.0)  # (HIDDEN_FEAT,)

    # Distance contribution: scalar * row-vector, broadcast over BLOCK_E
    dist_contribution = sq_dist[:, None] * W_d[None, :]  # (BLOCK_E, HIDDEN_FEAT)

    Y = dist_contribution + tl.dot(edge_feat, W_e)
    Y1 = tl.dot(n1, W_n1) + tl.dot(n2, W_n2) + Y
    Y2 = tl.dot(n2, W_n1) + tl.dot(n1, W_n2) + Y
    
    if MSG_ACT_TYPE == 0:
        X1 = swish(Y1, MSG_BETA) 
        X2 = swish(Y2, MSG_BETA)
    else:
        X1 = Y1 * tl.sigmoid(Y1)
        X2 = Y2 * tl.sigmoid(Y2)
    
    horizontal_h_sequence = tl.arange(0, OUT_FEAT)
    hl_w_addr = w2_msg_ptr + (horizontal_sequence[:, None] * OUT_FEAT) + horizontal_h_sequence[None, :]
    true_mask_hidden = (horizontal_sequence[:, None] >= 0) & (horizontal_h_sequence[None, :] >= 0)

    W_hl = tl.load(hl_w_addr, mask=true_mask_hidden, other=0.0)
    Y1 = tl.dot(X1, W_hl)
    Y2 = tl.dot(X2, W_hl)
    
    if MSG_ACT_TYPE == 0:
        X1 = swish(Y1, MSG_BETA) 
        X2 = swish(Y2, MSG_BETA)
    else:
        X1 = Y1 * tl.sigmoid(Y1)
        X2 = Y2 * tl.sigmoid(Y2)
    return X1, X2

    
@triton.jit
def movement_mlp(
    x1, x2,
    w1_mov_ptr, w2_mov_ptr,
    IN_FEAT: tl.constexpr,
    HIDDEN_FEAT: tl.constexpr,
    MOV_BETA: tl.constexpr,
    MOV_ACT_TYPE: tl.constexpr
):
    vertical_sequence = tl.arange(0, IN_FEAT)
    horizontal_sequence = tl.arange(0, HIDDEN_FEAT)
    
    w1_addr = w1_mov_ptr + (vertical_sequence[:, None] * HIDDEN_FEAT) + horizontal_sequence[None, :]
    true_mask_w1 = (vertical_sequence[:, None] >= 0) & (horizontal_sequence[None, :] >= 0)

    W1_edge = tl.load(w1_addr, mask=true_mask_w1, other=0.0)
    
    Y1 = tl.dot(x1, W1_edge)
    Y2 = tl.dot(x2, W1_edge)
    
    if MOV_ACT_TYPE == 0:
        X1 = swish(Y1, MOV_BETA) 
        X2 = swish(Y2, MOV_BETA)
    else:
        X1 = Y1 * tl.sigmoid(Y1)
        X2 = Y2 * tl.sigmoid(Y2)
    
    # w2_mov is (HIDDEN_FEAT, 1) — tl.dot can't handle dim=1 output.
    # Load as a (HIDDEN_FEAT,) vector and use elementwise sum instead.
    true_mask_w2 = horizontal_sequence >= 0
    w2_addr = w2_mov_ptr + horizontal_sequence
    W2_edge = tl.load(w2_addr, mask=true_mask_w2, other=0.0)  # (HIDDEN_FEAT,)

    # Dot product manually: sum(X * W2) over hidden dim → (BLOCK_E, 1)
    out1 = tl.sum(X1 * W2_edge[None, :], axis=1)[:, None]  # (BLOCK_E, 1)
    out2 = tl.sum(X2 * W2_edge[None, :], axis=1)[:, None]  # (BLOCK_E, 1)

    return out1, out2

    
@triton.jit
def swish(Y, BETA: tl.constexpr):
    return Y * tl.sigmoid(BETA * Y)