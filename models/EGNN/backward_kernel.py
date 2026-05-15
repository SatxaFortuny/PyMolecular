import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_NEIGHBORS': 16}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_NEIGHBORS': 32}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_NEIGHBORS': 32}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE_NEIGHBORS': 64}, num_warps=8, num_stages=1),
    ],
    key=['num_nodes', 'F_NODE', 'HIDDEN_FEATURES']
)
@triton.jit
def egnn_backward_kernel_node_parallel(
    # --- Pointers: Forward Inputs ---
    h_ptr, coord_ptr, edge_feat_ptr,

    # CSR Format
    row_ptrs_ptr,       # (N+1,) — start/end of neighbors for each node
    col_indices_ptr,    # (E,)   — source node IDs in CSR order
    edge_id_ptr,        # (E,)   — original edge index for edge_feat lookup

    # --- Pointers: Incoming Gradients ---
    grad_out_msg_ptr,   # (N, OUT_FEATURES) — dL/d(out_msg), node-indexed
    grad_out_mov_ptr,   # (N, 3)            — dL/d(out_mov), node-indexed

    # --- Pointers: Outgoing Gradients ---
    grad_h_ptr,         # (N, F_NODE)
    grad_coord_ptr,     # (N, 3)
    grad_w1_msg_ptr,    
    grad_w2_msg_ptr,    
    grad_w1_mov_ptr,    
    grad_w2_mov_ptr,    

    # --- Pointers: Weights ---
    w1_msg_ptr, w2_msg_ptr, w1_mov_ptr, w2_mov_ptr,

    # --- Strides ---
    stride_h_n, stride_coord_n, stride_edge_e,
    stride_msg_n, stride_mov_n,

    num_nodes,

    BLOCK_SIZE_NEIGHBORS: tl.constexpr,
    F_NODE: tl.constexpr,          
    F_EDGE: tl.constexpr,
    HIDDEN_FEATURES: tl.constexpr,
    OUT_FEATURES: tl.constexpr,
    HID_FEAT_MOV_MLP: tl.constexpr,
    MSG_BETA: tl.constexpr,
    MOV_BETA: tl.constexpr,
    MOV_ACT_TYPE: tl.constexpr,
    MSG_ACT_TYPE: tl.constexpr,
):
    # =======================================================
    # Section 1. Main node identification and feature load
    # =======================================================
    main_node_idx = tl.program_id(axis=0)
    if main_node_idx >= num_nodes:
        return

    node_feat_sequence = tl.arange(0, F_NODE)
    main_features_addr = h_ptr + (main_node_idx * stride_h_n) + node_feat_sequence
    main_features = tl.load(main_features_addr)  

    coord_sequence = tl.arange(0, 4)
    coord_mask = coord_sequence < 3
    main_coords_addr = coord_ptr + (main_node_idx * stride_coord_n) + coord_sequence
    main_coords = tl.load(main_coords_addr, mask=coord_mask, other=0.0) 

    # Broadcast to 2D for Triton dot products inside the loop
    main_features_2d = tl.broadcast_to(main_features[None, :], (BLOCK_SIZE_NEIGHBORS, F_NODE))
    main_coords_2d   = tl.broadcast_to(main_coords[None, :],   (BLOCK_SIZE_NEIGHBORS, 4))

    # =======================================================
    # Section 1.5: Weight Fetching (Load ONCE into Registers)
    # =======================================================
    vertical_n_sequence  = tl.arange(0, F_NODE)
    vertical_e_sequence  = tl.arange(0, F_EDGE)
    horizontal_sequence  = tl.arange(0, HIDDEN_FEATURES)
    horizontal_out_sequence = tl.arange(0, OUT_FEATURES)

    node_size        = F_NODE * HIDDEN_FEATURES
    node2_offset     = w1_msg_ptr + node_size
    dist_col_offset  = node2_offset + node_size
    edge_offset      = dist_col_offset + HIDDEN_FEATURES

    n1_w_addr   = w1_msg_ptr  + (vertical_n_sequence[:, None] * HIDDEN_FEATURES) + horizontal_sequence[None, :]
    n2_w_addr   = node2_offset + (vertical_n_sequence[:, None] * HIDDEN_FEATURES) + horizontal_sequence[None, :]
    edge_w_addr = edge_offset  + (vertical_e_sequence[:, None] * HIDDEN_FEATURES) + horizontal_sequence[None, :]
    dist_w_addr = dist_col_offset + horizontal_sequence
    w2_addr     = w2_msg_ptr + (horizontal_sequence[:, None] * OUT_FEATURES) + horizontal_out_sequence[None, :]

    true_mask_2d_n = (vertical_n_sequence[:, None] >= 0) & (horizontal_sequence[None, :] >= 0)
    true_mask_2d_e = (vertical_e_sequence[:, None] >= 0) & (horizontal_sequence[None, :] >= 0)
    true_mask_1d   = horizontal_sequence >= 0
    true_mask_w2   = (horizontal_sequence[:, None] >= 0) & (horizontal_out_sequence[None, :] >= 0)

    W_n1 = tl.load(n1_w_addr,   mask=true_mask_2d_n, other=0.0)
    W_n2 = tl.load(n2_w_addr,   mask=true_mask_2d_n, other=0.0)
    W_e  = tl.load(edge_w_addr, mask=true_mask_2d_e, other=0.0)
    W_d  = tl.load(dist_w_addr, mask=true_mask_1d,   other=0.0)
    W_hl = tl.load(w2_addr,     mask=true_mask_w2,   other=0.0)

    vertical_mov_sequence = tl.arange(0, OUT_FEATURES)
    horizontal_mov_sequence = tl.arange(0, HID_FEAT_MOV_MLP)
    
    w1_mov_addr = w1_mov_ptr + (vertical_mov_sequence[:, None] * HID_FEAT_MOV_MLP) + horizontal_mov_sequence[None, :]
    w2_mov_addr = w2_mov_ptr + horizontal_mov_sequence
    true_mask_w1_mov = (vertical_mov_sequence[:, None] >= 0) & (horizontal_mov_sequence[None, :] >= 0)
    true_mask_w2_mov = horizontal_mov_sequence >= 0

    W1_mov = tl.load(w1_mov_addr, mask=true_mask_w1_mov, other=0.0)
    W2_mov = tl.load(w2_mov_addr, mask=true_mask_w2_mov, other=0.0)

    # =======================================================
    # Section 1.8: Backward Variables & Accumulators
    # =======================================================
    grad_m_i_addr = grad_out_msg_ptr + (main_node_idx * stride_msg_n) + horizontal_out_sequence
    grad_m_i = tl.load(grad_m_i_addr, mask=(horizontal_out_sequence>=0), other=0.0) 
    
    grad_x_i_addr = grad_out_mov_ptr + (main_node_idx * stride_coord_n) + coord_sequence
    grad_x_i = tl.load(grad_x_i_addr, mask=coord_mask, other=0.0)
    grad_x_i_2d = tl.broadcast_to(grad_x_i[None, :], (BLOCK_SIZE_NEIGHBORS, 4))

    # Weight accumulators
    acc_grad_W_n1 = tl.zeros([F_NODE, HIDDEN_FEATURES], dtype=tl.float32)
    acc_grad_W_n2 = tl.zeros([F_NODE, HIDDEN_FEATURES], dtype=tl.float32)
    acc_grad_W_e  = tl.zeros([F_EDGE, HIDDEN_FEATURES], dtype=tl.float32)
    acc_grad_W_d  = tl.zeros([HIDDEN_FEATURES], dtype=tl.float32)
    acc_grad_W_hl = tl.zeros([HIDDEN_FEATURES, OUT_FEATURES], dtype=tl.float32)
    acc_grad_W1_mov = tl.zeros([OUT_FEATURES, HID_FEAT_MOV_MLP], dtype=tl.float32)
    acc_grad_W2_mov = tl.zeros([HID_FEAT_MOV_MLP], dtype=tl.float32)

    # Node accumulators
    acc_grad_main_feat = tl.zeros([F_NODE], dtype=tl.float32)
    acc_grad_main_coord = tl.zeros([4], dtype=tl.float32)

    # =======================================================
    # Section 2. Neighbor range from CSR
    # =======================================================
    neighbors_start = tl.load(row_ptrs_ptr + main_node_idx)
    neighbors_end   = tl.load(row_ptrs_ptr + main_node_idx + 1)
    neighbors_sequence  = tl.arange(0, BLOCK_SIZE_NEIGHBORS)
    edge_feat_sequence  = tl.arange(0, F_EDGE)

    for iter in range(neighbors_start, neighbors_end, BLOCK_SIZE_NEIGHBORS):
        neighbors_index = iter + neighbors_sequence
        neighbors_mask  = neighbors_index < neighbors_end

        # =======================================================
        # Section 3. Neighbor node indices, features, coords
        # =======================================================
        neighbors = tl.load(col_indices_ptr + neighbors_index, mask=neighbors_mask, other=0)

        node_feat_addr = h_ptr + (neighbors[:, None] * stride_h_n) + node_feat_sequence[None, :]
        neighbors_features = tl.load(node_feat_addr, mask=neighbors_mask[:, None], other=0.0)

        node_coo_addr  = coord_ptr + (neighbors[:, None] * stride_coord_n) + coord_sequence[None, :]
        coo_mask       = neighbors_mask[:, None] & coord_mask[None, :]
        neighbors_coo  = tl.load(node_coo_addr, mask=coo_mask, other=0.0)

        edge_ids    = tl.load(edge_id_ptr + neighbors_index, mask=neighbors_mask, other=0)
        e_feat_addr = edge_feat_ptr + (edge_ids[:, None] * stride_edge_e) + edge_feat_sequence[None, :]
        e_feat_mask = neighbors_mask[:, None] & (edge_feat_sequence[None, :] < F_EDGE)
        e_feat      = tl.load(e_feat_addr, mask=e_feat_mask, other=0.0)

        # =======================================================
        # Section 4. Forward Geometry
        # =======================================================
        delta   = main_coords_2d - neighbors_coo   
        sq_dist = tl.sum(delta * delta, axis=1)    

        # =======================================================
        # Section 5. Message & Movement MLP Forward Rematerialization
        # =======================================================
        msg, pre_act1, post_act1, pre_act2 = message_mlp_forward_remat(
            main_features_2d, neighbors_features, sq_dist, e_feat,
            W_n1, W_n2, W_e, W_d, W_hl,
            MSG_BETA, MSG_ACT_TYPE
        )

        force, mov_pre_act, mov_post_act = movement_mlp_forward_remat(
            msg, W1_mov, W2_mov,
            MOV_BETA, MOV_ACT_TYPE
        )

        # =======================================================
        # Section 6. The Backward Pass (Pure Math)
        # =======================================================
        
        # 6.1 Coordinates (Path A: The physical movement)
        grad_force = tl.sum(delta * grad_x_i_2d, axis=1)[:, None] 
        grad_delta_geom = grad_x_i_2d * force 

        # 6.2 Movement MLP Backward
        acc_grad_W2_mov += tl.sum(mov_post_act * grad_force, axis=0) 
        grad_mov_post = grad_force * W2_mov[None, :] 
        
        sig_mov_pre = tl.sigmoid(mov_pre_act)
        if MOV_ACT_TYPE == 0:
            grad_mov_pre = grad_mov_post * (sig_mov_pre + mov_pre_act * sig_mov_pre * (1.0 - sig_mov_pre)) * MOV_BETA
        else:
            grad_mov_pre = grad_mov_post * (sig_mov_pre + mov_pre_act * sig_mov_pre * (1.0 - sig_mov_pre))

        acc_grad_W1_mov += tl.dot(tl.trans(msg), grad_mov_pre)
        grad_msg_mov = tl.dot(grad_mov_pre, tl.trans(W1_mov))

        # 6.3 Message MLP Backward
        grad_msg_total = grad_m_i[None, :] + grad_msg_mov

        sig_pre2 = tl.sigmoid(pre_act2)
        if MSG_ACT_TYPE == 0:
            grad_pre_act2 = grad_msg_total * (sig_pre2 + pre_act2 * sig_pre2 * (1.0 - sig_pre2)) * MSG_BETA
        else:
            grad_pre_act2 = grad_msg_total * (sig_pre2 + pre_act2 * sig_pre2 * (1.0 - sig_pre2))
            
        acc_grad_W_hl += tl.dot(tl.trans(post_act1), grad_pre_act2)
        grad_post_act1 = tl.dot(grad_pre_act2, tl.trans(W_hl))

        sig_pre1 = tl.sigmoid(pre_act1)
        if MSG_ACT_TYPE == 0:
            grad_pre_act1 = grad_post_act1 * (sig_pre1 + pre_act1 * sig_pre1 * (1.0 - sig_pre1)) * MSG_BETA
        else:
            grad_pre_act1 = grad_post_act1 * (sig_pre1 + pre_act1 * sig_pre1 * (1.0 - sig_pre1))

        # 6.4 Inputs Backward
        acc_grad_W_n1 += tl.dot(tl.trans(main_features_2d), grad_pre_act1)
        acc_grad_W_n2 += tl.dot(tl.trans(neighbors_features), grad_pre_act1)
        acc_grad_W_e  += tl.dot(tl.trans(e_feat), grad_pre_act1)
        acc_grad_W_d  += tl.sum(grad_pre_act1 * sq_dist[:, None], axis=0)

        # 6.5 Coordinates (Path B: Distance MLP)
        grad_sq_dist = tl.sum(grad_pre_act1 * W_d[None, :], axis=1)
        grad_delta_dist = grad_sq_dist[:, None] * (2.0 * delta)
        
        # Confluence of Coordinate Gradients
        grad_delta_total = grad_delta_geom + grad_delta_dist
        acc_grad_main_coord += tl.sum(grad_delta_total, axis=0)
        grad_neigh_coord = -grad_delta_total

        # Confluence of Feature Gradients
        grad_main_feat_local = tl.dot(grad_pre_act1, tl.trans(W_n1))
        acc_grad_main_feat += tl.sum(grad_main_feat_local, axis=0)
        grad_neigh_feat = tl.dot(grad_pre_act1, tl.trans(W_n2))

        # =======================================================
        # Section 7. Atomic Writes for Neighbors
        # =======================================================
        # Features
        grad_neigh_feat_addr = grad_h_ptr + (neighbors[:, None] * stride_h_n) + node_feat_sequence[None, :]
        tl.atomic_add(grad_neigh_feat_addr, grad_neigh_feat, mask=neighbors_mask[:, None])
        
        # Coordinates
        grad_neigh_coo_addr = grad_coord_ptr + (neighbors[:, None] * stride_coord_n) + coord_sequence[None, :]
        tl.atomic_add(grad_neigh_coo_addr, grad_neigh_coord, mask=coo_mask)

    # =======================================================
    # Section 8. Post-Loop Main Node & Weight Evacuation
    # =======================================================
    # Main Node (Standard Store)
    tl.store(grad_h_ptr + (main_node_idx * stride_h_n) + node_feat_sequence, acc_grad_main_feat)
    tl.store(grad_coord_ptr + (main_node_idx * stride_coord_n) + coord_sequence, acc_grad_main_coord, mask=coord_mask)

    # Weights (Atomic Add)
    tl.atomic_add(n1_w_addr, acc_grad_W_n1, mask=true_mask_2d_n)
    tl.atomic_add(n2_w_addr, acc_grad_W_n2, mask=true_mask_2d_n)
    tl.atomic_add(edge_w_addr, acc_grad_W_e, mask=true_mask_2d_e)
    tl.atomic_add(dist_w_addr, acc_grad_W_d, mask=true_mask_1d)
    tl.atomic_add(w2_addr, acc_grad_W_hl, mask=true_mask_w2)
    tl.atomic_add(w1_mov_addr, acc_grad_W1_mov, mask=true_mask_w1_mov)
    tl.atomic_add(w2_mov_addr, acc_grad_W2_mov, mask=true_mask_w2_mov)


@triton.jit
def swish(Y, BETA: tl.constexpr):
    return Y * tl.sigmoid(BETA * Y)

@triton.jit
def message_mlp_forward_remat(
    n_dst, n_src, sq_dist, edge_feat,
    W_n1, W_n2, W_e, W_d, W_hl, 
    MSG_BETA: tl.constexpr, MSG_ACT_TYPE: tl.constexpr,
):
    dist_contribution = sq_dist[:, None] * W_d[None, :]
    pre_act1 = tl.dot(n_dst, W_n1) + tl.dot(n_src, W_n2) + dist_contribution + tl.dot(edge_feat, W_e)

    if MSG_ACT_TYPE == 0:
        post_act1 = swish(pre_act1, MSG_BETA)
    else:
        post_act1 = pre_act1 * tl.sigmoid(pre_act1)

    pre_act2 = tl.dot(post_act1, W_hl)

    if MSG_ACT_TYPE == 0:
        msg_out = swish(pre_act2, MSG_BETA)
    else:
        msg_out = pre_act2 * tl.sigmoid(pre_act2)

    return msg_out, pre_act1, post_act1, pre_act2

@triton.jit
def movement_mlp_forward_remat(
    msg, W1_mov, W2_mov,
    MOV_BETA: tl.constexpr, MOV_ACT_TYPE: tl.constexpr,
):
    mov_pre_act = tl.dot(msg, W1_mov) 

    if MOV_ACT_TYPE == 0:
        mov_post_act = swish(mov_pre_act, MOV_BETA)
    else:
        mov_post_act = mov_pre_act * tl.sigmoid(mov_pre_act)

    force = tl.sum(mov_post_act * W2_mov[None, :], axis=1)[:, None]
    return force, mov_pre_act, mov_post_act