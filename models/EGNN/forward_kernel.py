import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_NEIGHBORS': 16, 'TILE_H': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_NEIGHBORS': 32, 'TILE_H': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_NEIGHBORS': 32, 'TILE_H': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_NEIGHBORS': 64, 'TILE_H': 64}, num_warps=8, num_stages=2),
    ],
    key=['num_nodes', 'F_NODE', 'HIDDEN_FEATURES']
)
@triton.jit
def egnn_forward_kernel_node_parallel(
    # --- Pointers: Inputs ---
    h_ptr, coord_ptr, edge_feat_ptr,

    # CSR Format
    row_ptrs_ptr,       # (N+1,) — start/end of neighbors for each node
    col_indices_ptr,    # (E,)   — source node IDs in CSR order
    edge_id_ptr,        # (E,)   — original edge index for edge_feat lookup

    # --- Pointers: Outputs ---
    out_msg_ptr,        # (N, OUT_FEATURES)
    out_mov_ptr,        # (N, 4)

    # --- Pointers: Weights ---
    w1_msg_ptr, w2_msg_ptr, w1_mov_ptr, w2_mov_ptr,

    # --- Strides ---
    stride_h_n, stride_coord_n, stride_edge_e,
    stride_msg_n, stride_mov_n,

    num_nodes,

    BLOCK_SIZE_NEIGHBORS: tl.constexpr,
    TILE_H: tl.constexpr,          # Tiling block size for hidden dimension
    F_NODE: tl.constexpr,
    F_EDGE: tl.constexpr,
    HIDDEN_FEATURES: tl.constexpr,
    OUT_FEATURES: tl.constexpr,
    HID_FEAT_MOV_MLP: tl.constexpr,
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
    # Section 1.5: Weight Fetching (Movement MLP ONLY)
    # The Message MLP weights are loaded lazily inside the loop.
    # =======================================================
    vertical_mov_sequence = tl.arange(0, OUT_FEATURES)
    horizontal_mov_sequence = tl.arange(0, HID_FEAT_MOV_MLP)

    w1_mov_addr = w1_mov_ptr + (vertical_mov_sequence[:, None] * HID_FEAT_MOV_MLP) + horizontal_mov_sequence[None, :]
    w2_mov_addr = w2_mov_ptr + horizontal_mov_sequence

    true_mask_w1_mov = (vertical_mov_sequence[:, None] >= 0) & (horizontal_mov_sequence[None, :] >= 0)
    true_mask_w2_mov = horizontal_mov_sequence >= 0

    W1_mov = tl.load(w1_mov_addr, mask=true_mask_w1_mov, other=0.0)
    W2_mov = tl.load(w2_mov_addr, mask=true_mask_w2_mov, other=0.0)

    # =======================================================
    # Section 1.8: Forward Accumulators
    # =======================================================
    acc_msg = tl.zeros([OUT_FEATURES], dtype=tl.float32)
    acc_mov = tl.zeros([4], dtype=tl.float32)

    # =======================================================
    # Section 2. Neighbor range from CSR
    # =======================================================
    neighbors_start = tl.load(row_ptrs_ptr + main_node_idx)
    neighbors_end   = tl.load(row_ptrs_ptr + main_node_idx + 1)
    neighbors_sequence  = tl.arange(0, BLOCK_SIZE_NEIGHBORS)
    edge_feat_sequence  = tl.arange(0, F_EDGE)
    horizontal_out_sequence = tl.arange(0, OUT_FEATURES)

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
        sq_dist = tl.sum(delta * delta, axis=1)   # scalar (BLOCK_SIZE_NEIGHBORS,)

        # =======================================================
        # Section 5. Message & Movement MLPs
        # =======================================================
        msg = message_mlp_forward_tiled(
            main_features_2d, neighbors_features, sq_dist, e_feat,
            w1_msg_ptr, w2_msg_ptr,
            BLOCK_SIZE_NEIGHBORS, F_NODE, F_EDGE,
            HIDDEN_FEATURES, OUT_FEATURES, TILE_H,
        )

        force = movement_mlp_forward(
            msg, W1_mov, W2_mov,
        )

        # Mask out out-of-bounds neighbors before accumulating
        msg   = tl.where(neighbors_mask[:, None], msg,   0.0)
        force = tl.where(neighbors_mask[:, None], force, 0.0)

        # =======================================================
        # Section 6. Local Accumulation (NO ATOMICS)
        # =======================================================
        acc_msg += tl.sum(msg, axis=0)
        acc_mov += tl.sum(force * delta, axis=0)

    # =======================================================
    # Section 7. Final Global Memory Write
    # =======================================================
    out_msg_addr = out_msg_ptr + (main_node_idx * stride_msg_n) + horizontal_out_sequence
    out_mov_addr = out_mov_ptr + (main_node_idx * stride_mov_n) + coord_sequence

    tl.store(out_msg_addr, acc_msg, mask=(horizontal_out_sequence >= 0))
    tl.store(out_mov_addr, acc_mov, mask=coord_mask)


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def message_mlp_forward_tiled(
    n_dst, n_src, sq_dist, edge_feat,
    w1_msg_ptr, w2_msg_ptr,
    BLOCK_SIZE_NEIGHBORS: tl.constexpr, F_NODE: tl.constexpr, F_EDGE: tl.constexpr,
    HIDDEN_FEATURES: tl.constexpr, OUT_FEATURES: tl.constexpr, TILE_H: tl.constexpr,
):
    pre_act2 = tl.zeros([BLOCK_SIZE_NEIGHBORS, OUT_FEATURES], dtype=tl.float32)

    # Setup base pointer arithmetic offsets
    node_size        = F_NODE * HIDDEN_FEATURES
    node2_offset     = w1_msg_ptr + node_size
    dist_col_offset  = node2_offset + node_size
    edge_offset      = dist_col_offset + HIDDEN_FEATURES

    vertical_n_sequence     = tl.arange(0, F_NODE)
    vertical_e_sequence     = tl.arange(0, F_EDGE)
    horizontal_out_sequence = tl.arange(0, OUT_FEATURES)

    for h_offset in range(0, HIDDEN_FEATURES, TILE_H):
        h_sequence = h_offset + tl.arange(0, TILE_H)

        # 1. Pointers for this specific chunk
        n1_w_addr   = w1_msg_ptr   + (vertical_n_sequence[:, None] * HIDDEN_FEATURES) + h_sequence[None, :]
        n2_w_addr   = node2_offset + (vertical_n_sequence[:, None] * HIDDEN_FEATURES) + h_sequence[None, :]
        edge_w_addr = edge_offset  + (vertical_e_sequence[:, None] * HIDDEN_FEATURES) + h_sequence[None, :]
        dist_w_addr = dist_col_offset + h_sequence
        w2_addr     = w2_msg_ptr + (h_sequence[:, None] * OUT_FEATURES) + horizontal_out_sequence[None, :]

        # 2. Masks (prevents out-of-bounds on non-perfect multiples of TILE_H)
        mask_h  = h_sequence < HIDDEN_FEATURES
        mask_n1 = (vertical_n_sequence[:, None] < F_NODE) & mask_h[None, :]
        mask_n2 = mask_n1
        mask_e  = (vertical_e_sequence[:, None] < F_EDGE) & mask_h[None, :]
        mask_w2 = mask_h[:, None] & (horizontal_out_sequence[None, :] < OUT_FEATURES)

        # 3. Load just the TILE_H chunk
        W_n1_tile = tl.load(n1_w_addr, mask=mask_n1, other=0.0)
        W_n2_tile = tl.load(n2_w_addr, mask=mask_n2, other=0.0)
        W_e_tile  = tl.load(edge_w_addr, mask=mask_e, other=0.0)
        W_d_tile  = tl.load(dist_w_addr, mask=mask_h, other=0.0)
        W_hl_tile = tl.load(w2_addr, mask=mask_w2, other=0.0)

        # 4. Dot products for Layer 1
        dist_contribution = sq_dist[:, None] * W_d_tile[None, :]
        pre_act1_tile = (
            tl.dot(n_dst, W_n1_tile)
            + tl.dot(n_src, W_n2_tile)
            + dist_contribution
            + tl.dot(edge_feat, W_e_tile)
        )

        # 5. SiLU activation + accumulate into Layer 2
        pre_act2 += tl.dot(silu(pre_act1_tile), W_hl_tile)

    # Final SiLU outside the tile loop
    return silu(pre_act2)


@triton.jit
def movement_mlp_forward(
    msg, W1_mov, W2_mov,
):
    force = tl.sum(silu(tl.dot(msg, W1_mov)) * W2_mov[None, :], axis=1)[:, None]
    return force