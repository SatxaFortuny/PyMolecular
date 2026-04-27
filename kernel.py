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
    node_feat_ptr, stride_nf_n,    # node features
    coord_ptr, stride_coord_n,     # x y z coordinates
    src_idx_ptr,                   # source array
    dst_idx_ptr,                   # destiny array
    edge_feat_ptr, stride_edge_n,  # edge features
    
    # Weights
    w1_msg_ptr,                     # Message MLP weights layer 1
    w2_msg_ptr,                     # Message MLP weights layer 2
    w1_mov_ptr,                     # Movement MLP weights
    w2_mov_ptr,
    
    # RBF
    rbf_centers_ptr,                # position of the gaussian bells
    rbf_gamma,                      # width of the gaussian bells
    
    # Output
    out_msg1_ptr,                   # partial message for node 1
    out_msg2_ptr,                   # partial message for node 2
    out_mov1_ptr,                   # movement for node 1
    out_mov2_ptr,                   # movement for node 2
    
    num_edges,
    
    BLOCK_E: tl.constexpr,         # number of edges x process
    F_NODE: tl.constexpr,          # num features x node
    F_EDGE: tl.constexpr,          # num features x edge
    RBF_DIM: tl.constexpr,         # number of gauss bells  
    HIDDEN_FEATURES: tl.constexpr, # n features mlp message hidden layer
    OUT_FEATURES: tl.constexpr,    # n features mlp message output layer
    HID_FEAT_MOV_MLP: tl.constexpr,# n features mlp movement hidden layer
    MSG_BETA: tl.constexpr,        # beta value for RBF. Should be 1 for training compatibility
    MOV_BETA: tl.constexpr,
    MOV_ACT_TYPE: tl.constexpr,    # movement tunnable swish function
    MSG_ACT_TYPE: tl.constexpr     # message tunnable swish function
):

    # 1. Identification
    process_index = tl.program_id(axis=0)
    edge_index = process_index * BLOCK_E + tl.arange(0, BLOCK_E)
    
    # 2. COO node fetch
    node1_addr = src_idx_ptr + edge_index
    node2_addr = dst_idx_ptr + edge_index
    
    edges_mask = (edge_index < num_edges)
    
    node1 = tl.load(node1_addr, mask=edges_mask, other=0)
    node2 = tl.load(node2_addr, mask=edges_mask, other=0)
    
    # 3. Node and edges features and coordinates fetch
    n_feat_sequence = tl.arange(0, F_NODE)
    edge_sequence = tl.arange(0, F_EDGE)
    coord_sequence = tl.arange(0, 4)
    
    n1_feat_addr = node_feat_ptr + (node1[:, None] * stride_nf_n) + n_feat_sequence[None, :]
    n2_feat_addr = node_feat_ptr + (node2[:, None] * stride_nf_n) + n_feat_sequence[None, :]
    e_feat_addr = edge_feat_ptr + (edge_index[:, None] * stride_edge_n) + edge_sequence[None, :]
    n1_coord_addr = coord_ptr + (node1[:, None] * stride_coord_n) + coord_sequence[None, :]
    n2_coord_addr = coord_ptr + (node2[:, None] * stride_coord_n) + coord_sequence[None, :]
    
    node_mask = edges_mask[:, None] & (n_feat_sequence[None, :] < F_NODE)
    e_feat_mask = edges_mask[:, None] & (edge_sequence[None, :] < F_EDGE)
    coord_mask = edges_mask[:, None] & (coord_sequence[None, :] < 3)
    
    n1_feat = tl.load(n1_feat_addr, mask=node_mask, other=0.0)
    n2_feat = tl.load(n2_feat_addr, mask=node_mask, other=0.0)
    e_feat = tl.load(e_feat_addr, mask=e_feat_mask, other=0.0)
    n1_coord = tl.load(n1_coord_addr, mask=coord_mask, other=0.0)
    n2_coord = tl.load(n2_coord_addr, mask=coord_mask,  other=0.0)
    
    """
    # 4. RBF distances calculation
    rbf_seq = tl.arange(0, RBF_DIM)
    centers = tl.load(rbf_centers_ptr + rbf_seq)

    x_inc = n1_coord - n2_coord
    x = x_inc * x_inc
    x = tl.sum(x, axis=1)
    x = tl.sqrt(x)
    centered_x = x[:, None] - centers[None, :]
    distances = tl.exp(-rbf_gamma * (centered_x * centered_x))
    """
    x_inc = n1_coord - n2_coord
    sq_dist = tl.sum(x_inc * x_inc, axis=1)
    distances = sq_dist[:, None]
    
    # 5. MLP partial message obtention
    Y1, Y2 = message_mlp(F_NODE, F_EDGE, OUT_FEATURES, HIDDEN_FEATURES, RBF_DIM, 
                n1_feat, n2_feat, distances, e_feat, w1_msg_ptr, w2_msg_ptr, MSG_BETA, MSG_ACT_TYPE)
    
    # 6. MLP edges movement obtention
    mov1, mov2 = movement_mlp(Y1, Y2, w1_mov_ptr, w2_mov_ptr, OUT_FEATURES, HID_FEAT_MOV_MLP, MOV_BETA, MOV_ACT_TYPE)
    new_pos1 = x_inc * mov1
    new_pos2 = -x_inc * mov2
    
    # 7. Messages and distances saving
    horizontal_sequence = tl.arange(0, OUT_FEATURES)
    out_msg1_addr = out_msg1_ptr + (node1[:, None] * OUT_FEATURES) + horizontal_sequence[None, :]
    out_msg2_addr = out_msg2_ptr + (node2[:, None] * OUT_FEATURES) + horizontal_sequence[None, :]
    out_mov1_addr = out_mov1_ptr + (node1[:, None] * stride_coord_n) + coord_sequence[None, :]
    out_mov2_addr = out_mov2_ptr + (node2[:, None] * stride_coord_n) + coord_sequence[None, :]
    
    mov_mask = edges_mask[:, None] & (coord_sequence[None, :] < 3)
    
    tl.atomic_add(out_msg1_addr, Y1, mask=edges_mask[:, None])
    tl.atomic_add(out_msg2_addr, Y2, mask=edges_mask[:, None])
    tl.atomic_add(out_mov1_addr, new_pos1, mask=mov_mask)
    tl.atomic_add(out_mov2_addr, new_pos2, mask=mov_mask)
    
@triton.jit
def message_mlp(
    FEAT_N: tl.constexpr, FEAT_E: tl.constexpr, 
    OUT_FEAT: tl.constexpr, HIDDEN_FEAT: tl.constexpr, 
    RBF_DIM: tl.constexpr, 
    n1, n2, distance, edge_feat, 
    w1_msg_ptr, w2_msg_ptr,
    MSG_BETA: tl.constexpr,
    MSG_ACT_TYPE: tl.constexpr
    ):
    vertical_n_sequence = tl.arange(0, FEAT_N)
    vertical_e_sequence = tl.arange(0, FEAT_E)
    vertical_d_sequence = tl.arange(0, RBF_DIM)
    horizontal_sequence = tl.arange(0, HIDDEN_FEAT)
    
    node_size = FEAT_N * HIDDEN_FEAT
    distance_size = RBF_DIM * HIDDEN_FEAT
    node2_offset = w1_msg_ptr + node_size
    distance_offset = node2_offset + node_size
    edge_offset = distance_offset + distance_size
    
    n1_w_addr = w1_msg_ptr + (vertical_n_sequence[:, None] * HIDDEN_FEAT) + horizontal_sequence[None, :]
    n2_w_addr = node2_offset + (vertical_n_sequence[:, None] * HIDDEN_FEAT) + horizontal_sequence[None, :]
    d_w_addr = distance_offset + (vertical_d_sequence[:, None] * HIDDEN_FEAT) + horizontal_sequence[None, :]
    edge_w_addr = edge_offset + (vertical_e_sequence[:, None] * HIDDEN_FEAT) + horizontal_sequence[None, :]
    
    W_n1 = tl.load(n1_w_addr, other=0.0)
    W_n2 = tl.load(n2_w_addr, other=0.0)
    W_d = tl.load(d_w_addr, other=0.0)
    W_e = tl.load(edge_w_addr, other=0.0)
    
    Y = tl.dot(distance, W_d) + tl.dot(edge_feat, W_e)
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
    
    W_hl = tl.load(hl_w_addr, other=0.0)
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
    
    W1_edge = tl.load(w1_addr, other=0.0)
    
    Y1 = tl.dot(x1, W1_edge)
    Y2 = tl.dot(x2, W1_edge)
    
    if MOV_ACT_TYPE == 0:
        X1 = swish(Y1, MOV_BETA) 
        X2 = swish(Y2, MOV_BETA)
    else:
        X1 = Y1 * tl.sigmoid(Y1)
        X2 = Y2 * tl.sigmoid(Y2)
    
    seq_1 = tl.arange(0, 1)
    w2_addr = w2_mov_ptr + (horizontal_sequence[:, None] * 1) + seq_1[None, :]
    
    W2_edge = tl.load(w2_addr, other=0.0)
    
    return tl.dot(X1, W2_edge), tl.dot(X2, W2_edge)
    
@triton.jit
def swish(Y, BETA: tl.constexpr):
    return Y * tl.sigmoid(BETA * Y)
