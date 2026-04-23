import triton
import triton.language as tl

@triton.jit
def mpnn_forward(
    node_feat_ptr, stride_nf_n,    # node features
    coord_ptr, stride_coord_n,     # x y z coordinates
    src_idx_ptr,                   # source array
    dst_idx_ptr,                   # destiny array
    edge_feat_ptr, stride_edge_n,       # edge features
    
    # Weights
    w_msg_ptr,                       # Message MLP weights
    w_mov_ptr,                       # Movement MLP weights
    
    # RBF
    rbf_centers_ptr,
    rbf_gamma,
    
    # Output
    out_feat_ptr, stride_ofeat_n,
    out_coord_ptr, stride_ocoord_n,
    
    num_edges,
    
    BLOCK_E: tl.constexpr,         # number of edges x process
    F_NODE: tl.constexpr,          # num features x node
    F_EDGE: tl.constexpr,          # num features x edge
    RBF_DIM: tl.constexpr,         # number of gauss bells  
    HIDDEN_FEATURES: tl.constexpr,
    OUT_FEATURES: tl.constexpr
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
    
    # 4. RBF distances calculation
    rbf_seq = tl.arange(0, RBF_DIM)
    centers = tl.load(rbf_centers_ptr + rbf_seq)

    x_inc = n1_coord - n2_coord
    x_inc = x_inc*x_inc
    x = tl.sum(x_inc, axis=1)
    x = tl.sqrt(x)
    centered_x = x[:, None] - centers[None, :]
    distances = tl.exp(-rbf_gamma * (centered_x * centered_x))
    
    # 5. MLP partial message obtention
    w_m_sequence = tl.arange(0, OUT_FEATURES)
    n2_sequence = F_NODE + n_feat_sequence
    
    w_m_node_feat_addr = w_msg_ptr + (n_feat_sequence[:, None] * HIDDEN_FEATURES) + w_m_sequence[None, :]
    W_node_message = tl.load(w_m_node_feat_addr, other=0.0)
    
    # 6. MLP edges movement obtention
    
    # 7. Messages and distances saving
    
