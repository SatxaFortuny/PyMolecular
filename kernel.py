import triton
import triton.language as tl

@triton.jit
def mpnn_forward(
    node_feat_ptr, stride_nf_n,    # node features
    coord_ptr, stride_coord_n,     # x y z coordinates
    row_ptr,                       # index array
    col_ptr,                       # destiny node
    edge_ptr, stride_edge_n,       # edge features
    
    # Weights
    w_msg_ptr,                       # Message MLP weights
    w_mov_ptr,                       # Movement MLP weights
    w_nod_ptr,                       # Node MLP weights
    
    # RBF
    rbf_centers_ptr,
    rbf_gamma,
    
    # Output
    out_feat_ptr, stride_ofeat_n,
    out_coord_ptr, stride_ocoord_n,
    
    num_nodes,
    
    BLOCK_N: tl.constexpr,         # number of nodes x process
    BLOCK_NEIGHBORS: tl.constexpr, # max neighbors x node
    F_NODE: tl.constexpr,          # num features x node
    F_EDGE: tl.constexpr,          # num features x edge
    RBF_DIM: tl.constexpr,         # number of gauss bells  
):

    # 1. Identification
    process_index = tl.program_id(axis=0)
    node_index = process_index * BLOCK_N * tl.arange(0, BLOCK_N)
    
    # 2. Features and coordinates load
    x_nod_base_addr = node_feat_ptr + process_index * BLOCK_N * stride_nf_n
    c_nod_base_addr = coord_ptr + process_index * BLOCK_N * stride_nf_n
    
    vertical_sequence = tl.arange(0, BLOCK_N)
    horizontal_x_sequence = tl.arange(0, F_NODE)
    horizontal_c_sequence = tl.arange(0, 3)
    
    x_nod_addr = x_nod_base_addr + (vertical_sequence[:, None] * stride_nf_n) + (horizontal_x_sequence[None, :])
    c_nod_addr = c_nod_base_addr + (vertical_sequence[:, None] * stride_coord_n) + (horizontal_c_sequence[None, :])
    
    x_nod_mask = (node_index[:, None] < num_nodes) & (horizontal_x_sequence[None, :] < F_NODE)
    c_nod_mask = (node_index[:, None] < num_nodes) & (horizontal_c_sequence[None, :] < 3)
    
    X_node = tl.load(x_nod_addr, mask=x_nod_mask, other=0.0)
    C_node = tl.load(c_nod_addr, mask=c_nod_mask, other=0.0)
    # 3. Neighbors fetch
    
    # 4. Neighbors features load
    
    # 5. Distance calculator
    
    # 6. Message obtention
    
    # 7. Movement obtention
    
    # 8. Final message calculation
    
    # 9. New features obtention and storing
    