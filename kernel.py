import triton
import triton.language as tl

@triton.jit
def mpnn_forward(
    node_feat_ptr, stride_nf_n,    # node features
    coord_ptr, stride_coord_n,     # x y z coordinates
    row_ptr,                       # index array
    col_ptr,                       # source node
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
    node_index = process_index * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 2. Features, coordinates and csr load
    x_nod_base_addr = node_feat_ptr + process_index * BLOCK_N * stride_nf_n
    c_nod_base_addr = coord_ptr + process_index * BLOCK_N * stride_coord_n
    row_addr = row_ptr + node_index
    source_addr = row_ptr + node_index + 1

    vertical_sequence = tl.arange(0, BLOCK_N)
    horizontal_x_sequence = tl.arange(0, F_NODE)
    horizontal_c_sequence = tl.arange(0, 3)

    x_nod_addr = x_nod_base_addr + (vertical_sequence[:, None] * stride_nf_n) + (horizontal_x_sequence[None, :])
    c_nod_addr = c_nod_base_addr + (vertical_sequence[:, None] * stride_coord_n) + (horizontal_c_sequence[None, :])
    
    # the mask compares the indexes of the tensor, not the contents
    x_nod_mask = (node_index[:, None] < num_nodes) & (horizontal_x_sequence[None, :] < F_NODE)
    c_nod_mask = (node_index[:, None] < num_nodes) & (horizontal_c_sequence[None, :] < 3)
    csr_mask = (node_index < num_nodes)

    X_node = tl.load(x_nod_addr, mask=x_nod_mask, other=0.0)
    C_node = tl.load(c_nod_addr, mask=c_nod_mask, other=0.0)
    row = tl.load(row_addr, mask=csr_mask, other=0)
    source = tl.load(source_addr, mask=csr_mask, other=0)
    
    # 3. Neighbors fetch
    neighbor_sequence = tl.arange(0, BLOCK_NEIGHBORS)
    neighbors_indexes = row[:, None] + neighbor_sequence[None, :]
    neighbors_addrs = col_ptr + neighbors_indexes
    neighbor_mask = (node_index[:, None] < num_nodes) & (neighbors_indexes < source[:, None])
    neighbors = tl.load(neighbors_addrs, mask=neighbor_mask, other=0)

    # 4. Neighbors features and coordinates load
    n_feat_addr = node_feat_ptr + (neighbors[:, :, None] * stride_nf_n) + horizontal_x_sequence[None, None, :]
    n_coord_addr = coord_ptr + (neighbors[:, :, None] * stride_coord_n) + horizontal_c_sequence[None, None, :]

    n_feat_mask = neighbor_mask[:, :, None] & (horizontal_x_sequence[None, None, :] < F_NODE)
    n_coord_mask = neighbor_mask[:, :, None] & (horizontal_c_sequence[None, None, :] < 3)

    n_features = tl.load(neighbors_feat_addr, mask=n_feat_mask, other=0.0)
    n_coordinates = tl.load(neighbors_coord_addr, mask=n_coord_mask, other=0.0)

    # 5. Distance calculator
    rbf_seq = tl.arange(0, RBF_DIM)
    centers = tl.load(rbf_centers_ptr + rbf_seq)

    x_inc = n_coordinates - C_node[:, None, :]
    x_inc = x_inc*x_inc
    x = tl.sum(x_inc, axis=2)
    x = tl.sqrt(x)
    centered_x = x[:, :, None] - centers[None, None, :]
    distances = tl.exp(-rbf_gamma * (centered_x * centered_x))

    # 6. Message obtention
    horizontal_e_sequence = tl.arange(0, F_EDGE)
    e_features_addr = edge_ptr + ()
    
    # 7. Movement obtention
    
    # 8. Final message calculation
    
    # 9. New features obtention and storing
    
