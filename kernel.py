import triton
import triton.language as tl

# =========================================================================== #
#  MESSAGE PASSING FORWARD KERNEL                                             #
# =========================================================================== #
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ATOMS": 32,  "BLOCK_OUTPUT": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ATOMS": 32,  "BLOCK_OUTPUT": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ATOMS": 64,  "BLOCK_OUTPUT": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_ATOMS": 64,  "BLOCK_OUTPUT": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_ATOMS": 64,  "BLOCK_OUTPUT": 256}, num_warps=8, num_stages=3),
    ],
    key=["num_molecules", "actual_A", "actual_X", "actual_O"],
)
@triton.jit
def fused_message_mpnn_forward(
    # Pointers to the tensors
    X_pointer, A_pointer, W_pointer, O_pointer,
           
    # Widths
    molecule_width, atom_width, feature_width,
    mol_adj_width, adj_row_width, adj_col_width,
    w_row_width, w_col_width,
    o_mol_width, o_row_width, o_col_width,
    
    # Actual limits, necessary in order to not process garbage
    num_molecules,
    actual_A, actual_X, actual_O,
    
    
    BLOCK_ATOMS: tl.constexpr,
    BLOCK_FEATURES: tl.constexpr,
    BLOCK_OUTPUT: tl.constexpr,
):

    # SETUP: identification, memory preparation, masks.
    
    molecule_index = tl.program_id(axis=0)
    feature_index = tl.program_id(axis=1)
    if (molecule_index >= num_molecules):
        return
        
    
    # mem = base + id * width    
    X_base = X_pointer + molecule_index * molecule_width
    A_base = A_pointer + molecule_index * mol_adj_width
    W_base = W_pointer
    O_base = O_pointer + molecule_index * o_mol_width

    rows_sequence = tl.arange(0, BLOCK_ATOMS)
    cols_sequence = tl.arange(0, BLOCK_FEATURES)
    features_sequence = tl.arange(0, BLOCK_OUTPUT) + (BLOCK_OUTPUT * feature_index)
    X_pointers = X_base + (rows_sequence[:, None] * atom_width) + (cols_sequence[None, :] * feature_width)
    A_pointers = A_base + (rows_sequence[:, None] * adj_row_width) + (rows_sequence[None, :] * adj_col_width)
    W_pointers = W_base + (cols_sequence[:, None] * w_row_width) + (features_sequence[None, :] * w_col_width)
    O_pointers = O_base + (rows_sequence[:, None] * o_row_width) + (features_sequence[None, :] * o_col_width)
    
    X_mask = (rows_sequence[:, None] < actual_A) & (cols_sequence[None, :] < actual_X)
    A_mask = (rows_sequence[:, None] < actual_A) & (rows_sequence[None, :] < actual_A)
    W_mask = (cols_sequence[:, None] < actual_X) & (features_sequence[None, :] < actual_O)
    O_mask = (rows_sequence[:, None] < actual_A) & (features_sequence[None, :] < actual_O)
    
    X = tl.load(X_pointers, mask=X_mask, other=0.0)
    A = tl.load(A_pointers, mask=A_mask, other=0.0)
    W = tl.load(W_pointers, mask=W_mask, other=0.0)
    
    # Weight multiplication, message passing, activation function
    
    temp = tl.dot(X, W)
    msgs = tl.dot(A, temp)
    O = tl.maximum(0.0, msgs)
    
    # SRAM -> VRAM
    
    tl.store(O_pointers, O, mask=O_mask)
    
    
# =========================================================================== #
#  FORWARD KERNEL                                                             #
# =========================================================================== #
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ATOMS": 32,  "BLOCK_OUTPUT": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ATOMS": 32,  "BLOCK_OUTPUT": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ATOMS": 64,  "BLOCK_OUTPUT": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_ATOMS": 64,  "BLOCK_OUTPUT": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_ATOMS": 64,  "BLOCK_OUTPUT": 256}, num_warps=8, num_stages=3),
    ],
    key=["num_molecules", "actual_A", "actual_X", "actual_O"],
)
@triton.jit
def fused_mpnn_forward(
    # Pointers to the tensors
    X_pointer, W_pointer, O_pointer,
           
    # Widths
    molecule_width, atom_width, feature_width,
    w_row_width, w_col_width,
    o_mol_width, o_row_width, o_col_width,
    
    # Actual limits, necessary in order to not process garbage
    num_molecules,
    actual_A, actual_X, actual_O,
    
    
    BLOCK_ATOMS: tl.constexpr,
    BLOCK_FEATURES: tl.constexpr,
    BLOCK_OUTPUT: tl.constexpr,
):

    # SETUP: identification, memory preparation, masks.
    
    molecule_index = tl.program_id(axis=0)
    feature_index = tl.program_id(axis=1)
    if (molecule_index >= num_molecules):
        return
        
    
    # mem = base + id * width    
    X_base = X_pointer + molecule_index * molecule_width
    W_base = W_pointer
    O_base = O_pointer + molecule_index * o_mol_width

    rows_sequence = tl.arange(0, BLOCK_ATOMS)
    cols_sequence = tl.arange(0, BLOCK_FEATURES)
    features_sequence = tl.arange(0, BLOCK_OUTPUT) + (BLOCK_OUTPUT * feature_index)
    X_pointers = X_base + (rows_sequence[:, None] * atom_width) + (cols_sequence[None, :] * feature_width)
    W_pointers = W_base + (cols_sequence[:, None] * w_row_width) + (features_sequence[None, :] * w_col_width)
    O_pointers = O_base + (rows_sequence[:, None] * o_row_width) + (features_sequence[None, :] * o_col_width)
    
    X_mask = (rows_sequence[:, None] < actual_A) & (cols_sequence[None, :] < actual_X)
    W_mask = (cols_sequence[:, None] < actual_X) & (features_sequence[None, :] < actual_O)
    O_mask = (rows_sequence[:, None] < actual_A) & (features_sequence[None, :] < actual_O)
    
    X = tl.load(X_pointers, mask=X_mask, other=0.0)
    W = tl.load(W_pointers, mask=W_mask, other=0.0)
    
    # Weight multiplication, activation function
    
    temp = tl.dot(X, W)
    O = tl.maximum(0.0, temp)
    
    # SRAM -> VRAM
    
    tl.store(O_pointers, O, mask=O_mask)
    