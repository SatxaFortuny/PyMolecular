import torch
import triton

class EGNNNodeParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, coord, edge_feat, row_ptrs, col_indices, edge_ids, 
                w1_msg, w2_msg, w1_mov, w2_mov, ...): # Pass your hyperparams too
        
        # 1. Run your standard PyTorch forward pass here!
        # ... (Calculate msg, force, updated coords, updated features)
        
        # 2. Save the inputs and the aggregated message for the backward pass
        ctx.save_for_backward(h, coord, edge_feat, row_ptrs, col_indices, edge_ids,
                              w1_msg, w2_msg, w1_mov, w2_mov)
        
        # Save dimensions and hyperparams (not tensors)
        ctx.num_nodes = h.shape[0]
        ctx.BLOCK_SIZE_NEIGHBORS = 64 # Tune this!
        # ... save other dimensions
        
        return h_updated, coord_updated, m_i_aggregated # Return the hook!

    @staticmethod
    def backward(ctx, grad_h_updated, grad_coord_updated, grad_m_i):
        # 1. Retrieve the saved tensors
        h, coord, edge_feat, row_ptrs, col_indices, edge_ids, \
        w1_msg, w2_msg, w1_mov, w2_mov = ctx.saved_tensors
        
        # 2. Initialize empty gradient tensors (These will be passed as pointers)
        grad_h = torch.zeros_like(h)
        grad_coord = torch.zeros_like(coord)
        grad_w1_msg = torch.zeros_like(w1_msg)
        grad_w2_msg = torch.zeros_like(w2_msg)
        grad_w1_mov = torch.zeros_like(w1_mov)
        grad_w2_mov = torch.zeros_like(w2_mov)
        
        # 3. Define the Grid (1 block per node)
        grid = lambda meta: (ctx.num_nodes,)
        
        # 4. LAUNCH THE TRITON KERNEL
        egnn_backward_kernel_node_parallel[grid](
            # Inputs
            h, coord, edge_feat, row_ptrs, col_indices, edge_ids,
            # Incoming Gradients (grad_m_i is your Option 2 PyTorch Hook!)
            grad_m_i, grad_coord_updated,
            # Outgoing Gradients
            grad_h, grad_coord, 
            grad_w1_msg, grad_w2_msg, grad_w1_mov, grad_w2_mov,
            # Weights
            w1_msg, w2_msg, w1_mov, w2_mov,
            # Strides (e.g., h.stride(0))
            h.stride(0), coord.stride(0), edge_feat.stride(0),
            grad_m_i.stride(0), grad_coord_updated.stride(0),
            # Dimensions & Constexpr
            ctx.num_nodes,
            BLOCK_SIZE_NEIGHBORS=ctx.BLOCK_SIZE_NEIGHBORS,
            # ... Pass the rest of your constexprs (F_NODE, OUT_FEATURES, etc.)
        )
        
        # 5. Return gradients in the exact same order as the forward() arguments
        # Return None for non-tensor arguments (like row_ptrs)
        return grad_h, grad_coord, None, None, None, None, \
               grad_w1_msg, grad_w2_msg, grad_w1_mov, grad_w2_mov, None