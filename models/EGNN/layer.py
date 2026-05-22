import torch
import torch.nn as nn
import triton
from kernel import mpnn_forward


class EGNN_Triton_Layer(nn.Module):
    def __init__(
        self,
        f_node,             # num features x node
        f_edge,             # num features x edge
        msg_hidden_dim,     # n features mlp message hidden layer
        msg_out_feat,       # n features mlp message output layer
        mov_hidden_dim,     # n features mlp movement hidden layer
        node_hidden_dim,    # n features mlp node update hidden layer
    ):
        super().__init__()
        self.F_NODE = f_node
        self.F_EDGE = f_edge
        self.HIDDEN_FEATURES = msg_hidden_dim
        self.OUT_FEATURES = msg_out_feat
        self.HID_FEAT_MOV_MLP = mov_hidden_dim

        # w1_msg layout in the kernel: [W_n1 (F_NODE, H) | W_n2 (F_NODE, H) | W_dist (H,) | W_edge (F_EDGE, H)]
        # The distance is a scalar per edge, so its weight is a single row vector (1 * H),
        # stored flat as (H,) — total param count: 2*F_NODE*H + H + F_EDGE*H
        msg_in_dim = (f_node * 2) + 1 + f_edge   # 1 for the scalar distance weight row
        self.w1_msg = nn.Parameter(torch.randn(msg_in_dim, msg_hidden_dim) / msg_hidden_dim**0.5)
        self.w2_msg = nn.Parameter(torch.randn(msg_hidden_dim, msg_out_feat) / msg_out_feat**0.5)

        self.w1_mov = nn.Parameter(torch.randn(msg_out_feat, mov_hidden_dim) / mov_hidden_dim**0.5)
        self.w2_mov = nn.Parameter(torch.randn(mov_hidden_dim, 1) / 1.0)

        node_mlp_in = f_node + msg_out_feat
        self.node_mlp = nn.Sequential(
            nn.Linear(node_mlp_in, node_hidden_dim),
            nn.SiLU(),
            nn.Linear(node_hidden_dim, f_node)
        )

    def forward(self, node_feat, coord, edge_index, edge_feat):
        num_nodes = node_feat.shape[0]
        num_edges = edge_index.shape[1]

        out_msg1 = torch.zeros((num_nodes, self.OUT_FEATURES), device=node_feat.device, dtype=node_feat.dtype)
        out_msg2 = torch.zeros_like(out_msg1)

        out_mov1 = torch.zeros((num_nodes, coord.shape[1]), device=coord.device, dtype=coord.dtype)
        out_mov2 = torch.zeros_like(out_mov1)

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        ones   = torch.ones(num_edges, dtype=coord.dtype, device=coord.device)
        degree = torch.zeros(num_nodes, dtype=coord.dtype, device=coord.device)

        # Because the kernel processes bi-directionally, a node gets a coordinate
        # update both when it's the source (out_mov1) and destination (out_mov2)
        degree.scatter_add_(0, src_idx, ones)
        degree.scatter_add_(0, dst_idx, ones)

        # Clamp to 1.0 to prevent divide-by-zero for disconnected nodes
        degree = torch.clamp(degree, min=1.0)

        grid = lambda meta: (triton.cdiv(num_edges, meta['BLOCK_E']),)

        mpnn_forward[grid](
            node_feat, node_feat.stride(0),
            coord, coord.stride(0),
            src_idx,
            dst_idx,
            edge_feat, edge_feat.stride(0),

            self.w1_msg,
            self.w2_msg,
            self.w1_mov,
            self.w2_mov,

            out_msg1,
            out_msg2,
            out_mov1,
            out_mov2,

            num_edges,

            F_NODE=self.F_NODE,
            F_EDGE=self.F_EDGE,
            HIDDEN_FEATURES=self.HIDDEN_FEATURES,
            OUT_FEATURES=self.OUT_FEATURES,
            HID_FEAT_MOV_MLP=self.HID_FEAT_MOV_MLP,
        )

        total_mov = out_mov1 + out_mov2
        total_mov = total_mov / degree.unsqueeze(-1)
        new_coord = coord + total_mov

        total_msg = out_msg1 + out_msg2

        node_mlp_input = torch.cat([node_feat, total_msg], dim=-1)
        update_feat = self.node_mlp(node_mlp_input)
        new_feat = node_feat + update_feat

        return new_feat, new_coord