import torch
import torch.nn as nn
import triton

class TunableSwish(nn.Module):
    def __init__(self, beta_init=1.0, trainable=True):
        super().__init__()
        self.beta = nn.Parameter(
            torch.tensor([beta_init], dtype=torch.float32), 
            requires_grad=trainable
        )

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class EGNN_Triton_Layer(nn.Module):
    def __init__(
        self,         
        f_node,                   # num features x node
        f_edge,                   # num features x edge
        msg_hidden_dim,           # n features mlp message hidden layer
        msg_out_feat,             # n features mlp message output layer
        mov_hidden_dim,           # n features mlp movement hidden layer
        node_hidden_dim,          # n features mlp node update hidden layer
        rbf_dim,                  # number of gauss bells
        rbf_gamma,                # width of the gaussian bells
        rbf_max_dist=10.0,        # max distance gaussian bells
        custom_centers=None,      # specify gaussian bells positions
        move_centers=False,
        msg_beta=1.0,
        mov_beta=1.0,             # swish function parameter
        node_beta=1.0,
        msg_trainable_beta=False, # mrain msg mlp beta
        mov_trainable_beta=False, # train movement mlp beta
        node_trainable_beta=False # train node mlp beta 
    ):
        super().__init__()
        self.F_NODE = f_node
        self.F_EDGE = f_edge
        self.HIDDEN_FEATURES = msg_hidden_dim
        self.OUT_FEATURES = msg_out_feat
        self.HID_FEAT_MOV_MLP = mov_hidden_dim
        self.RBF_DIM = rbf_dim
        self.MSG_BETA = msg_beta
        self.MOV_BETA = mov_beta
        self.rbf_gamma = rbf_gamma

        self.MOV_ACT_TYPE = 0 if (mov_trainable_beta or mov_beta != 1.0) else 1
        self.MSG_ACT_TYPE = 0 if (msg_trainable_beta or msg_beta != 1.0) else 1

        self.msg_beta = nn.Parameter(torch.tensor([msg_beta], dtype=torch.float32), requires_grad=msg_trainable_beta)
        self.mov_beta = nn.Parameter(torch.tensor([mov_beta], dtype=torch.float32), requires_grad=mov_trainable_beta)

        msg_in_dim = (f_node * 2) + rbf_dim + f_edge
        self.w1_msg = nn.Parameter(torch.randn(msg_in_dim, msg_hidden_dim) / msg_hidden_dim**0.5)
        self.w2_msg = nn.Parameter(torch.randn(msg_hidden_dim, msg_out_feat) / msg_out_feat**0.5)
        
        self.w1_mov = nn.Parameter(torch.randn(msg_out_feat, mov_hidden_dim) / mov_hidden_dim**0.5)
        self.w2_mov = nn.Parameter(torch.randn(mov_hidden_dim, 1) / 1.0)
        
        if custom_centers is not None:
            centers = torch.tensor(custom_centers, dtype=torch.float32)
        else:
            centers = torch.linspace(0.0, rbf_max_dist, rbf_dim)
        if move_centers:
            self.rbf_centers = nn.Parameter(centers)
        else:
            self.rbf_centers = nn.Parameter(centers, requires_grad=False)
        node_mlp_in = f_node + msg_out_feat
        
        node_act = TunableSwish(node_beta, node_trainable_beta)  if (node_trainable_beta or node_beta != 1.0) else nn.SiLU()
            
        self.node_mlp = nn.Sequential(
            nn.Linear(node_mlp_in, node_hidden_dim),
            node_act,
            nn.Linear(node_hidden_dim, f_node)
        )
        
    def forward(self, node_feat, coord, edge_index, edge_feat):
        num_nodes = node_feat.shape[0]
        num_edges = edge_index.shape[1]
        
        out_msg1 = torch.zeros((num_nodes, self.OUT_FEATURES), device=node_feat.device, dtype=node_feat.dtype)
        out_msg2 = torch.zeros_like(out_msg1)
        
        out_mov1 = torch.zeros((num_nodes, 3), device=coord.device, dtype=coord.dtype)
        out_mov2 = torch.zeros_like(out_mov1)
        
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        
        grid = lambda meta: (triton.cdiv(num_edges, meta['BLOCK_E']), )
        
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
            
            self.rbf_centers, 
            self.rbf_gamma,
            
            out_msg1, 
            out_msg2, 
            out_mov1, 
            out_mov2,
            
            num_edges,
            
            F_NODE=self.F_NODE,
            F_EDGE=self.F_EDGE,
            RBF_DIM=self.RBF_DIM,
            HIDDEN_FEATURES=self.HIDDEN_FEATURES,
            OUT_FEATURES=self.OUT_FEATURES,
            HID_FEAT_MOV_MLP=self.HID_FEAT_MOV_MLP,
            MSG_BETA=self.msg_beta.item(),
            MOV_BETA=self.mov_beta.item(),
            MOV_ACT_TYPE=self.MOV_ACT_TYPE,
            MSG_ACT_TYPE=self.MSG_ACT_TYPE
        )
        
        total_mov = out_mov1 + out_mov2
        new_coord = coord + total_mov
        
        total_msg = out_msg1 + out_msg2
        
        node_mlp_input = torch.cat([node_feat, total_msg], dim=-1)
        
        update_feat = self.node_mlp(node_mlp_input)
        
        new_feat = node_feat + update_feat
        
        return new_feat, new_coord