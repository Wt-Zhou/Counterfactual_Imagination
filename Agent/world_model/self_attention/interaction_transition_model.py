
from Agent.world_model.self_attention.predmlp import TrajPredMLP
from Agent.world_model.self_attention.selfatten import SelfAttentionLayer


class Interaction_Transition_Model(nn.Module):
    """
    Self_attention GNN with trajectory prediction MLP
    """

    def __init__(self, in_channels, out_channels, global_graph_width=8, traj_pred_mlp_width=8):
        super(Attention_GNN, self).__init__()
        self.polyline_vec_shape = in_channels
        self.self_atten_layer = SelfAttentionLayer(
            self.polyline_vec_shape, global_graph_width, need_scale=False)
        self.traj_pred_mlp = TrajPredMLP(
            global_graph_width, out_channels, traj_pred_mlp_width)

    def forward(self, obs_with_action):
        """
        args: 
            data (Data): [x, y, cluster, edge_index, valid_len]

        """

        valid_lens = obs_with_action.valid_len 
        out = self.self_atten_layer(obs_with_action.x, valid_lens)
        # print("gnn_out",out.squeeze(0)[:, ].squeeze(1))
        pred = self.traj_pred_mlp(out.squeeze(0)[:, ].squeeze(1))
        # print("pred",pred)
        return pred
