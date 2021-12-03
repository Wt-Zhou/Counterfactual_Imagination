import torch
import torch.nn as nn
import math
import numpy as np
from Agent.world_model.self_attention.predmlp import TrajPredMLP
from Agent.world_model.self_attention.selfatten import SelfAttentionLayer
from Agent.world_model.agent_model.KinematicBicycleModel.kinematic_model import KinematicBicycleModel, KinematicBicycleModel_Pytorch


class Interaction_Transition_Model(nn.Module):
    """
    Self_attention GNN with trajectory prediction MLP
    """

    def __init__(self, in_channels, out_channels, global_graph_width=8, traj_pred_mlp_width=8):
        super(Interaction_Transition_Model, self).__init__()
        self.polyline_vec_shape = in_channels
        self.self_atten_layer = SelfAttentionLayer(
            self.polyline_vec_shape, global_graph_width)
        self.traj_pred_mlp = TrajPredMLP(
            global_graph_width+2, out_channels, traj_pred_mlp_width)
        
        # Vehicle Model
        self.wheelbase = 2.96
        self.max_steer = np.deg2rad(60)
        self.dt = 0.2
        self.c_r = 0.1
        self.c_a = 0.5
        self.vehicle_model = KinematicBicycleModel(self.wheelbase, self.max_steer, self.dt, self.c_r, self.c_a)
        self.vehicle_model_torch = KinematicBicycleModel_Pytorch(self.wheelbase, self.max_steer, self.dt, self.c_r, self.c_a)

    def forward(self, obs, action_torch):
        """
        args: 
            data (Data): [x, y, cluster, edge_index, valid_len]

        """
        # print("Debug", obs, action_torch)
        out = self.self_atten_layer(obs)
        action_torch = torch.cat((action_torch, action_torch, action_torch, action_torch), dim=0)
        # print("out",out)
        # concat out and control_action
        out_with_action = torch.cat((out, action_torch),dim=1)
        # print("out_with_action",out_with_action)
        pred_action = self.traj_pred_mlp(out_with_action)
        # pred_action = self.traj_pred_mlp(out_with_action.squeeze(0)[:, ].squeeze(1))
        # print("pred_action", pred_action)
        for i in range(len(pred_action)):
            x = obs[i][0].cpu().numpy()
            y = obs[i][1].cpu().numpy()
            yaw = obs[i][4].cpu().numpy()
            v = math.sqrt(obs[i][2].cpu().numpy() ** 2 + obs[i][3].cpu().numpy() ** 2)
            print("before vehicle",x,y,yaw,v)
            x, y, yaw, v, _, _ = self.vehicle_model.kinematic_model(x, y, yaw, v, pred_action[0][0].detach().cpu().numpy(), pred_action[0][1].detach().cpu().numpy())
            print("after vehicle",x,y,yaw,v)
            
            x = obs[i][0]
            y = obs[i][1]
            yaw = obs[i][4]
            v = torch.tensor(math.sqrt(obs[i][2] ** 2 + obs[i][3] ** 2))
            x, y, yaw, v, _, _ = self.vehicle_model_torch.kinematic_model(x, y, yaw, v, pred_action[0][0], pred_action[0][1])
            print("after vehicle",x,y,yaw,v)

        
        return pred_state
