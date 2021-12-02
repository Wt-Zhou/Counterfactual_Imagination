import gym
import numpy as np
import os
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from Test_Scenarios.TestScenario_Town02 import CarEnv_02_Intersection_fixed
from Agent.world_model.world_model import World_Model
# from Agent.world_model.self_attention.self_atten_world_model import GNN_World_Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = CarEnv_02_Intersection_fixed()

world_model = World_Model(obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape, 
            state_space_dim=env.state_dimension,
            device=device,
            env=env)
world_model.update(env, load_step=20000, train_step=30000)




# world_model = GNN_World_Model(obs_shape=env.observation_space.shape,
#             action_shape=[1], # discrete, 1 dimension!
#             state_space_dim=env.state_dimension,
#             device=device,
#             env=env)
# world_model.train_world_model(env, load_step=20000, train_step=30000)


