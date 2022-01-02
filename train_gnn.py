import gym
import numpy as np
import os
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from Test_Scenarios.TestScenario_Town02 import CarEnv_02_Intersection_fixed
from Test_Scenarios.TestScenario_Town03_cut_in import CarEnv_03_Cut_In
from Agent.world_model.world_model import World_Model
# from Agent.world_model.self_attention.self_atten_world_model import GNN_World_Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = CarEnv_03_Cut_In()

world_model = World_Model(obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape, 
            state_space_dim=env.state_dimension,
            device=device,
            env=env)
world_model.collect_buffer(env, load_step=0, train_step=1000)
# world_model.learn_from_buffer(env, load_step=100, train_step=500000)




