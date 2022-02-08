import gym
import numpy as np
import os
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from Test_Scenarios.TestScenario_Town02 import CarEnv_02_Intersection_fixed
from Test_Scenarios.TestScenario_Town03_cut_in import CarEnv_03_Cut_In
from Agent.world_model.world_model import World_Model
from Agent.Planning_with_world_model import Planning_with_World_Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = CarEnv_03_Cut_In()

world_model = World_Model(device=device, env=env)
planner = Planning_with_World_Model(world_model)
planner.test(env, load_step=0)
# world_model.test(env, load_step=0, test_step=1000)
# world_model.learn_from_buffer(env, load_step=0, train_step=4000)




