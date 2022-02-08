import numpy as np
import math
import torch
import os
import time
from tqdm import tqdm

from Agent.zzz.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from Agent.zzz.controller import Controller
from Agent.zzz.dynamic_map import DynamicMap
from Agent.zzz.actions import LaneAction
from Agent.zzz.actions import TrajectoryAction
from Agent.world_model.world_model import World_Model
from Agent.zzz.tools import *

class Planning_with_World_Model(object):
    def __init__(self, world_model):
        self.world_model = world_model
        self.heads_num = world_model.args.heads_num
        
        # Planner
        self.trajectory_planner = JunctionTrajectoryPlanner()
        self.controller = Controller()
        self.dynamic_map = DynamicMap()
        
    def test(self, env, load_step, test_step=10000):
        model_dir = self.world_model.make_dir(os.path.join(self.world_model.args.work_dir, 'world_model'))
        load_step = self.world_model.load(model_dir, load_step)
        
        episode, episode_reward, done = 0, 0, True
        
        for steps in tqdm(range(1, test_step + 1), unit='steps'):
            if done:
                self.trajectory_planner.clear_buff(clean_csp=False)
                obs = env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                reward = 0   
                    
            obs = np.array(obs)
            
            # Rule-based Planner
            self.dynamic_map.update_map_from_obs(obs, env)
            rule_trajectory, rule_action = self.trajectory_planner.trajectory_update(self.dynamic_map) 
            output_action_list = range(1, len(self.trajectory_planner.all_trajectory)+1)
            control_action_list = []
            trajectory = self.trajectory_planner.trajectory_update_CP(rule_action, rule_trajectory, update=False)

            control_action =  self.controller.get_control(self.dynamic_map,  trajectory.trajectory, trajectory.desired_speed)
            rule_output_action = [control_action.acc, control_action.steering]
            
            for action in range(0, len(self.trajectory_planner.all_trajectory) + 1):
                # Control
                trajectory = self.trajectory_planner.trajectory_update_CP(action, rule_trajectory, update=False)
                control_action =  self.controller.get_control(self.dynamic_map,  trajectory.trajectory, trajectory.desired_speed)
                output_action = [control_action.acc, control_action.steering]
                control_action_list.append(output_action)
                
            worst_case_action, index = self.worst_case_planning(obs, output_action_list, control_action_list)
            trajectory = self.trajectory_planner.trajectory_update_CP(index, rule_trajectory, update=True)

            new_obs, reward, done, info = env.step(worst_case_action)
                       
            episode_reward += reward
            normal_new_obs = (new_obs - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
            normal_obs = (obs - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)

            obs = new_obs
            episode_step += 1
    
    def worst_case_planning(self, obs, action_list, control_action_list):
        reward_action_list = []
        reward_action_list.append(-800) # for action 0, brake trajectory
        # # print("len(self.trajectory_planner.all_trajectory)", len(self.trajectory_planner.all_trajectory))
        # print("action_list",action_list)
        # print("control_action_list",control_action_list)
        for action in action_list:
            reward_action = 10000
            accumulate_reward = 0
            # print('debug', control_action_list[action],action)
            trans_prediction_list = self.world_model.get_trans_prediction(obs, control_action_list[action])
            accumulate_reward = self.get_reward_prediction(trans_prediction_list, action) 
            reward_action = np.array(accumulate_reward)[np.where(accumulate_reward==np.min(accumulate_reward))[0][0]]

            reward_action_list.append(reward_action)
            
        # print("reward_action_list",reward_action_list)
        action = np.array(control_action_list)[np.where(reward_action_list==np.max(reward_action_list))[0]]
        # print("worst_case_action",np.where(reward_action_list==np.max(reward_action_list))[0][0])
        # print("worst_case_action",action)

        return action[0], np.where(reward_action_list==np.max(reward_action_list))[0][0]
    
    def get_reward_prediction(self, obs_list, expected_action_index):
        reward_list = []
        for obs in obs_list[0]:
            if math.sqrt((obs[0] - obs[5]) ** 2 + (obs[1] - obs[6])** 2) < 10:
                reward = -1000
            else:
                reward = -0.1 * self.trajectory_planner.all_trajectory[expected_action_index-1][1]
                # print('reward',reward)
            reward_list.append(reward)
        return reward_list
