# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import argparse
import torch
import os
import math
import copy
import random
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(profile='short')
from Agent.world_model.single_transition_model import make_transition_model
from Agent.world_model.self_attention.interaction_transition_model import Interaction_Transition_Model

from Agent.zzz.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from Agent.zzz.controller import Controller
from Agent.zzz.dynamic_map import DynamicMap
from Agent.zzz.actions import LaneAction
from Agent.world_model.agent_model.KinematicBicycleModel.kinematic_model import KinematicBicycleModel

class World_Model(object):
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        state_space_dim,
        env,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.001,
        ego_transition_learn=False,
        transition_lr=0.001,
        transition_weight_lambda=0.0,
        
    ):
        
        self.device = device
        self.state_space_dim = state_space_dim
        self.args = self.parse_args()
        self.env = env
        self.replay_buffer = World_Buffer(obs_shape=env.observation_space.shape,
            action_shape=action_shape, # discrete, 1 dimension!
            capacity= self.args.replay_buffer_capacity,
            batch_size= self.args.batch_size,
            device=device)
        
        # Ego Vehicle Transition
        self.ego_transition_learn = ego_transition_learn
        if self.ego_transition_learn:
            self.init_NN_ego_transition_model()
        else:
            # Vehicle Dynamics Model Parameter
            self.wheelbase = 2.96
            self.max_steer = np.deg2rad(60)
            self.dt = env.dt
            self.c_r = 0.0
            self.c_a = 0.0
            self.ego_transition_model = KinematicBicycleModel(self.wheelbase, self.max_steer, self.dt, self.c_r, self.c_a)
        
        # Env Agent Transition
        self.obs_scale = self.args.obs_scale
        self.throttle_scale = self.args.throttle_scale
        self.ensemble_env_transition_model = []
        self.ensemble_env_trans_optimizer = []
        for i in range(self.args.heads_num):
            env_transition_model = Interaction_Transition_Model(5, 2, self.obs_scale).to(self.device)
            env_transition_model.apply(self.weight_init)
            self.ensemble_env_transition_model.append(env_transition_model)
            self.ensemble_env_trans_optimizer.append(torch.optim.Adam(env_transition_model.parameters(), lr=transition_lr))
            env_transition_model.train()
        # Planner
        self.trajectory_planner = JunctionTrajectoryPlanner()
        self.controller = Controller()
        self.dynamic_map = DynamicMap()
        self.target_speed = 30/3.6 
        
        self.train()
    
    def parse_args(self):
        parser = argparse.ArgumentParser()
        # environment
        parser.add_argument('--domain_name', default='carla')
        parser.add_argument('--task_name', default='run')
        # replay buffer
        parser.add_argument('--replay_buffer_capacity', default=1000000, type=int)
        # train
        parser.add_argument('--agent', default='bisim', type=str, choices=['baseline', 'bisim', 'deepmdp'])
        parser.add_argument('--init_steps', default=1, type=int)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--hidden_dim', default=256, type=int)
        # eval
        parser.add_argument('--eval_freq', default=100, type=int)  
        parser.add_argument('--num_eval_episodes', default=20, type=int)
        parser.add_argument('--discount', default=0.99, type=float)
        parser.add_argument('--init_temperature', default=0.01, type=float)
        # misc
        parser.add_argument('--seed', default=1, type=int)
        parser.add_argument('--work_dir', default='.', type=str)
        parser.add_argument('--save_model', default=True, action='store_true')
        parser.add_argument('--save_buffer', default=True, action='store_true')
        parser.add_argument('--transition_model_type', default='probabilistic', type=str, choices=['', 'deterministic', 'probabilistic', 'ensemble'])
        parser.add_argument('--port', default=2000, type=int)
        
        # ensemble
        parser.add_argument('--heads_num', default=10, type=int)
        parser.add_argument('--obs_scale', default=10, type=int)
        parser.add_argument('--throttle_scale', default=5, type=int)
        args = parser.parse_args()
        return args
    
    def train(self, training=True):
        self.training = training

    def init_NN_ego_transition_model(self, ego_state_dim = 5):
        transition_model_type = 'probabilistic'
        self.ego_transition_model = make_transition_model(
            transition_model_type, ego_state_dim, action_shape
        ).to(self.device)
        self.ego_transition_optimizer = torch.optim.Adam(
            list(self.ego_transition_model.parameters()),
            lr=transition_lr,
            weight_decay=transition_weight_lambda
        )
        
    def update_ego_transition_model(self, replay_buffer, step):
        
        obs, action, _, reward, next_obs, not_done = replay_buffer.sample()
        # To ego center:(x,y,yaw)
        next_obs[0][0] -= obs[0][0] 
        next_obs[0][1] -= obs[0][1]
        next_obs[0][4] -= obs[0][4]
        next_obs[0][0] *= 10
        next_obs[0][1] *= 10
        next_obs[0][4] *= 10
        
        obs[0][0] = 0
        obs[0][1] = 0
        obs[0][4] = 0
        # print("deebug2",obs,next_obs)

        ego_obs = torch.take(obs, torch.tensor([[0,1,2,3,4]]).to(device=self.device))
        ego_obs_with_action = torch.cat([ego_obs, action], dim=1)
        next_ego_obs = torch.take(next_obs, torch.tensor([[0,1,2,3,4]]).to(device=self.device))

        pred_next_latent_mu, pred_next_latent_sigma = self.ego_transition_model(ego_obs_with_action)
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        diff = (pred_next_latent_mu - next_ego_obs.detach()) / pred_next_latent_sigma
        loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
        print('pred_next_latent_mu', pred_next_latent_mu, next_ego_obs.detach())
        print('ego_transition_loss2', loss, step)
        self.ego_transition_optimizer.zero_grad()
        loss.backward()
        self.ego_transition_optimizer.step()

    def update_env_transition_model_with_state(self, replay_buffer):
        num = random.randint(0,999)
        obs, action, _, reward, next_obs, not_done = replay_buffer.get(num)
        obs /= self.obs_scale
        next_obs /= self.obs_scale
        for i in range(1,2):
            obs[0][0+i*5] -= obs[0][0] 
            obs[0][1+i*5] -= obs[0][1] 
            next_obs[0][0+i*5] -= obs[0][0] 
            next_obs[0][1+i*5] -= obs[0][1] 
            
            
        next_obs[0][0] -= obs[0][0] 
        next_obs[0][1] -= obs[0][1] 
        obs[0][0] = 0
        obs[0][1] = 0

        obs_torch = torch.reshape(obs, [2,5])
        action_torch = torch.reshape(action, [1,2])

        y = torch.reshape(next_obs, [2,5])
        # y = y[torch.arange(y.size(0))!=0] #exclude ego av

        print("y",y)

        for i in range(self.args.heads_num):
            predict_state = self.ensemble_env_transition_model[i](obs_torch, action_torch)
            print("predict_state",predict_state)

            env_trans_loss = F.mse_loss(y, predict_state)
            print("------------env_trans_loss",env_trans_loss)
                         
            self.ensemble_env_trans_optimizer[i].zero_grad()
            env_trans_loss.backward()
            self.ensemble_env_trans_optimizer[i].step()
        
    def update_env_transition_model_with_action(self, replay_buffer):
        num = random.randint(0,999)
        obs, action, _, reward, next_obs, not_done = replay_buffer.get(num)
        obs /= self.obs_scale
        next_obs /= self.obs_scale
        for i in range(1,2):
            obs[0][0+i*5] -= obs[0][0] 
            obs[0][1+i*5] -= obs[0][1] 
            next_obs[0][0+i*5] -= obs[0][0] 
            next_obs[0][1+i*5] -= obs[0][1] 
            
            
        next_obs[0][0] -= obs[0][0] 
        next_obs[0][1] -= obs[0][1] 
        obs[0][0] = 0
        obs[0][1] = 0

        obs_torch = torch.reshape(obs, [2,5])
        action_torch = torch.reshape(action, [1,2])
        print("obs",obs_torch,action_torch)

        y = torch.reshape(next_obs, [2,5])
        # y = y[torch.arange(y.size(0))!=0] #exclude ego av
        
        expected_action = []
        for i in range(len(y)):
            x1 = torch.mul(obs_torch[i][0], self.obs_scale)
            y1 = torch.mul(obs_torch[i][1], self.obs_scale)
            yaw1 = torch.mul(obs_torch[i][4], self.obs_scale)
            v1 = torch.tensor(math.sqrt(torch.mul(obs_torch[i][2], self.obs_scale) ** 2 + torch.mul(obs_torch[i][3], self.obs_scale) ** 2))
            x2 = torch.mul(y[i][0], self.obs_scale)
            y2 = torch.mul(y[i][1], self.obs_scale)
            yaw2 = torch.mul(y[i][4], self.obs_scale)
            v2 = torch.tensor(math.sqrt(torch.mul(y[i][2], self.obs_scale) ** 2 + torch.mul(y[i][3], self.obs_scale) ** 2))
            throttle, delta = self.ensemble_env_transition_model[0].vehicle_model_torch.calculate_a_from_data(x1, y1, yaw1, v1, x2, y2, yaw2, v2)
            tensor_list = [torch.div(throttle,self.throttle_scale).to(device=self.device), delta]
            action = torch.stack((tensor_list))
            expected_action.append(action)
        expected_action = torch.stack(expected_action).unsqueeze(0)
        print("expected_action",expected_action)
        for i in range(self.args.heads_num):
            predict_action = self.ensemble_env_transition_model[i](obs_torch, action_torch)
            # predict_action = predict_action.squeze(0)
            # predict_action[0] = predict_action[0][torch.arange(predict_action[0].size(0))!=0] #exclude ego av
            print("predict_action",predict_action)

            env_trans_loss = F.mse_loss(expected_action, predict_action)
            print("------------env_trans_loss",env_trans_loss)
            # with open("gnn_loss.txt", 'a') as fw: 
            #     fw.write(str(env_trans_loss)) 
            #     fw.write("\n")
            #     fw.close()      
                         
            self.ensemble_env_trans_optimizer[i].zero_grad()
            env_trans_loss.backward()
            self.ensemble_env_trans_optimizer[i].step()
          
    def predict_env_vehicle_state_with_action(self, obs_torch, predict_action):  
        next_vehicle_state_list = []
        for i in range(len(predict_action[0])):
            throttle = torch.mul(predict_action[0][i][0],self.throttle_scale)
            delta = predict_action[0][i][1]
            x1 = obs_torch[i][0]#torch.mul(obs_torch[i][0], self.obs_scale)
            y1 = obs_torch[i][1]#torch.mul(obs_torch[i][1], self.obs_scale)
            yaw1 = obs_torch[i][4]#torch.mul(obs_torch[i][4], self.obs_scale)
            # v1 = torch.tensor(math.sqrt(torch.mul(obs_torch[i][2], self.obs_scale) ** 2 + torch.mul(obs_torch[i][3], self.obs_scale) ** 2))
            v1 = torch.tensor(math.sqrt(obs_torch[i][2] ** 2 + obs_torch[i][3] ** 2))

            x2, y2, yaw2, v2, _, _ = self.ensemble_env_transition_model[0].vehicle_model_torch.kinematic_model(x1, y1, yaw1, v1, throttle, delta)
            
            
            next_vehicle_state_list.append(x2.cpu().detach().numpy().tolist())
            next_vehicle_state_list.append(y2.cpu().detach().numpy().tolist())
            next_vehicle_state_list.append(torch.div(torch.mul(v2, torch.cos(yaw2)), self.obs_scale).cpu().detach().numpy().tolist())
            next_vehicle_state_list.append(torch.div(torch.mul(v2, torch.sin(yaw2)), self.obs_scale).cpu().detach().numpy().tolist())
            next_vehicle_state_list.append(yaw2.cpu().detach().numpy().tolist())
            # tensor_list = [torch.div(x2, self.obs_scale), torch.div(y2, self.obs_scale), torch.div(torch.mul(v2, torch.cos(yaw2)), self.obs_scale),
            #                torch.div(torch.mul(v2, torch.sin(yaw2)), self.obs_scale), torch.div(yaw2, self.obs_scale)]
            # next_vehicle_state = torch.stack(tensor_list)
            # print("x, y, yaw, v, _, _ ",x2, y2, yaw2, v2)
        return next_vehicle_state_list
            
    def test_vehicle_model(self, obs_torch, y):
        # Using ego transition model (not pytorch) to calculate throttle and delta 
        for i in range(len(y)):
            x1 = torch.mul(obs_torch[i][0], self.obs_scale).cpu().numpy()
            y1 = torch.mul(obs_torch[i][1], self.obs_scale).cpu().numpy()
            yaw1 = torch.mul(obs_torch[i][4], self.obs_scale).cpu().numpy()
            v1 = torch.tensor(math.sqrt(torch.mul(obs_torch[i][2], self.obs_scale) ** 2 + torch.mul(obs_torch[i][3], self.obs_scale) ** 2)).cpu().numpy()
            x2 = torch.mul(y[i][0], self.obs_scale).cpu().numpy()
            y2 = torch.mul(y[i][1], self.obs_scale).cpu().numpy()
            yaw2 = torch.mul(y[i][4], self.obs_scale).cpu().numpy()
            v2 = torch.tensor(math.sqrt(torch.mul(y[i][2], self.obs_scale) ** 2 + torch.mul(y[i][3], self.obs_scale) ** 2)).cpu().numpy()
            throttle, delta = self.ego_transition_model.calculate_a_from_data(x1, y1, yaw1, v1, x2, y2, yaw2, v2)
            print("------------numpy model",throttle, delta)

    def collect_buffer(self, env, load_step, train_step):
        model_dir = self.make_dir(os.path.join(self.args.work_dir, 'world_model'))
        buffer_dir = self.make_dir(os.path.join(self.args.work_dir, 'world_buffer'))

        # Collected data and train
        episode, episode_reward, done = 0, 0, True
        
        try:
            self.load(model_dir, load_step)
            print("[World_Model] : Load learned model successful, step=",load_step)

        except:
            load_step = 0
            print("[World_Model] : No learned model, Creat new model")

        for step in range(train_step + 1):
            if done:
                self.trajectory_planner.clear_buff(clean_csp=False)
                obs = env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                reward = 0   
            
            # save agent periodically
            if step % self.args.eval_freq == 0:
                if self.args.save_model:
                    print("[World_Model] : Saved Model! Step:",step + load_step)
                    self.save(model_dir, step + load_step)
                if self.args.save_buffer:
                    self.replay_buffer.save(buffer_dir)
                    print("[World_Model] : Saved Buffer!")

            # run training update
            if step >= self.args.init_steps:
                num_updates = self.args.init_steps if step == self.args.init_steps else 5
                for _ in range(num_updates):
                    if self.ego_transition_learn:
                        self.update_ego_transition_model(self.replay_buffer, step) 
                    # self.update_env_transition_model(self.replay_buffer) 
                    
            obs = np.array(obs)
            curr_reward = reward
            
            # Rule-based Planner
            self.dynamic_map.update_map_from_obs(obs, env)
            rule_trajectory, action = self.trajectory_planner.trajectory_update(self.dynamic_map)
            # action = np.array(random.randint(3,6)) #FIXME:Action space
            # Control
            trajectory = self.trajectory_planner.trajectory_update_CP(action, rule_trajectory)
            control_action =  self.controller.get_control(self.dynamic_map,  rule_trajectory.trajectory, rule_trajectory.desired_speed)
            output_action = [control_action.acc, control_action.steering]
            new_obs, reward, done, info = env.step(output_action)


            print("Predicted Reward:",self.get_reward_prediction(new_obs))
            print("Actual Reward:",reward)
            print("Predicted State:", self.get_trans_prediction(obs, output_action))
            print("Actual State:",new_obs)
            
            episode_reward += reward
            normal_new_obs = (new_obs - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
            normal_obs = (obs - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
            self.replay_buffer.add(normal_obs, output_action, curr_reward, reward, normal_new_obs, done)

            obs = new_obs
            episode_step += 1

    def learn_from_buffer(self, env, load_step, train_step):
        model_dir = self.make_dir(os.path.join(self.args.work_dir, 'world_model'))
        buffer_dir = self.make_dir(os.path.join(self.args.work_dir, 'world_buffer'))
        # Collected data and train
        episode, episode_reward, done = 0, 0, True
        
        self.replay_buffer.load(buffer_dir)
        print("[World_Model] : Load Buffer!",self.replay_buffer.idx)
        try:
            self.load(model_dir, load_step)
            print("[World_Model] : Load learned model successful, step=",load_step)

        except:
            load_step = 0
            print("[World_Model] : No learned model, Creat new model")
        # for steps in tqdm(range(1, 1000 + 1), unit='Steps'):
        for step in tqdm(range(1, train_step + 1), unit='steps'):
            self.update_env_transition_model_with_action(self.replay_buffer) 
            if step % 5000 == 0:
                print("[World_Model] : Saved Model! Step:",step + load_step)
                self.save(model_dir, step + load_step)
                    
    def get_reward_prediction(self, obs):
        return 0

    def get_trans_prediction(self, obs, control_action):
        v = math.sqrt(obs[2]**2 + obs[3]**2)
        x, y, yaw, v, _, _ = self.ego_transition_model.kinematic_model(obs[0], obs[1], obs[4], v, control_action[0], control_action[1])
        
        obs = torch.as_tensor(obs).to(device=self.device).float()

        control_action = torch.as_tensor(control_action).to(device=self.device).float()
        origin_obs_torch = torch.reshape(copy.deepcopy(obs), [2,5])
        obs /= self.obs_scale
        for i in range(1,2):
            obs[0+i*5] -= obs[0] 
            obs[1+i*5] -= obs[1] 
        obs[0] = 0
        obs[1] = 0

        scale_obs_torch = torch.reshape(obs, [2,5])
        action_torch = torch.reshape(control_action, [1,2])
        trans_prediction_list = []

        for i in range(self.args.heads_num):
            predict_action = self.ensemble_env_transition_model[i](scale_obs_torch, action_torch)
            #FIXME: the calculation is so fucking slowly using pytorch, use numpy for vehicle model!
            next_vehicle_state = self.predict_env_vehicle_state_with_action(origin_obs_torch, predict_action)
            next_vehicle_state[0] = x
            next_vehicle_state[1] = y
            next_vehicle_state[2] = v * math.cos(yaw)
            next_vehicle_state[3] = v * math.sin(yaw)
            next_vehicle_state[4] = yaw
            trans_prediction_list.append(next_vehicle_state)
            
        return trans_prediction_list
            
    def save(self, model_dir, step):
        if self.ego_transition_learn:
            torch.save(
                self.ego_transition_model.state_dict(),
                '%s/ego_transition_model_%s.pt' % (model_dir, step)
            )
        for i in range(self.args.heads_num):
            
            torch.save(
                self.ensemble_env_transition_model[i].state_dict(),
                '%s/transition_model_%s_%s.pt' % (model_dir, step, i)
            )
        
    def load(self, model_dir, step):
        if self.ego_transition_learn:
            self.ego_transition_model.load_state_dict(
            torch.load('%s/ego_transition_model_%s.pt' % (model_dir, step))
            )
        
        for i in range(self.args.heads_num):
            self.ensemble_env_transition_model[i].load_state_dict(
            torch.load('%s/transition_model_%s_%s.pt' % (model_dir, step, i))
            )

    def make_dir(self, dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            pass
        return dir_path

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.uniform_(m.weight, a=-0.1, b=0.1)
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式 
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def worst_case_planning(self, obs_torch, action_list):
        reward_action_list = []
        for action in action_list:
            reward_action = -10000
            for i in range(self.args.heads_num):
                action_torch = torch.tensor(action)
                accumulate_reward = 0
                predict_action = self.ensemble_env_transition_model[i](obs_torch, action_torch)
                next_predict_state = self.ensemble_env_transition_model[i].get_vehicle_prediction(predict_action)
                accumulate_reward += self.get_reward(next_predict_state)
                if accumulate_reward > reward_action:
                    reward_action = accumulate_reward
            reward_action_list.append(reward_action)
        
        action = action_list[np.where(reward_action_list==np.max(reward_action_list))]
        
        return action
        
class World_Buffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.k_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.curr_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, curr_reward, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.curr_rewards[self.idx], curr_reward)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, k=False):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=self.batch_size) 
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        curr_rewards = torch.as_tensor(self.curr_rewards[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        if k:
            return obses, actions, rewards, next_obses, not_dones, torch.as_tensor(self.k_obses[idxs], device=self.device)
        return obses, actions, curr_rewards, rewards, next_obses, not_dones
    
    def get(self, idxs, k=False):
        idxs = np.array([idxs]) #FIXME
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        curr_rewards = torch.as_tensor(self.curr_rewards[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        if k:
            return obses, actions, rewards, next_obses, not_dones, torch.as_tensor(self.k_obses[idxs], device=self.device)
        return obses, actions, curr_rewards, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.curr_rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.curr_rewards[start:end] = payload[4]
            self.not_dones[start:end] = payload[5]
            self.idx = end

