# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import argparse
import torch
import os
import random
import torch.nn as nn
import torch.nn.functional as F

from Agent.world_model.single_transition_model import make_transition_model

from Agent.zzz.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from Agent.zzz.controller import Controller
from Agent.zzz.dynamic_map import DynamicMap
from Agent.zzz.actions import LaneAction

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
        init_temperature=0.01,
        transition_lr=0.001,
        transition_weight_lambda=0.0,
    ):
        
        self.device = device
        self.discount = discount
        self.state_space_dim = state_space_dim
        self.action_shape = action_shape
        self.args = self.parse_args()

        # Ego Vehicle Transition
        ego_state_dim = 5
        transition_model_type = 'probabilistic'
        self.ego_transition_model = make_transition_model(
            transition_model_type, ego_state_dim, action_shape
        ).to(self.device)
        self.ego_transition_optimizer = torch.optim.Adam(
            list(self.ego_transition_model.parameters()),
            lr=transition_lr,
            weight_decay=transition_weight_lambda
        )
        
        # Env Agent Transition

        # Planner
        self.trajectory_planner = JunctionTrajectoryPlanner()
        self.controller = Controller()
        self.dynamic_map = DynamicMap()
        self.target_speed = 30/3.6 
        
        self.train()
        self.env = env
        self.replay_buffer = World_Buffer(obs_shape=env.observation_space.shape,
            action_shape=action_shape, # discrete, 1 dimension!
            capacity= self.args.replay_buffer_capacity,
            batch_size= self.args.batch_size,
            device=device)
    
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--decision_count", type=int, default=1, help="how many steps for a decision")

        # environment
        parser.add_argument('--domain_name', default='carla')
        parser.add_argument('--task_name', default='run')
        parser.add_argument('--image_size', default=84, type=int)
        parser.add_argument('--action_repeat', default=1, type=int)
        parser.add_argument('--frame_stack', default=1, type=int) #3
        parser.add_argument('--resource_files', type=str)
        parser.add_argument('--eval_resource_files', type=str)
        parser.add_argument('--img_source', default=None, type=str, choices=['color', 'noise', 'images', 'video', 'none'])
        parser.add_argument('--total_frames', default=1000, type=int)
        # replay buffer
        parser.add_argument('--replay_buffer_capacity', default=1000000, type=int)
        # train
        parser.add_argument('--agent', default='bisim', type=str, choices=['baseline', 'bisim', 'deepmdp'])
        parser.add_argument('--init_steps', default=1, type=int)
        parser.add_argument('--num_train_steps', default=1000, type=int)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--hidden_dim', default=256, type=int)
        parser.add_argument('--k', default=3, type=int, help='number of steps for inverse model')
        parser.add_argument('--bisim_coef', default=0.5, type=float, help='coefficient for bisim terms')
        parser.add_argument('--load_encoder', default=None, type=str)
        # eval
        parser.add_argument('--eval_freq', default=1000, type=int)  # TODO: master had 10000
        parser.add_argument('--num_eval_episodes', default=20, type=int)
        # critic
        parser.add_argument('--critic_lr', default=1e-3, type=float)
        parser.add_argument('--critic_beta', default=0.9, type=float)
        parser.add_argument('--critic_tau', default=0.005, type=float)
        parser.add_argument('--critic_target_update_freq', default=2, type=int)
        # actor
        parser.add_argument('--actor_lr', default=1e-3, type=float)
        parser.add_argument('--actor_beta', default=0.9, type=float)
        parser.add_argument('--actor_log_std_min', default=-10, type=float)
        parser.add_argument('--actor_log_std_max', default=2, type=float)
        parser.add_argument('--actor_update_freq', default=2, type=int)
        # encoder/decoder
        parser.add_argument('--encoder_type', default='pixelCarla098', type=str, choices=['pixel', 'pixelCarla096', 'pixelCarla098', 'identity'])
        parser.add_argument('--encoder_feature_dim', default=50, type=int)
        parser.add_argument('--encoder_lr', default=1e-3, type=float)
        parser.add_argument('--encoder_tau', default=0.005, type=float)
        parser.add_argument('--encoder_stride', default=1, type=int)
        parser.add_argument('--decoder_type', default='pixel', type=str, choices=['pixel', 'identity', 'contrastive', 'reward', 'inverse', 'reconstruction'])
        parser.add_argument('--decoder_lr', default=1e-3, type=float)
        parser.add_argument('--decoder_update_freq', default=1, type=int)
        parser.add_argument('--decoder_weight_lambda', default=0.0, type=float)
        parser.add_argument('--num_layers', default=4, type=int)
        parser.add_argument('--num_filters', default=32, type=int)
        # sac
        parser.add_argument('--discount', default=0.99, type=float)
        parser.add_argument('--init_temperature', default=0.01, type=float)
        parser.add_argument('--alpha_lr', default=1e-3, type=float)
        parser.add_argument('--alpha_beta', default=0.9, type=float)
        # misc
        parser.add_argument('--seed', default=1, type=int)
        parser.add_argument('--work_dir', default='.', type=str)
        parser.add_argument('--save_tb', default=False, action='store_true')
        parser.add_argument('--save_model', default=True, action='store_true')
        parser.add_argument('--save_buffer', default=True, action='store_true')
        parser.add_argument('--save_video', default=False, action='store_true')
        parser.add_argument('--transition_model_type', default='probabilistic', type=str, choices=['', 'deterministic', 'probabilistic', 'ensemble'])
        parser.add_argument('--render', default=False, action='store_true')
        parser.add_argument('--port', default=2000, type=int)
        args = parser.parse_args()
        return args
    
    def train(self, training=True):
        self.training = training
    
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

    def update(self, env, load_step, train_step):
        model_dir = self.make_dir(os.path.join(self.args.work_dir, 'world_model'))

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
                # if self.args.save_buffer:
                #     self.replay_buffer.save(buffer_dir)
                #     print("[World_Model] : Saved Buffer!")

            # run training update
            if step >= self.args.init_steps:
                num_updates = self.args.init_steps if step == self.args.init_steps else 10
                for _ in range(num_updates):
                    self.update_ego_transition_model(self.replay_buffer, step) 


            obs = np.array(obs)
            curr_reward = reward
            

            # Rule-based Planner
            self.dynamic_map.update_map_from_obs(obs, env)
            rule_trajectory, action = self.trajectory_planner.trajectory_update(self.dynamic_map)
            action = np.array(random.randint(0,6)) #FIXME:Action space
            # Control
            trajectory = self.trajectory_planner.trajectory_update_CP(action, rule_trajectory)
            control_action =  self.controller.get_control(self.dynamic_map,  trajectory.trajectory, trajectory.desired_speed)
            output_action = [control_action.acc, control_action.steering]
            new_obs, reward, done, info = env.step(output_action)


            # print("Predicted Reward:",self.get_reward_prediction(obs, action))
            # print("Actual Reward:",reward, step)
            # print("Predicted State:",self.get_trans_prediction(obs, action)* (env.observation_space.high - env.observation_space.low) + env.observation_space.low)
            # print("Actual State:",new_obs)
            episode_reward += reward
            normal_new_obs = (new_obs - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
            normal_obs = (obs - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
            self.replay_buffer.add(normal_obs, output_action, curr_reward, reward, normal_new_obs, done)

            obs = new_obs
            episode_step += 1

    def get_reward_prediction(self, obs, action):
        obs = (obs - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low)

        np_obs = np.empty((1, self.state_space_dim), dtype=np.float32)
        np.copyto(np_obs[0], obs)
        obs = torch.as_tensor(np_obs, device=self.device).float()
        np_action = np.empty((1, 1), dtype=np.float32)
        np.copyto(np_action[0], action)
        action = torch.as_tensor(np_action, device=self.device)

        with torch.no_grad():
            obs_with_action = torch.cat([obs, action], dim=1).to("cuda:0")
            return self.reward_decoder(obs_with_action)

    def get_trans_prediction(self, obs, action):
        obs = (obs - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low)

        np_obs = np.empty((1, self.state_space_dim), dtype=np.float32)
        np.copyto(np_obs[0], obs)
        obs = torch.as_tensor(np_obs, device=self.device).float()
        np_action = np.empty((1, 1), dtype=np.float32)
        np.copyto(np_action[0], action)
        action = torch.as_tensor(np_action, device=self.device)
        with torch.no_grad():
            obs_with_action = torch.cat([obs, action], dim=1).to("cuda:0")
            return self.ego_transition_model(obs_with_action)
            
    def save(self, model_dir, step):
        torch.save(
            self.ego_transition_model.state_dict(),
            '%s/transition_model%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.ego_transition_model.load_state_dict(
            torch.load('%s/transition_model%s.pt' % (model_dir, step))
        )

    def make_dir(self, dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            pass
        return dir_path

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
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

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

