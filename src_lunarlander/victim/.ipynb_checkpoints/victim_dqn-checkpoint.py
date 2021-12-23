import sys
import gym
import math
import random
import numpy as np
import seaborn as sns
import copy
import time

from collections import namedtuple
from collections import defaultdict
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import os
from os.path import dirname, abspath

import sys
if "../" not in sys.path:
    sys.path.append("../") 

from utils import utils_buf
from envs import LunarLander

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0

''' import configuration '''
from yacs.config import CfgNode as CN
yaml_name = os.path.join(dirname(dirname(abspath(__file__))), "config", "config_default.yaml")
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()

LEN_TRAJECTORY = config.AE.LEN_TRAJECTORY
MEMORY_SIZE = config.AE.MEMORY_SIZE
T_max = config.VICTIM.TMAX

"""
Replay Buffer
"""
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



"""
QNetwork
"""
class DQN(nn.Module):

    def __init__(self, state_size, action_size, fc1_units=150, fc2_units=120):
        """Initialize parameters and build model."""
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units) # first fully-connected layer fc1
        self.fc2 = nn.Linear(fc1_units, fc2_units)  # second fully-connected layer fc2
        self.fc3 = nn.Linear(fc2_units, action_size) # third fully-connected layer fc3
        
    def forward(self, x):
        x = x.to(device)
        """Build a network that maps state -> action values."""
        x = self.fc1(x)
        x = F.relu(x) # 1-st rectified nonlinear layer, state_size = 8, fc1_units = 64, tensor x is 64x8 = 512 units 
        x = self.fc2(x)
        x = F.relu(x) # 2-st rectified nonlinear layer
        x = self.fc3(x)
        return x


class Victim_DQN():
    
    def __init__(self, env, MEMORY_SIZE, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TARGET_UPDATE, n_states, n_actions, learning_rate=0.001):
        self.env = env
        self.MEMORY_SIZE = MEMORY_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TARGET_UPDATE = TARGET_UPDATE
        self.learning_rate = learning_rate
        
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.policy_net = DQN(n_states, n_actions).to(device)
        self.target_net = DQN(n_states, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(),lr=learning_rate)
        self.reply_buffer = ReplayMemory(1000000)
        self.MEM = utils_buf.Memory(MEMORY_SIZE)
        self.steps_done = 0
        
    def reset(self):
        self.policy_net = DQN(self.n_states, self.n_actions).to(device)
        self.target_net = DQN(self.n_states, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(),lr=self.learning_rate)
        self.reply_buffer = ReplayMemory(1000000)
        self.MEM = utils_buf.Memory(self.MEMORY_SIZE)
        
        self.steps_done = 0


    def select_action(self, state):
        global steps_done
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)



    def optimize_model(self):
        if len(self.reply_buffer) < self.BATCH_SIZE:
            return
        transitions = self.reply_buffer.sample(self.BATCH_SIZE)
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. 
        state_action_values = self.policy_net(state_batch.float()).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states.float()).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        
        
    def train_model(self, num_episodes):
        
        for i_episode in range(num_episodes):
            # log
            score = 0
            trajectory_list = []
            
            # reset env and victim
            state = self.env.reset()
            state = torch.from_numpy(state).view(1,self.n_states).to(device)
            done = False
            
            for t in range(T_max):
                # log:
                t_sample = []
                
                action = self.select_action(state).to(device)
#                 self.env.render()
                next_state, reward, done, _ = self.env.step(action.item())
                next_state = torch.from_numpy(next_state).view(1,self.n_states).to(device)
                reward = torch.tensor([reward], dtype=torch.float32, device=device)

                # Store the transition (s,a,s',r) in memory
                self.reply_buffer.push(state, action, next_state, reward)
                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                # store the transition(s,a,s',a') 
                if t != 0:
                    t_sample.append(pre_state)
                    t_sample.append(pre_action)
                    t_sample.append(state)
                    t_sample.append(action)
                    trajectory_list.append(t_sample)
                    
                # Move to the next state
                pre_state = copy.deepcopy(state)
                pre_action = copy.deepcopy(action)
                state = copy.deepcopy(next_state)
                score += reward

                if done:
#                     print("DQN-LunarLander: episode = {} | t = {} | score = {}".format(i_episode, t, round(score.item(),0)))
                    break
                if t+1 == T_max:
#                     print("DQN-LunarLander: episode = {} | t = {} | score = {}".format(i_episode, t, round(score.item(),0)))
                    break
                    
            # Trajectory to MEMORY with head_padding
            if len(trajectory_list) < LEN_TRAJECTORY:
                padding_state = torch.zeros(state.shape, device=device)
                padding_action = torch.zeros(action.shape, device=device)
                n_padding = LEN_TRAJECTORY - len(trajectory_list)
                for i in range(n_padding):
                    self.MEM.push(padding_state, padding_action, padding_state, padding_action)
                for i in range(len(trajectory_list)):
                    pre_state = trajectory_list[i][0]
                    pre_action = trajectory_list[i][1]
                    state = trajectory_list[i][2]
                    action = trajectory_list[i][3]
                    self.MEM.push(pre_state, pre_action, state, action)
            else:
                for i in range(LEN_TRAJECTORY):
                    pre_state = trajectory_list[i][0]
                    pre_action = trajectory_list[i][1]
                    state = trajectory_list[i][2]
                    action = trajectory_list[i][3]
                    self.MEM.push(pre_state, pre_action, state, action)

            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
        self.env.close()



    def eval_model(self):
        state = self.env.reset()
        state = torch.from_numpy(state).view(1,self.n_states)
        
        score = 0
        T_max = 2000
        done = False
        for t in range(T_max):

            action = self.policy_net(state).max(1)[1].view(1, 1).to(device)
#             self.env.render()
            next_state, reward, done, _ = self.env.step(action.item())
            next_state = torch.from_numpy(next_state).view(1,self.n_states)
            reward = torch.tensor([reward], dtype=torch.float32)

            # Move to the next state
            state = next_state
            score += reward

            if done:
#                 print(f".....Victim Evaluation : Timesteps ={t} | Scores ={score.item():.2f}")
                break
            if t+1 == T_max:
#                 print(f".....Victim Evaluation : Timesteps ={t} | Scores ={score.item():.2f}")
                break
                
        self.env.close()
        
        
        
    def eval_model_memory(self):
        
        score = 0
        trajectory_list = []
        
        T_max = 2000
        done = False
        
        state = self.env.reset()
        state = torch.from_numpy(state).view(1,self.n_states)
        
        for t in range(T_max):
            t_sample = []
            
            action = self.policy_net(state).max(1)[1].view(1, 1).to(device)
#             self.env.render()
            next_state, reward, done, _ = self.env.step(action.item())
            next_state = torch.from_numpy(next_state).view(1,self.n_states)
            reward = torch.tensor([reward], dtype=torch.float32)
            
            # store the transition(s,a,s',a') 
            if t != 0:
                t_sample.append(pre_state)
                t_sample.append(pre_action)
                t_sample.append(state)
                t_sample.append(action)
                trajectory_list.append(t_sample)

            # Move to the next state
            pre_state = copy.deepcopy(state)
            pre_action = copy.deepcopy(action)
            state = next_state
            score += reward

            if done and t+1 == T_max:
#                 print(f".....DQN-LunarLander: Timesteps ={t} | Scores ={score.item():.2f}")
                break
                
        # Trajectory to MEMORY with head_padding
        if len(trajectory_list) < LEN_TRAJECTORY:
            padding_state = torch.zeros(state.shape, device=device)
            padding_action = torch.zeros(action.shape, device=device)
            n_padding = LEN_TRAJECTORY - len(trajectory_list)
            for i in range(n_padding):
                self.MEM.push(padding_state, padding_action, padding_state, padding_action)
            for i in range(len(trajectory_list)):
                pre_state = trajectory_list[i][0]
                pre_action = trajectory_list[i][1]
                state = trajectory_list[i][2]
                action = trajectory_list[i][3]
                self.MEM.push(pre_state, pre_action, state, action)
        else:
            for i in range(LEN_TRAJECTORY):
                pre_state = trajectory_list[i][0]
                pre_action = trajectory_list[i][1]
                state = trajectory_list[i][2]
                action = trajectory_list[i][3]
                self.MEM.push(pre_state, pre_action, state, action)
                
        self.env.close()
        

        
if __name__ == "__main__":
    
    env = LunarLander()
    env.clear_attack()
    
    v_BATCH_SIZE = 256
    v_GAMMA = .99
    v_EPS_START = 1.0
    v_EPS_END = .01
    v_EPS_DECAY = .996
    v_TARGET_UPDATE = 10

    v_n_actions = env.action_space.n
    v_n_states = env.observation_space.shape[0]
    
    victim_args = {
        "env": env, 
        "MEMORY_SIZE": MEMORY_SIZE, 
        "BATCH_SIZE": v_BATCH_SIZE, 
        "GAMMA": v_GAMMA, 
        "EPS_START": v_EPS_START, 
        "EPS_END": v_EPS_END, 
        "EPS_DECAY": v_EPS_DECAY, 
        "TARGET_UPDATE": v_TARGET_UPDATE, 
        "n_states": v_n_states, 
        "n_actions": v_n_actions, 
        "learning_rate": 0.001,
    }
    
    victim = Victim_DQN(**victim_args)
    
    victim.train_model(1)
#     print(f"Size of Memory = {victim.MEM.__len__()}")
#     print(f"First 5 Trajectory is \n{victim.MEM.memory[0:5]}")
    victim.eval_model()
    victim.eval_model_memory()
