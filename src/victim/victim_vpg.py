# import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym
from collections import deque
import copy

import os
from os.path import dirname, abspath
import sys
if "../" not in sys.path:
    sys.path.append("../") 

# self-defined
from utils import utils_buf

''' import configuration '''
from yacs.config import CfgNode as CN
yaml_name = os.path.join(dirname(dirname(abspath(__file__))), "config", "config_default.yaml")
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()

LEN_TRAJECTORY = config.AE.LEN_TRAJECTORY
MEMORY_SIZE = config.AE.MEMORY_SIZE
T_max = config.VICTIM.TMAX


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





''' Policy_Net definition '''
class Policy_Net(nn.Module):
    def __init__(self, nS, nA, fc1_units=150, fc2_units=120): # nS: state space size, nH: n. of neurons in hidden layer, nA: size action space
        super(Policy_Net, self).__init__()
        self.fc1 = nn.Linear(nS, fc1_units) # first fully-connected layer fc1
        self.fc2 = nn.Linear(fc1_units, fc2_units)  # second fully-connected layer fc2
        self.fc3 = nn.Linear(fc2_units, nA) # third fully-connected layer fc3

    # define forward pass with one hidden layer with ReLU activation and sofmax after output layer
    def forward(self, x):
        x = x.to(device)
        """Build a network that maps state -> action values."""
        x = self.fc1(x)
        x = F.relu(x) # 1-st rectified nonlinear layer, state_size = 8, fc1_units = 64, tensor x is 64x8 = 512 units 
        x = self.fc2(x)
        x = F.relu(x) # 2-st rectified nonlinear layer
#         x = self.fc3(x)
        x = F.softmax(self.fc3(x), dim=1)
        return x
    
    
class Victim_VPG():
    def __init__(self, env, MEMORY_SIZE, GAMMA, MAX_LEN, n_states, n_actions, learning_rate=0.001):
        # create environment
        self.env = env
        self.n_states = n_states
        self.n_actions = n_actions
        
        # instantiate the policy and optimizer
        self.policy_net = Policy_Net(n_states, n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        self.MEMORY_SIZE = MEMORY_SIZE
        self.MEM = utils_buf.Memory(MEMORY_SIZE)
        
        # initialize hyper-parameters
        self.MEMORY_SIZE = MEMORY_SIZE
        self.GAMMA= GAMMA
        self.MAX_LEN = MAX_LEN
        self.learning_rate = learning_rate
        self.returns = deque(maxlen=MAX_LEN)
        
        # initialize logger
        self.n_episode = 1
        
    def reset(self):
        self.policy_net = Policy_Net(self.n_states, self.n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.MEM = utils_buf.Memory(self.MEMORY_SIZE)
        
        self.returns = deque(maxlen=self.MAX_LEN)
        self.n_episode = 1
        
        
    def train_model(self, max_episodes_num):
        
        self.policy_net.train()
        
        for i_episode in range(max_episodes_num):
            # logger
            trajectory_list = []
            score = 0
            
            rewards = []
            actions = []
            states  = []
            
            # reset environment
            state = self.env.reset()
            done = False
            
            for t in range(T_max):
                # log
                t_sample = []

                # calculate probabilities of taking each action
                probs = self.policy_net(torch.tensor(state, device=device).unsqueeze(0).float())
                # sample an action from that set of probs
                sampler = Categorical(probs)
                action = sampler.sample()
                
#                 self.env.render()
                
                # use that action in the environment
                next_state, reward, done, info = self.env.step(action.item())
                
                # store state, action and reward
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                # store the transition(s,a,s',a') 
                if t != 0:
                    pre_state_tensor = torch.from_numpy(pre_state).view(1,self.n_states).to(device)
                    state_tensor = torch.from_numpy(state).view(1,self.n_states).to(device)
                    t_sample.append(pre_state_tensor)
                    t_sample.append(pre_action)
                    t_sample.append(state_tensor)
                    t_sample.append(action)
                    trajectory_list.append(t_sample)
                
                # update state
                pre_state = copy.deepcopy(state)
                pre_action = copy.deepcopy(action)
                state = copy.deepcopy(next_state)
                score += reward
                
                if done or t+1==T_max:
#                     print("LunarLander: episode = {} | t = {} | score = {}".format(i_episode, t, round(score.item(),0)))
                    break
                    
            # Trajectory to MEMORY with head_padding
            if len(trajectory_list) < LEN_TRAJECTORY:
                padding_state = torch.zeros(state.shape, device=device).view(1,self.n_states)
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

            # preprocess rewards
            rewards = np.array(rewards)
            # calculate rewards to go for less variance
            R = torch.tensor([np.sum(rewards[i:]*(self.GAMMA**np.array(range(i, len(rewards))))) for i in range(len(rewards))]).to(device)
            # or uncomment following line for normal rewards
            #R = torch.sum(torch.tensor(rewards))

            # preprocess states and actions
            states = torch.tensor(states).float().to(device)
            actions = torch.tensor(actions).to(device)

            # calculate gradient
            probs = self.policy_net(states)
            sampler = Categorical(probs)
            log_probs = -sampler.log_prob(actions)   # "-" because it was built to work with gradient descent, but we are using gradient ascent
            pseudo_loss = torch.sum(log_probs * R) # loss that when differentiated with autograd gives the gradient of J(θ)
            
            # update policy weights
            self.optimizer.zero_grad()
            pseudo_loss.backward()
            self.optimizer.step()

            # calculate average return and print it out
            self.returns.append(np.sum(rewards))
#             print("VPG-lunarlander : Episode: {:6d}\tAvg. Return: {:6.2f}".format(self.n_episode, np.mean(self.returns)))
#             print("Episode: {:6d}Avg. | Timesteps: {:6d} | Score:\t{:6.2f} | Return:\t{:6.2f}".format(self.n_episode, t, score.item(), np.mean(self.returns)))
            self.n_episode += 1

        # close environment
        self.env.close()
        
        
    def eval_model_memory(self):
        
        self.policy_net.eval()
        
        trajectory_list = []
        score = 0

        rewards = []
        actions = []
        states  = []

        # reset environment
        state = self.env.reset()
        done = False

        for t in range(T_max):
            # log
            t_sample = []

            # calculate probabilities of taking each action
            probs = self.policy_net(torch.tensor(state, device=device).unsqueeze(0).float())
            action = torch.argmax(probs, dim = -1)

#                 self.env.render()

            # use that action in the environment
            next_state, reward, done, info = self.env.step(action.item())

            # store state, action and reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # store the transition(s,a,s',a') 
            if t != 0:
                pre_state_tensor = torch.from_numpy(pre_state).view(1,self.n_states).to(device)
                state_tensor = torch.from_numpy(state).view(1,self.n_states).to(device)
                t_sample.append(pre_state_tensor)
                t_sample.append(pre_action)
                t_sample.append(state_tensor)
                t_sample.append(action)
                trajectory_list.append(t_sample)

            # update state
            pre_state = copy.deepcopy(state)
            pre_action = copy.deepcopy(action)
            state = copy.deepcopy(next_state)
            score += reward

            if done or t+1==T_max:
#                 print(".....VPG-LunarLander: episode = {} | t = {} | score = {}".format(i_episode, t, round(score.item(),0)))
                break
    
        # Trajectory to MEMORY with head_padding
        if len(trajectory_list) < LEN_TRAJECTORY:
            padding_state = torch.zeros(state.shape, device=device).view(1,self.n_states)
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
        
        # preprocess rewards
        rewards = np.array(rewards)
        returns = np.sum(rewards)
#         print("Episode: {:6d}\tAvg. Return: {:6.2f}".format(self.n_episode, np.mean(returns)))

        # close environment
        self.env.close()
        
        
    def save(self):
        torch.save(self.policy_net.state_dict(), f"victim_model_vpg")
        
        
if __name__ == "__main__":
    
    from envs import LunarLander
    env = LunarLander()
    env.clear_attack()
    
    v_GAMMA = .99
    v_MAX_LEN = 500

    v_n_actions = env.action_space.n
    v_n_states = env.observation_space.shape[0]
    
    victim_args = {
        "env": env, 
        "MEMORY_SIZE": MEMORY_SIZE, 
        "GAMMA": v_GAMMA, 
        "MAX_LEN": v_MAX_LEN,
        "n_states": v_n_states, 
        "n_actions": v_n_actions, 
        "learning_rate": 0.001,
    }
    
    victim = Victim_VPG(**victim_args)
    
    victim.train_model(10)
    
#     victim.save()
    
#     print(f"Size of Memory = {victim.MEM.__len__()}")
#     print(f"First 5 Trajectory is \n{victim.MEM.memory[0:5]}")
#     print(f"Last 5 Trajectory is \n{victim.MEM.memory[4995:5000]}")

    ''' debug GPU device issue '''
    from ae.autoencoder import AutoEncoder
    EMBEDDING_SIZE = 100
    SEQ_LEN = 100
    
    ae_enc_IN = env.observation_space.shape[0] + 1 # env.state.size + env.select_action.size
    ae_enc_OUT = EMBEDDING_SIZE # embedding_size
    ae_enc_N_layer = 1

    ae_dec_fc_IN = ae_enc_OUT + env.observation_space.shape[0] # embedding + env.state.size
    ae_dec_fc_OUT = env.action_space.n # env.action.size

    ae_dec_lstm_IN = ae_enc_OUT + env.observation_space.shape[0] + 1 # embedding + env.state.size + env.select_action.size
    ae_dec_lstm_OUT = env.observation_space.shape[0] # env.state.size
    ae_dec_lstm_N_layer = 1
    
    ae_args = {
        "env": env, 
        "enc_in_size": ae_enc_IN, 
        "enc_out_size": ae_enc_OUT, 
        "enc_num_layer": ae_enc_N_layer, 
        "dec_fc_in_size": ae_dec_fc_IN, 
        "dec_fc_out_size": ae_dec_fc_OUT, 
        "dec_lstm_in_size": ae_dec_lstm_IN, 
        "dec_lstm_out_size": ae_dec_lstm_OUT, 
        "dec_lstm_num_layer": ae_dec_lstm_N_layer, 
        "seq_len": SEQ_LEN, 
        "embedding_len": EMBEDDING_SIZE,
        "n_epochs": 2, 
        "lr": 0.001, 
    }
    
    ae_object = AutoEncoder(**ae_args)
    
    ae_object._train(victim.MEM, victim.MEM)
