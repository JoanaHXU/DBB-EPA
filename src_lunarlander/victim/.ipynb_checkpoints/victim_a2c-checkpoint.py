import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
import gym
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


class MLP(nn.Module):
	def __init__(self, state_size, action_size, fc1_units=150, fc2_units=120, dropout = 0.5):
		super().__init__()
		self.fc1 = nn.Linear(state_size, fc1_units) # first fully-connected layer fc1
		self.fc2 = nn.Linear(fc1_units, fc2_units)  # second fully-connected layer fc2
		self.fc3 = nn.Linear(fc2_units, action_size) # third fully-connected layer fc3
# 		self.dropout = nn.Dropout(dropout)
		
	def forward(self, x):
		x = x.to(device)
		x = self.fc1(x)
# 		x = self.dropout(x)
		x = F.relu(x) # 1-st rectified nonlinear layer, state_size = 8, fc1_units = 64, tensor x is 64x8 = 512 units 
		x = self.fc2(x)
# 		x = self.dropout(x)
		x = F.relu(x) # 2-st rectified nonlinear layer
		x = self.fc3(x)
		return x

class ActorCritic(nn.Module):
	def __init__(self, actor, critic):
		super().__init__()
		
		self.actor = actor
		self.critic = critic
		
	def forward(self, state):
		
		action_pred = self.actor(state)
		value_pred = self.critic(state)
		
		return action_pred, value_pred
	

def init_weights(m):
	if type(m) == nn.Linear:
		torch.nn.init.xavier_normal_(m.weight)
		m.bias.data.fill_(0)

	
	
class Victim_A2C():
	
	def __init__(self, env, MEMORY_SIZE, GAMMA, n_states, n_actions, learning_rate = 0.001):
		
		self.env = env
		self.n_states = n_states
		self.n_actions = n_actions
		self.GAMMA = GAMMA
		self.learning_rate = learning_rate

		self.actor = MLP(n_states, n_actions)
		self.critic = MLP(n_states, 1)

		self.policy_net = ActorCritic(self.actor, self.critic).to(device)
		self.optimizer = optim.Adam(self.policy_net.parameters(), lr = learning_rate)
		
		self.MEMORY_SIZE = MEMORY_SIZE
		self.MEM = utils_buf.Memory(MEMORY_SIZE)
		
		
	def reset(self):
		self.actor = MLP(self.n_states, self.n_actions)
		self.critic = MLP(self.n_states, 1)

		self.policy_net = ActorCritic(self.actor, self.critic).to(device)
		self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.learning_rate)
		
		self.MEM = utils_buf.Memory(MEMORY_SIZE)
		

	def calculate_returns(self, rewards, normalize = True):
		returns = []
		
		R = 0
		for r in reversed(rewards):
			R = r + R * self.GAMMA
			returns.insert(0, R)

		returns = torch.tensor(returns).to(device)
		if normalize:
			returns = (returns - returns.mean()) / returns.std()

		return returns


	def calculate_advantages(self, returns, values, normalize = True):
		advantages = returns - values
		if normalize:
			advantages = (advantages - advantages.mean()) / advantages.std()

		return advantages


	def update_policy(self, advantages, log_prob_actions, returns, values):

		advantages = advantages.detach()
		returns = returns.detach()

		policy_loss = - (advantages * log_prob_actions).sum()

		value_loss = F.smooth_l1_loss(returns, values).sum()

		self.optimizer.zero_grad()

		policy_loss.backward()
		value_loss.backward()

		self.optimizer.step()

#         return policy_loss.item(), value_loss.item()


	def train_model(self, max_episodes_num):

		self.policy_net.train()
		
		for i_episode in range(max_episodes_num):
			# logger
			trajectory_list = []
			
			# reset
			log_prob_actions = []
			values = []
			rewards = []
			done = False
			episode_reward = 0

			state = self.env.reset()
			state = torch.from_numpy(state).view(1,self.n_states).to(device)
			
			for t in range(T_max):
				
				# log:
				t_sample = []

				action_pred = self.actor(state)
				value_pred = self.critic(state)

				action_prob = F.softmax(action_pred, dim = -1)
				dist = distributions.Categorical(action_prob)
				action = dist.sample()
				
				# log
				log_prob_action = dist.log_prob(action)
				
# 				self.env.render()

				next_state, reward, done, _ = self.env.step(action.item())
				
				next_state = torch.from_numpy(next_state).view(1,self.n_states).to(device)
				reward = torch.tensor([reward], dtype=torch.float32, device=device)
				

				log_prob_actions.append(log_prob_action)
				values.append(value_pred)
				rewards.append(reward)
				
				# save to AutoEncoder Memo
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

				episode_reward += reward
				
				if done or t+1==T_max:
# 					print("A2C-LunarLander: episode = {} | t = {} | score = {}".format(i_episode, t, round(episode_reward.item(),0)))
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

			# update policy network
			log_prob_actions = torch.cat(log_prob_actions)
			values = torch.cat(values).squeeze(-1)

			returns = self.calculate_returns(rewards)
			advantages = self.calculate_advantages(returns, values)
			
			self.update_policy(advantages, log_prob_actions, returns, values)
			
		self.env.close()
		

	def eval_model_memory(self):

		self.policy_net.eval()
		
		# logger
		trajectory_list = []

		done = False
		episode_reward = 0
		
		state = self.env.reset()
		state = torch.from_numpy(state).view(1,self.n_states).to(device)

		for t in range(T_max):
			
			# log:
			t_sample = []

			with torch.no_grad():
				action_pred, _ = self.policy_net(state)
				action_prob = F.softmax(action_pred, dim = -1)
			action = torch.argmax(action_prob, dim = -1)
			
			next_state, reward, done, _ = self.env.step(action.item())
			
			next_state = torch.from_numpy(next_state).view(1,self.n_states).to(device)
			reward = torch.tensor([reward], dtype=torch.float32, device=device)
			
			# save to AutoEncoder Memo
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

			episode_reward += reward
			
			if done or t+1==T_max:
#                 print(".....A2C-LunarLander: t = {} | score = {}".format(t, round(episode_reward.item(),0)))
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
		
	def save(self):
		torch.save(self.policy_net.state_dict(), f"victim_model_a2c")

		
	
	
if __name__ == "__main__":
	
	from envs import LunarLander
	env = LunarLander()
	env.clear_attack()
	
	v_n_actions = env.action_space.n
	v_n_states = env.observation_space.shape[0]
	
	victim_args = {
		"env": env, 
		"MEMORY_SIZE": MEMORY_SIZE,
		"GAMMA": 0.99,
		"n_states": v_n_states, 
		"n_actions": v_n_actions, 
		"learning_rate": 0.001,
	}
	
	victim = Victim_A2C(**victim_args)
	
	victim.train_model(1000)
	victim.eval_model_memory()
	
#     victim.save()
	
#     print(f"Size of Memory = {victim.MEM.__len__()}")
#     print(f"First 5 Trajectory is \n{victim.MEM.memory[0:5]}")
#     print(f"Last 5 Trajectory is \n{victim.MEM.memory[4995:5000]}")

	# ''' debug GPU device issue '''
	# from ae.autoencoder import AutoEncoder
	# EMBEDDING_SIZE = 100
	# SEQ_LEN = 100
	
	# ae_enc_IN = env.observation_space.shape[0] + 1 # env.state.size + env.select_action.size
	# ae_enc_OUT = EMBEDDING_SIZE # embedding_size
	# ae_enc_N_layer = 1

	# ae_dec_fc_IN = ae_enc_OUT + env.observation_space.shape[0] # embedding + env.state.size
	# ae_dec_fc_OUT = env.action_space.n # env.action.size

	# ae_dec_lstm_IN = ae_enc_OUT + env.observation_space.shape[0] + 1 # embedding + env.state.size + env.select_action.size
	# ae_dec_lstm_OUT = env.observation_space.shape[0] # env.state.size
	# ae_dec_lstm_N_layer = 1
	
	# ae_args = {
	#     "env": env, 
	#     "enc_in_size": ae_enc_IN, 
	#     "enc_out_size": ae_enc_OUT, 
	#     "enc_num_layer": ae_enc_N_layer, 
	#     "dec_fc_in_size": ae_dec_fc_IN, 
	#     "dec_fc_out_size": ae_dec_fc_OUT, 
	#     "dec_lstm_in_size": ae_dec_lstm_IN, 
	#     "dec_lstm_out_size": ae_dec_lstm_OUT, 
	#     "dec_lstm_num_layer": ae_dec_lstm_N_layer, 
	#     "seq_len": SEQ_LEN, 
	#     "embedding_len": EMBEDDING_SIZE,
	#     "n_epochs": 2, 
	#     "lr": 0.001, 
	# }
	
	# ae_object = AutoEncoder(**ae_args)
	
	# ae_object._train(victim.MEM, victim.MEM)
