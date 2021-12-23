
import math
import os
from os.path import dirname, abspath
import sys
if "../" not in sys.path:
	sys.path.append("../") 
	
from victim import *
from ae.autoencoder import AutoEncoder
import utils.utils_attack as util

''' import configuration '''
from yacs.config import CfgNode as CN
yaml_name = os.path.join(dirname(dirname(abspath(__file__))), "config", "config_default.yaml")
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()

LEN_TRAJECTORY = config.AE.LEN_TRAJECTORY
MEMORY_SIZE = config.AE.MEMORY_SIZE

print(f"LEN_TRAJECTORY = {LEN_TRAJECTORY}")

''' import target_Memory '''
from target.target_def import MEM_target

class System():
	def __init__(self, victim_args, ae_args):
		self.victim = Victim_DQN(**victim_args)
		self.ae = AutoEncoder(**ae_args)
		
	def train(self, victim_episodes_num = 50, ae_episodes_num = 1):
		self.ae.n_epochs = ae_episodes_num
		for i_episode in range(victim_episodes_num):
			self.victim.train_model(1)
			if self.victim.MEM.__len__() >= MEMORY_SIZE:
				self.ae._train(self.victim.MEM, MEM_target)
				
	def eval_victim(self):
		self.victim.eval_model()
		
	def eval_autoencoder(self):
		self.ae._eval(self.victim.MEM, MEM_target)
		
		
class System_General():
	def __init__(self, ae_object, victim_args, victim_type = 'DQN'):
		
		self.ae = ae_object
				
		if victim_type == 'DQN':    
			self.victim = Victim_DQN(**victim_args)
		elif victim_type == 'PPO':
			self.victim = Victim_PPO(**victim_args)
		elif victim_type == 'VPG':
			self.victim = Victim_VPG(**victim_args)
		elif victim_type == 'A2C':
			self.victim = Victim_A2C(**victim_args)
		else:
			print("Unknown Victm Type")
		
		
	def train(self, victim_episodes_num = 50, ae_episodes_num = 1):
		self.ae.n_epochs = ae_episodes_num
		for i_episode in range(victim_episodes_num):
			self.victim.train_model(1)
			if self.victim.MEM.__len__() >= MEMORY_SIZE:
				self.ae._train(self.victim.MEM, MEM_target)
				
	def eval_victim(self):
		self.victim.eval_model_memory()
		
	def eval_autoencoder(self):
		self.ae._eval(self.victim.MEM, MEM_target)
		
		
		
class System_Identify_Attack_Chance():
	def __init__(self, ae_object, victim_args, victim_type = 'DQN'):
		
		self.ae = ae_object
		self.victim_type = victim_type
				
		if victim_type == 'DQN':    
			self.victim = Victim_DQN(**victim_args)
		elif victim_type == 'PPO':
			self.victim = Victim_PPO(**victim_args)
		elif victim_type == 'VPG':
			self.victim = Victim_VPG(**victim_args)
		elif victim_type == 'A2C':
			self.victim = Victim_A2C(**victim_args)
		else:
			print("Unknown Victm Type")
		
		
	def train(self, counter_bar = 10, max_victim_N = 200, ae_episodes_num = 1):
		self.ae.n_epochs = ae_episodes_num
		
		timestep = 0
		D_tmp = 0
		counter = 0
		
		while True:
			timestep += 1
			
			# victim's Learning
			self.victim.train_model(1)
			
			if self.victim.MEM.__len__() >= MEMORY_SIZE:
				# AutoEncoder Training
				self.ae._train(self.victim.MEM, MEM_target)
			
				# Obtain Embedding
				Z, Z_target = system.ae._embedding_batch(system.victim.MEM, MEM_target)
				
				# Measure Deviation between Z and Z*
				D = util.Deviation_of_Z(Z, Z_target)

				# Identification Rules: Z|Z* == c
				temporal_difference = abs(D - D_tmp)
				if temporal_difference < 0.05:
					counter += 1
				else:
					counter = 0

				print(f"T={timestep} : Deviation = {D} | TD = {temporal_difference} | counter = {counter}")

				D_tmp = D

			if counter_bar == None:
				if timestep == max_victim_N:
					break
			else:
				if counter >= counter_bar or timestep == max_victim_N:
					break
                    
        return Z, Z_target

				
    def eval_victim(self):
        self.victim.eval_model_memory()
		
    def eval_autoencoder(self):
        self.ae._eval(self.victim.MEM, MEM_target)


if __name__ == "__main__":
	
	from ae.autoencoder import AutoEncoder
	from envs import LunarLander
	
	env = LunarLander()
	
	
	""" AutoEncoder Hyper-parameter """
	
	ae_enc_IN = env.observation_space.shape[0] + 1 # env.state.size + env.select_action.size
	ae_enc_OUT = 100 # embedding_size
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
		"seq_len": 1000, 
		"embedding_len": 100,
		"n_epochs": 2, 
		"lr": 0.001, 
	}
	
	ae_object = AutoEncoder(**ae_args)
	
	""" Victim Hyper-parameter """
	
	victim_args_dqn = {
		"env": env, 
		"MEMORY_SIZE": 5000, 
		"BATCH_SIZE": 512, 
		"GAMMA": 0.99, 
		"EPS_START": 1.0, 
		"EPS_END": 0.01, 
		"EPS_DECAY": 0.996, 
		"TARGET_UPDATE": 10, 
		"n_states": env.observation_space.shape[0], 
		"n_actions": env.action_space.n,
		"learning_rate": 0.001,
	}

	# ''' System Class '''
	# system = System(victim_args_dqn, ae_args)
	# system.train()

	# ''' System_General Class '''
	# system = System_General(ae_object, victim_args_dqn, victim_type="DQN")
	# system.train()

	
	''' System_Identify_Attack_Chance '''
	system = System_Identify_Attack_Chance(ae_object, victim_args_dqn, victim_type="DQN")
	''' pre-train '''
	print("\n ... Pre-Train ..")
	system.train(None)
	''' development '''
	print("\n ... Develop ..")
	system.train(10)
	
	
	
		