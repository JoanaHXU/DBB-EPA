import numpy as np
import torch
import gym
import argparse
import os
import copy
import math
import sys
import time
import datetime
import tensorboardX

from collections import defaultdict
from itertools import count
from os.path import dirname, abspath

import sys
if "../" not in sys.path:
	sys.path.append("../") 

from attack.DDPG import DDPG
from attack.TD3 import TD3

from victim.system import System_General as system
from ae.autoencoder import AutoEncoder
import utils

# env & target
from envs import LunarLander
from target.target_def import MEM_target

env = LunarLander()
env_init_hyper = env.hyper_condition

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' import configuration '''
from yacs.config import CfgNode as CN

yaml_name_default = 'config/config_default.yaml'
fcfg = open(yaml_name_default)
config_default = CN.load_cfg(fcfg)
config_default.freeze()

SEQ_LEN = config_default.AE.LEN_TRAJECTORY
EMBEDDING_SIZE = config_default.AE.EMBEDDING_SIZE
MEMORY_SIZE = config_default.AE.MEMORY_SIZE

yaml_name_victim = 'config/config_victim.yaml'
fcfg = open(yaml_name_victim)
config_victim = CN.load_cfg(fcfg)
config_victim.freeze()


""" Victim Hyper-parameter """
	
victim_args_dqn = {
	"env": env, 
	"MEMORY_SIZE": MEMORY_SIZE, 
	"BATCH_SIZE": config_victim.dqn.BATCH_SIZE, 
	"GAMMA": config_victim.dqn.GAMMA, 
	"EPS_START": config_victim.dqn.EPS_START, 
	"EPS_END": config_victim.dqn.EPS_END, 
	"EPS_DECAY": config_victim.dqn.EPS_DECAY, 
	"TARGET_UPDATE": config_victim.dqn.TARGET_UPDATE, 
	"n_states": env.observation_space.shape[0], 
	"n_actions": env.action_space.n,
	"learning_rate": config_victim.dqn.LEARNING_RATE,
}

victim_args_ppo = {
	"env": env,
	"MEMORY_SIZE": MEMORY_SIZE, 
	"GAMMA": config_victim.ppo.GAMMA, 
	"LAMBD": config_victim.ppo.LAMBD,
	"n_states": env.observation_space.shape[0], 
	"n_actions": env.action_space.n,
	"BATCH_SIZE": config_victim.ppo.BATCH_SIZE, 
	"CLIP": config_victim.ppo.CLIP, 
	"learning_rate": config_victim.ppo.LEARNING_RATE,
}

victim_args_vpg = {
	"env": env,
	"MEMORY_SIZE": MEMORY_SIZE, 
	"GAMMA": config_victim.vpg.GAMMA, 
	"MAX_LEN": config_victim.vpg.MAX_LEN,
	"n_states": env.observation_space.shape[0], 
	"n_actions": env.action_space.n,
	"learning_rate": config_victim.vpg.LEARNING_RATE,
}

victim_args_a2c = {
	"env": env,
	"MEMORY_SIZE": MEMORY_SIZE,
	"GAMMA": config_victim.a2c.GAMMA,
	"n_states": env.observation_space.shape[0], 
	"n_actions": env.action_space.n,
	"learning_rate": config_victim.a2c.LEARNING_RATE,
}



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# attack network
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--model_dir", default=None)               # TensorBoard folder
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_episodes", default=50, type=int)  # Time steps initial random policy is used
	parser.add_argument("--max_timesteps", default=15, type=int)   # Max time steps to run environment
	parser.add_argument("--max_episodes_num", default=1000, type=int)   # Max episodes to run environment
	parser.add_argument("--eval_freq_episode", default=50, type=int)        # How often (time steps) we evaluate
	parser.add_argument("--expl_noise", default=0.1, type=float)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=512, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2, type=float)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5, type=float)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	# attack implementation
	parser.add_argument("--done_threshold", default=0.05, type=float)  # Threshold for attack_done
	parser.add_argument("--weight", default=0.1, type=float)    # Weight of env_deviations in attack_computing
	# victim hyper-parameter
	parser.add_argument("--victim_algo", default="DQN")
	parser.add_argument("--victim_n_episodes", default=60, type=int)  # number of victim's training episodes in each attack epoch
	# autoencoder hyper-parameter
	parser.add_argument("--ae_n_epochs", default=1, type=int)  # number of victim's training episodes in each attack epoch
	args = parser.parse_args()

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	''' TensorBoard Settings '''
	# Set run dir
	date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
	default_model_name = f"results_{date}"
	model_name = args.model_dir or default_model_name
	model_dir = utils.get_model_dir(model_name)
	# Load loggers and Tensorboard writer
	txt_logger = utils.get_txt_logger(model_dir)
	csv_file, csv_logger = utils.get_csv_logger(model_dir)
	tb_writer = tensorboardX.SummaryWriter(model_dir)
	# Log command and all script arguments
	txt_logger.info("{}\n".format(" ".join(sys.argv)))
	txt_logger.info("{}\n".format(args))
	
# 	try:
# 		status = utils.get_status(model_dir)
# 	except OSError:
# 		status = {"num_frames": 0, "update": 0}
	txt_logger.info("Training status loaded\n")
	
	""" Intialize Attack Network """
	state_dim = EMBEDDING_SIZE + env.attack_space.shape[0]
	action_dim = env.attack_space.shape[0]
	min_action = float(env.attack_space.low[0])
	max_action = float(env.attack_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}
	
	if args.policy == "TD3":
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		Policy = TD3(**kwargs)
		txt_logger.info("TD3 Attack NetWork - Initialize")    
	elif args.policy == "DDPG":
		Policy = DDPG(**kwargs)
		txt_logger.info("DDPG Attack NetWork - Initialize")
	
	Buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	""" AutoEncoder Hyper-parameter """
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
	
# 	if "ae_encoder" in status:
# 		ae_object.Enc_Model.load_state_dict(status["ae_encoder"])
# 	if "ae_decoder_FC" in status:
# 		ae_object.Dec_FC_Model.load_state_dict(status["ae_decoder_FC"])
# 		print("... load actor")
# 	if "ae_decoder_LSTM" in status:
# 		ae_object.Dec_LSTM_Model.load_state_dict(status["ae_decoder_LSTM"])
		
	txt_logger.info("AutoEncoder Initialize")
	
	
	''' Initialize victim-autoencoder System '''
	if args.victim_algo == "DQN":
		system = system(ae_object, victim_args_dqn, victim_type="DQN")
	if args.victim_algo == "PPO":
		system = system(ae_object, victim_args_ppo, victim_type="PPO")
	if args.victim_algo == "VPG":
		system = system(ae_object, victim_args_vpg, victim_type="VPG")
	if args.victim_algo == "A2C":
		system = system(ae_object, victim_args_a2c, victim_type="A2C")
		
	txt_logger.info("System - Initialize")
		
	''' Pre-train AutoEncoder '''
	print("------ PreTrain of AutoEncoder ------")
	system.train(300, args.ae_n_epochs)
	system.eval_autoencoder()
	txt_logger.info("AutoEncoder - Pretrain")
	
	
	""" Training Setting """
	done = False
	# csv log
	header = ["Length of Episode", "Cost of Episode"]
	csv_logger.writerow(header)
	

	""" Training """
	for i_episode in range(args.max_episodes_num):

		txt_logger.info(f"\n----------------- Episode = {i_episode+1} -----------------")

		cumulative_cost = 0
		tic_episode = time.time()

		# reset environment
		env.reset()
		env.clear_attack()
		system.victim.reset()

		## initial embedding
		victim_info = np.zeros((1,EMBEDDING_SIZE))
		victim_tensor = torch.from_numpy(victim_info)
		victim_tensor_4d = victim_tensor.unsqueeze(0).unsqueeze(0)

		env_info = np.array([env.hyper_condition])
		env_tensor = torch.from_numpy(env_info)
		env_tensor_4d = env_tensor.unsqueeze(0).unsqueeze(0)

		x = torch.cat((victim_tensor_4d, env_tensor_4d), 3)

		for t in range(args.max_timesteps):

			## attack action: u
			if i_episode < args.start_episodes:
				u = env.attack_space.sample()
			else:
				u = (
					Policy.select_action(np.array(x))
					+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
				).clip(-max_action, max_action)

			## implement attack u
			u_attack = env.attack(u)
			txt_logger.info(f"H-Env: {np.round(env.hyper_condition,2)}")

			## update victim & autoEncoder
			tic = time.time()
			
			system.train(args.victim_n_episodes, args.ae_n_epochs)
			
# 			system.eval_victim()
			system.eval_autoencoder()
			toc = time.time()
			print(f".....Running_Time = {toc-tic}s")

			# get embeddings of a <BATCH>
			Z, Z_target = system.ae._embedding_batch(system.victim.MEM, MEM_target)
			
			# get attack_cost, attack_done, success_rate <BATCH>
			cost, done, success_rate = utils.Attack_Cost_Done_batch(env, Z, Z_target, env.hyper_condition, env_init_hyper, args.done_threshold, args.weight)
			cumulative_cost += cost

			# log training data
			data = [i_episode+1, t, done, cost, success_rate]
			txt_logger.info("Episode.{}-T.{} : done: {:.0f} | cost: {:.3f} | success: {:.2f}\n".format(*data))

			''' next input: {x'}  '''
			next_victim_info = torch.from_numpy(Z[-1]).unsqueeze(0)
#             next_victim_info = torch.from_numpy(Z)
			next_victim_tensor = next_victim_info.data.type(torch.DoubleTensor)
			next_victim_tensor_4d = next_victim_tensor.unsqueeze(0).unsqueeze(0)

			next_env_info = np.array([env.hyper_condition])
			next_env_tensor = torch.from_numpy(next_env_info)
			next_env_tensor_4d = next_env_tensor.unsqueeze(0).unsqueeze(0)

			next_x = torch.cat((next_victim_tensor_4d, next_env_tensor_4d), 3)

			''' reply buffer '''
			cost_attack = cost*(-1)
			Buffer.add(x.view(state_dim), u_attack, next_x.view(state_dim), cost_attack, done)

			''' update input '''
			x = copy.deepcopy(next_x)

			''' train Policy '''
			if i_episode >= args.start_episodes:
				Policy.train(Buffer, args.batch_size)

			''' recording '''
			if done: 
				# training time of episode
				toc_episode = time.time()
				time_episode = toc_episode - tic_episode
				# txt_logger
				data = [i_episode+1, t+1, time_episode, cumulative_cost]
				txt_logger.info("=== Done === Episode Num: {} | Episode T: {} | Running Time: {:.3f} | Cost: {:.3f} ".format(*data))
				# csv log
				data = [t+1, cumulative_cost]
				csv_logger.writerow(data)
				csv_file.flush()
				# tensorboard log
				tb_writer.add_scalar('TimeSteps/train', t+1, i_episode+1)
				tb_writer.add_scalar('Cost/train', cumulative_cost, i_episode+1)

				# Reset statistics
				done = False
				episode_reward = 0
				
				''' save victim's policy_net '''
				victim_model_title = str(i_episode+1) + "_victim_model"
				torch.save(system.victim.policy_net.state_dict(), f"./{model_dir}/{victim_model_title}")
				txt_logger.info("Victim's policy network - saved")

				break

			if t+1==args.max_timesteps:
				# training time of episode
				toc_episode = time.time()
				time_episode = toc_episode - tic_episode
				# txt_logger
				data = [i_episode+1, t+1, time_episode, cumulative_cost]
				txt_logger.info("=== Tmax === Episode Num: {} | Episode T: {} | Running Time: {:.3f} | Cost: {:.3f} ".format(*data))
				# csv log
				data = [t+1, cumulative_cost]
				csv_logger.writerow(data)
				csv_file.flush()
				# tensorboard log
				tb_writer.add_scalar('TimeSteps/train', t+1, i_episode+1)
				tb_writer.add_scalar('Cost/train', cumulative_cost, i_episode+1)

				# Reset statistics
				episode_reward = 0
				
				break
				
		''' save status'''
		if (i_episode + 1) % (args.eval_freq_episode//2) == 0:
			## save Status
			status = {"critic_state": Policy.critic.state_dict(), "critic_optimizer_state": Policy.critic_optimizer.state_dict(),
					 "actor_state": Policy.actor.state_dict(), "actor_optimizer_state": Policy.actor_optimizer.state_dict(),
					 "ae_encoder": system.ae.Enc_Model.state_dict(), 
					 "ae_decoder_FC": system.ae.Dec_FC_Model.state_dict(), "ae_decoder_LSTM": system.ae.Dec_LSTM_Model.state_dict()}
			utils.save_status(status, model_dir)
			## logger
			txt_logger.info("Status - saved")
		
		''' save model '''
		if (i_episode + 1) % args.eval_freq_episode == 0:
			## save the model
			model_no = str(i_episode + 1)
			Policy.save(f"./{model_dir}/{model_no}")
			# save LSTM-AutoEncoder 
			system.ae._save(f"./{model_dir}/{model_no}")
			txt_logger.info("Policy Model and AutoEncoeder Model - saved")
			
			''' save victim's policy-net '''
			victim_model_title = str(i_episode+1) + "_check_" + "_victim_model"
			torch.save(system.victim.policy_net.state_dict(), f"./{model_dir}/{victim_model_title}")
			txt_logger.info("Victim's policy network - saved")
			
			