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
if "../" not in sys.path:
    sys.path.append("../") 

# utils & tools
import ae.autoencoder as AE
import utils

# env & target
from envs import LunarLander
from target.target_def import MEM_target

env = LunarLander()
env.clear_attack()
ENV_INIT_CONDITION = env.hyper_condition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' import configuration '''
from yacs.config import CfgNode as CN
yaml_name='../config/config_default.yaml'
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()

LEN_TRAJECTORY = config.AE.LEN_TRAJECTORY
SEQ_LEN = config.AE.LEN_TRAJECTORY
MEMORY_SIZE = config.AE.MEMORY_SIZE
EMBEDDING_SIZE = config.AE.EMBEDDING_SIZE
TARGET_FOLDER = config.ATTACK.TARGET_FOLDER
TARGET_POLICY_TITLE = config.ATTACK.TARGET_POLICY_TITLE



if __name__ == "__main__":
    
    # parser argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    # attack implementation
    parser.add_argument("--done_threshold", default=0.01, type=float)  # Threshold for attack_done
    parser.add_argument("--weight", default=0.01, type=float)    # Weight of env_deviations in attack_computing
    # victim hyper-parameter
    parser.add_argument("--victim_n_episodes", default=80, type=int)  # number of victim's training episodes in each attack epoch
    
    args = parser.parse_args()

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    """ Initialize Victim Network """
    v_BATCH_SIZE = 64
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
        "learning_rate": 0.0001,
    }
    
    victim = Victim_DQN(**victim_args)
    model_path = os.path.join(dirname(dirname(abspath(__file__))), TARGET_FOLDER, TARGET_POLICY_TITLE)
    
    """ Intialize AutoEncoder """
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
        "n_epochs": 50, 
        "lr": 0.01, 
    }

    ae_Model = AE.AutoEncoder(**ae_args)
    ae_Memory = utils.Memory(MEMORY_SIZE)
    
    ae_Model._load("model_AutoEncoder")
    
    """ training & evaluation of AutoEncoder """
    
    ''' I. victim's trajectory from target policy '''
    print("............. evaluate on TARGET policy ...............")
    
    victim.reset()
    victim.policy_net.load_state_dict(torch.load(model_path))

    for i in range(5):
        score = 0
        trajectory_list = []
        
        state = victim.env.reset()
        state = torch.from_numpy(state).view(1,victim.n_states).to(device)
        done = False
        
        for t in range(2000):
            t_sample = []
            
            action = victim.select_action(state).to(device)
            next_state, reward, done, _ = victim.env.step(action.item())
            next_state = torch.from_numpy(next_state).view(1, victim.n_states).to(device)
            reward = torch.tensor([reward], dtype=torch.float32, device=device)

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
                print("--- LunarLander -- episode = {} | t = {} | score = {}".format(i, t, round(score.item(),2)))
                break
            if t+1 == 2000:
                print("--- LunarLander -- episode = {} | t = {} | score = {}".format(i, t, round(score.item(),2)))
                break
                
        # Trajectory to MEMORY with head_padding
        if len(trajectory_list) < LEN_TRAJECTORY:
            padding_state = torch.zeros(state.shape, device=device)
            padding_action = torch.zeros(action.shape, device=device)
            n_padding = LEN_TRAJECTORY - len(trajectory_list)
            for i in range(n_padding):
                victim.MEM.push(padding_state, padding_action, padding_state, padding_action)
            for i in range(len(trajectory_list)):
                pre_state = trajectory_list[i][0]
                pre_action = trajectory_list[i][1]
                state = trajectory_list[i][2]
                action = trajectory_list[i][3]
                victim.MEM.push(pre_state, pre_action, state, action)
        else:
            for i in range(LEN_TRAJECTORY):
                pre_state = trajectory_list[i][0]
                pre_action = trajectory_list[i][1]
                state = trajectory_list[i][2]
                action = trajectory_list[i][3]
                victim.MEM.push(pre_state, pre_action, state, action)


    print(f"Size of Actual_Memory = {victim.MEM.__len__()}")
    print(f"Size of Target_Memory = {MEM_target.__len__()}")

    ae_Model._eval(victim.MEM, MEM_target)

    ''' embedding by BATCH-trajectories '''
    Z_batch, Z_target_batch = ae_Model._embedding_batch(victim.MEM, MEM_target)
    cost, done, success_rate = utils.Attack_Cost_Done_batch(Z_batch, Z_target_batch, env.hyper_condition, ENV_INIT_CONDITION, args.done_threshold, args.weight)
    print(f"BATCH: cost = {cost:.2f} | success_rate = {success_rate:.2f} ")

    ''' embedding by SINGLE-trajectory'''
    Z_single, Z_target_single = ae_Model._embedding_single(victim.MEM, MEM_target)
    cost, done, success_rate = utils.Attack_Cost_Done(Z_single, Z_target_single, env.hyper_condition, ENV_INIT_CONDITION, args.done_threshold, args.weight)
    print(f"SINGLE: cost = {cost:.2f} | success_rate = {success_rate:.2f} ")

        
    ''' II. victim's trajectory from random policy '''

    print("............. evaluate on RANDOM policy ...............")
    victim.reset()
    victim.train_model(5)
    
    print(f"Size of Actual_Memory = {victim.MEM.__len__()}")
    print(f"Size of Target_Memory = {MEM_target.__len__()}")

    ae_Model._eval(victim.MEM, MEM_target)
        
    ''' embedding by BATCH-trajectories '''
    Z_batch, Z_target_batch = ae_Model._embedding_batch(victim.MEM, MEM_target)
    print(f"length of Z_batch is {len(Z_batch)}")
    cost, done, success_rate = utils.Attack_Cost_Done_batch(Z_batch, Z_target_batch, env.hyper_condition, ENV_INIT_CONDITION, args.done_threshold, args.weight)
    print(f"BATCH: cost = {cost:.2f} | success_rate = {success_rate:.2f} ")
        
    ''' embedding by SINGLE-trajectory'''
    Z_single, Z_target_single = ae_Model._embedding_single(victim.MEM, MEM_target)
        
    cost, done, success_rate = utils.Attack_Cost_Done(Z_single, Z_target_single, env.hyper_condition, ENV_INIT_CONDITION, args.done_threshold, args.weight)
    print(f"SINGLE: cost = {cost:.2f} | success_rate = {success_rate:.2f} ")
        
    
