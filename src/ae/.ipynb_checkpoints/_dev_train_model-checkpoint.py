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

# attacker & victim algo
from attack.DDPG import DDPG
from victim.victim_dqn import Victim_DQN

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
    
    ''' ... Train AuotEncoder Model ... '''
    
    
    max_episodes_num = 500
    ae_Model.n_epochs = 10
    
    for iteration in range(5):
        
        victim.reset()
        
        for i_episode in range(max_episodes_num):
            print(f'--- episode = {i_episode} ---')
            victim.train_model(1)

            if victim.MEM.__len__() >= MEMORY_SIZE:
                ae_Model._train(victim.MEM, MEM_target)

            ae_Model._eval(victim.MEM, MEM_target)

            if (i_episode+1)%50 == 0:
                victim.eval_model()

        ae_Model._save("model")