import math
import random
import numpy as np
import copy

from collections import namedtuple
from collections import defaultdict
from itertools import count
import itertools

import os
from os.path import dirname, abspath

import sys
if "../" not in sys.path:
    sys.path.append("../") 

from envs.env3D_6x6 import GridWorld_3D_env
from utils import utils_buffer


''' import configuration '''
from yacs.config import CfgNode as CN
yaml_name = os.path.join(dirname(dirname(abspath(__file__))), "config", "config_default.yaml")
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()

LEN_TRAJECTORY = config.AE.LEN_TRAJECTORY
MEMORY_SIZE = config.AE.MEMORY_SIZE
T_max = config.VICTIM.TMAX


class VictimAgent():
    
    def __init__(self, env, MEMORY_SIZE, discount_factor=1.0, alpha=0.1, epsilon=0.1):
        self.env = env
        self.memory_size = MEMORY_SIZE
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.epsilon = epsilon
        
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.MEM = utils_buffer.Memory(MEMORY_SIZE)
        
    def reset(self):
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.MEM = utils_buffer.Memory(self.memory_size)
        

    """
    Q-learning algorithm with transfer_Q
    """
    def MakeEpsilonGreedyPolicy(self):
        nA = self.env.action_space.n
        def policy_fn(observation):
            A = np.ones(nA, dtype = float) * self.epsilon/nA
            best_action = np.argmax(self.Q[observation])
            A[best_action] += (1.0 - self.epsilon)
            return A
        return policy_fn

    def Train_Model(self, num_episodes):

        # The policy we're following
        policy = self.MakeEpsilonGreedyPolicy()

        for i_episode in range(num_episodes):
            # logger
            trajectory_list = []
            score = 0
            
            # Reset the environment and pick the first action
            state = self.env.reset()

            # One step in the environment
            for t in itertools.count():
                # logger
                t_sample = []

                # Take a step
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = self.env.step(action)

                # TD Update
                best_next_action = np.argmax(self.Q[next_state])    
                td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
                td_delta = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_delta
                
                # save transitions
                if t!=0:
                    t_sample.append(pre_state)
                    t_sample.append(pre_action)
                    t_sample.append(state)
                    t_sample.append(action)
                    trajectory_list.append(t_sample)
                    
                # Move to the next state
                pre_state = copy.deepcopy(state)
                pre_action = copy.deepcopy(action)
                # update state
                state = copy.deepcopy(next_state)
                score += reward

                if done:
#                     print(f"Timsteps = {t} | Scores = {score}")
                    break

            # Trajectory to MEMORY with head_padding
            if len(trajectory_list) < LEN_TRAJECTORY:
                padding_state = 0
                padding_action = 0
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
                
#             # display training progress
#             if (i_episode + 1) % 100 == 0:
#                 print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
#                 sys.stdout.flush()

#         return self.Q


    """
    Function to generate trajectories
    """
    def Eval_Model(self, n_episode=1):
        # logger
        score = 0

        policy = self.Get_Policy()

        for i_episode in range(n_episode):

            # Reset the environment and pick the first action
            state = self.env.reset()

            # One step in the environment
            for t in range(T_max):

                # Take a step
                action = policy[state].argmax()
                next_state, reward, done, _ = self.env.step(action)
                score += reward

                if done or t+1==T_max:
                    print(f"Q-Victim: Timsteps = {t} | Scores = {score}")
                    break

                # update state
                state = copy.deepcopy(next_state)
                

    """
    Show policy from Q
    """
    def Show_PolicyQ(self):
        policy_Q = np.zeros(self.env.shape)

        for i in self.Q:
            best_action = np.argmax(self.Q[i])
            policy_Q[np.unravel_index(i, self.env.shape)] = best_action

        print(f"\n {policy_Q}")

    """
    Get policy-table from Q-dictionary
    """
    def Get_Policy(self):
        policy = np.zeros((self.env.nS, self.env.nA))
        for i in self.Q:
            best_action = np.argmax(self.Q[i])
            policy[i][best_action] = 1

        return policy
    
                
                
if __name__ == "__main__":
    
    env = GridWorld_3D_env()
    
    victim_args = {
        "env": env, 
        "MEMORY_SIZE": 100, 
        "discount_factor": 1.0, 
        "alpha": 0.1, 
        "epsilon": 0.1,
    }
    
    victim = VictimAgent(**victim_args)
    victim.Train_Model(200)
    victim.Show_PolicyQ()
    victim.Eval_Model()
    
    print(f"Size of Memory = {victim.MEM.__len__()}")
    print(f"First 5 Trajectory is \n{victim.MEM.memory[0:5]}")
    
    