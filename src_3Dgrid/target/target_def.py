import os
from os.path import dirname, abspath

import sys
if "../" not in sys.path:
    sys.path.append("../") 

from collections import defaultdict
from collections import namedtuple
from itertools import count
import itertools
import copy
import numpy as np

from envs.env3D_6x6 import GridWorld_3D_env
env = GridWorld_3D_env()

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

UP = np.array([1, 0, 0, 0])
RIGHT = np.array([0, 1, 0, 0])
DOWN = np.array([0, 0, 1, 0])
LEFT = np.array([0, 0, 0, 1])

""" Define partial target policy """
target = defaultdict(lambda: np.zeros(env.action_space.n))

target[0] = DOWN
target[6] = DOWN
target[12] = DOWN
target[18] = DOWN
target[24] = DOWN
target[30] = RIGHT
target[31] = RIGHT
target[32] = RIGHT
target[33] = RIGHT


''' save Target_Trajectories into MEM_target '''

MEM_Target = utils_buffer.Memory(MEMORY_SIZE)

def MakeEpsilonGreedyPolicy(Q, epsilon):
    nA = env.action_space.n
    def policy_fn(observation):
        A = np.ones(nA, dtype = float) * epsilon/nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


policy = MakeEpsilonGreedyPolicy(target, epsilon=0.1)
n_episodes = MEMORY_SIZE//LEN_TRAJECTORY + 1

for i_episode in range(n_episodes):
    # logger
    trajectory_list = []

    # Reset the environment and pick the first action
    state = env.reset()

    # One step in the environment
    for t in range(T_max):
        # logger
        t_sample = []

        # Take a step
        action_probs = policy(state)
        action = np.argmax(action_probs)
        next_state, reward, done, _ = env.step(action)

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

        if done or t+1==T_max:
            break

    # Trajectory to MEMORY with head_padding
    if len(trajectory_list) < LEN_TRAJECTORY:
        padding_state = 0
        padding_action = 0
        n_padding = LEN_TRAJECTORY - len(trajectory_list)
        for i in range(n_padding):
            MEM_Target.push(padding_state, padding_action, padding_state, padding_action)
        for i in range(len(trajectory_list)):
            pre_state = trajectory_list[i][0]
            pre_action = trajectory_list[i][1]
            state = trajectory_list[i][2]
            action = trajectory_list[i][3]
            MEM_Target.push(pre_state, pre_action, state, action)
    else:
        for i in range(LEN_TRAJECTORY):
            pre_state = trajectory_list[i][0]
            pre_action = trajectory_list[i][1]
            state = trajectory_list[i][2]
            action = trajectory_list[i][3]
            MEM_Target.push(pre_state, pre_action, state, action)

