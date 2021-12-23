import pickle
import torch
import copy

import os
from os.path import dirname, abspath

import sys
if "../" not in sys.path:
    sys.path.append("../") 
from utils import utils_buf

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' import configuration '''
from yacs.config import CfgNode as CN
yaml_name = os.path.join(dirname(dirname(abspath(__file__))), "config", "config_default.yaml")
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()

LEN_TRAJECTORY = config.AE.LEN_TRAJECTORY
MEMORY_SIZE = config.AE.MEMORY_SIZE
TARGET_TITLE = config.ATTACK.TARGET_TITLE
TARGET_FOLDER = config.ATTACK.TARGET_FOLDER


''' file_path of .pkl '''
FILE_PATH = os.path.join(dirname(dirname(abspath(__file__))), TARGET_FOLDER, TARGET_TITLE)

''' read from .pkl '''
open_file = open(FILE_PATH, "rb")
target_trajectory_list = pickle.load(open_file)
open_file.close()

len_target_trajetory = len(target_trajectory_list)


''' save trajectories into MEM_target '''
MEM_target = utils_buf.Memory(MEMORY_SIZE)

num_trajectories = int(MEMORY_SIZE/LEN_TRAJECTORY)

for n in range(num_trajectories):
    trajectory_list = []
    length = 0
    for i in range(len_target_trajetory):
        t_sample = []
        state = target_trajectory_list[i][0]
        action = target_trajectory_list[i][1]

        if i != 0:
            t_sample.append(pre_state)
            t_sample.append(pre_action)
            t_sample.append(state)
            t_sample.append(action)
            trajectory_list.append(t_sample)

        pre_state = copy.deepcopy(state)
        pre_action = copy.deepcopy(action)
            
    # Trajectory to MEMORY with head_padding
    if len(trajectory_list) < LEN_TRAJECTORY:
        padding_state = torch.zeros(state.shape)
        padding_action = torch.zeros(action.shape)
        n_padding = LEN_TRAJECTORY - len(trajectory_list)
        for i in range(n_padding):
            MEM_target.push(padding_state, padding_action, padding_state, padding_action)
        for i in range(len(trajectory_list)):
            pre_state = trajectory_list[i][0]
            pre_action = trajectory_list[i][1]
            state = trajectory_list[i][2]
            action = trajectory_list[i][3]
            MEM_target.push(pre_state, pre_action, state, action)
    else:
        for i in range(LEN_TRAJECTORY):
            pre_state = trajectory_list[i][0]
            pre_action = trajectory_list[i][1]
            state = trajectory_list[i][2]
            action = trajectory_list[i][3]
            MEM_target.push(pre_state, pre_action, state, action)


if __name__ == "__main__":
    print(f"Length of Memory: {MEM_target.__len__()}")
    print(f"Samples:\n {MEM_target.sample(2)}")
    print(f"First two samples:\n {MEM_target.memory[0:2]}")
    

