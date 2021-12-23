import numpy as np
import torch
import gym
import argparse
import os
import copy
import math
import sys
import time

from collections import defaultdict
from itertools import count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import ae.autoencoder as AutoEncoder
from attack.DDPG import DDPG
from utils import utils_log, utils_buffer, utils_attack

from victim.system import System
from target.target_def import MEM_Target, target

# env
from envs.env3D_6x6 import GridWorld_3D_env
env = GridWorld_3D_env()
env_init_altitude = env.altitude.reshape(-1)


# TensorBoard
import tensorboardX
import datetime

# Configuration
from yacs.config import CfgNode as CN
yaml_name='config/config_default.yaml'
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()

SEQ_LEN = config.AE.SEQ_LEN
EMBEDDING_SIZE = config.AE.EMBEDDING_SIZE
MEMORY_SIZE = config.AE.MEMORY_SIZE


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DDPG")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--model_dir", default=None)               # TensorBoard folder
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_episodes", default=100, type=int)  # Time steps initial random policy is used
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
    parser.add_argument("--weight", default=0.1, type=float)    # Weight of env_deviations in attack_computing
    # victim hyper-parameter
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
    model_dir = utils_log.get_model_dir(model_name)

    # Load loggers and Tensorboard writer
    txt_logger = utils_log.get_txt_logger(model_dir)
    csv_file, csv_logger = utils_log.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)
    
    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))
    

    """ Intialize Attack Network """
    state_dim = EMBEDDING_SIZE + env.nS
    action_dim = env.Attack_ActionSpace.shape[0]
    max_action = float(env.Attack_ActionSpace.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }
    Policy = DDPG(**kwargs)
    Buffer = utils_buffer.ReplayBuffer(state_dim, action_dim, max_size=int(1e5))
    
    txt_logger.info("Attack Strategy NetWork Initialized")


    """ Intialize System (victim, autoencoder) """
    ae_enc_IN = 2
    ae_enc_OUT = EMBEDDING_SIZE # embedding size
    ae_enc_N_layer = 1

    ae_dec_fc_IN = ae_enc_OUT + 1
    ae_dec_fc_OUT = env.nA

    ae_dec_lstm_IN = ae_enc_OUT + 2
    ae_dec_lstm_OUT = env.nS
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
    
    victim_args = {
        "env": env, 
        "MEMORY_SIZE": MEMORY_SIZE, 
        "discount_factor": 1.0, 
        "alpha": 0.1, 
        "epsilon": 0.1,
    }
    
    system = System(victim_args, ae_args)


    """ Pre-Train Embedding Network """
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

        txt_logger.info(f"-------- Episode = {i_episode+1} ----------")

        cumulative_cost = 0
        tic_episode = time.time()

        # reset environment
        env.reset_altitude()
        system.victim.reset()


        ''' define initial embedding '''
        victim_info = np.zeros((1,EMBEDDING_SIZE))
        victim_tensor = torch.from_numpy(victim_info)
        victim_tensor_4d = victim_tensor.unsqueeze(0).unsqueeze(0)

        env_info = env.altitude.copy()
        env_tensor = torch.from_numpy(env_info)
        env_tensor = env_tensor.view(1,36)
        env_tensor_4d = env_tensor.unsqueeze(0).unsqueeze(0)

        x = torch.cat((victim_tensor_4d, env_tensor_4d), 3)

        for t in range(args.max_timesteps):

            ''' attack action: u '''
            if i_episode < args.start_episodes:
                u = env.Attack_ActionSpace.sample()
            else:
                u = (
                    Policy.select_action(np.array(x))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

            ''' implement attack u '''
            env.Attack_Env(u)

            ''' update victim policy '''
            system.train(args.victim_n_episodes, args.ae_n_epochs)
            system.eval_victim()
            system.eval_autoencoder()

            Z, Z_target = system.ae._embedding_BATCH(system.victim.MEM, MEM_Target)

            ''' attack cost '''
            cost = (-1) * utils_attack.Attack_Cost_BATCH(env, Z, Z_target, env.altitude.reshape(-1), env_init_altitude, args.weight)
            cumulative_cost += cost

            ''' attack done '''
            done, success_rate = utils_attack.Attack_Done(env, target, system.victim.Q)

            # log training data
            data = [i_episode, t, cost, done, success_rate]
            txt_logger.info("episode = {} | t = {} | cost: {:.4f} | done: {} | success: {:.3f}".format(*data))

            ''' next input: {x'}  '''
            next_victim_info = torch.from_numpy(Z[-1]).unsqueeze(0)
            next_victim_tensor = next_victim_info.data.type(torch.DoubleTensor)
            next_victim_tensor_4d = next_victim_tensor.unsqueeze(0).unsqueeze(0)

            next_env_info = env.altitude.copy()
            next_env_tensor = torch.from_numpy(next_env_info)
            next_env_tensor = next_env_tensor.view(1,36)
            next_env_tensor_4d = next_env_tensor.unsqueeze(0).unsqueeze(0)

            next_x = torch.cat((next_victim_tensor_4d, next_env_tensor_4d), 3)

            ''' reply buffer '''
            Buffer.add(x.view(state_dim), u, next_x.view(state_dim), cost, done)

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
                txt_logger.info("*** Done *** Episode Num: {} | Episode T: {} | Running Time: {:.3f} | Cost: {:.3f} ".format(*data))
                # csv log
                data = [t+1, -cumulative_cost]
                csv_logger.writerow(data)
                csv_file.flush()
                # tensorboard log
                tb_writer.add_scalar('TimeSteps/train', t+1, i_episode+1)
                tb_writer.add_scalar('Cost/train', -cumulative_cost, i_episode+1)

                # Reset statistics
                done = False
                episode_reward = 0

                break

            if t+1==args.max_timesteps:
                # training time of episode
                toc_episode = time.time()
                time_episode = toc_episode - tic_episode
                # txt_logger
                data = [i_episode+1, t+1, time_episode, cumulative_cost]
                txt_logger.info("*** Tmax *** Episode Num: {} | Episode T: {} | Running Time: {:.3f} | Cost: {:.3f} ".format(*data))
                # csv log
                data = [t+1, -cumulative_cost]
                csv_logger.writerow(data)
                csv_file.flush()
                # tensorboard log
                tb_writer.add_scalar('TimeSteps/train', t+1, i_episode+1)
                tb_writer.add_scalar('Cost/train', -cumulative_cost, i_episode+1)

                # Reset statistics
                episode_reward = 0

                break

        ''' save model & status '''
        if (i_episode + 1) % args.eval_freq_episode == 0:
            ## Save the model
            model_no = str(i_episode + 1)
            Policy.save(f"./{model_dir}/{model_no}")
            # save LSTM-AutoEncoder 
            system.ae._save(f"./{model_dir}/{model_no}")
            txt_logger.info("Policy Model and AutoEncoeder Model - saved")
#             ## Save Status
#             status = {"critic_state": Policy.critic.state_dict(), "critic_optimizer_state": Policy.critic_optimizer.state_dict(),
#                      "actor_state": Policy.actor.state_dict(), "actor_optimizer_state": Policy.actor_optimizer.state_dict()}
#             utils_log.save_status(status, model_dir)
            ## log
#             txt_logger.info("Status saved")
            