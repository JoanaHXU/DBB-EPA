import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

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



'''------------- memory Class -------------------'''

def cumulative_sum(array, gamma=1.0):
    curr = 0
    cumulative_array = []

    for a in array[::-1]:
        curr = a + gamma * curr
        cumulative_array.append(curr)

    return cumulative_array[::-1]


class Episode:
    def __init__(self, gamma=0.99, lambd=0.95):
        self.observations = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.values = []
        self.log_probabilities = []
        self.gamma = gamma
        self.lambd = lambd

    def append(self, observation, action, reward, value, log_probability, reward_scale=20):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward / reward_scale)
        self.values.append(value)
        self.log_probabilities.append(log_probability)

    def end_episode(self, last_value):
        rewards = np.array(self.rewards + [last_value])
        values = np.array(self.values + [last_value])

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantages = cumulative_sum(deltas.tolist(), gamma=self.gamma * self.lambd)

        self.rewards_to_go = cumulative_sum(rewards.tolist(), gamma=self.gamma)[:-1]


def normalize_list(array):
    array = np.array(array)
    array = (array - np.mean(array)) / (np.std(array) + 1e-5)
    return array.tolist()


class History(Dataset):
    def __init__(self):
        self.episodes = []
        self.observations = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.log_probabilities = []

    def free_memory(self):
        del self.episodes[:]
        del self.observations[:]
        del self.actions[:]
        del self.advantages[:]
        del self.rewards[:]
        del self.rewards_to_go[:]
        del self.log_probabilities[:]

    def add_episode(self, episode):
        self.episodes.append(episode)

    def build_dataset(self):
        for episode in self.episodes:
            self.observations += episode.observations
            self.actions += episode.actions
            self.advantages += episode.advantages
            self.rewards += episode.rewards
            self.rewards_to_go += episode.rewards_to_go
            self.log_probabilities += episode.log_probabilities

        assert (
            len(
                {
                    len(self.observations),
                    len(self.actions),
                    len(self.advantages),
                    len(self.rewards),
                    len(self.rewards_to_go),
                    len(self.log_probabilities),
                }
            )
            == 1
        )

        self.advantages = normalize_list(self.advantages)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (
            self.observations[idx],
            self.actions[idx],
            self.advantages[idx],
            self.log_probabilities[idx],
            self.rewards_to_go[idx],
        )


'''## ---------------- Policy_NetWork ------------------'''

class PolicyNetwork(torch.nn.Module):
    
    def __init__(self, nS, nA, fc1_units=150, fc2_units=120):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(nS, fc1_units) # first fully-connected layer fc1
        self.fc2 = nn.Linear(fc1_units, fc2_units)  # second fully-connected layer fc2
        self.fc3 = nn.Linear(fc2_units, nA) # third fully-connected layer fc3

    def forward(self, x):
        x = x.to(device)
        x = self.fc1(x)
        x = F.relu(x) # 1-st rectified nonlinear layer, state_size = 8, fc1_units = 64, tensor x is 64x8 = 512 units 
        x = self.fc2(x)
        x = F.relu(x) # 2-st rectified nonlinear layer
#         x = self.fc3(x)
        x = F.softmax(self.fc3(x), dim=-1)
    
        return x
    

    def sample_action(self, state):
        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)
        if len(state.size()) == 1:
            state = state.unsqueeze(0)
        y = self(state)
        dist = Categorical(y)
        action = dist.sample()
        log_probability = dist.log_prob(action)

        return action.item(), log_probability.item()

    
    def best_action(self, state):
        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)
        if len(state.size()) == 1:
            state = state.unsqueeze(0)
        y = self(state).squeeze()
        action = torch.argmax(y)

        return action.item()
    

    def evaluate_actions(self, states, actions):
        y = self(states)
        dist = Categorical(y)
        entropy = dist.entropy()
        log_probabilities = dist.log_prob(actions)

        return log_probabilities, entropy

'''## ------------ Value Network ---------------'''
    
class ValueNetwork(torch.nn.Module):

    def __init__(self, nS, fc1_units=150, fc2_units=120):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(nS, fc1_units) # first fully-connected layer fc1
        self.fc2 = nn.Linear(fc1_units, fc2_units)  # second fully-connected layer fc2
        self.fc3 = nn.Linear(fc2_units, 1) # third fully-connected layer fc3

    def forward(self, x):
        x = x.to(device)
        x = self.fc1(x)
        x = F.relu(x) # 1-st rectified nonlinear layer, state_size = 8, fc1_units = 64, tensor x is 64x8 = 512 units 
        x = self.fc2(x)
        x = F.relu(x) # 2-st rectified nonlinear layer
        x = self.fc3(x)
        x = x.squeeze(1)
        return x

    def state_value(self, state):

        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)
        if len(state.size()) == 1:
            state = state.unsqueeze(0)
        y = self(state)

        return y.item()
    
    
'''## ---------------- Victim_PPO -------------------- ##'''

class Victim_PPO():
    
    def __init__(self, env, MEMORY_SIZE, GAMMA, LAMBD, n_states, n_actions, BATCH_SIZE=32, CLIP=0.2, learning_rate=0.001):
        self.env = env
        self.n_states = n_states
        self.n_actions = n_actions
        self.GAMMA = GAMMA
        self.LAMBD = LAMBD
        
        self.reward_scale = 20.0
        self.state_scale=1.0
        
        self.batch_size = BATCH_SIZE
        self.clip = CLIP
        self.learning_rate = learning_rate
        
        self.n_epoch = 4
        
        self.value_model = ValueNetwork(n_states).to(device)
        self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=self.learning_rate)

        self.policy_net = PolicyNetwork(n_states, n_actions).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        self.MEMORY_SIZE = MEMORY_SIZE
        self.MEM = utils_buf.Memory(MEMORY_SIZE)
        self.history = History()

        
    def reset(self):
        
        self.value_model = ValueNetwork(self.n_states).to(device)
        self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=self.learning_rate)

        self.policy_net = PolicyNetwork(self.n_states, self.n_actions).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        self.MEM = utils_buf.Memory(self.MEMORY_SIZE)
        self.history = History()


    def train_value_network(self, data_loader, epochs=4):
        epochs_losses = []

        for i in range(self.n_epoch):

            losses = []

            for observations, _, _, _, rewards_to_go in data_loader:
                observations = observations.float().to(device)
                rewards_to_go = rewards_to_go.float().to(device)

                self.value_optimizer.zero_grad()

                values = self.value_model(observations)

                loss = F.mse_loss(values, rewards_to_go)

                loss.backward()

                self.value_optimizer.step()

                losses.append(loss.item())

            mean_loss = np.mean(losses)

            epochs_losses.append(mean_loss)

        return epochs_losses


    def ac_loss(self, new_log_probabilities, old_log_probabilities, advantages):
        
        probability_ratios = torch.exp(new_log_probabilities - old_log_probabilities)
        clipped_probabiliy_ratios = torch.clamp(probability_ratios, 1 - self.clip, 1 + self.clip)

        surrogate_1 = probability_ratios * advantages
        surrogate_2 = clipped_probabiliy_ratios * advantages

        return -torch.min(surrogate_1, surrogate_2)


    def train_policy_network(self, data_loader):
        epochs_losses = []

        c1 = 0.01

        for i in range(self.n_epoch):

            losses = []

            for observations, actions, advantages, log_probabilities, _ in data_loader:
                observations = observations.float().to(device)
                actions = actions.long().to(device)
                advantages = advantages.float().to(device)
                old_log_probabilities = log_probabilities.float().to(device)

                self.policy_optimizer.zero_grad()

                new_log_probabilities, entropy = self.policy_net.evaluate_actions(
                    observations, actions
                )

                loss = (
                    self.ac_loss(
                        new_log_probabilities,
                        old_log_probabilities,
                        advantages,
                    ).mean()
                    - c1 * entropy.mean()
                )

                loss.backward()

                self.policy_optimizer.step()

                losses.append(loss.item())

            mean_loss = np.mean(losses)

            epochs_losses.append(mean_loss)

        return epochs_losses
    
    
    def train_model(self, max_episode_n = 200):
        
        self.policy_net.train()
        
        for i_episode in range(max_episode_n):
            
            # log
            score = 0
            trajectory_list = []

            state = self.env.reset()
            
            episode = Episode(self.GAMMA, self.LAMBD)
            done = False

            for timestep in range(T_max):
                
                # log:
                t_sample = []

                action, log_probability = self.policy_net.sample_action(state / self.state_scale)
                value = self.value_model.state_value(state / self.state_scale)

#                 self.env.render()

                next_state, reward, done, info = self.env.step(action)

                episode.append(
                    observation=state / self.state_scale,
                    action=action,
                    reward=reward,
                    value=value,
                    log_probability=log_probability,
                    reward_scale=self.reward_scale,
                )
                
                if timestep != 0:
                    pre_state_tensor = torch.from_numpy(pre_state).view(1,self.n_states).to(device)
                    pre_action_tensor = torch.tensor([pre_action], dtype=torch.float32, device=device)
                    state_tensor = torch.from_numpy(state).view(1,self.n_states).to(device)
                    action_tensor = torch.tensor([action], dtype=torch.float32, device=device)
                    t_sample.append(pre_state_tensor)
                    t_sample.append(pre_action_tensor)
                    t_sample.append(state_tensor)
                    t_sample.append(action_tensor)
                    trajectory_list.append(t_sample)
                    
                # Move to the next state
                pre_state = copy.deepcopy(state)
                pre_action = copy.deepcopy(action)
                state = copy.deepcopy(next_state)
                score += reward

                if done:
                    episode.end_episode(last_value=0)
#                     print("PPO-LunarLander: episode = {} | t = {} | score = {}".format(i_episode, timestep, round(score.item(),0)))
                    break

                if timestep + 1 == T_max:
                    value = self.value_model.state_value(state / self.state_scale)
                    episode.end_episode(last_value=value)
#                     print("PPO-LunarLander: episode = {} | t = {} | score = {}".format(i_episode, timestep, round(score.item(),0)))
            
            self.history.add_episode(episode)
            
            # Trajectory to MEMORY with head_padding
            if len(trajectory_list) < LEN_TRAJECTORY:
                # action type is int, convert to tensor for shape recognition
                action = torch.tensor([action], dtype=torch.float32, device=device)
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
            

            self.history.build_dataset()
            data_loader = DataLoader(self.history, batch_size=self.batch_size, shuffle=True)

            policy_loss = self.train_policy_network(data_loader)
            value_loss = self.train_value_network(data_loader)

            self.history.free_memory()
            
        self.env.close()
        
        return policy_loss, value_loss

        
    def eval_model_memory(self):
        
        self.policy_net.eval()
        
        # log
        score = 0
        trajectory_list = []

        state = self.env.reset()
        done = False

        for timestep in range(T_max):

            # log:
            t_sample = []

            action = self.policy_net.best_action(state / self.state_scale)
#             self.env.render()
            next_state, reward, done, info = self.env.step(action)

            if timestep != 0:
                pre_state_tensor = torch.from_numpy(pre_state).view(1,self.n_states).to(device)
                pre_action_tensor = torch.tensor([pre_action], dtype=torch.float32, device=device)
                state_tensor = torch.from_numpy(state).view(1,self.n_states).to(device)
                action_tensor = torch.tensor([action], dtype=torch.float32, device=device)
                t_sample.append(pre_state_tensor)
                t_sample.append(pre_action_tensor)
                t_sample.append(state_tensor)
                t_sample.append(action_tensor)
                trajectory_list.append(t_sample)

            # Move to the next state
            pre_state = copy.deepcopy(state)
            pre_action = copy.deepcopy(action)
            state = copy.deepcopy(next_state)
            score += reward

            if done or timestep + 1 == T_max:
#                 print(".....PPO-LunarLander: t = {:6d} | score = {:.3f}".format(timestep, round(score.item(),0)))
                break


        # Trajectory to MEMORY with head_padding
        if len(trajectory_list) < LEN_TRAJECTORY:
            # action type is int, convert to tensor for shape recognition
            action = torch.tensor([action], dtype=torch.float32, device=device)
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
                
        self.env.close()

        
        
    def save(self):
        torch.save(self.policy_net.state_dict(), f"victim_model_ppo")



        
if __name__ == "__main__":
    
    # TensorBoard
    import tensorboardX
    from utils import utils_log
    
    ## ... Tensorboard Settings ...

    # Set run dir
    model_name = "ppo_debug"
    model_dir = utils_log.get_model_dir(model_name)

    # Load loggers and Tensorboard writer
    txt_logger = utils_log.get_txt_logger(model_dir)
    csv_file, csv_logger = utils_log.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    
    
    from envs import LunarLander
    env = LunarLander()
    env.clear_attack()

    v_n_actions = env.action_space.n
    v_n_states = env.observation_space.shape[0]
    
    victim_args = {
        "env": env, 
        "MEMORY_SIZE": MEMORY_SIZE, 
        "GAMMA": 0.99, 
        "LAMBD": 0.95,
        "n_states": v_n_states, 
        "n_actions": v_n_actions, 
        "BATCH_SIZE": 256, 
        "CLIP": 0.2, 
        "learning_rate": 0.001,
    }
    
    victim = Victim_PPO(**victim_args)
    
    for i_episode in range(2000):
        print(f"Training_Episode = {i_episode}")
        policy_loss, value_loss = victim.train_model(1)
    
        tb_writer.add_scalar('PolicyLoss/train', sum(policy_loss), i_episode+1)
        tb_writer.add_scalar('ValueLoss/train', sum(value_loss), i_episode+1)
