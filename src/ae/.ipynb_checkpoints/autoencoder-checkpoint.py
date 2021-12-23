import argparse
import time
import copy

import os
from os.path import dirname, abspath

import sys
if "../" not in sys.path:
    sys.path.append("../") 

# attacker & victim algo
from attack.DDPG import DDPG
from victim.victim_dqn import Victim_DQN

# utils & tools
from utils import *

# env & target
from envs import LunarLander
from target.target_def import MEM_target

# for class definition
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

''' import configuration '''
from yacs.config import CfgNode as CN
yaml_name = os.path.join(dirname(dirname(abspath(__file__))), "config", "config_default.yaml")
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()

LEN_TRAJECTORY = config.AE.LEN_TRAJECTORY
MEMORY_SIZE = config.AE.MEMORY_SIZE
TARGET_FOLDER = config.ATTACK.TARGET_FOLDER
TARGET_POLICY_TITLE = config.ATTACK.TARGET_POLICY_TITLE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" LSTM-AutoEncoder Memory """
Transition_AutoEncoder = namedtuple('Transition_AutoEncoder', ('pre_state', 'pre_action', 'state', 'action'))


""" Define the Encoder(LSTM) Network"""

class Encoder_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer):
        super(Encoder_LSTM, self).__init__()
        
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layer, x.size(1), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layer, x.size(1), self.hidden_size).to(device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        embedding = hn[-1, :, :]
        
        return embedding.data
    
    
""" Define the Decoder(FC) Network"""

class Decoder_FC(nn.Module):
    def __init__(self, input_size, output_size, fc1_units=128, fc2_units=128):
        super(Decoder_FC, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.ln1 = nn.LayerNorm(fc1_units)
        self.fc2 = nn.Linear(fc2_units, output_size)
        
    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.fc2(x)
        x = F.softmax(x, dim=2)
        return x
    

""" Define the Decoder(LSTM) Network """

class Decoder_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, output_size):
        super(Decoder_LSTM, self).__init__()
        
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layer, x.size(1), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layer, x.size(1), self.hidden_size).to(device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))

        out = self.fc(out[:, :, :])
        
        return out
        

""" Construct Encoder + Decoder """

class Encoder_Decoder(nn.Module):
    def __init__(self, encoder, decoder_fc, decoder_lstm):
        super(Encoder_Decoder, self).__init__()
        self.encoder = encoder
        self.decoder_fc = decoder_fc
        self.decoder_lstm = decoder_lstm
        
    def forward(self, enc_in, dec_fc_in, dec_lstm_in):
        # encoder
        embedding = self.encoder(enc_in)

        # decoder-fc
        z = embedding.unsqueeze(1)
        z = torch.cat(enc_in.size(0)*[z],1)
        x1 = torch.cat((dec_fc_in, z), 2) 
        y1 = self.decoder_fc(x1)
        
        # decoder-lstm
        z = embedding.unsqueeze(0)
        z = torch.cat(enc_in.size(0)*[z],0)
        x2 = torch.cat((dec_lstm_in, z), 2)
        y2 = self.decoder_lstm(x2)
        
        return y1, y2
    
    
class AutoEncoder():
    
    def __init__(self, env, enc_in_size, enc_out_size, enc_num_layer, dec_fc_in_size, dec_fc_out_size, dec_lstm_in_size, dec_lstm_out_size, dec_lstm_num_layer, seq_len, embedding_len, n_epochs=50, lr=0.1):
        
        self.env_state_dim = env.observation_space.shape[0]

        self.Enc_Model = Encoder_LSTM(enc_in_size, enc_out_size, enc_num_layer).to(device)
        self.Dec_FC_Model = Decoder_FC(dec_fc_in_size, dec_fc_out_size).to(device)
        self.Dec_LSTM_Model = Decoder_LSTM(dec_lstm_in_size, dec_lstm_out_size, dec_lstm_num_layer, dec_lstm_out_size).to(device)

        self.Model = Encoder_Decoder(self.Enc_Model, self.Dec_FC_Model, self.Dec_LSTM_Model).to(device)
        
        self.seq_len = seq_len
        self.embedding_len = embedding_len
        
        self.n_epochs = n_epochs
        self.lr = lr
        

    def _train(self, MEM, MEM_target):

        criterion_class = nn.CrossEntropyLoss()
        criterion_reg = nn.MSELoss()

        optimizer = torch.optim.SGD(self.Model.parameters(), self.lr)

        self.Model.train()

        n_trajectory = MEM.__len__()//self.seq_len

        for epoch in range(self.n_epochs):

            train_loss = 0.0  # monitor training loss

            for n in range(n_trajectory-1):
                i_start = n*self.seq_len
                i_end = i_start + self.seq_len

                '''  Input to Encoder  '''
                # ---------------------- Actual(A) Set
                A_transitions_enc = MEM.memory[i_start: i_end]
                A_batch_enc = Transition_AutoEncoder(*zip(*A_transitions_enc))
                
                A_state_tuple = A_batch_enc.state
                A_state_enc = A_state_tuple[0]
                for i in range(1, len(A_state_tuple)):
                    A_state_enc = torch.cat((A_state_enc, A_state_tuple[i]))
                A_state_enc = A_state_enc.to(device)
                
                A_action_enc = torch.FloatTensor([A_batch_enc.action]).view(-1,1).to(device)

                A_enc_IN = torch.cat((A_state_enc, A_action_enc),1).unsqueeze(1)
                
                # ----------------- Desired (B) Set
                B_transitions_enc = MEM_target.memory[0:self.seq_len]
                B_batch_enc = Transition_AutoEncoder(*zip(*B_transitions_enc))
                
                B_state_tuple = B_batch_enc.state
                B_state_enc = B_state_tuple[0]
                for i in range(1, len(B_state_tuple)):
                    B_state_enc = torch.cat((B_state_enc, B_state_tuple[i]))
                B_state_enc = B_state_enc.to(device)
                
                B_action_enc = torch.FloatTensor([B_batch_enc.action]).view(-1,1).to(device)

                B_enc_IN = torch.cat((B_state_enc, B_action_enc),1).unsqueeze(1)
                
                # ------------------
                enc_IN = torch.cat((A_enc_IN, B_enc_IN), 1)

                '''  Input to Decoder  '''
                # ------------------ Actual (A) Set
                A_transition_dec = MEM.memory[i_start+self.seq_len : i_end+self.seq_len]
                A_batch_dec = Transition_AutoEncoder(*zip(*A_transition_dec))
                # pre-state & pre-action
                A_pre_state_tuple = A_batch_dec.pre_state
                A_pre_state_dec = A_pre_state_tuple[0]
                for i in range(1, len(A_pre_state_tuple)):
                    A_pre_state_dec = torch.cat((A_pre_state_dec, A_pre_state_tuple[i]))
                A_pre_state_dec = A_pre_state_dec.to(device)
                
                A_pre_action_dec = torch.FloatTensor([A_batch_dec.pre_action]).view(-1,1).to(device)
                
                # state & action
                A_state_tuple = A_batch_dec.state
                A_state_dec = A_state_tuple[0]
                for i in range(1, len(A_state_tuple)):
                    A_state_dec = torch.cat((A_state_dec, A_state_tuple[i]))
                A_state_dec = A_state_dec.to(device)
                A_action_dec = torch.FloatTensor([A_batch_dec.action]).view(-1,1).to(device)
                
                # fully-connected decoder
                A_dec_fc_IN = A_state_dec
                A_dec_fc_TARGET = A_action_dec.long().view(1, self.seq_len)
                
                # lstm decoder
                A_dec_lstm_IN = torch.cat((A_pre_state_dec, A_pre_action_dec),1)
                A_dec_lstm_TARGET = A_state_dec.unsqueeze(0)

                # ---------------- Desired (B) Set
                B_transition_dec = MEM_target.memory[0:self.seq_len]
                B_batch_dec = Transition_AutoEncoder(*zip(*B_transition_dec))
                
                # pre-state & pre-action
                B_pre_state_tuple = B_batch_dec.pre_state
                B_pre_state_dec = B_pre_state_tuple[0]
                for i in range(1, len(B_pre_state_tuple)):
                    B_pre_state_dec = torch.cat((B_pre_state_dec, B_pre_state_tuple[i]))
                B_pre_state_dec = B_pre_state_dec.to(device)
                
                B_pre_action_dec = torch.FloatTensor([B_batch_dec.pre_action]).view(-1,1).to(device)
                
                # state & action
                B_state_tuple = B_batch_dec.state
                B_state_dec = B_state_tuple[0]
                for i in range(1, len(B_state_tuple)):
                    B_state_dec = torch.cat((B_state_dec, B_state_tuple[i]))
                B_state_dec = B_state_dec.to(device)
                B_action_dec = torch.FloatTensor([B_batch_dec.action]).view(-1,1).to(device)
                
                # fully-connected decoder
                B_dec_fc_IN = B_state_dec
                B_dec_fc_TARGET = B_action_dec.long().view(1, self.seq_len)
                # lstm decoder
                B_dec_lstm_IN = torch.cat((B_pre_state_dec, B_pre_action_dec),1)
                B_dec_lstm_TARGET = B_state_dec.unsqueeze(0)

                ## --- combine batch.A and batch.B
                dec_fc_IN = torch.cat((A_dec_fc_IN.unsqueeze(0), B_dec_fc_IN.unsqueeze(0)), 0)
                dec_lstm_IN = torch.cat((A_dec_lstm_IN.unsqueeze(1), B_dec_lstm_IN.unsqueeze(1)), 1)

                dec_fc_TARGET = torch.cat((A_dec_fc_TARGET, B_dec_fc_TARGET), 0).view(-1)
                dec_lstm_TARGET = torch.cat((A_dec_lstm_TARGET, B_dec_lstm_TARGET),0).view(-1)

                ''' Optimize '''
                # clear the gradients of all optimized variables
                optimizer.zero_grad()

                # forward pass: compute predicted outputs by passing inputs to the model
                out1, out2 = self.Model(enc_IN, dec_fc_IN, dec_lstm_IN)
                # reshape output & target
                out1 = torch.cat((out1[0,:,:], out1[1,:,:]), 0)
                out2 = torch.cat((out2[:,0,:], out2[:,1,:]), 0).view(-1)

                # calculate the loss
                loss = criterion_class(out1, dec_fc_TARGET) + criterion_reg(out2, dec_lstm_TARGET)
                # backward pass
                loss.backward()

                # perform a single optimization step (parameter update)
                optimizer.step()

                # update running training loss
                train_loss += loss.item()

            total_size = (n_trajectory-1)*self.seq_len*2

            train_loss = train_loss/total_size
#             print(f'.....................autoencoder loss = {train_loss:.6f}')
            # print('autoencoder: epoch = {} | loss = {:.6f}'.format(epoch+1, train_loss))

    
    def _eval(self, MEM, MEM_target):

        criterion_class = nn.CrossEntropyLoss()
        criterion_reg = nn.MSELoss()

        test_loss = 0.0
        correct_stat_1 = 0
        correct_stat_2 = 0

        self.Model.eval() # prep model for *evaluation*

        n_trajectory = MEM.__len__()//self.seq_len

        for n in range(n_trajectory):
            i_start = n*self.seq_len
            i_end = i_start + self.seq_len

            '''  Input to Encoder  '''
            # ---------------------- Actual(A) Set
            A_transitions_enc = MEM.memory[i_start: i_end]
            A_batch_enc = Transition_AutoEncoder(*zip(*A_transitions_enc))
            A_state_tuple = A_batch_enc.state
            A_state_enc = A_state_tuple[0]
            for i in range(1, len(A_state_tuple)):
                A_state_enc = torch.cat((A_state_enc, A_state_tuple[i]))
            A_state_enc = A_state_enc.to(device)
            A_action_enc = torch.FloatTensor([A_batch_enc.action]).view(-1,1).to(device)

            A_enc_IN = torch.cat((A_state_enc, A_action_enc),1).unsqueeze(1)

            # ----------------- Desired (B) Set
            B_transitions_enc = MEM_target.memory[0:self.seq_len]
            B_batch_enc = Transition_AutoEncoder(*zip(*B_transitions_enc))
            B_state_tuple = B_batch_enc.state
            B_state_enc = B_state_tuple[0]
            for i in range(1, len(B_state_tuple)):
                B_state_enc = torch.cat((B_state_enc, B_state_tuple[i]))
            B_state_enc = B_state_enc.to(device)
            B_action_enc = torch.FloatTensor([B_batch_enc.action]).view(-1,1).to(device)

            B_enc_IN = torch.cat((B_state_enc, B_action_enc),1).unsqueeze(1)

            # ------------------
            enc_IN = torch.cat((A_enc_IN, B_enc_IN), 1)

            '''  Input to Decoder  '''
            # ------------------ Actual (A) Set
#             A_transition_dec = MEM.memory[i_start+self.seq_len : i_end+self.seq_len]
            A_transition_dec = MEM.memory[i_start: i_end]
            A_batch_dec = Transition_AutoEncoder(*zip(*A_transition_dec))
            A_pre_state_tuple = A_batch_dec.pre_state
            A_pre_state_dec = A_pre_state_tuple[0]
            for i in range(1, len(A_pre_state_tuple)):
                A_pre_state_dec = torch.cat((A_pre_state_dec, A_pre_state_tuple[i]))
            A_pre_state_dec = A_pre_state_dec.to(device)
            A_pre_action_dec = torch.FloatTensor([A_batch_dec.pre_action]).view(-1,1).to(device)
            
            A_state_tuple = A_batch_dec.state
            A_state_dec = A_state_tuple[0]
            for i in range(1, len(A_state_tuple)):
                A_state_dec = torch.cat((A_state_dec, A_state_tuple[i]))
            A_state_dec = A_state_dec.to(device)
            A_action_dec = torch.FloatTensor([A_batch_dec.action]).view(-1,1).to(device)

            A_dec_fc_IN = A_state_dec
            A_dec_fc_TARGET = A_action_dec.long().view(1, self.seq_len)

            A_dec_lstm_IN = torch.cat((A_pre_state_dec, A_pre_action_dec),1)
            A_dec_lstm_TARGET = A_state_dec.unsqueeze(0)

            # ---------------- Desired (B) Set
            B_transition_dec = MEM_target.memory[0:self.seq_len]
            B_batch_dec = Transition_AutoEncoder(*zip(*B_transition_dec))
            B_pre_state_tuple = B_batch_dec.pre_state
            B_pre_state_dec = B_pre_state_tuple[0]
            for i in range(1, len(B_pre_state_tuple)):
                B_pre_state_dec = torch.cat((B_pre_state_dec, B_pre_state_tuple[i]))
            B_pre_state_dec = B_pre_state_dec.to(device)
            B_pre_action_dec = torch.FloatTensor([B_batch_dec.pre_action]).view(-1,1).to(device)
            
            B_state_tuple = B_batch_dec.state
            B_state_dec = B_state_tuple[0]
            for i in range(1, len(B_state_tuple)):
                B_state_dec = torch.cat((B_state_dec, B_state_tuple[i]))
            B_state_dec = B_state_dec.to(device)
            B_action_dec = torch.FloatTensor([B_batch_dec.action]).view(-1,1).to(device)

            B_dec_fc_IN = B_state_dec
            B_dec_fc_TARGET = B_action_dec.long().view(1, self.seq_len)

            B_dec_lstm_IN = torch.cat((B_pre_state_dec, B_pre_action_dec),1)
            B_dec_lstm_TARGET = B_state_dec.unsqueeze(0)

            ## -----------------
            dec_fc_IN = torch.cat((A_dec_fc_IN.unsqueeze(0), B_dec_fc_IN.unsqueeze(0)), 0)
            dec_lstm_IN = torch.cat((A_dec_lstm_IN.unsqueeze(1), B_dec_lstm_IN.unsqueeze(1)), 1)

            dec_fc_TARGET = torch.cat((A_dec_fc_TARGET, B_dec_fc_TARGET), 0).view(-1)
            dec_lstm_TARGET = torch.cat((A_dec_lstm_TARGET, B_dec_lstm_TARGET),0).view(-1)

            # forward pass: compute predicted outputs by passing inputs to the model
            out1, out2 = self.Model(enc_IN, dec_fc_IN, dec_lstm_IN)
            # reshape output & target
            out1 = torch.cat((out1[0,:,:], out1[1,:,:]), 0)
            out2 = torch.cat((out2[:,0,:], out2[:,1,:]), 0).view(-1)

            # calculate the loss
            loss = criterion_class(out1, dec_fc_TARGET) + criterion_reg(out2, dec_lstm_TARGET)
            # update test loss 
            test_loss += loss.item()

            """ correctness of \pi(a|s) """
            # convert output probabilities to predicted class
            _, pred = torch.max(out1, 1)

            # compare predictions to true label
            correct_1 = np.squeeze(pred.eq(dec_fc_TARGET.data.view_as(pred)))

            for i in range(len(correct_1)):
                if correct_1[i].item() == True:
                    correct_stat_1 += 1

            """ correctness of T(s'|s,a) """
            correct_2 = criterion_reg(out2, dec_lstm_TARGET)
            correct_stat_2 += correct_2

        # calculate and print avg test loss
        total_size = (n_trajectory)*self.seq_len*2

        test_loss = test_loss/total_size
        accuracy_rate_1 = (correct_stat_1/total_size)*100
        accuracy_delta_2 = (correct_stat_2/(n_trajectory-1))
        print(f'.....Encoder Evaluation: Policy={accuracy_rate_1:.2f}% | Dynamics={accuracy_delta_2:.6f}')
        
        
        
    ''' Embedding : a batch of trajectories '''
    def _embedding_batch(self, MEM, MEM_target):
        memory_size = MEM.__len__()
        n_trajectory = memory_size//self.seq_len

        zA = []
        zB = []

        for n in range(n_trajectory):
            # Input Data
            i_start = n*self.seq_len
            i_end = i_start + self.seq_len

            # Actual(A) Set
            A_transitions = MEM.memory[i_start: i_end]
            A_batch = Transition_AutoEncoder(*zip(*A_transitions))
            
            A_state_tuple = A_batch.state
            A_state = A_state_tuple[0]
            for i in range(1, len(A_state_tuple)):
                A_state = torch.cat((A_state, A_state_tuple[i]))
            A_state = A_state.to(device)
            A_action = torch.FloatTensor([A_batch.action]).view(-1,1).to(device)

            A_IN = torch.cat((A_state, A_action),1).unsqueeze(1)

            # Desired (B) Set
            B_transitions = MEM_target.memory[i_start: i_end] 
            B_batch = Transition_AutoEncoder(*zip(*B_transitions))
            
            B_state_tuple = B_batch.state
            B_state = B_state_tuple[0]
            for i in range(1, len(B_state_tuple)):
                B_state = torch.cat((B_state, B_state_tuple[i]))
            B_state = B_state.to(device)
            B_action = torch.FloatTensor([B_batch.action]).view(-1,1).to(device)

            B_IN = torch.cat((B_state, B_action),1).unsqueeze(1)

            # Input to Encoder
            IN = torch.cat((A_IN, B_IN), 1)

            # Get embedding
            z = self.Enc_Model(IN)

            zA.append(z[0].cpu().data.numpy())
            zB.append(z[1].cpu().data.numpy())

        zA = np.vstack(zA)
        zB = np.vstack(zB)

        return zA, zB

        
    ''' Embedding : single trajectories '''
    def _embedding_single(self, MEM, MEM_target):
        
        # Input Data
        i_start = 0
        i_end = self.seq_len

        # Actual(A) Set
#         A_transitions = MEM.memory[i_start: i_end]
        A_transitions = MEM.memory[-self.seq_len: ]
        A_batch = Transition_AutoEncoder(*zip(*A_transitions))
        
        A_state_tuple = A_batch.state
        A_state = A_state_tuple[0]
        for i in range(1, len(A_state_tuple)):
            A_state = torch.cat((A_state, A_state_tuple[i]))
        A_state = A_state.to(device)
        
        A_action = torch.FloatTensor([A_batch.action]).view(-1,1).to(device)

        A_IN = torch.cat((A_state, A_action),1).unsqueeze(1)

        # Desired (B) Set
#         B_transitions = MEM_target.memory[i_start: i_end]
        B_transitions = MEM_target.memory[-self.seq_len: ]
        B_batch = Transition_AutoEncoder(*zip(*B_transitions))
        
        B_state_tuple = B_batch.state
        B_state = B_state_tuple[0]
        for i in range(1, len(B_state_tuple)):
            B_state = torch.cat((B_state, B_state_tuple[i]))
        B_state = B_state.to(device)
        
        B_action = torch.FloatTensor([B_batch.action]).view(-1,1).to(device)

        B_IN = torch.cat((B_state, B_action),1).unsqueeze(1)

        # Input to Encoder
        IN = torch.cat((A_IN, B_IN), 1)

        # Get embedding
        z = self.Enc_Model(IN)

        zA = z[0].cpu().data.numpy()
        zB = z[1].cpu().data.numpy()

        return zA, zB

    def _save(self, filename):
        torch.save(self.Model.state_dict(), filename + "_AutoEncoder")

    def _load(self, filename):
        load_model = self.Model.load_state_dict(torch.load(filename))

        return load_model
    
    


if __name__ == "__main__":
    
    # env object
    env = LunarLander()
    env.clear_attack()
    # global settings
    ENV_INIT_CONDITION = env.hyper_condition
    EMBEDDING_SIZE = 128
    SEQ_LEN = LEN_TRAJECTORY
    print(f"LEN_TRAJECOTRY = {LEN_TRAJECTORY}")
    
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
        "learning_rate": 0.001,
    }
    
    victim = Victim_DQN(**victim_args)
    model_path = os.path.join(dirname(dirname(abspath(__file__))), TARGET_FOLDER, TARGET_POLICY_TITLE)
    
            
    victim.reset()
    victim.policy_net.load_state_dict(torch.load(model_path))
    
    
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
        "lr": 0.001, 
    }

    ae_Model = AutoEncoder(**ae_args)
    ae_Memory = Memory(MEMORY_SIZE)
    
    """ training & evaluation of AutoEncoder """
    
#     ''' I. victim's trajectory from target policy '''
#     print("............. evaluate on TARGET policy ...............")
    
#     victim.policy_net.load_state_dict(torch.load(model_path))
    
#     for i in range(5):
#         score = 0
#         trajectory_list = []
        
#         state = victim.env.reset()
#         state = torch.from_numpy(state).view(1,victim.n_states).to(device)
#         done = False
        
#         for t in range(2000):
#             t_sample = []
            
#             action = victim.select_action(state).to(device)
#             next_state, reward, done, _ = victim.env.step(action.item())
#             next_state = torch.from_numpy(next_state).view(1, victim.n_states).to(device)
#             reward = torch.tensor([reward], dtype=torch.float32, device=device)

#             # store the transition(s,a,s',a') 
#             if t != 0:
#                 t_sample.append(pre_state)
#                 t_sample.append(pre_action)
#                 t_sample.append(state)
#                 t_sample.append(action)
#                 trajectory_list.append(t_sample)

#             # Move to the next state
#             pre_state = copy.deepcopy(state)
#             pre_action = copy.deepcopy(action)
#             state = copy.deepcopy(next_state)
#             score += reward
#             if done:
#                 print("--- LunarLander -- episode = {} | t = {} | score = {}".format(i, t, round(score.item(),2)))
#                 break
#             if t+1 == 2000:
#                 print("--- LunarLander -- episode = {} | t = {} | score = {}".format(i, t, round(score.item(),2)))
#                 break
                
#         # Trajectory to MEMORY with head_padding
#         if len(trajectory_list) < LEN_TRAJECTORY:
#             padding_state = torch.zeros(state.shape, device=device)
#             padding_action = torch.zeros(action.shape, device=device)
#             n_padding = LEN_TRAJECTORY - len(trajectory_list)
#             for i in range(n_padding):
#                 victim.MEM.push(padding_state, padding_action, padding_state, padding_action)
#             for i in range(len(trajectory_list)):
#                 pre_state = trajectory_list[i][0]
#                 pre_action = trajectory_list[i][1]
#                 state = trajectory_list[i][2]
#                 action = trajectory_list[i][3]
#                 victim.MEM.push(pre_state, pre_action, state, action)
#         else:
#             for i in range(LEN_TRAJECTORY):
#                 pre_state = trajectory_list[i][0]
#                 pre_action = trajectory_list[i][1]
#                 state = trajectory_list[i][2]
#                 action = trajectory_list[i][3]
#                 victim.MEM.push(pre_state, pre_action, state, action)

#     print(f"Size of Actual_Memory = {victim.MEM.__len__()}")
#     print(f"Size of Target_Memory = {MEM_target.__len__()}")

#     tic = time.time()
#     ae_Model.n_epochs = 50
#     ae_Model._train(victim.MEM, MEM_target)
#     toc = time.time()
#     print(f"Training Time of AutoEncoder is {toc-tic}s")

#     ae_Model._eval(victim.MEM, MEM_target)
        
#     ''' embedding by BATCH-trajectories '''
#     Z_batch, Z_target_batch = ae_Model._embedding_batch(victim.MEM, MEM_target)
#     cost, done, success_rate = Attack_Cost_Done_batch(Z_batch, Z_target_batch, env.hyper_condition, ENV_INIT_CONDITION, args.done_threshold, args.weight)
#     print(f"BATCH: cost = {cost:.2f} | success_rate = {success_rate:.2f} ")
        
#     ''' embedding by SINGLE-trajectory'''
#     Z_single, Z_target_single = ae_Model._embedding_single(victim.MEM, MEM_target)
#     cost, done, success_rate = Attack_Cost_Done(Z_single, Z_target_single, env.hyper_condition, ENV_INIT_CONDITION, args.done_threshold, args.weight)
#     print(f"SINGLE: cost = {cost:.2f} | success_rate = {success_rate:.2f} ")
    
# #     ae_Model._save("model")
        
        
    ''' II. victim's trajectory from random policy '''

    print("............. evaluate on RANDOM policy ...............")
    for i_iteration in range(1):
        print(f"\n .... iteration = {i_iteration} .... ")
        
        victim.reset()
        victim.train_model(6)

        tic = time.time()
        ae_Model.n_epochs = 50
        ae_Model._train(victim.MEM, MEM_target)
        toc = time.time()
        print(f"Training Time of AutoEncoder is {toc-tic}s")

        ae_Model._eval(victim.MEM, MEM_target)
        
        ''' embedding by BATCH-trajectories '''
        Z_batch, Z_target_batch = ae_Model._embedding_batch(victim.MEM, MEM_target)
        cost, done, success_rate = Attack_Cost_Done_batch(Z_batch, Z_target_batch, env.hyper_condition, ENV_INIT_CONDITION, args.done_threshold, args.weight)
        print(f"BATCH: cost = {cost:.2f} | success_rate = {success_rate:.2f} ")
        
        ''' embedding by SINGLE-trajectory'''
        Z_single, Z_target_single = ae_Model._embedding_single(victim.MEM, MEM_target)
        
        cost, done, success_rate = Attack_Cost_Done(Z_single, Z_target_single, env.hyper_condition, ENV_INIT_CONDITION, args.done_threshold, args.weight)
        print(f"SINGLE: cost = {cost:.2f} | success_rate = {success_rate:.2f} ")
        




        