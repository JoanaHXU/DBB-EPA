import os
from os.path import dirname, abspath
import sys
if "../" not in sys.path:
    sys.path.append("../") 
    
from victim.victim_Q import VictimAgent
from ae.autoencoder import AutoEncoder

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
from target.target_def import MEM_Target

class System():
    def __init__(self, victim_args, ae_args):
        self.victim = VictimAgent(**victim_args)
        self.ae = AutoEncoder(**ae_args)
        
    def train(self, victim_episodes_num = 50, ae_episodes_num = 1):
        self.ae.n_epochs = ae_episodes_num
        
        for i_episode in range(victim_episodes_num):
            
            self.victim.Train_Model(1)
            
            if self.victim.MEM.__len__() >= MEMORY_SIZE:
                self.ae._train(self.victim.MEM, MEM_Target)
                
    def eval_victim(self):
        self.victim.Eval_Model()
        
    def eval_autoencoder(self):
        self.ae._eval(self.victim.MEM, MEM_Target)