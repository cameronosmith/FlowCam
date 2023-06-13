import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import torch
from torch import nn
from einops import repeat,rearrange

def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

class ResMLP(nn.Module):
    def __init__(self, ch_in, ch_mod, out_ch, num_res_block=1 ):
        super().__init__()

        self.res_blocks = nn.ModuleList([
          nn.Sequential(nn.Linear(ch_mod,ch_mod),nn.ReLU(),
                        nn.LayerNorm([ch_mod], elementwise_affine=True),
                        nn.Linear(ch_mod,ch_mod),nn.ReLU())
            for _ in range(num_res_block)
        ])  

        self.proj_in = nn.Linear(ch_in,ch_mod)
        self.out = nn.Linear(ch_mod,out_ch)

    def forward(self,x):

        x = self.proj_in(x)

        for i,block in enumerate(self.res_blocks):

            x_in = x

            x = block(x)

            if i!=len(self.res_blocks)-1: x = x + x_in

        return self.out(x)

# FILM, but just the biases, not scalings - featurewise additive modulation
# "x" is the input coordinate and "y" is the conditioning feature (img features, for exmaple)
class ResFAMLP(nn.Module):
    def __init__(self, ch_in_x,ch_in_y, ch_mod, out_ch, num_res_block=1, last_res=False):
        super().__init__()

        self.res_blocks = nn.ModuleList([
          nn.Sequential(nn.Linear(ch_mod,ch_mod),nn.ReLU(),
                        nn.LayerNorm([ch_mod], elementwise_affine=True),
                        nn.Linear(ch_mod,ch_mod),nn.ReLU())
            for _ in range(num_res_block)
        ])  

        self.last_res=last_res
        self.proj_in = nn.Linear(ch_in_x,ch_mod)
        self.modulators = nn.ModuleList([nn.Linear(ch_in_y,ch_mod) for _ in range(num_res_block)])
        self.out = nn.Linear(ch_mod,out_ch)

    def forward(self,x,y):

        x = self.proj_in(x)

        for i,(block,modulator) in enumerate(zip(self.res_blocks,self.modulators)):

            x_in = x + modulator(y)
            
            x = block(x)

            if i!=len(self.res_blocks)-1 or self.last_res: x = x + x_in

        return self.out(x)


