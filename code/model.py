import torch
import torchvision
import torch.nn as nn

#discriminator model
class Discriminator(nn.Module):
    def __init__(self,in_size):
        super().__init__()
        self.dmodel = nn.Sequential(
                        nn.Linear(in_size,512),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(0.3),
                        nn.Linear(512,256),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(0.3),
                        nn.Linear(256,128),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(0.3),
                        nn.Linear(128,1))
                        #nn.Sigmoid())
    def forward(self,x):
        out = self.dmodel(x)
        return out

#generator model
class Generator(nn.Module):
    def __init__(self,noise_dim,in_size):
        super().__init__()
        self.gmodel = nn.Sequential(
                        nn.Linear(noise_dim,128),
                        nn.LeakyReLU(0.2),
                        nn.Linear(128,256),
                        nn.LeakyReLU(0.2),
                        nn.Linear(256,512),
                        nn.LeakyReLU(0.2),
                        nn.Linear(512,in_size),
                        nn.Tanh())
    def forward(self,x):
        out = self.gmodel(x)
        return out
