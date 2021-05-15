import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import Discriminator, Generator

#hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.0001
noise_dim = 128
in_size = 784
batch_size = 64

disc = Discriminator(in_size).to(device)
gen = Generator(noise_dim,in_size).to(device)

def generate_noise(b,n):
    return torch.randn((b,n)).to(device)

noise = generate_noise(batch_size,noise_dim)
