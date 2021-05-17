#imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import Discriminator, Generator
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

#hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
lr = 0.0001
noise_dim = 128
in_size = 784
batch_size = 64
epochs = 10

#initialize disc and gen models
disc = Discriminator(in_size).to(device)
gen = Generator(noise_dim,in_size).to(device)

#function to generate fake noise
def generate_noise(b,n):
    return torch.randn((b,n)).to(device)

#loading data
noise = generate_noise(batch_size,noise_dim)
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.13,),(0.13,))])
data = datasets.MNIST(root=r'dataset/',download=False,transform=transform)
train = DataLoader(data,batch_size=batch_size,shuffle=True)

#optimizer
optim_disc = Adam(disc.parameters(), lr=lr)
optim_gen = Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

#tensorboard
board_fake = SummaryWriter(f'run/SimpleGAN/fake')
board_real = SummaryWriter(f'run/SimpleGAN/real')
step = 0

print('Starting the training process: ')
for epoch in range(epochs):
    for idx, (real,_) in enumerate(train):
        real = real.view(-1,784).to(device)
        batch_size = real.shape[0]
        noise = noise.to(device)
        fake = gen(noise)

        #discriminator
        dreal = disc(real).view(-1)
        dloss_real = criterion(dreal,torch.ones_like(dreal))
        dfake = disc(fake).view(-1)
        dloss_fake = criterion(dfake,torch.zeros_like(dfake))
        dloss = (dloss_fake + dloss_real)/2
        disc.zero_grad()
        dloss.backward(retain_graph=True)
        optim_disc.step()

        #generator
        gout = disc(fake).view(-1)
        gloss = criterion(gout,torch.ones_like(gout))
        gen.zero_grad()
        gloss.backward()
        optim_gen.step()

        #print loss values
        if idx == 0:
            print('Epoch: ',epoch,'Batch: ',idx,'Discriminator Loss: ',dloss,'Generator Loss: ',gloss)

            with torch.no_grad():
                fake = gen(noise).reshape(-1,1,28,28)
                actual = real.reshape(-1,1,28,28)
                fake_grid = make_grid(fake,normalize=True)
                real_grid = make_grid(actual,normalize=True)

                board_fake.add_image('MNIST FAKE IMAGES',fake_grid,global_step=step)
                board_real.add_image('MNIST REAL IMAGES',real_grid,gloabl_step=step)

                step+=1
print('Training complete.')
