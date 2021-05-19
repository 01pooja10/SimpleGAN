#imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import Discriminator, Generator
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import warnings
warnings.filterwarnings('ignore')

#hyperparameters
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
'''
device = 'cuda'
lr = 0.0001
noise_dim = 128
in_size = 784
batch_size = 64
epochs = 20

#initialize disc and gen models
disc = Discriminator(in_size).to(device)
gen = Generator(noise_dim,in_size).to(device)

#function to generate fake noise
def generate_noise(b,n):
    return torch.randn((b,n)).to(device)

#loading data
noise = generate_noise(batch_size,noise_dim)
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
data = datasets.MNIST(root=r'dataset/',download=False,transform=transform)
train = DataLoader(data,batch_size=batch_size,shuffle=True)

#initialize the gradient scaler
dscaler = GradScaler()
gscaler = GradScaler()

#optimizer
optim_disc = Adam(disc.parameters(), lr=lr)
optim_gen = Adam(gen.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

#tensorboard
board_fake = SummaryWriter(f'logs/fake images')
board_real = SummaryWriter(f'logs/real images')
step = 0

print('Starting the training process: ')
for epoch in range(epochs):
    for idx, (real,_) in enumerate(train):
        real = real.view(-1,784).to(device)
        batch_size = real.shape[0]
        noise = noise.to(device)
        fake = gen(noise)
        '''
        training in mixed precision
        using amp
        '''
        #discriminator
        with autocast():
            dreal = disc(real).view(-1)
            dloss_real = criterion(dreal,torch.ones_like(dreal))
            dfake = disc(fake).view(-1)
            dloss_fake = criterion(dfake,torch.zeros_like(dfake))
            dloss = (dloss_fake + dloss_real)/2
        disc.zero_grad()
        dscaler.scale(dloss).backward(retain_graph=True)
        dscaler.step(optim_disc)
        dscaler.update()

        #generator
        with autocast():
            gout = disc(fake).view(-1)
            gloss = criterion(gout,torch.ones_like(gout))
        gen.zero_grad()
        gscaler.scale(gloss).backward()
        gscaler.step(optim_gen)
        gscaler.update()

        #print loss values
        if idx == 0:
            print('Epoch: ',epoch,'Discriminator Loss: ',dloss.item(),'Generator Loss: ',gloss.item())

            with torch.no_grad():
                fake = gen(noise).reshape(-1,1,28,28)
                actual = real.reshape(-1,1,28,28)
                fake_grid = make_grid(fake,normalize=True)
                real_grid = make_grid(actual,normalize=True)

                board_fake.add_image('MNIST FAKE IMAGES',fake_grid,global_step=step)
                board_real.add_image('MNIST REAL IMAGES',real_grid,global_step=step)

                step += 1

print('Final loss for D: ',dloss)
print('Final loss for G: ',gloss)
print('Training complete.')


d_save = 'disc_model.pth'
torch.save(disc.state_dict(), d_save)

g_save = 'gen_model.pth'
torch.save(gen.state_dict(), g_save)
