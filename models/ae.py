import torch
from torch import nn
from torch.nn import functional as F
from models.generator import MNISTGenerator

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
        

class UnFlatten(nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
        
    def forward(self, input):
        return torch.reshape(input, (input.size(0),*self.target_shape))


class MNISTEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        ngf = 32
        nz = latent_dim
        nc = 1
        self.main = nn.Sequential(
            nn.Conv2d(nc, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(),

            nn.Conv2d(ngf, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(),
            
            nn.Conv2d(ngf*2, ngf*4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(),
            
            nn.Conv2d(ngf*4, ngf*8, 2, 2, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(),
    
            nn.Conv2d(ngf*8, nz, 2, 2, 0, bias=False),
            nn.Flatten()
        )

    def forward(self, inp):
        return self.main(inp)

class MNISTAutoencoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = MNISTEncoder(latent_dim=latent_dim)
        self.decoder = MNISTGenerator(latent_dim=latent_dim)

    def forward(self, inp):
        return self.decoder(self.encoder(inp))