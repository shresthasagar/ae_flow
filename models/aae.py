from models.ae import MNISTAutoencoder
import torch
from torch import nn
from torch.nn import functional as F
from models.ae import MNISTAutoencoder

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
        

class UnFlatten(nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
        
    def forward(self, input):
        return torch.reshape(input, (input.size(0),*self.target_shape))

class AAEDiscriminator(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Linear(256, 1), 
            nn.Sigmoid()
        ) 
    
    def forward(self, inp):
        return self.main(inp)

class AAESupervisedDiscriminator(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim+10, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Linear(256, 1), 
            nn.Sigmoid()
        ) 
    
    def forward(self, inp):
        return self.main(inp)


class AAESemiSupervised(nn.Module):
    def __init__(self):
        super().__init__()
        self.autoencoder = MNISTAutoencoder()
        self.discriminator = AAESupervisedDiscriminator()

    def forward(self, inp):
        return self.autoencoder(inp)

class AAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.autoencoder = MNISTAutoencoder(latent_dim=latent_dim)
        self.discriminator = AAEDiscriminator(latent_dim=latent_dim)

    def forward(self, inp):
        return self.autoencoder(inp)

    # def forward_discriminator(self, inp):
    #     return self.autoencoder(inp)

