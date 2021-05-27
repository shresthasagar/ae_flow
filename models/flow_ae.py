import torch
import torch.nn as nn
from ae import MNISTAutoencoder

class MNISTFlowAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.autoencoder = MNISTAutoencoder(latent_dim=latent_dim)
        self.flow = Flow(latent_dim=latent_dim)
        self.latent_dim = latent_dim

    def sample(self, num_samples):
        y = torch.randn((num_samples, self.latent_dim), dtype=torch.float32)
        z = self.flow.reverse(y)
        x_hat = self.autoencoder.decoder(z)
        return x_hat

    def forward(self, inp, flow_mode=False):
        if flow_mode:
            return self.flow(inp)
        else:
            return self.autoencoder(inp)