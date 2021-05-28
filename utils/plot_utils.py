import os
from os.path import join, dirname, exists
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

def show_latent(samples, labels, fname=None, title='latent space'):
    plt.figure()
    plt.title(title)
    plt.scatter(samples[:, 0], samples[:, 1], s=1, c=labels)
    plt.xlabel('z1')
    plt.ylabel('z2')
    if fname:
        # TODO: save the figure in file
        pass

