import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm

import cv2
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


def add_noise(inputs, noise_factor=0.3):


    blur_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.GaussianBlur(kernel_size=3),  # Adjust the kernel size as needed
        # transforms.RandomVerticalFlip(p=0.2),
        # transforms.GaussianBlur(kernel_size=5),  # Adjust the kernel size as needed
        transforms.ToTensor()
    ])

    noise = inputs + torch.randn_like(inputs) * noise_factor
    noise = torch.clip(noise, 0., 1.)

    noisyblur = []
    i = 0
    for image in noise:
        if i % 2 == 0:
            blurry = blur_transform(image)
            noisyblur.append(blurry)
        else:
            noisyblur.append(image)
        i += 1

    noisyblur = torch.stack(noisyblur)
    noise = noisyblur + torch.randn_like(noisyblur) * 0.1
    noise = torch.clip(noise, 0., 1.)

    return noise
