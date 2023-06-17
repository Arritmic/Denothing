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



class Encoder(nn.Module):

    def __init__(self, encoded_space_dim ,fc2_input_dim):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            # nn.BatchNorm2d(8),
            nn.ReLU(True),
            # nn.MaxPool2d(2),  # Add a MaxPooling layer here

            # Second convolutional layer
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # nn.MaxPool2d(2),  # Add a MaxPooling layer here

            # Third convolutional layer
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.Dropout(0.25),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, stride=2, padding=0),
            # nn.BatchNorm2d(64),
            nn.SiLU(True),

        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # Linear section
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(2 * 32, fc2_input_dim),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(fc2_input_dim, encoded_space_dim)
        )



    def forward(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = self.flatten(x)
        # print(x.shape)
        # # Apply linear layers
        x = self.encoder_lin(x)
        # print(x.shape)
        # print(".....")
        return x



class Decoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()

        # Linear section
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(encoded_space_dim, fc2_input_dim),
            nn.SiLU(True),
            # Second linear layer
            nn.Linear(fc2_input_dim, 3 * 3 * 32),
            nn.ReLU(True)
        )

        # Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        # Convolutional section
        self.decoder_conv = nn.Sequential(
            # First transposed convolution
            # nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1),  # Update input and output channels
            # nn.BatchNorm2d(32),
            # nn.SiLU(True),

            # Additional convolution
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),  # Update input and output channels
            nn.BatchNorm2d(16),
            nn.SiLU(True),

            # Second transposed convolution
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.SiLU(True),

            # Third transposed convolution
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
            # nn.MaxPool2d(2),  # Add a MaxPooling layer here
        )


    def forward(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # print(x.shape)
        # print("******")
        # Unflatten
        x = self.unflatten(x)
        # print(x.shape)
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        x = torch.sigmoid(x)
        return x