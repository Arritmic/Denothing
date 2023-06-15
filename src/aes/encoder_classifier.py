import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn

class Classifier(nn.Module):

    def __init__(self, fc2_input_dim):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            # nn.BatchNorm2d(8),
            nn.ReLU(True),
            # Second convolutional layer
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
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
            nn.Linear(2 * 32, 128),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(128, 47)  # Changed to output 10 classes
        )

        # Add a softmax layer for the output
        self.softmax = nn.LogSoftmax(dim=1)  # or nn.Softmax(dim=1)

    def forward(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = self.flatten(x)
        # Apply linear layers
        x = self.encoder_lin(x)
        # Apply softmax
        x = self.softmax(x)
        return x
