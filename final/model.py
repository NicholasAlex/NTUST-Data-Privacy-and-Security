"""
model.py — Models for FL system + DLG attack

TinyCNN:
  - 2 conv layers, NO BatchNorm
  - Designed to be vulnerable to DLG attack
  - Follows the original DLG paper's experimental setup
  - Used for BOTH Step 1 (FL training) and Step 2 (attack)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCNN(nn.Module):
    """
    Small CNN for CIFAR-10, no BatchNorm — DLG compatible.

    Input:  3 x 32 x 32 (CIFAR-10 RGB images)
    Output: 10 classes

    Architecture:
      Conv(3→32) → ReLU → Pool (16x16)
      Conv(32→64) → ReLU → Pool (8x8)
      FC(64*8*8 → 256) → ReLU → FC(256 → 10)

    Why no BatchNorm:
      BN mixes batch statistics into gradients, making DLG inversion
      mathematically ill-posed. Without BN, each sample's gradients
      uniquely determine the input — exactly what DLG exploits.
    """

    def __init__(self, num_classes: int = 10):
        super(TinyCNN, self).__init__()

        # Block 1: 3 → 32 channels, 32x32 → 16x16
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: 32 → 64 channels, 16x16 → 8x8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Classifier
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_weights(self):
        return [p.data.clone() for p in self.parameters()]

    def set_weights(self, weights):
        for p, w in zip(self.parameters(), weights):
            p.data.copy_(w)
