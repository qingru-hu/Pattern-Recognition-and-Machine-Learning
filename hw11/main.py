import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
from torch import optim
import torch.nn.functional as F


# Prepare the CIFAR-100 dataset and the dataloader
# TODO

# Finish the multi-stage ResNet model. Here, you are required to finish the DownSamplingBlock class.
class BasicBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(dim)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DownSamplingBlock(nn.Module):
    def __init__(self, dim_in, dim):
        super().__init__()
        # TODO

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.blocks = nn.Sequential(
            BasicBlock(16),
            BasicBlock(16),
            DownSamplingBlock(16, 32),
            BasicBlock(32),
            DownSamplingBlock(32, 64),
            BasicBlock(64)
        )
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
net = ResNet().cuda()

# Define the optimizer and loss function.
# TODO

# Write the training loop.
# TODO

# Evaluate the model on the test dataset
# TODO
