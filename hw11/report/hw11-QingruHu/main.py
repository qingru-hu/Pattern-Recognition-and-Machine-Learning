import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
from torch import optim
import torch.nn.functional as F


# Prepare the CIFAR-100 dataset and the dataloader
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 128
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)

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
        dim = int(2 * dim_in)
        self.conv1 = nn.Conv2d(dim_in, dim, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(dim)
        self.shortcut = nn.Sequential(
            nn.Conv2d(dim_in, dim, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(dim)
        )

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
lossfunc = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Write the training loop.
for epoch in range(10):
    epoch_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        images, labels = data[0].cuda(), data[1].cuda()
        optimizer.zero_grad()
        outputs = net(images)
        loss = lossfunc(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    # print epoch loss
    print(f'Epoch: {epoch + 1}, Loss: {epoch_loss / 200:.3f}')

# Evaluate the model on the test dataset
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].cuda(), data[1].cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy of the ResNet on 10000 test images is {100*correct/total:.2f} %.')
