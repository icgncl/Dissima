import torch.nn as nn
import torch
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.reflection_pad = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(8, 8, kernel_size=(3, 3))
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm1 = nn.BatchNorm2d(4)
        self.batch_norm2 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(8 * 250 * 250, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 4)

    def forward_one_branch(self, x):
        x = self.batch_norm1(self.relu(self.conv1(self.reflection_pad(x))))
        x = self.batch_norm2(self.relu(self.conv2(self.reflection_pad(x))))
        x = self.batch_norm2(self.relu(self.conv3(self.reflection_pad(x))))
        x = x.view(x.size()[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one_branch(input1)
        output2 = self.forward_one_branch(input2)

        return output1, output2


class ConstrastiveLoss(nn.Module):

    def __init__(self, margin=2.0):
        super(ConstrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distance = F.pairwise_distance(output1, output2, keepdim=True)
        contrastive_loss = torch.mean((1 - label) * torch.pow(distance, 2)
                                      + label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))

        return contrastive_loss
