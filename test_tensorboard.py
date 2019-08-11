import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)
        self.bn = nn.BatchNorm2d(20)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x),2)
        x = F.relu(x) + F.relu(-x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = self.bn(x)
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,training=self.training)
        x = self.fc2(x)
        x = F.softmax(x,dim=1)
        return x

dummy_input = torch.rand(13,1,28,28)

model = Net1()

with SummaryWriter(log_dir='test_tensorboard') as writer:
    # for epoch in range(100):
    #     writer.add_scalar('scalar/test',np.random.rand(),epoch)
    #     writer.add_scalar('scalar/test',epoch*np.sin(epoch))
    #     writer.add_scalars('scalar/0/scalars_test',{'xsinx':epoch*np.sin(epoch)},epoch)
    #     writer.add_scalars('scalar/0/scalars_test',{'xcosx': epoch * np.cos(epoch)}, epoch)
    #     writer.add_scalars('scalar/1/scalars_test', {'xsinx': epoch * np.sin(epoch)}, epoch)
    #     writer.add_scalars('scalar/1/scalars_test', {'xcosx': epoch * np.cos(epoch)}, epoch)
    writer.add_graph(model,(dummy_input,))
    writer.close()