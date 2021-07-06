import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

import numpy as np
from collections import OrderedDict

from torch.nn.modules.loss import GaussianNLLLoss


class Generator(nn.Module):
    def __init__(self, noise_dim, activation=nn.ReLU, dp_rate=0.3):
        super().__init__()
        self.activation = activation
        self.stem = nn.Sequential(OrderedDict([
            ('linear',  nn.Linear(noise_dim, 512*4*4, bias=False)),
            ('bn',      nn.LazyBatchNorm2d()),
            ('act',     activation(inplace=True)),
            ('dropout', nn.Dropout(dp_rate)),       # try 2d for spatial
        ]))

        self.stacks = nn.Sequential(OrderedDict([
            ('layer1',  self.upsample(512, dp_rate=dp_rate)),
            ('layer2',  self.upsample(256, dp_rate=dp_rate)),
            ('layer3',  self.upsample(128, dp_rate=dp_rate)),
            ('layer4',  self.upsample( 64, dp_rate=0))
        ]))

        self.gen = nn.Sequential(OrderedDict([
            ('conv',    nn.LazyConvTranspose2d(3, kernel_size=4, stride=2, padding=1)),
            ('act',     nn.Tanh()),
        ]))

    def upsample(self, num_filters, bn=True, dp_rate=0.3):
        layers = [nn.LazyConvTranspose2d(num_filters, kernel_size=4, stride=2, bias=not bn, padding=1)]
        if bn:
            layers.append(nn.BatchNorm2d(num_filters))
        layers.append(self.activation(inplace=True))
        if dp_rate > 0:
            layers.append(nn.Dropout(dp_rate))      # try 2d for spatial
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = x.view(-1, 512, 4, 4)
        x = self.stacks(x)
        x = self.gen(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, noise_dim, activation=nn.LeakyReLU, dp_rate=0.3):
        super().__init__()
        self.activation = activation
        self.stacks = nn.Sequential(OrderedDict([
            ('layer1',  self.downsample( 32, bn=False, stride=1)),
            ('layer2',  self.downsample( 64)),
            ('layer3',  self.downsample(128)),
            ('layer4',  self.downsample(256)),
            ('layer5',  self.downsample(512))
        ]))

        self.head = nn.Sequential(OrderedDict([
            ('gauss', GaussianNoise()),
            ('linear', nn.LazyLinear(1)),
            ('act', nn.Sigmoid()),
        ]))

    def downsample(self, num_filters, bn=True, stride=2):
        layers = [
            GaussianNoise(),
            nn.LazyConv2d(num_filters, kernel_size=4, stride=stride, bias=not bn, padding=1)
        ]
        if bn:
            layers.append(nn.BatchNorm2d(num_filters))
        layers.append(self.activation(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stacks(x)
        x = x.flatten(1)
        x = self.head(x)
        return x

noise_dim = 128
netG = Generator(noise_dim)
netD = Discriminator(noise_dim)

criterion = nn.BCELoss()

batch_size = 128

def train_step():
    noise = torch.randn(batch_size, noise_dim)
    gen = netG(noise)

    rl = netD(real)
    fk = netD(gen)

    torch.empty(3,4, dtype=torch.float32, device='cuda').uniform_(0.9,1.)
    torch.empty(3,4, dtype=torch.float32, device='cuda').uniform_(0.,0.1)
    gl = criterion(fk, torch.ones_like(fk))
    dl = criterion(rl, torch.ones_like(rl)) + criterion(fk, torch.zeros_like(fk))

    grad(gl, netG.parameters(), retain_graph=True)
    grad(dl, netD.parameters())

    netD.requires_grad_(False); netG.requires_grad_(True)
    gl.backward(retain_graph=True)
    netD.requires_grad_(True); netG.requires_grad_(False)
    dl.backward()
