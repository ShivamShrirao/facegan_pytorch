import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.cuda import amp

import random
import numpy as np
from collections import OrderedDict


def seed_everything(seed=33):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

seed_everything()


class GaussianNoise(nn.Module):
    def __init__(self, mean=0, std=0.1):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return x + torch.empty_like(x).normal_(mean=self.mean, std=self.std)


class Generator(nn.Module):
    def __init__(self, noise_dim, activation=nn.ReLU, dp_rate=0.3):
        super().__init__()
        self.activation = activation
        self.stem = nn.Sequential(OrderedDict([
            ('linear',  nn.Linear(noise_dim, 512*4*4, bias=False)),
            ('bn',      nn.LazyBatchNorm1d()),
            ('act',     activation(inplace=True)),
            ('dropout', nn.Dropout(dp_rate)),       # try 2d for spatial
        ]))

        self.stacks = nn.Sequential(*[
            self.upsample(512, dp_rate=dp_rate),
            self.upsample(256, dp_rate=dp_rate),
            self.upsample(128, dp_rate=dp_rate),
            self.upsample(64, dp_rate=0)
        ])

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
    def __init__(self, activation=nn.LeakyReLU, std=0.1):
        super().__init__()
        self.std = std
        self.activation = activation
        self.stacks = nn.Sequential(*[
            self.downsample(32, bn=False, stride=1),
            self.downsample(64),
            self.downsample(128),
            self.downsample(256),
            self.downsample(512)
        ])

        self.head = nn.Sequential(OrderedDict([
            ('gauss', GaussianNoise(self.std)),
            ('linear', nn.LazyLinear(1)),
            ('act', nn.Sigmoid()),
        ]))

    def downsample(self, num_filters, bn=True, stride=2):
        layers = [
            GaussianNoise(self.std),
            nn.LazyConv2d(num_filters, kernel_size=4,
                          stride=stride, bias=not bn, padding=1)
        ]
        if bn:
            layers.append(nn.BatchNorm2d(num_filters))
        layers.append(self.activation(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stacks(x)
        x = x.flatten(1)
        x = self.head(x)
        return x


@torch.no_grad()
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


noise_dim = 128

def train_step(real_images, netG, netD, optG, optD, scaler, use_amp=True):
    noise = torch.randn(real_images.size(0), noise_dim, dtype=real_images.dtype,
                        device=real_images.device)
    # Update generator
    with amp.autocast(enabled=use_amp):
        fake_out = netD(netG(noise))
        lossG = criterion(fake_out, torch.ones_like(fake_out))  # Treat fake images as real to train the Generator.

    # grad(lossG, netG.parameters(), retain_graph=True)     # this can also be used to calculate grads wrt specific parameters and update them for parameters manually.
    netG.requires_grad_(True)           # Only calculate gradients for Generator.
    netD.requires_grad_(False)          # Do not calculate gradients for Discriminator.
    scaler.scale(lossG).backward(retain_graph=True) # retain graph cause fake_out is also used to calculate loss for Discriminator.
    scaler.step(optG)
    optG.zero_grad(set_to_none=True)

    # Update Discriminator
    with amp.autocast(enabled=use_amp):
        real_out = netD(real_images)
        lossD = (criterion(real_out, torch.empty_like(real_out).uniform_(0.9, 1.0))
                 + criterion(fake_out, torch.empty_like(fake_out).uniform_(0.0, 0.1)))   # Treat real as real and fake as fake to train Discriminator.

    # grad(lossD, netD.parameters())
    netG.requires_grad_(False)          # Do not calculate gradients for Generator.
    netD.requires_grad_(True)           # Only calculate gradients for Discriminator.
    scaler.scale(lossD).backward()
    scaler.step(optD)
    optD.zero_grad(set_to_none=True)

    scaler.update()


if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64

    netG = Generator(noise_dim).to(DEVICE)
    netD = Discriminator().to(DEVICE)

    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()

    lr = 2e-4
    beta1 = 0.5
    optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    use_amp = True
    scaler = amp.GradScaler(enabled=use_amp)

    noise = torch.randn(batch_size, noise_dim, device=DEVICE)
    gen = netG(noise)
    o = netD(gen)
    print(gen.shape)
    print(o.shape)
