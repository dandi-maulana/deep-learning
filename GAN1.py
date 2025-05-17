# -*- coding: utf-8 -*-
"""
Created on Thu May 15 19:55:08 2025

@author: ARNES
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.nn import Upsample
import matplotlib.pyplot as plt
import numpy as np
import time

# Gunakan GPU jika tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformasi: normalize ke [-1, 1] karena Tanh digunakan di Generator
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST
dataloader = DataLoader(
    datasets.MNIST(root="./data", train=True, download=True, transform=transform),
    batch_size=128,
    shuffle=True
)
#
#Arsitektur Generator dan Discriminator
#
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz

        self.fc = nn.Sequential(
            nn.Linear(nz, 128 * 7 * 7),
            nn.ReLU(True)
        )

        self.reshape_size = (128, 7, 7)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 7x7 -> 14x14
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.ReLU(True)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 14x14 -> 28x28
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.ReLU(True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(-1, *self.reshape_size)  # Reshape to (batch, 128, 7, 7)
        out = self.up1(out)
        out = self.up2(out)
        img = self.final_conv(out)
        return img

#
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Input: (1, 28, 28)
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),   # → (64, 14, 14)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # → (128, 7, 7)
            nn.ZeroPad2d((0, 1, 0, 1)),                               # → (128, 8, 8)
            nn.BatchNorm2d(128, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            nn.Flatten(),                                            # → (128 * 8 * 8,)
            nn.Linear(128 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

#
#Inisialisasi Model dan Optimizer
nz = 100  # Dimensi noise
G = Generator(nz).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
lr = 0.001

optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

#
#Proses Pelatihan
epochs = 10
fixed_noise = torch.randn(64, nz, device=device)
start_time1 = time.time()
for epoch in range(epochs):
    for i, (real, _) in enumerate(dataloader):
        batch_size = real.size(0)
        real = real.to(device)

        # === Update Discriminator ===
        D.zero_grad()
        label_real = torch.ones(batch_size, 1, device=device)
        label_fake = torch.zeros(batch_size, 1, device=device)

        # Real loss
        output_real = D(real)
        loss_real = criterion(output_real, label_real)

        # Fake loss
        noise = torch.randn(batch_size, nz, device=device)
        fake = G(noise)
        output_fake = D(fake.detach())
        loss_fake = criterion(output_fake, label_fake)

        # Total D loss
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizerD.step()

        # === Update Generator ===
        G.zero_grad()
        output = D(fake)
        loss_G = criterion(output, label_real)  # Target: seolah-olah asli
        loss_G.backward()
        optimizerG.step()

        # === Logging ===
        if i % 200 == 0:
            print(f"[{epoch+1}/{epochs}] Batch {i}/{len(dataloader)} \
                  Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")
    
    # Tampilkan hasil sementara
    with torch.no_grad():
        fake_images = G(fixed_noise).detach().cpu()
    grid = vutils.make_grid(fake_images, padding=2, normalize=True)
    plt.imshow(np.transpose(grid, (1,2,0)))
    plt.title(f"Epoch {epoch+1} GAN dengan Convolution Layer")
    plt.show()
elapsed_time1 = time.time() - start_time1
elapsed_time1=elapsed_time1/60
print(f'Total Waktu pelatihan GAN1 = {elapsed_time1} menit')