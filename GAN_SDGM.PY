# -*- coding: utf-8 -*-
"""
Created on Thu May 15 20:03:33 2025

@author: ARNES
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import time

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations: normalize to [-1, 1] because Tanh is used in the Generator
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
dataloader = DataLoader(
    datasets.MNIST(root="./data", train=True, download=True, transform=transform),
    batch_size=128,
    shuffle=True
)

# Architecture of Generator and Discriminator
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(nz, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 28*28),
            nn.Tanh()  # Output should be in the range [-1, 1]
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(-1, 1, 28, 28)  # Reshape output to image format

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability of being real
        )

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        return self.net(x)

# Initialize Model and Optimizer
nz = 100  # Dimension of noise
G = Generator(nz).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()  # Binary Cross Entropy Loss
lr = 0.001  # Learning rate

optimizerG = optim.SGD(G.parameters(), lr=lr)  # Optimizer for Generator
optimizerD = optim.SGD(D.parameters(), lr=lr)  # Optimizer for Discriminator

# Training Process
epochs = 1
fixed_noise = torch.randn(64, nz, device=device)  # Fixed noise for generating images
start_time1 = time.time()
for epoch in range(epochs):
    for i, (real, _) in enumerate(dataloader):
        batch_size = real.size(0)
        real = real.to(device)  # Move real images to device

        # === Update Discriminator ===
        D.zero_grad()  # Zero the gradients for Discriminator
        label_real = torch.ones(batch_size, 1, device=device)  # Real labels
        label_fake = torch.zeros(batch_size, 1, device=device)  # Fake labels

        # Real loss
        output_real = D(real)  # Discriminator output for real images
        loss_real = criterion(output_real, label_real)  # Loss for real images

        # Fake loss
        noise = torch.randn(batch_size, nz, device=device)  # Generate noise
        fake = G(noise)  # Generate fake images
        output_fake = D(fake.detach())  # Discriminator output for fake images
        loss_fake = criterion(output_fake, label_fake)  # Loss for fake images

        # Total Discriminator loss
        loss_D = loss_real + loss_fake
        loss_D.backward()  # Backpropagation
        optimizerD.step()  # Update Discriminator

        # === Update Generator ===
        G.zero_grad()  # Zero the gradients for Generator
        output = D(fake)  # Discriminator output for fake images
        loss_G = criterion(output, label_real)  # Target: pretend fake images are real
        loss_G.backward()  # Backpropagation
        optimizerG.step()  # Update Generator

        # === Logging ===
        if i % 200 == 0:
                print(f"[{epoch+1}/{epochs}] Batch {i}/{len(dataloader)} \
                Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

    # Display intermediate results
    with torch.no_grad():
        fake_images = G(fixed_noise).detach().cpu()  # Generate images from fixed noise
    grid = vutils.make_grid(fake_images, padding=2, normalize=True)  # Create a grid of images
    plt.imshow(np.transpose(grid, (1, 2, 0)))  # Transpose for correct image format
    plt.title(f"Epoch {epoch+1} GAN with Linear Layers")  # Title for the plot
    plt.show()  # Show the generated images

elapsed_time1 = time.time() - start_time1  # Calculate elapsed time
elapsed_time1 = elapsed_time1 / 60  # Convert to minutes
print(f'Total training time for GAN with Linear Layers = {elapsed_time1:.2f} minutes')