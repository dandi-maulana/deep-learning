"""
Created on Thu Apr 24 12:15:25 2025

@author: ARNES
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
from PIL import Image

# Location of the dataset and label storage folder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Dataset ---
data_dir = r"C:\deep_learning\dataset\flower-dataset"  # Change to your dataset path
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "test")

# Transformations to be applied to the dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to 32x32 pixels
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
])

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Batch size for training
batch_size = 64

# Define CNN architecture
class CNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 4 * 4, num_classes)  # Fully connected layer

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)  # Flatten the tensor
        x = self.fc(x)  # Fully connected layer
        return x

# Initialize model
model = CNN()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                        eps=1e-8, weight_decay=1e-2, amsgrad=False)

# Number of epochs
num_epochs = 50

# Training the model
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # Get predicted classes
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)  # Average training loss
    train_acc = 100 * correct / total  # Training accuracy
    train_losses.append(train_loss)
    train_accs.append(train_acc)

        # Evaluation
    model.eval()  # Set model to evaluation mode
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # Get predicted classes
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)  # Average validation loss
    val_acc = 100 * correct / total  # Validation accuracy
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# Save Model ---
torch.save(model.state_dict(), "CNN1.pth")

# Plot Loss & Accuracy ---
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs+1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accs, label="Train Accuracy")
plt.plot(range(1, num_epochs+1), val_accs, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.grid()

plt.show()

# Model Evaluation (Confusion Matrix & Classification Report) ---
model.eval()
true_labels, pred_labels = [], []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predicted.cpu().numpy())

# Check if true_labels and pred_labels have the same length
if len(true_labels) > 0 and len(pred_labels) > 0 and len(true_labels) == len(pred_labels):
    # --- Confusion Matrix ---
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    # --- Classification Report ---
    print("Classification Report:\n")
    print(classification_report(true_labels, pred_labels, target_names=train_dataset.classes))
else:
    print("⚠️ ERROR: No data processed in validation! Check DataLoader and dataset.")
