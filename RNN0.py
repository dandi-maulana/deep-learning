# -*- coding: utf-8 -*-
"""
Created on Fri May 30 00:10:10 2025

@author: ARNES

Note:
  Use the following command in Anaconda Prompt to install yfinance:
  pip install yfinance
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

import yfinance as yf

# Fetch stock data for Apple (AAPL) from Yahoo Finance
# Date range: January 1, 2019 to January 1, 2024
df = yf.download("AAPL", start="2019-01-01", end="2024-01-01")

# Data Preprocessing
# We'll use only the 'Close' price column
# Normalize data using MinMaxScaler (scales values between 0 and 1)
# Create sequences of data (e.g., last 60 days to predict day 61)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[['Close']].values)

def create_sequences(data, seq_length):
    """Create input sequences and corresponding target values for time series prediction.
    
    Args:
        data: The time series data
        seq_length: Length of input sequences (number of time steps)
        
    Returns:
        xs: Input sequences (each of length seq_length)
        ys: Target values (next value after each sequence)
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LEN = 60  # Use last 60 days to predict the next day
X, y = create_sequences(data_scaled, SEQ_LEN)

# Split Dataset into training and testing sets
# 80% for training, 20% for testing
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Create PyTorch Dataset and DataLoader for efficient batch processing
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Build RNN Model for Price Prediction
class PriceRNN(nn.Module):
    """Recurrent Neural Network for stock price prediction.
    
    Architecture:
        - RNN layer with specified hidden size and number of layers
        - Fully connected layer for final prediction
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(PriceRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output layer for single value prediction

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        out, _ = self.rnn(x)  # out shape: (batch_size, seq_length, hidden_size)
        return self.fc(out[:, -1, :])  # Only use the last time step's output for prediction

# Train the Model
model = PriceRNN()
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

# Training loop
for epoch in range(100):  # Number of training epochs
    for X_batch, y_batch in train_loader:
        # Reshape input for RNN (batch_size, seq_length, input_size)
        X_batch = X_batch.view(-1, SEQ_LEN, 1)
        
        # Forward pass
        output = model(X_batch)
        loss = criterion(output, y_batch)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print progress
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluate Model
# Calculate RMSE and plot predictions vs actual values
model.eval()  # Set model to evaluation mode

# Prepare test data
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, SEQ_LEN, 1)

# Make predictions
with torch.no_grad():  # Disable gradient calculation for evaluation
    predicted = model(X_test_tensor).detach().numpy()

# Rescale predictions and actual values back to original price range
predicted_rescaled = scaler.inverse_transform(predicted)
actual_rescaled = scaler.inverse_transform(y_test)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(actual_rescaled, label='Actual Price')
plt.plot(predicted_rescaled, label='Predicted Price')
plt.xlabel("Time Step (Days)")          # X-axis label
plt.ylabel("Price (USD)")              # Y-axis label
plt.title("Actual vs Predicted Stock Prices")  # Chart title
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()