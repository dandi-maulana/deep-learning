# -*- coding: utf-8 -*-
"""
Created on Thu May 29 23:01:48 2025

@author: ARNES
--> Additional feature to save two Excel files (Normalized & Original + Date columns)

Notes:
1. Use the following command in Anaconda Prompt to install yfinance:
   pip install yfinance
2. Use the following command to install library for saving datasets to Excel:
   pip install openpyxl
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

# Fetch Apple (AAPL) stock data from Yahoo Finance
# Date range: January 1, 2019 to January 1, 2024
df = yf.download("AAPL", start="2019-01-01", end="2024-01-01")
data = df[['Close']].copy()  # We'll use only the closing prices
dates = df.index  # Store the date index for later use

# Data Preprocessing
# Normalize the data using MinMaxScaler (scales values between 0 and 1)
# Create sequences of data (e.g., last 60 days to predict day 61)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences_with_dates(data, dates, seq_length):
    """Create input sequences with corresponding dates for time series prediction.
    
    Args:
        data: The time series data (normalized or original)
        dates: Corresponding dates for the data points
        seq_length: Length of input sequences (number of time steps)
        
    Returns:
        xs: Input sequences (each of length seq_length)
        ys: Target values (next value after each sequence)
        x_dates: Dates corresponding to input sequences
        y_dates: Dates corresponding to target values
    """
    xs, ys, x_dates, y_dates = [], [], [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
        x_dates.append(dates[i:i+seq_length])
        y_dates.append(dates[i+seq_length])
    return np.array(xs), np.array(ys), np.array(x_dates), np.array(y_dates)

SEQ_LEN = 60  # Use last 60 days to predict the next day
X, y, X_dates, y_dates = create_sequences_with_dates(data_scaled, dates, SEQ_LEN)
X_original, y_original, _, _ = create_sequences_with_dates(data.values, dates, SEQ_LEN)  # Non-normalized version

# Split Dataset into training and testing sets
# 80% for training, 20% for testing
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]
X_orig_train, y_orig_train = X_original[:split_idx], y_original[:split_idx]
X_orig_test, y_orig_test = X_original[split_idx:], y_original[split_idx:]
X_dates_train, y_dates_train = X_dates[:split_idx], y_dates[:split_idx]
X_dates_test, y_dates_test = X_dates[split_idx:], y_dates[split_idx:]

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
    
    Args:
        input_size: Number of features in input (1 for univariate time series)
        hidden_size: Number of hidden units in RNN layer
        num_layers: Number of stacked RNN layers
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(PriceRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output layer for single value prediction

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        out, _ = self.rnn(x)  # out shape: (batch_size, seq_length, hidden_size)
        return self.fc(out[:, -1, :])  # Only use the last time step's output for prediction

# Initialize model, loss function and optimizer
model = PriceRNN()
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with smaller learning rate

# Training loop
for epoch in range(50):  # Number of training epochs
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
    
    # Print training progress
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Model Evaluation
# Calculate RMSE and plot predictions vs actual values
model.eval()  # Set model to evaluation mode

# Prepare test data tensor
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

# Excel Export Functions
def sequences_to_long_df(X, y, x_dates, y_dates, set_name):
    """Convert sequences to long format DataFrame for Excel export.
    
    Args:
        X: Input sequences
        y: Target values
        x_dates: Dates for input sequences
        y_dates: Dates for target values
        set_name: Name of the dataset ('train' or 'test')
        
    Returns:
        DataFrame in long format suitable for Excel export
    """
    rows = []
    for i in range(X.shape[0]):
        for t in range(X.shape[1]):
            rows.append({
                'set': set_name,
                'sequence_id': i,
                'date': x_dates[i, t],
                'time_step': t + 1,
                'x': X[i, t, 0],
                'y': y[i][0] if t == X.shape[1] - 1 else np.nan,
                'target_date': y_dates[i] if t == X.shape[1] - 1 else pd.NaT
            })
    return pd.DataFrame(rows)

# Create DataFrames (normalized data)
train_norm_df = sequences_to_long_df(X_train, y_train, X_dates_train, y_dates_train, "train")
test_norm_df = sequences_to_long_df(X_test, y_test, X_dates_test, y_dates_test, "test")

# Create DataFrames (original data)
train_orig_df = sequences_to_long_df(X_orig_train, y_orig_train, X_dates_train, y_dates_train, "train")
test_orig_df = sequences_to_long_df(X_orig_test, y_orig_test, X_dates_test, y_dates_test, "test")

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Save normalized data to Excel
with pd.ExcelWriter("output/train_test_normalized.xlsx", engine='openpyxl') as writer:
    train_norm_df.to_excel(writer, sheet_name="train", index=False)
    test_norm_df.to_excel(writer, sheet_name="test", index=False)

# Save original data to Excel
with pd.ExcelWriter("output/train_test_original.xlsx", engine='openpyxl') as writer:
    train_orig_df.to_excel(writer, sheet_name="train", index=False)
    test_orig_df.to_excel(writer, sheet_name="test", index=False)

print("✅ Two Excel files successfully created:")
print("- output/train_test_normalized.xlsx (for training)")
print("- output/train_test_original.xlsx (original data with dates)")

### PREDICTION PHASE
# Convert test data to tensor
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create Dataset and DataLoader for test set
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Make predictions and save results to Excel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 1: Make predictions and collect results
y_pred = []
with torch.no_grad():
    for x_batch, _ in test_loader:
        x_batch = x_batch.to(device)
        output = model(x_batch)
        y_pred.extend(output.cpu().numpy())  # collect predictions

# Step 2: Convert to numpy arrays
y_pred = np.array(y_pred).reshape(-1, 1)
y_test_np = y_test.reshape(-1, 1)

# Step 3: Get target dates (y_dates_test)
target_dates = y_dates_test  # length should match y_test

# Step 4: Save normalized predictions to DataFrame
predict_df_norm = pd.DataFrame({
    'target_date': target_dates,
    'y_true': y_test_np.flatten(),
    'y_pred': y_pred.flatten()
})

# Step 5: Save to 'predict' sheet in normalized Excel file
norm_path = "output/train_test_normalized.xlsx"
with pd.ExcelWriter(norm_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    predict_df_norm.to_excel(writer, sheet_name='predict', index=False)

# Step 6: Convert back to original scale
y_true_orig = scaler.inverse_transform(y_test_np)
y_pred_orig = scaler.inverse_transform(y_pred)

predict_df_orig = pd.DataFrame({
    'target_date': target_dates,
    'y_true': y_true_orig.flatten(),
    'y_pred': y_pred_orig.flatten()
})

# Step 7: Save to 'predict' sheet in original Excel file
orig_path = "output/train_test_original.xlsx"
with pd.ExcelWriter(orig_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    predict_df_orig.to_excel(writer, sheet_name='predict', index=False)

print("✅ Prediction results successfully added to both Excel files (sheet 'predict').")