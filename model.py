import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load and preprocess the data
class TemperatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Load data from CSV file
data = pd.read_csv('D:/PythonProject/infrared_temperature_measurement/data/data.csv')  # Replace with your CSV file path

# Assume the last column is the target temperature, and others are features
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalize features and target values
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
y = y.reshape(-1, 1)
y_scaler = MinMaxScaler(feature_range=(0, 1))
y = y_scaler.fit_transform(y).flatten()

# Create Dataset object
dataset = TemperatureDataset(X, y)

# Define KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Step 2: Define the ITM1D-CNN model
class ITM1DCNN(nn.Module):
    def __init__(self, input_size):
        super(ITM1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32 * (input_size // 2), 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Step 3: Cross-Validation Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()

# To store loss values for all folds
fold_train_losses = []
fold_val_losses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f'Fold {fold + 1}')

    # Create train and validation datasets
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    # Create DataLoader objects
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    # Initialize the model for each fold
    model = ITM1DCNN(input_size=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    epochs = 100
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation step
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                total_loss += loss.item() * inputs.size(0)

        avg_loss = total_loss / len(val_loader.dataset)
        val_losses.append(avg_loss)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {avg_loss:.4f}')

    # Store fold losses
    fold_train_losses.append(train_losses)
    fold_val_losses.append(val_losses)

# Step 4: Plot Training and Validation Loss for All Folds
for fold in range(len(fold_train_losses)):
    plt.plot(range(1, epochs + 1), fold_train_losses[fold], label=f'Fold {fold + 1} Train Loss')
    plt.plot(range(1, epochs + 1), fold_val_losses[fold], label=f'Fold {fold + 1} Val Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss for Each Fold')
plt.legend()
plt.show()

# Step 5: Predict Temperatures on Test Set
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
model.eval()
predictions = []
actuals = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(labels.numpy())

# Inverse transform the predictions and actual values to original scale
predictions = y_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
actuals = y_scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

# Display some of the predictions
for i in range(10):
    print(f'Actual: {actuals[i]:.2f}, Predicted: {predictions[i]:.2f}')