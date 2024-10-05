import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# Step 1: Load and preprocess the data
class TemperatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Load and normalize the data
data = pd.read_csv('D:/PythonProject/infrared_temperature_measurement/data/data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
y = y.reshape(-1, 1)
y_scaler = MinMaxScaler(feature_range=(0, 1))
y = y_scaler.fit_transform(y).flatten()

dataset = TemperatureDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 2: Define Transformer-based model
class TransformerTemperaturePredictor(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2):
        super(TransformerTemperaturePredictor, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: [batch_size, input_size]
        x = self.embedding(x)  # [batch_size, input_size] -> [batch_size, input_size, d_model]
        x = x.unsqueeze(1)  # Add sequence dimension -> [batch_size, 1, d_model]
        x = self.transformer_encoder(x)  # [batch_size, 1, d_model]
        x = x.mean(dim=1)  # Average pooling across the sequence length
        x = self.fc(x)  # [batch_size, 1]
        return x


# Step 3: Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerTemperaturePredictor(input_size=X.shape[1]).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 300
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')

# Step 4: Predict temperatures
model.eval()
with torch.no_grad():
    inputs = torch.tensor(X, dtype=torch.float32).to(device)
    predictions = model(inputs).cpu().numpy()
    predictions = y_scaler.inverse_transform(predictions)

# Display some of the predictions
for i in range(10):
    print(f'Actual: {y_scaler.inverse_transform(y.reshape(-1, 1))[i][0]:.2f}, Predicted: {predictions[i][0]:.2f}')
