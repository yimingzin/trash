import torch
import torch.nn as nn
import torch.nn.functional as F


# Step 2: Define Transformer-based model
class TransformerTemperaturePredictor(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2):
        super(TransformerTemperaturePredictor, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

        nn.init.xavier_uniform_(self.fc.weight)
    def forward(self, x):
        # x shape: [batch_size, input_size]
        x = self.embedding(x)  # [batch_size, input_size] -> [batch_size, input_size, d_model]
        x = self.transformer_encoder(x.unsqueeze(1))  # [batch_size, 1, d_model]
        x = self.fc(x.squeeze(1))  # [batch_size, 1]
        return x


# 位置编码类，用于为输入的特征提供位置信息
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
