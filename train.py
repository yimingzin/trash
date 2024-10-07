import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from model import *
from dataset import *

# 3. 初始化模型参数
input_dim = X_train.shape[1]
d_model = 64
n_heads = 4
num_encoder_layers = 2
dim_feedforward = 256
dropout = 0.1

model = TransformerTemperaturePredictor(input_dim, d_model, n_heads, num_encoder_layers)

# 4. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 5. 训练和验证过程
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证过程
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

# 6. 开始训练
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=300)


# 测试集评估并输出预测值
def evaluate_model_and_predict(model, test_loader, criterion, y_scaler):
    model.eval()
    test_loss = 0.0
    all_predictions = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

            # 将预测值保存起来并进行反归一化
            predictions = y_scaler.inverse_transform(outputs.cpu().numpy())
            all_predictions.extend(predictions)

    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')

    # 打印所有预测值
    for i, prediction in enumerate(all_predictions):
        print(f"Sample {i + 1}: Predicted Temperature: {prediction[0]:.2f}")

# 使用测试集评估模型并输出预测值
evaluate_model_and_predict(model, test_loader, criterion, y_scaler)

