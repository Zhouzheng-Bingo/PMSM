import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from ResNetLSTMAttentionModel import ResNetLSTMAttentionModel

# 检查是否有可用的 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# 读取数据
data = pd.read_csv("./data/电机转动角度(弧度).csv", encoding='gbk')
print(data.columns)

# 创建滞后变量
lags = 5
for i in range(1, lags + 1):
    data[f'指令_lag_{i}'] = data['指令'].shift(i)
    data[f'实际_lag_{i}'] = data['实际'].shift(i)
data.dropna(inplace=True)

# 分割数据
features = ['指令'] + [f'指令_lag_{i}' for i in range(1, lags + 1)] + [f'实际_lag_{i}' for i in range(1, lags + 1)]
X = data[features]
y = data['实际']

# 数据归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 转为tensor并移动到 GPU
X_train = torch.tensor(X_train, dtype=torch.float32).view(X_train.shape[0], 1, -1).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).view(X_test.shape[0], 1, -1).to(device)
y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test.values, dtype=torch.float32).to(device)

# 创建数据加载器
batch_size = 1024
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# 构建LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=0.6)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out.view(x.size(0), -1))
        return y_pred


if __name__ == '__main__':
    best_hidden_dim = 246
    best_lr = 0.002483
    # 构建 ResNetLSTMAttentionModel 模型
    input_channels = X_train.shape[2]  # 根据你的输入特征数量
    num_residual_blocks = 3
    lstm_hidden_dim = 128
    model = ResNetLSTMAttentionModel(input_channels, num_residual_blocks, lstm_hidden_dim).to(device)

    # model = LSTMModel(X_train.shape[2], best_hidden_dim).to(device)  # 使用最佳的hidden_dim
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_lr)  # 使用最佳的学习率

    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    # 用于保存每个epoch的损失值
    losses = []

    # 训练模型
    epochs = 20
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_X).squeeze()
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")

        # 更新学习率
        scheduler.step(avg_loss)

    # 预测
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)

    # 将预测结果从 GPU 移到 CPU
    predictions = predictions.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    # 绘制收敛图和预测结果图
    plt.figure(figsize=(15, 5))

    # 绘制收敛图
    plt.subplot(1, 2, 1)
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Convergence Graph")
    plt.legend()

    # 绘制预测结果图
    plt.subplot(1, 2, 2)
    plt.plot(predictions, label="Predictions", color="red")
    plt.plot(y_test_np, label="Actual Values", color="blue")
    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.title("Predictions vs Actual Values")
    plt.legend()

    plt.tight_layout()
    plt.show()
