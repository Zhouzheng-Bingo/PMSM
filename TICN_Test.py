import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class InceptionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionLayer, self).__init__()
        self.conv1 = nn.Conv1d(16, out_channels, kernel_size=1)
        self.conv3 = nn.Conv1d(16, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(16, out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        return x1 + x3 + x5


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation * (kernel_size - 1),
                              dilation=dilation)

    def forward(self, x):
        return self.conv(x)[:, :, :-self.conv.padding[0]]


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Inception Layer
        self.inception1 = InceptionLayer(16, 16)
        self.inception2 = InceptionLayer(16, 16)

        # WeightNorm and ReLU
        # self.norm1 = nn.utils.weight_norm(nn.Conv1d(16, 16, kernel_size=1))
        # self.norm2 = nn.utils.weight_norm(nn.Conv1d(16, 16, kernel_size=1))
        self.norm1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(16, 16, kernel_size=1))
        self.norm2 = nn.utils.parametrizations.weight_norm(nn.Conv1d(16, 16, kernel_size=1))

        # Dilated Causal Conv Layer
        self.dilated_conv1 = CausalConv1d(16, 16, kernel_size=3, dilation=2)
        self.dilated_conv2 = CausalConv1d(16, 16, kernel_size=3, dilation=2)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        self.embedding_layer = nn.Linear(23, 16)

        # Multi-head attention layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=16, num_heads=4)

        # Dense Layer
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        # 将输入维度从 [batch_size, 23, seq_len] 转换为 [batch_size, 16, seq_len]
        x = self.embedding_layer(x.transpose(1, 2)).transpose(1, 2)

        x1 = self.inception1(x)
        x1 = F.relu(self.norm1(x1))
        x1 = self.dropout(x1)
        x1 = self.dilated_conv1(x1)

        x2 = self.inception2(x)
        x2 = F.relu(self.norm2(x2))
        x2 = self.dropout(x2)
        x2 = self.dilated_conv2(x2)

        # Combining outputs
        x = x1 + x2

        # Transposing for multihead attention
        # 确保数据形状是 [seq_len, batch_size, 16]
        x = x.permute(2, 0, 1)

        # print("Shape of x before attention:", x.shape)  # 打印数据形状
        attn_output, _ = self.multihead_attn(x, x, x)

        # Passing through dense layer
        # 确保数据形状回到 [batch_size, seq_len, 16] 以便通过全连接层
        x = self.fc(attn_output.permute(1, 0, 2)).squeeze(1)

        return x


# # Instantiate model
# model = Model()
# print(model)

# 数据预处理部分
def load_and_preprocess_data(file_path, lags=1):
    data = pd.read_csv(file_path)

    # 选择相关的列
    relevant_columns = ['time', 'id_command', 'id_feedback', 'iq_command', 'iq_feedback']
    data = data[relevant_columns]

    # 创建反馈值的滞后变量
    for i in range(1, lags + 1):
        data[f'id_feedback_lag_{i}'] = data['id_feedback'].shift(i)
        data[f'iq_feedback_lag_{i}'] = data['iq_feedback'].shift(i)

    data.dropna(inplace=True)

    # 数据归一化
    feature_cols = ['time', 'id_command', 'iq_command'] + [f'id_feedback_lag_{i}' for i in range(1, lags + 1)] + [
        f'iq_feedback_lag_{i}' for i in range(1, lags + 1)]
    output_cols = ['id_feedback', 'iq_feedback']

    X = data[feature_cols]
    y = data[output_cols]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test


# 创建数据加载器
def create_loader(X, y):
    X_tensor = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32)
    y_tensor = torch.tensor(y.values if isinstance(y, pd.DataFrame) else y, dtype=torch.float32)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)


if __name__ == '__main__':
    # 加载和预处理数据
    X_train, X_test, y_train, y_test = load_and_preprocess_data("./data/电流_id_iq.csv", lags=10)
    # print("Input shape:", X_train.shape)
    train_loader = create_loader(X_train, y_train)
    test_loader = create_loader(X_test, y_test)

    # 定义模型
    model = Model()

    # 训练模型
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.view(inputs.size(0), 23, -1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.view(inputs.size(0), 23, -1)
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test Loss: {test_loss/len(test_loader):.4f}")
