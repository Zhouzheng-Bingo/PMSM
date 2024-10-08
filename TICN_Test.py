import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


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
        self.norm1 = nn.utils.weight_norm(nn.Conv1d(16, 16, kernel_size=1))
        self.norm2 = nn.utils.weight_norm(nn.Conv1d(16, 16, kernel_size=1))
        # self.norm1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(16, 16, kernel_size=1))
        # self.norm2 = nn.utils.parametrizations.weight_norm(nn.Conv1d(16, 16, kernel_size=1))
        self.batch_norm1 = nn.BatchNorm1d(16)  # 添加批处理标准化
        self.batch_norm2 = nn.BatchNorm1d(16)  # 添加批处理标准化

        # Dilated Causal Conv Layer
        self.dilated_conv1 = CausalConv1d(16, 16, kernel_size=3, dilation=2)
        self.dilated_conv2 = CausalConv1d(16, 16, kernel_size=3, dilation=2)

        # Dropout
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.embedding_layer = nn.Linear(201, 16)

        # Multi-head attention layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=16, num_heads=4)

        # Dense Layer
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        # 将输入维度从 [batch_size, 23, seq_len] 转换为 [batch_size, 16, seq_len]
        # print(x.shape)
        x = self.embedding_layer(x.transpose(1, 2)).transpose(1, 2)
        # print(x.shape)
        x1 = self.inception1(x)
        x1 = F.relu(self.norm1(x1))
        x1 = self.batch_norm1(x1)  # 使用批处理标准化
        x1 = self.dropout1(x1)
        x1 = self.dilated_conv1(x1)

        x2 = self.inception2(x)
        x2 = F.relu(self.norm2(x2))
        x2 = self.batch_norm2(x2)  # 使用批处理标准化
        x2 = self.dropout2(x2)
        x2 = self.dilated_conv2(x2)

        # Combining outputs
        x = x1 + x2
        # print(x.shape)
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
    lag_data = pd.concat([data[col].shift(i) for i in range(1, lags + 1) for col in ['id_feedback', 'iq_feedback']],
                         axis=1)
    lag_columns = [f'{col}_lag_{i}' for i in range(1, lags + 1) for col in ['id_feedback', 'iq_feedback']]
    lag_data.columns = lag_columns
    data = pd.concat([data, lag_data], axis=1)

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
    X_train, X_test, y_train, y_test = load_and_preprocess_data("./data/电流_id_iq.csv", lags=99)
    train_loader = create_loader(X_train, y_train)
    test_loader = create_loader(X_test, y_test)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # 201:99个历史id和iq，1个当前id和iq，1个时间，99*2+1*2+1=201
    # 定义模型
    model = Model()

    # 训练模型
    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    # 添加L2正则化 (weight decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # L2

    # 如果想添加L1正则化，可以使用如下方法：
    # 但是注意，L1正则化可能会使得一些权重变为0，导致模型部分参数无效。
    def l1_penalty(var):
        return torch.norm(var, 1)

    # l1_weight = 1e-5

    epochs = 10
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for inputs, targets in train_loader:
            # print(targets.shape)
            inputs = inputs.view(inputs.size(0), 201, -1)
            # print(inputs.shape)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs.shape)
            # l1_loss = sum(l1_penalty(param) for param in model.parameters())
            # loss = criterion(outputs, targets) + l1_weight * l1_loss  # 在损失中添加L1正则化
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            print(epoch_train_loss)
        train_losses.append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_test_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                # print(inputs.shape)
                inputs = inputs.view(inputs.size(0), 201, -1) # 确保数据形状是 [batch_size, seq_len, 16]
                outputs = model(inputs)
                epoch_test_loss += criterion(outputs, targets).item()
                all_preds.append(outputs)
                all_targets.append(targets)

        test_losses.append(epoch_test_loss / len(test_loader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

    # 绘制测试集的电流iq和电流id的预测值和真实值
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # 绘制训练和测试损失、测试集的电流id的预测值和真实值、测试集的电流iq的预测值和真实值
    plt.figure(figsize=(18, 5))

    # 绘制训练和测试损失
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制测试集的电流id的预测值和真实值
    plt.subplot(1, 3, 2)
    plt.plot(all_preds[:, 0], label='Predicted id', linestyle='--', color='blue')
    plt.plot(all_targets[:, 0], label='True id', color='lightsalmon')
    plt.title('Test Set Predictions for id')
    plt.xlabel('Sample')
    plt.ylabel('Current id')
    plt.legend()

    # 绘制测试集的电流iq的预测值和真实值
    plt.subplot(1, 3, 3)
    plt.plot(all_preds[:, 1], label='Predicted iq', linestyle='--', color='red')
    plt.plot(all_targets[:, 1], label='True iq', color='lightblue')
    plt.title('Test Set Predictions for iq')
    plt.xlabel('Sample')
    plt.ylabel('Current iq')
    plt.legend()

    plt.tight_layout()
    plt.show()