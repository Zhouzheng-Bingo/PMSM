import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from data_preprocessing import load_and_preprocess_data
from TICN_Att import Network

def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):  # 修改这里
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i + seq_length - 1])  # 确保不会越界
    return np.array(X_seq), np.array(y_seq)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())

    # 加载并预处理数据
    X_train_np, X_test_np, y_train_np, y_test_np = load_and_preprocess_data("./data/电机转动角度(弧度).csv")

    # 转换为时间序列格式
    seq_length = 10  # 你可以根据需要调整这个值
    X_train_np, y_train_np = create_sequences(X_train_np, y_train_np.values, seq_length)
    X_test_np, y_test_np = create_sequences(X_test_np, y_test_np.values, seq_length)

    # 确保 X_train_np 和 X_test_np 的形状是 (num_samples, seq_len, 2)
    # 这里假设 '指令' 和 '实际' 是 X_train_np 和 X_test_np 的最后两列
    X_train_np = X_train_np[:, :, -2:]
    X_test_np = X_test_np[:, :, -2:]

    # 将数据转换为 torch.Tensor
    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).to(device).view(-1, 1)
    y_test = torch.tensor(y_test_np, dtype=torch.float32).to(device).view(-1, 1)

    # 调整数据的形状以匹配网络结构
    X_train = X_train.permute(0, 2, 1)
    X_test = X_test.permute(0, 2, 1)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 初始化模型、损失函数、优化器和学习率调度器
    model = Network(input_dim=2, output_dim=1, seq_len=seq_length).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    losses = []

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
        scheduler.step(avg_loss)

    model.eval()
    with torch.no_grad():
        predictions = model(X_test)

    predictions = predictions.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Convergence Graph")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(predictions, label="Predictions", color="red")
    plt.plot(y_test_np, label="Actual Values", color="blue")
    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.title("Predictions vs Actual Values")
    plt.legend()

    plt.tight_layout()
    plt.show()
