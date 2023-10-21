from bayes_opt import BayesianOptimization

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from baseline import LSTMModel, X_train, train_loader, X_test, y_test

# 检查是否有可用的 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


def objective(hidden_dim, lr):
    hidden_dim = int(hidden_dim)

    # 构建模型
    model = LSTMModel(X_train.shape[2], hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=False)

    # 训练模型
    epochs = 10  # 减少epoch数量以加速搜索过程
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_X).squeeze()
            # print("y_pred shape:", y_pred.shape) y_pred shape: torch.Size([1024])
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)

        # 更新学习率
        scheduler.step(avg_loss)
    # print("X_test shape:", X_test.shape)
    # 预测
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    predictions = predictions.view(-1).cpu().numpy()
    # ("y_test shape:", y_test.shape)
    y_test_np = y_test.view(-1).cpu().numpy()

    # 返回负MSE，因为我们希望最小化这个值
    return -np.mean((predictions - y_test_np) ** 2)

if __name__ == '__main__':

    # 定义超参数的范围
    pbounds = {
        'hidden_dim': (32, 256),
        'lr': (0.001, 0.01),
    }

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        verbose=2,  # 1: print only when a maximum is observed, 0: silent
        random_state=1,
    )

    # 开始优化，这里我们进行10次迭代
    optimizer.maximize(
        init_points=2,
        n_iter=8,
    )

    # optimizer.maximize(
    #     init_points=1,
    #     n_iter=0,
    # )

    print(optimizer.max)
