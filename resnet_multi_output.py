import pandas as pd
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from data_preprocessing import load_and_preprocess_data_multi_output
import os
import json


class Res2NetBlock(nn.Module):
    def __init__(self, dim, num_splits=4, kernel_size=3, padding=1):
        super(Res2NetBlock, self).__init__()
        self.num_splits = num_splits
        self.dim_split = dim // num_splits  # Determine the number of channels per split
        self.splits = nn.ModuleList([nn.Conv1d(self.dim_split, self.dim_split, kernel_size, padding=padding) for _ in range(num_splits)])
        self.relu = nn.ReLU()

    def forward(self, x):
        chunks = torch.chunk(x, self.num_splits, dim=1)  # Split the feature map into multiple sub-features
        output = []
        for i in range(self.num_splits):
            out_i = chunks[i] + self.splits[i](chunks[i])  # Apply the conv op independently to each split
            output.append(out_i)
        output = torch.cat(output, dim=1)  # Concatenate the processed splits back together
        return self.relu(output)


# ResNet Module Definition
class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks=3):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(*[Res2NetBlock(hidden_dim) for _ in range(num_blocks)])
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, input_dim, seq_len) -> (batch_size, hidden_dim, seq_len)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.blocks(x)
        # (batch_size, seq_len, hidden_dim)
        return x.transpose(1, 2)


class CombinedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=3, num_heads=4):
        super(CombinedModel, self).__init__()

        # ResNet Layer
        self.resnet = ResNet(input_dim, hidden_dim, num_blocks)

        # LSTM层
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)

        # 多头注意力模块
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # 输出层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = self.tcn(x)
        x = self.resnet(x)
        x, _ = self.lstm(x)
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.fc(attn_output[:, -1, :])
        return x


def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    # X_train_np, X_test_np, y_train_np, y_test_np = load_and_preprocess_data("./data/电机转动角度(弧度).csv")
    X_train_np, X_test_np, y_train_np, y_test_np = load_and_preprocess_data_multi_output("./data/多数据源位置预测_all.csv")
    # X_train_np, X_test_np, y_train_np, y_test_np = load_and_preprocess_data_multi_output_default_lags("./data/多数据源位置预测_all.csv")
    # print(X_train_np.shape, X_test_np.shape, y_train_np.shape, y_test_np.shape)
    # Convert data to torch.Tensor and adjust shape to fit LSTM
    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device).view(-1, X_train_np.shape[1], 1) # (batch_size, seq_len, input_dim)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device).view(-1, X_test_np.shape[1], 1)
    # y_train = torch.tensor(y_train_np.values, dtype=torch.float32).to(device).view(-1, 1)
    # y_test = torch.tensor(y_test_np.values, dtype=torch.float32).to(device).view(-1, 1)
    y_train = torch.tensor(y_train_np.values, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test_np.values, dtype=torch.float32).to(device)
    print(y_train.shape, y_test.shape)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # Initialize the model with more residual blocks
    num_residual_blocks = 22  # for example, to have 22 residual blocks
    train_flag = 1  # 训练并评估：train_flag = 1&test_flag = 0
    test_flag = 0  # 测试：train_flag = 0&test_flag = 1

    if train_flag == 1 and test_flag == 0:
        # model = CombinedModel(input_dim=1, hidden_dim=64, output_dim=1).to(device)
        model = CombinedModel(input_dim=1, hidden_dim=64, output_dim=4, num_blocks=num_residual_blocks, num_heads=4).to(device)

        criterion = nn.MSELoss()
        # learning_rate = 0.000864293053363452
        # weight_decay = 0.02915625436117011
        # learning_rate = 0.0002382091521841793
        learning_rate = 0.0000001
        weight_decay = 0.06454389416796091

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

        # To save loss for each epoch
        losses = []

        # 在训练循环外定义一个变量来存储使用模型预测的概率
        sampling_prob = 0.0  # 初始时完全依赖于真实数据

        # Train the model
        epochs = 20
        for epoch in range(epochs):
            model.train()
            epoch_losses = []
            for batch_X, batch_y in train_loader:
                # print(batch_X.shape, batch_y.shape)
                # 使用模型预测作为输入的概率
                use_model_pred = torch.bernoulli(torch.tensor([sampling_prob])).bool().item()

                if use_model_pred:
                    # 初始化序列
                    sequence = batch_X.clone().to(device)

                    # 存储模型的预测
                    y_pred = []

                    for i in range(sequence.size(2)):  # 遍历特征维度
                        # 使用当前序列获取模型的预测
                        pred = model(sequence).squeeze()

                        # 存储预测
                        y_pred.append(pred)

                        # 更新序列
                        if i + 1 < sequence.size(2):
                            sequence[:, :, i + 1] = pred.unsqueeze(-1)

                    # 将预测列表转换为张量
                    y_pred = y_pred[-1].squeeze(-1)

                    # 计算损失
                    loss = criterion(y_pred, batch_y)

                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    optimizer.step()

                    # 存储本批次的平均损失
                    epoch_losses.append(loss.item())
                else:
                    optimizer.zero_grad()
                    # y_pred = model(batch_X).squeeze()
                    y_pred = model(batch_X) # 新做的特征工程，形状和原来不一致，去掉了squeeze()
                    # print(y_pred.shape, batch_y.shape)
                    loss = criterion(y_pred, batch_y)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                    optimizer.step()
                    epoch_losses.append(loss.item())
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")
            # 在每个epoch后增加使用模型预测的概率
            sampling_prob = min(sampling_prob + 0.01, 1.0)  # 增加概率，但不超过1.0
            # Update the learning rate
            scheduler.step(avg_loss)

        # 保存模型
        model_save_path = './model/'
        model_filename = 'model_checkpoint_epoch2_lagsEx5_block22.pth'

        # 检查是否存在保存模型的目录，如果不存在则创建
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        # 保存模型
        torch.save(model.state_dict(), os.path.join(model_save_path, model_filename))
        print(f"Model saved to {os.path.join(model_save_path, model_filename)}")

        # 在训练循环结束后保存损失值
        loss_directory = 'loss'
        if not os.path.exists(loss_directory):
            os.makedirs(loss_directory)

        with open(os.path.join(loss_directory, 'loss_values.json'), 'w') as f:
            json.dump(losses, f)

        model.eval()
        with torch.no_grad():
            # Initialize the sequence with history data
            sequence = X_test.clone().to(device)

            # To store model's predictions
            predictions = []

            for i in range(X_test.size(0)):
                # For each sample in the test set, we iterate over its features
                sample_seq = sequence[i:i + 1]

                for j in range(X_test.size(2) - 1):  # iterating over feature dimension except the last one
                    # Get the model's prediction for the current sequence
                    pred = model(sample_seq[:, :, :j + 1]).squeeze()

                    # Update the sequence with the model's prediction
                    sample_seq[:, :, j + 1] = pred.unsqueeze(-1)

                # After iterating over all features, get the final prediction
                final_pred = model(sample_seq).squeeze()

                # Store the final prediction
                predictions.append(final_pred.cpu().numpy())

            # Convert the list of predictions to a numpy array
            predictions = np.array(predictions)

        # Now, 'predictions' contains the model's predictions for the entire test set
        print("Predictions Min:", predictions.min())
        print("Predictions Max:", predictions.max())
        y_test_np = y_test.cpu().numpy()
        print("Actual Values Min:", y_test_np.min())
        print("Actual Values Max:", y_test_np.max())

    elif train_flag == 0 and test_flag == 1:
        model = CombinedModel(input_dim=1, hidden_dim=64, output_dim=4, num_blocks=num_residual_blocks, num_heads=4).to(
            device)
        # 加载保存的模型参数
        model.load_state_dict(torch.load('./model/model_checkpoint_epoch2_lagsEx5_block22.pth'))

        # model.load_state_dict(torch.load('./model/model_checkpoint_epoch20_lags50_block22.pth')) # 如果模型没在内存加载模型

        # # Make predictions
        # model.eval()
        # with torch.no_grad():
        #     predictions = model(X_test)
        # # Transfer predictions from GPU to CPU
        # predictions = predictions.cpu().numpy()
        # # print(predictions)
        # y_test_np = y_test.cpu().numpy()
        #
        # print("Predictions Min:", predictions.min())
        # print("Predictions Max:", predictions.max())
        # print("Actual Values Min:", y_test_np.min())
        # print("Actual Values Max:", y_test_np.max())

        # Make predictions using the model's own previous predictions
        model.eval()
        with torch.no_grad():
            # Initialize the sequence with history data
            sequence = X_test.clone().to(device)

            # To store model's predictions
            predictions = []

            for i in range(X_test.size(0)):
                # For each sample in the test set, we iterate over its features
                sample_seq = sequence[i:i + 1]

                for j in range(X_test.size(2) - 1):  # iterating over feature dimension except the last one
                    # Get the model's prediction for the current sequence
                    pred = model(sample_seq[:, :, :j + 1]).squeeze()

                    # Update the sequence with the model's prediction
                    sample_seq[:, :, j + 1] = pred.unsqueeze(-1)

                # After iterating over all features, get the final prediction
                final_pred = model(sample_seq).squeeze()

                # Store the final prediction
                predictions.append(final_pred.cpu().numpy())

            # Convert the list of predictions to a numpy array
            predictions = np.array(predictions)

        # Now, 'predictions' contains the model's predictions for the entire test set
        print("Predictions Min:", predictions.min())
        print("Predictions Max:", predictions.max())
        y_test_np = y_test.cpu().numpy()
        print("Actual Values Min:", y_test_np.min())
        print("Actual Values Max:", y_test_np.max())
    else:
        print("Invalid combination of train_flag and test_flag.")

    plt.figure(figsize=(15, 10))

    # 读取损失值
    loss_directory = 'loss'
    loss_filename = 'loss_values.json'
    loss_file_path = os.path.join(loss_directory, loss_filename)

    if os.path.exists(loss_file_path):
        with open(loss_file_path, 'r') as f:
            losses = json.load(f)
    else:
        losses = None
        print("Loss values file not found. Convergence graph will not be plotted.")

    # 在这里，确保 y_test_np 是一个 NumPy array
    y_test_np = y_test_np.values if isinstance(y_test_np, pd.DataFrame) else y_test_np

    # Plot actual values vs predictions for each output
    for i in range(4):
        plt.subplot(2, 3, i + 1)
        plt.plot(y_test_np[:, i], label="Actual Values", color="blue")
        plt.plot(predictions[:, i], label="Predictions", color="red")
        plt.xlabel("Samples")
        plt.ylabel(f"Output {i + 1}")
        plt.title(f"Actual vs Predictions for Output {i + 1}")
        plt.legend()

        # 计算预测值和实际值的最小值和最大值
        min_value = min(predictions[:, i].min(), y_test_np[:, i].min())
        max_value = max(predictions[:, i].max(), y_test_np[:, i].max())
        # 设置y轴的限制
        plt.ylim(min_value, max_value)

    # Plot the convergence graph
    if losses is not None:
        plt.subplot(2, 3, 6)
        plt.plot(losses, label="Training Loss", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Convergence Graph")
        plt.legend()

    plt.tight_layout()

    if not os.path.exists('./pic'):
        os.makedirs('./pic')
    plt.savefig('./pic/model_checkpoint_epoch2_lagsEx5_block22.png')

    plt.show()
