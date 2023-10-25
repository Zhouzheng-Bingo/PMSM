import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from data_preprocessing import load_and_preprocess_data


# Residual Block Definition
class ResidualBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


# ResNet Module Definition
class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks=3):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, input_dim, seq_len) -> (batch_size, hidden_dim, seq_len)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.blocks(x)
        # (batch_size, seq_len, hidden_dim)
        return x.transpose(1, 2)


class CombinedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=3):
        super(CombinedModel, self).__init__()

        # ResNet Layer
        self.resnet = ResNet(input_dim, hidden_dim, num_blocks)

        # LSTM层
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)

        # 输出层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = self.tcn(x)
        x = self.resnet(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    # X_train_np, X_test_np, y_train_np, y_test_np = load_and_preprocess_data("./data/电机转动角度(弧度).csv")
    X_train_np, X_test_np, y_train_np, y_test_np = load_and_preprocess_data("./data/电机转速(转每秒).csv")
    # print(X_train_np.shape, X_test_np.shape, y_train_np.shape, y_test_np.shape)
    # Convert data to torch.Tensor and adjust shape to fit LSTM
    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device).view(-1, X_train_np.shape[1], 1)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device).view(-1, X_test_np.shape[1], 1)
    y_train = torch.tensor(y_train_np.values, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test_np.values, dtype=torch.float32).to(device)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize the model with more residual blocks
    num_residual_blocks = 10  # for example, to have 10 residual blocks
    # model = CombinedModel(input_dim=1, hidden_dim=64, output_dim=1).to(device)
    model = CombinedModel(input_dim=1, hidden_dim=64, output_dim=1, num_blocks=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    # To save loss for each epoch
    losses = []

    # Train the model
    epochs = 5
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for batch_X, batch_y in train_loader:
            # print(batch_X.shape, batch_y.shape)
            optimizer.zero_grad()
            y_pred = model(batch_X).squeeze()
            # print(y_pred.shape, batch_y.shape)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")

        # Update the learning rate
        scheduler.step(avg_loss)

    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    # Transfer predictions from GPU to CPU
    predictions = predictions.cpu().numpy()
    # print(predictions)
    y_test_np = y_test.cpu().numpy()

    print("Predictions Min:", predictions.min())
    print("Predictions Max:", predictions.max())
    print("Actual Values Min:", y_test_np.min())
    print("Actual Values Max:", y_test_np.max())

    # Plot convergence graph and predictions
    plt.figure(figsize=(15, 5))

    # Plot the convergence graph
    plt.subplot(1, 2, 1)
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Convergence Graph")
    plt.legend()

    # Plot predictions vs. actual values
    plt.subplot(1, 2, 2)
    plt.plot(predictions, label="Predictions", color="red")
    plt.plot(y_test_np, label="Actual Values", color="blue")
    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.title("Predictions vs Actual Values")
    plt.legend()

    # 计算预测值和实际值的最小值和最大值
    min_value = min(predictions.min(), y_test_np.min())
    max_value = max(predictions.max(), y_test_np.max())
    # 打印最小值和最大值
    print("Min value:", min_value)
    print("Max value:", max_value)
    plt.ylim(min_value, max_value)

    plt.tight_layout()
    plt.show()
