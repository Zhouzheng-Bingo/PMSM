import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from data_preprocessing import load_and_preprocess_data
from TICN_Att import Network


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    X_train_np, X_test_np, y_train_np, y_test_np = load_and_preprocess_data("./data/电机转动角度(弧度).csv")

    # 将数据转换为 torch.Tensor 并调整形状以适应 TICN_Att
    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device).view(-1, X_train_np.shape[1], 1)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device).view(-1, X_test_np.shape[1], 1)
    y_train = torch.tensor(y_train_np.values, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test_np.values, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = Network(X_train.shape[2], 1).to(device)  # 调整模型的输入维度
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
