import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from data_preprocessing import load_and_preprocess_data


class TCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=2, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        # (batch_size, seq_len, output_dim)
        return x.transpose(1, 2)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        inner_product = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        keys_dim = torch.tensor(self.head_dim, dtype=torch.float32, requires_grad=False)
        scaled_inner_product = inner_product / torch.sqrt(keys_dim)
        attention = torch.nn.functional.softmax(scaled_inner_product, dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)

        return self.fc_out(out)


class WaveNetBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=2, dilation=1):
        super(WaveNetBlock, self).__init__()
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, dilation=dilation)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        tanh_out = self.tanh(self.conv(x))
        sigmoid_out = self.sigmoid(self.conv(x))
        x = tanh_out * sigmoid_out
        # (batch_size, seq_len, output_dim)
        return x.transpose(1, 2)


class CombinedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CombinedModel, self).__init__()

        # TCN层
        self.tcn = TCNBlock(input_dim, hidden_dim)

        # Attention层
        self.multi_head_attention = MultiHeadAttention(embed_size=hidden_dim, heads=8)

        # WaveNet层
        self.wavenet = WaveNetBlock(hidden_dim, hidden_dim)

        # LSTM层
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)

        # 输出层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.tcn(x)
        x = self.multi_head_attention(x, x, x)
        x = self.wavenet(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    X_train_np, X_test_np, y_train_np, y_test_np = load_and_preprocess_data("./data/电机转动角度(弧度).csv")

    # Convert data to torch.Tensor and adjust shape to fit LSTM
    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device).view(-1, X_train_np.shape[1], 1)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device).view(-1, X_test_np.shape[1], 1)
    y_train = torch.tensor(y_train_np.values, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test_np.values, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize SimpleWaveNet model
    model = CombinedModel(input_dim=X_train.shape[2], hidden_dim=64, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    # To save loss for each epoch
    losses = []

    # Train the model
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

        # Update the learning rate
        scheduler.step(avg_loss)

    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)

    # Transfer predictions from GPU to CPU
    predictions = predictions.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

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

    plt.tight_layout()
    plt.show()
