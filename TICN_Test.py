import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)

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
        self.inception1 = InceptionLayer(1, 16)
        self.inception2 = InceptionLayer(1, 16)

        # WeightNorm and ReLU
        self.norm1 = nn.utils.weight_norm(nn.Conv1d(16, 16, kernel_size=1))
        self.norm2 = nn.utils.weight_norm(nn.Conv1d(16, 16, kernel_size=1))

        # Dilated Causal Conv Layer
        self.dilated_conv1 = CausalConv1d(16, 16, kernel_size=3, dilation=2)
        self.dilated_conv2 = CausalConv1d(16, 16, kernel_size=3, dilation=2)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Multi-head attention layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=16, num_heads=4)

        # Dense Layer
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
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
        x = x.transpose(0, 1)
        attn_output, _ = self.multihead_attn(x, x, x)

        # Passing through dense layer
        x = self.fc(attn_output.transpose(0, 1))

        return x


# Instantiate model
model = Model()
print(model)
