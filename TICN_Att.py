import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(InceptionLayer, self).__init__()

        # Inception structures with 1x1, 2x8, and 4x16 kernels
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(input_channels, output_channels, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(input_channels, output_channels, kernel_size=4, padding=2)

        self.weight_norm = nn.utils.weight_norm

    def forward(self, x):
        x1 = self.weight_norm(self.conv1(x))
        x2 = self.weight_norm(self.conv2(x))
        x3 = self.weight_norm(self.conv3(x))

        # Concatenate along channels axis
        return torch.cat([x1, x2, x3], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, dilation):
        super(ResidualBlock, self).__init__()

        self.inception = InceptionLayer(input_channels, 32)  # As per your description
        self.conv = nn.utils.weight_norm(nn.Conv1d(32 * 3, 32, kernel_size=2, padding=dilation, dilation=dilation))
        self.downsample = nn.Conv1d(input_channels, output_channels, kernel_size=1)  # 1x1 conv
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.inception(x)
        out = F.relu(out)
        out = F.dropout(out, 0.5)
        out = self.conv(out)
        out = F.relu(out)
        out = F.dropout(out, 0.5)
        if residual.size(1) != out.size(1):
            residual = self.downsample(residual)
        out += residual  # Residual connection

        return out


class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Network, self).__init__()

        dilations = [1, 2, 4, 8, 16]
        self.blocks = nn.ModuleList(
            [ResidualBlock(input_dim if i == 0 else 32 * 3, 32 * 3, d) for i, d in enumerate(dilations)])

        # Multi-head Attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=32 * 3, num_heads=2)

        # Dense Layer
        self.fc = nn.Linear(32 * 3, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change to N, C, L

        for block in self.blocks:
            x = block(x)

        # Multihead Attention
        attn_output, _ = self.multihead_attn(x, x, x)
        x = attn_output

        # Pass through dense layer
        x = self.fc(x.permute(0, 2, 1))  # Change back to N, L, C

        return x


# Modify the code to include the model
# model = Network(X_train.shape[2], 1).to(device)

# You can continue with your existing training code
