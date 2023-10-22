import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(InceptionLayer, self).__init__()

        # Inception structures with 1x1, 2x1, and 4x1 kernels
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(input_channels, output_channels, kernel_size=1))
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(input_channels, output_channels, kernel_size=2, padding=1))
        self.conv3 = nn.utils.weight_norm(nn.Conv1d(input_channels, output_channels, kernel_size=4, padding=2))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        # Ensure all tensors have the same spatial dimensions
        min_len = min(x1.size(2), x2.size(2), x3.size(2))
        x1, x2, x3 = x1[:, :, :min_len], x2[:, :, :min_len], x3[:, :, :min_len]

        # Concatenate along channels axis
        return torch.cat([x1, x2, x3], dim=1)



class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, dilation):
        super(ResidualBlock, self).__init__()

        self.inception = InceptionLayer(input_channels, 32)
        self.conv = nn.utils.weight_norm(nn.Conv1d(32 * 3, output_channels, kernel_size=2, padding=dilation, dilation=dilation))
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

        if residual.size(1) != out.size(1) or residual.size(2) != out.size(2):
            residual = self.downsample(residual)

        # Ensure the spatial dimensions are the same
        min_len = min(residual.size(2), out.size(2))
        residual, out = residual[:, :, :min_len], out[:, :, :min_len]

        out += residual  # Residual connection
        return out


class Network(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len):
        super(Network, self).__init__()

        self.input_dim = input_dim
        self.embed_dim = 32 * 3
        self.seq_len = seq_len
        # 添加一维卷积层以处理时间序列数据
        self.conv1d = nn.Conv1d(self.input_dim, self.embed_dim, 1)
        self.input_linear = nn.Linear(self.input_dim, self.embed_dim)
        self.pre_attn_linear = nn.Linear(self.embed_dim, self.embed_dim)

        # 修改权重初始化形状
        self.input_linear.weight.data.normal_(0, 0.01)
        self.input_linear.bias.data.zero_()

        dilations = [1, 2, 4, 8, 16]
        self.blocks = nn.ModuleList(
            [ResidualBlock(self.embed_dim, self.embed_dim, d) for d in dilations])

        # Multi-head Attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=2)

        # Dense Layer
        self.fc = nn.Linear(self.embed_dim, output_dim)

    def forward(self, x):
        # 首先通过一维卷积层
        x = self.conv1d(x)
        x = x.view(x.size(0), -1)
        x = self.input_linear(x)
        x = self.pre_attn_linear(x)

        for block in self.blocks:
            x = block(x)

        # Multihead Attention
        attn_output, _ = self.multihead_attn(x, x, x)
        x = attn_output

        x = self.fc(x.permute(0, 2, 1))

        return x

