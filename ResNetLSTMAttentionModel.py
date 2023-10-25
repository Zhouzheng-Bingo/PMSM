import torch.nn as nn
from res_block import Residual, FirstBlock, LastBlock


class ResNetLSTMAttentionModel(nn.Module):
    def __init__(self, input_channels, num_residual_blocks, lstm_hidden_dim, output_dim=1):
        super(ResNetLSTMAttentionModel, self).__init__()

        # ResNet Blocks
        self.first_block = FirstBlock(input_channels, 64, 32)
        self.residual_blocks = nn.Sequential(
            *[Residual(64 if i == 0 else 128, 128, 128) for i in range(num_residual_blocks)]
        )
        self.last_block = LastBlock(128, 3072, output_dim)  # Updated to include output_dim

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=3072, hidden_size=lstm_hidden_dim, num_layers=1, batch_first=True)

        # Attention Layer
        self.attention = nn.MultiheadAttention(embed_dim=lstm_hidden_dim, num_heads=1, batch_first=True)

        # Output Layer
        self.output_layer = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, x):
        # Apply ResNet Blocks
        x = self.first_block(x)
        x = self.residual_blocks(x)
        x = self.last_block(x)

        # Reshape for LSTM
        x = x.permute(0, 2, 1)
        x = x.reshape(x.size(0), -1, 3072)

        # Apply LSTM
        lstm_out, _ = self.lstm(x)

        # Apply Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Apply Output Layer
        output = self.output_layer(attn_out[:, -1, :])

        return output
