import torch
import torch.nn as nn
from app.core.logging import get_logger

logger = get_logger(__name__)


class CNN_BiLSTM(nn.Module):
    def __init__(
        self,
        n_features: int = 14,
        seq_len: int = 30,
        conv_filters: int = 64,
        kernel_size: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len

        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels=n_features,
                out_channels=conv_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(conv_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.bilstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        logger.info(
            f"CNN_BiLSTM creado — features={n_features}, "
            f"conv={conv_filters}, lstm={lstm_hidden}x{lstm_layers}, "
            f"dropout={dropout}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        x = x.permute(0, 2, 1)
        out, _ = self.bilstm(x)
        out = out[:, -1, :]
        return self.fc(out)
