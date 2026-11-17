"""Modelo LSTM para previsão um passo à frente."""
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        # Dropout interno do LSTM só atua entre camadas (num_layers > 1)
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):                        # x: (B, window, F)
        out, _ = self.lstm(x)                    # out: (B, window, H)
        return self.head(self.drop(out[:, -1]))  # usa o último estado oculto -> (B, 1)