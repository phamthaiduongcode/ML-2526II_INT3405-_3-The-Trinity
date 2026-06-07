"""
src/models/eeg_bilstm.py
Chứa định nghĩa kiến trúc các mạng Neural — KHÔNG chứa code Train.
"""

import torch
import torch.nn as nn


class EEG_BiLSTM(nn.Module):
    """
    Model dùng cho Exp 1 (2-class) và Exp 2 (4-class), vẽ t-SNE.

    Input : (batch, 128, 32)  — 128 timesteps, 32 kênh
    Output: (batch, n_classes) — raw logits, KHÔNG có Softmax
    """
    def __init__(self, input_size=32, hidden_size=64, num_layers=2, n_classes=2, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        out, _ = self.bilstm(x)
        out    = out[:, -1, :]
        out    = self.dropout(out)
        return self.classifier(out)

    def get_hidden(self, x):
        """Trả về hidden state trước Dense — dùng để vẽ t-SNE."""
        out, _ = self.bilstm(x)
        out    = out[:, -1, :]
        out    = self.dropout(out)
        return nn.functional.relu(self.classifier[0](out))


class BiLSTM_Model(nn.Module):
    """
    Model dùng riêng cho Exp 3 (LOSO) — kiến trúc 2 lớp LSTM xếp chồng.

    Input : (batch, 32, 128) — 32 kênh, 128 timesteps (sẽ permute nội bộ)
    Output: (batch, num_classes) — raw logits
    """
    def __init__(self, input_size=32, hidden_size=64, num_classes=2, dropout=0.5):
        super(BiLSTM_Model, self).__init__()
        self.lstm1    = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2    = nn.LSTM(hidden_size * 2, hidden_size // 2, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc       = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x        = x.permute(0, 2, 1)   # (batch, 128, 32)
        out, _   = self.lstm1(x)
        out      = self.dropout1(out)
        out, _   = self.lstm2(out)
        out      = self.dropout2(out)
        out      = out[:, -1, :]
        return self.fc(out)
