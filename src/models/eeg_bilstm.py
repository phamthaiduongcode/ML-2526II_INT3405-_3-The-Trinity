"""
src/models/eeg_bilstm.py
Chứa định nghĩa kiến trúc các mạng Neural — KHÔNG chứa code Train.
Đã được FIX lỗi index BiLSTM cho cả mạng đơn và mạng stacked.
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
        self.hidden_size = hidden_size  # Lưu lại để slice tensor chuẩn xác
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

    def _get_bilstm_output(self, x):
        """Hàm helper lấy đúng đầu ra hội tụ của cả 2 chiều BiLSTM"""
        out, _ = self.bilstm(x)
        h = self.hidden_size
        
        forward_out  = out[:, -1, :h]  # Chiều xuôi lấy ở timestep cuối cùng
        backward_out = out[:,  0, h:]  # Chiều ngược lấy ở timestep đầu tiên
        
        return torch.cat((forward_out, backward_out), dim=-1)

    def forward(self, x):
        out = self._get_bilstm_output(x)
        out = self.dropout(out)
        return self.classifier(out)

    def get_hidden(self, x):
        """Trả về hidden state trước Dense — dùng để vẽ t-SNE."""
        out = self._get_bilstm_output(x)
        out = self.dropout(out)
        return nn.functional.relu(self.classifier[0](out))


class BiLSTM_Model(nn.Module):
    """
    Model dùng riêng cho Exp 3 (LOSO) — kiến trúc 2 lớp LSTM xếp chồng.

    Input : (batch, 32, 128) — 32 kênh, 128 timesteps (sẽ permute nội bộ)
    Output: (batch, num_classes) — raw logits
    """
    def __init__(self, input_size=32, hidden_size=64, num_classes=2, dropout=0.5):
        super(BiLSTM_Model, self).__init__()
        # Lớp lstm2 có hidden_size gốc là hidden_size // 2
        self.hidden_size2 = hidden_size // 2 
        
        self.lstm1    = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        
        # Đầu vào của lstm2 là hidden_size * 2 (do lstm1 là bidirectional)
        self.lstm2    = nn.LSTM(hidden_size * 2, self.hidden_size2, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)
        
        # Kích thước đầu ra của lstm2 sau khi cat 2 chiều sẽ là (hidden_size // 2) * 2 = hidden_size
        self.fc       = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x        = x.permute(0, 2, 1)   # Đưa về dạng (batch, 128 timesteps, 32 channels)
        
        # Lớp 1: Trả về chuỗi kết quả (N, 128, hidden_size * 2) để làm đầu vào cho lớp 2
        out, _   = self.lstm1(x)
        out      = self.dropout1(out)
        
        # Lớp 2: Trả về chuỗi kết quả (N, 128, hidden_size)
        out, _   = self.lstm2(out)
        out      = self.dropout2(out)
        
        # FIX INDEXING CHO LỚP 2:
        h2 = self.hidden_size2
        forward_out  = out[:, -1, :h2]  # Chiều xuôi lớp 2 tại timestep cuối
        backward_out = out[:,  0, h2:]  # Chiều ngược lớp 2 tại timestep đầu
        
        # Gộp lại thành vector có độ dài h2 * 2 = hidden_size
        out = torch.cat((forward_out, backward_out), dim=-1)
        
        return self.fc(out)