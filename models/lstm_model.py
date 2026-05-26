"""
EmoWave — Người 3: LSTM / BiLSTM Model
=========================================
LSTM nắm bắt sự thay đổi EEG theo thời gian.

Input shape: (batch, 128 timesteps, 32 features)

TODO (Người 3):
  [ ] Thiết kế LSTM / BiLSTM architecture
  [ ] Train + tune hyperparameters
  [ ] Thử BiLSTM nếu kịp
  [ ] So sánh kết quả 2 lớp vs 4 lớp
  [ ] Lưu kết quả vào results/lstm_results.json
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.deap_loader import load_all_subjects, prepare_for_lstm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EEG_LSTM(nn.Module):
    def __init__(self, input_size=32, hidden_size=128,
                 num_layers=2, n_classes=2, bidirectional=False, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= bidirectional,
            dropout      = dropout if num_layers > 1 else 0,
        )
        lstm_out = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, 128, 32)
        out, _ = self.lstm(x)
        out     = out[:, -1, :]    # lấy timestep cuối
        return self.classifier(out)


def train_model(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        out  = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == y_batch).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    for X_batch, y_batch in loader:
        out      = model(X_batch.to(DEVICE)).cpu()
        correct += (out.argmax(1) == y_batch).sum().item()
    return correct / len(loader.dataset)


if __name__ == "__main__":
    EPOCHS     = 30
    BATCH_SIZE = 64
    LR         = 1e-3

    for label_type, n_cls in [("2class", 2), ("4class", 4)]:
        for bidirectional in [False, True]:
            name = "BiLSTM" if bidirectional else "LSTM"
            print(f"\n{'='*50}")
            print(f"{name} — {label_type}")
            print("="*50)

            X, y, _ = load_all_subjects(label_type=label_type)
            X_tr, X_te, y_tr, y_te = prepare_for_lstm(X, y)

            train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
            test_ds  = TensorDataset(torch.tensor(X_te), torch.tensor(y_te))
            train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

            model     = EEG_LSTM(n_classes=n_cls, bidirectional=bidirectional).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(1, EPOCHS + 1):
                loss, tr_acc = train_model(model, train_dl, optimizer, criterion)
                te_acc       = evaluate_model(model, test_dl)
                if epoch % 5 == 0:
                    print(f"  Epoch {epoch:02d} | loss: {loss:.4f} | train: {tr_acc*100:.1f}% | test: {te_acc*100:.1f}%")

            print(f"  Final test accuracy: {te_acc*100:.2f}%")
