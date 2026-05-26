"""
EmoWave — Người 2: CNN 1D Model
=================================
CNN học đặc trưng trực tiếp từ EEG,
không cần trích xuất features thủ công.

Input shape: (batch, 32 channels, 128 timesteps)

TODO (Người 2):
  [ ] Thiết kế CNN architecture
  [ ] Train + tune hyperparameters
  [ ] So sánh kết quả 2 lớp vs 4 lớp
  [ ] Lưu kết quả vào results/cnn_results.json
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.deap_loader import load_all_subjects, prepare_for_cnn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EEG_CNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),               # 128 → 64

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),               # 64 → 32

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),       # → (batch, 256, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.conv_block(x))


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
        print(f"\n{'='*50}")
        print(f"CNN — {label_type}")
        print("="*50)

        X, y, _ = load_all_subjects(label_type=label_type)
        X_tr, X_te, y_tr, y_te = prepare_for_cnn(X, y)

        train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        test_ds  = TensorDataset(torch.tensor(X_te), torch.tensor(y_te))
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

        model     = EEG_CNN(n_classes=n_cls).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, EPOCHS + 1):
            loss, tr_acc = train_model(model, train_dl, optimizer, criterion)
            te_acc       = evaluate_model(model, test_dl)
            if epoch % 5 == 0:
                print(f"  Epoch {epoch:02d} | loss: {loss:.4f} | train: {tr_acc*100:.1f}% | test: {te_acc*100:.1f}%")

        print(f"  Final test accuracy: {te_acc*100:.2f}%")
