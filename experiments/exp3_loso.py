"""
experiments/exp3_loso.py
Exp 3: Leave-One-Subject-Out (LOSO) Cross-Validation, 2-class

Chạy từ thư mục root:
    python -m experiments.exp3_loso

Output:
    result/logs/loso_checkpoint.json
    result/plots/cm_valence_S*.png
    result/plots/cm_arousal_S*.png
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.utils.dataset import set_seed
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.models.eeg_bilstm import BiLSTM_Model

# ── ĐƯỜNG DẪN ──
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_NAME = "bilstm" 
LOG_DIR    = os.path.join(ROOT_DIR, "result", MODEL_NAME, "logs")
PLOT_DIR   = os.path.join(ROOT_DIR, "result", MODEL_NAME, "plots")

for d in [LOG_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── HYPERPARAMS ──
BATCH_SIZE = 256
MAX_EPOCHS = 60
PATIENCE   = 15
LR         = 0.001
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


# ══════════════════════════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════════════════════════

def normalize_channel_wise(X_train, X_test):
    """Chuẩn hóa channel-wise, fit trên train, transform cả hai."""
    N_train, C, T = X_train.shape
    N_test        = X_test.shape[0]
    scaler        = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.transpose(0, 2, 1).reshape(-1, C))
    X_test_scaled  = scaler.transform(X_test.transpose(0, 2, 1).reshape(-1, C))
    return (X_train_scaled.reshape(N_train, T, C).transpose(0, 2, 1),
            X_test_scaled.reshape(N_test, T, C).transpose(0, 2, 1))


def plot_cm(y_true, y_pred, subject_idx, target_name):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred),
        display_labels=['Low', 'High']
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f"{target_name} — Subject {subject_idx + 1}")
    plt.savefig(os.path.join(PLOT_DIR, f"cm_{target_name.lower()}_S{subject_idx + 1}.png"))
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    set_seed(42)  # BẮT BUỘC GỌI CÁI NÀY ĐỂ KHOÁ RANDOM
    print(f"🖥️  Device: {DEVICE}")
    X              = np.load(os.path.join(DATA_DIR, 'X_epochs.npy'))
    y_val          = np.load(os.path.join(DATA_DIR, 'y_valence.npy'))
    y_aro          = np.load(os.path.join(DATA_DIR, 'y_arousal.npy'))
    subject_groups = np.load(os.path.join(DATA_DIR, 'subject_groups.npy'))
    print(f"✅ Tải dữ liệu: X={X.shape}")

    loso_results = {"Valence": {}, "Arousal": {}, "Skipped": []}

    for target_name, y_full in [("Valence", y_val), ("Arousal", y_aro)]:
        print(f"\n{'='*60}\n🚀 BẮT ĐẦU LOSO: {target_name}\n{'='*60}")

        for test_subject in range(32):
            start_time = time.time()
            test_idx  = subject_groups == test_subject
            train_idx = subject_groups != test_subject

            X_train, y_train = X[train_idx], y_full[train_idx]
            X_test,  y_test  = X[test_idx],  y_full[test_idx]

            if len(np.unique(y_test)) == 1:
                print(f"⚠️  BỎ QUA Subject {test_subject+1} — test set chỉ có 1 class!")
                loso_results["Skipped"].append(f"{target_name}_S{test_subject+1}")
                continue

            X_train, X_test = normalize_channel_wise(X_train, X_test)

            train_loader = DataLoader(
                TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long)),
                batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
            )
            test_loader = DataLoader(
                TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                              torch.tensor(y_test, dtype=torch.long)),
                batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
            )

            # Class weights
            classes = np.unique(y_train)
            weights = np.clip(compute_class_weight('balanced', classes=classes, y=y_train), 0, 5.0)
            class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            model     = BiLSTM_Model().to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            best_val_loss    = float('inf')
            best_f1          = 0.0
            best_acc         = 0.0
            patience_counter = 0
            best_preds       = []
            best_trues       = []

            for epoch in range(MAX_EPOCHS):
                # Train
                model.train()
                for bx, by in train_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()

                # Eval
                model.eval()
                val_loss, all_preds, all_trues = 0.0, [], []
                with torch.no_grad():
                    for bx, by in test_loader:
                        bx, by = bx.to(DEVICE), by.to(DEVICE)
                        logits = model(bx)
                        val_loss += criterion(logits, by).item()
                        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                        all_trues.extend(by.cpu().numpy())

                val_loss /= len(test_loader)

                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    best_f1          = f1_score(all_trues, all_preds, average='macro')
                    best_acc         = accuracy_score(all_trues, all_preds)
                    best_preds       = all_preds.copy()
                    best_trues       = all_trues.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= PATIENCE:
                    print(f"    🛑 Early stopping tại epoch {epoch+1}")
                    break

            exec_time = round((time.time() - start_time) / 60, 2)
            print(f"✅ S{test_subject+1} ({exec_time} phút) | Acc: {best_acc:.4f} | F1-Macro: {best_f1:.4f}")

            plot_cm(best_trues, best_preds, test_subject, target_name)

            loso_results[target_name][f"Subject_{test_subject+1}"] = {
                "Acc"          : float(best_acc),
                "F1_Macro"     : float(best_f1),
                "Best_Val_Loss": float(best_val_loss),
                "Epochs_Ran"   : epoch + 1,
            }

            # Lưu checkpoint sau mỗi subject
            with open(os.path.join(LOG_DIR, "loso_checkpoint.json"), "w") as f:
                json.dump(loso_results, f, indent=4)

    print(f"\n🎉 HOÀN TẤT EXP 3 (LOSO). Kết quả tại: result/logs/loso_checkpoint.json")
