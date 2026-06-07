"""
experiments/exp1_2class.py
Exp 1: Subject-Dependent, K-Fold (k=5), 2-class (Valence & Arousal)

Chạy từ thư mục root:
    python -m experiments.exp1_2class

Output:
    result/logs/bilstm_exp1_valence.json
    result/logs/bilstm_exp1_arousal.json
    result/plots/cm_bilstm_exp1_*.png
    result/plots/lc_bilstm_exp1_*.png
    result/checkpoints/exp1_best_model_*.pth
    result/checkpoints/exp1_best_train_idx_*.npy
    result/checkpoints/exp1_best_test_idx_*.npy
"""

import os
import sys
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold

# ── Đưa root project vào sys.path để import src ──
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.models.eeg_bilstm import EEG_BiLSTM
from src.utils.dataset import set_seed, get_dataloaders
from src.utils.metrics import evaluate_metrics, plot_confusion_matrix, plot_learning_curves
from src.data_pipeline.preprocess import normalize_after_split, get_dynamic_class_weights

# ── ĐƯỜNG DẪN ──
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_NAME = "bilstm" 
LOG_DIR    = os.path.join(ROOT_DIR, "result", MODEL_NAME, "logs")
PLOT_DIR   = os.path.join(ROOT_DIR, "result", MODEL_NAME, "plots")
CKPT_DIR   = os.path.join(ROOT_DIR, "result", MODEL_NAME, "checkpoints")

for d in [LOG_DIR, PLOT_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── HYPERPARAMS ──
N_SPLITS            = 5
EPOCHS              = 50
BATCH_SIZE          = 64
LR                  = 5e-4
SEED                = 42
AUGMENT_NOISE_STD   = 0.01
EARLY_STOP_PATIENCE = 15
LABEL_SMOOTHING     = 0.1


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL 1 EPOCH
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, augment=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        if augment and AUGMENT_NOISE_STD > 0:
            X_batch = X_batch + torch.randn_like(X_batch) * AUGMENT_NOISE_STD
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss += loss.item()
        preds       = logits.argmax(1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    return total_loss / len(loader), correct / total, np.array(all_preds), np.array(all_labels)


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN 1 FOLD
# ══════════════════════════════════════════════════════════════════════════════

def train_fold(X_train, X_test, y_train, y_test, fold_idx, label_name):
    train_loader, test_loader = get_dataloaders(X_train, X_test, y_train, y_test, batch_size=BATCH_SIZE)

    weights   = get_dynamic_class_weights(y_train, n_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    model     = EEG_BiLSTM(n_classes=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss    = float('inf')
    best_metrics     = None
    best_preds       = None
    best_labels      = None
    best_model_wts   = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc          = train_one_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_acc, preds, labels = eval_one_epoch(model, test_loader, criterion)
        scheduler.step(va_loss)

        train_losses.append(tr_loss);  val_losses.append(va_loss)
        train_accs.append(tr_acc);     val_accs.append(va_acc)
        metrics = evaluate_metrics(labels, preds)

        if va_loss < best_val_loss:
            best_val_loss    = va_loss
            best_metrics     = metrics
            best_preds, best_labels = preds.copy(), labels.copy()
            best_model_wts   = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"      Epoch {epoch:02d} | loss: {tr_loss:.4f}/{va_loss:.4f} | "
                  f"acc: {tr_acc*100:.1f}%/{va_acc*100:.1f}% | f1: {metrics['f1_score']:.4f}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"      🛑 Early Stopping tại Epoch {epoch}!")
            break

    cm_path = os.path.join(PLOT_DIR, f"cm_bilstm_exp1_{label_name}_fold{fold_idx}.png")
    plot_confusion_matrix(best_labels, best_preds,
                          classes=['Low', 'High'],
                          title=f"BiLSTM Exp1 {label_name} — Fold {fold_idx}",
                          save_path=cm_path)

    return best_metrics, train_losses, val_losses, train_accs, val_accs, best_model_wts


# ══════════════════════════════════════════════════════════════════════════════
# RUN EXP 1
# ══════════════════════════════════════════════════════════════════════════════

def run_exp1(X, y, label_name):
    print(f"\n{'='*60}\n  Exp 1 — BiLSTM — {label_name.upper()} — {N_SPLITS}-Fold CV\n{'='*60}")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_metrics = []
    all_tr_l, all_va_l, all_tr_a, all_va_a = [], [], [], []
    global_best_f1 = 0.0

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n  ── Fold {fold_idx}/{N_SPLITS} ──")
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test         = y[train_idx], y[test_idx]

        X_train_norm, X_test_norm, _ = normalize_after_split(X_train_raw, X_test_raw, mode='channel')
        X_train = X_train_norm.transpose(0, 2, 1).copy()
        X_test  = X_test_norm.transpose(0, 2, 1).copy()

        metrics, tr_l, va_l, tr_a, va_a, best_wts = train_fold(
            X_train, X_test, y_train, y_test, fold_idx, label_name
        )
        fold_metrics.append(metrics)
        all_tr_l.append(tr_l); all_va_l.append(va_l)
        all_tr_a.append(tr_a); all_va_a.append(va_a)

        print(f"  Fold {fold_idx}: Acc={metrics['accuracy']:.4f} | F1={metrics['f1_score']:.4f}")

        if metrics['f1_score'] > global_best_f1:
            global_best_f1 = metrics['f1_score']
            torch.save(best_wts, os.path.join(CKPT_DIR, f"exp1_best_model_{label_name}.pth"))
            np.save(os.path.join(CKPT_DIR, f"exp1_best_train_idx_{label_name}.npy"), train_idx)
            np.save(os.path.join(CKPT_DIR, f"exp1_best_test_idx_{label_name}.npy"),  test_idx)

    avg_metrics  = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    std_metrics  = {f"{k}_std": float(np.std([m[k] for m in fold_metrics])) for k in fold_metrics[0]}

    best_fold_idx = int(np.argmax([m['f1_score'] for m in fold_metrics]))
    lc_path = os.path.join(PLOT_DIR, f"lc_bilstm_exp1_{label_name}.png")
    plot_learning_curves(all_tr_l[best_fold_idx], all_va_l[best_fold_idx],
                         all_tr_a[best_fold_idx], all_va_a[best_fold_idx],
                         title=f"BiLSTM Exp1 {label_name} — Best Fold {best_fold_idx+1}",
                         save_path=lc_path)

    result_data = {
        "experiment" : "Exp1_KFold_2class",
        "model"      : "BiLSTM",
        "label"      : label_name,
        "n_splits"   : N_SPLITS,
        "epochs"     : EPOCHS,
        "tuning"     : {
            "label_smoothing"         : LABEL_SMOOTHING,
            "gaussian_noise_std"      : AUGMENT_NOISE_STD,
            "early_stopping_patience" : EARLY_STOP_PATIENCE,
        },
        "avg_metrics" : avg_metrics,
        "std_metrics" : std_metrics,
        "fold_metrics": fold_metrics,
    }
    json_path = os.path.join(LOG_DIR, f"bilstm_exp1_{label_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"\n  📊 KẾT QUẢ TRUNG BÌNH {label_name.upper()} ({N_SPLITS}-Fold):")
    print(f"     Accuracy : {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy_std']:.4f}")
    print(f"     F1-macro : {avg_metrics['f1_score']:.4f} ± {std_metrics['f1_score_std']:.4f}")
    return result_data


if __name__ == "__main__":
    set_seed(SEED)
    X         = np.load(os.path.join(DATA_DIR, "X_epochs.npy"))
    y_valence = np.load(os.path.join(DATA_DIR, "y_valence.npy")).astype(np.int64)
    y_arousal = np.load(os.path.join(DATA_DIR, "y_arousal.npy")).astype(np.int64)

    results = {}
    for label_name, y in [("valence", y_valence), ("arousal", y_arousal)]:
        results[label_name] = run_exp1(X, y, label_name)

    print("\n✅ HOÀN TẤT EXPERIMENT 1 — Model tốt nhất đã lưu tại result/checkpoints/")
