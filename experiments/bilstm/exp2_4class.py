"""
experiments/exp2_4class.py
Exp 2: Subject-Dependent, K-Fold (k=5), 4-class (HVHA / HVLA / LVHA / LVLA)

Chạy từ thư mục root:
    python -m experiments.exp2_4class

Output:
    result/logs/bilstm_exp2_4class.json
    result/plots/cm_bilstm_exp2_4class_fold*.png
    result/plots/lc_bilstm_exp2_4class.png
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

# Đảm bảo nhận diện đúng thư mục root của project (lùi 3 bước: bilstm -> experiments -> root)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
for d in [LOG_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── HYPERPARAMS ──
N_SPLITS            = 5
EPOCHS              = 50
BATCH_SIZE          = 64
LR                  = 5e-4
SEED                = 42
N_CLASSES           = 4
AUGMENT_NOISE_STD   = 0.01
EARLY_STOP_PATIENCE = 15
LABEL_SMOOTHING     = 0.1


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL 1 EPOCH
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        if AUGMENT_NOISE_STD > 0:
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

def train_fold(X_train, X_test, y_train, y_test, fold_idx):
    train_loader, test_loader = get_dataloaders(X_train, X_test, y_train, y_test, batch_size=BATCH_SIZE)

    weights   = get_dynamic_class_weights(y_train, n_classes=4).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    model     = EEG_BiLSTM(n_classes=N_CLASSES).to(DEVICE)
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
        tr_loss, tr_acc                = train_one_epoch(model, train_loader, optimizer, criterion)
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

    cm_path = os.path.join(PLOT_DIR, f"cm_bilstm_exp2_4class_fold{fold_idx}.png")
    plot_confusion_matrix(best_labels, best_preds,
                          classes=['LVLA', 'LVHA', 'HVLA', 'HVHA'],
                          title=f"BiLSTM Exp2 (4-class) — Fold {fold_idx}",
                          save_path=cm_path)

    return best_metrics, train_losses, val_losses, train_accs, val_accs


if __name__ == "__main__":
    set_seed(SEED)
    print(f"🖥️  Device: {DEVICE}")

    X         = np.load(os.path.join(DATA_DIR, "X_epochs.npy"))
    y_valence = np.load(os.path.join(DATA_DIR, "y_valence.npy")).astype(np.int64)
    y_arousal = np.load(os.path.join(DATA_DIR, "y_arousal.npy")).astype(np.int64)

    # Gộp Valence & Arousal → 4 class
    y_4class = y_valence * 2 + y_arousal
    labels_name = ['LVLA(0)', 'LVHA(1)', 'HVLA(2)', 'HVHA(3)']
    counts      = np.bincount(y_4class, minlength=4)
    print("\n📊 Phân bố dữ liệu 4-class:")
    for lbl, cnt in zip(labels_name, counts):
        print(f"   {lbl}: {cnt:>6} mẫu ({cnt/len(y_4class)*100:.1f}%)")

    print(f"\n{'='*60}\n  Exp 2 — BiLSTM — 4-CLASS — {N_SPLITS}-Fold CV\n{'='*60}")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_metrics = []
    all_tr_l, all_va_l, all_tr_a, all_va_a = [], [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_4class), start=1):
        print(f"\n  ── Fold {fold_idx}/{N_SPLITS} ──")
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test         = y_4class[train_idx], y_4class[test_idx]

        X_train_norm, X_test_norm, _ = normalize_after_split(X_train_raw, X_test_raw, mode='channel')
        X_train = X_train_norm.transpose(0, 2, 1).copy()
        X_test  = X_test_norm.transpose(0, 2, 1).copy()

        metrics, tr_l, va_l, tr_a, va_a = train_fold(X_train, X_test, y_train, y_test, fold_idx)
        fold_metrics.append(metrics)
        all_tr_l.append(tr_l); all_va_l.append(va_l)
        all_tr_a.append(tr_a); all_va_a.append(va_a)
        print(f"  Fold {fold_idx}: Acc={metrics['accuracy']:.4f} | F1={metrics['f1_score']:.4f}")

    avg_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    std_metrics = {f"{k}_std": float(np.std([m[k] for m in fold_metrics])) for k in fold_metrics[0]}

    print(f"\n  📊 KẾT QUẢ TRUNG BÌNH EXP 2 ({N_SPLITS}-Fold):")
    print(f"     Accuracy : {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy_std']:.4f}")
    print(f"     F1-macro : {avg_metrics['f1_score']:.4f} ± {std_metrics['f1_score_std']:.4f}")

    best_fold_idx = int(np.argmax([m['f1_score'] for m in fold_metrics]))
    plot_learning_curves(all_tr_l[best_fold_idx], all_va_l[best_fold_idx],
                         all_tr_a[best_fold_idx], all_va_a[best_fold_idx],
                         title=f"BiLSTM Exp2 (4-class) — Best Fold {best_fold_idx+1}",
                         save_path=os.path.join(PLOT_DIR, "lc_bilstm_exp2_4class.png"))

    result = {
        "experiment"        : "Exp2_KFold_4class",
        "model"             : "BiLSTM",
        "tuning"            : {
            "label_smoothing"         : LABEL_SMOOTHING,
            "gaussian_noise_std"      : AUGMENT_NOISE_STD,
            "early_stopping_patience" : EARLY_STOP_PATIENCE,
            "learning_rate"           : LR,
        },
        "class_distribution": {lbl: int(cnt) for lbl, cnt in zip(labels_name, counts)},
        "avg_metrics"       : avg_metrics,
        "std_metrics"       : std_metrics,
        "fold_metrics"      : fold_metrics,
    }
    json_path = os.path.join(LOG_DIR, "bilstm_exp2_4class.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ HOÀN TẤT EXPERIMENT 2 — Kết quả đã lưu tại result/")
