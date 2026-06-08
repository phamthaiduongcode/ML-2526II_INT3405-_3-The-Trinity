# File: models/train_engine.py
"""
Shared Training Engine cho 3 Experiments (DEAP EEG Emotion Recognition).

  - Exp1: Valence binary      (num_classes=2, label="valence")
  - Exp2: 4-class V×A          (num_classes=4, label="4class")
  - Exp3: Arousal binary       (num_classes=2, label="arousal")

Hàm chính: run_experiment(config: dict)
Hàm phụ : train_one_epoch(), evaluate()
"""

import os
import sys
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

# ── Thêm project root vào sys.path để import nội bộ ─────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.cnn import EEGNet2D
from src.data_pipeline.preprocess import normalize_after_split, get_dynamic_class_weights
from src.utils.dataset import set_seed, EEGDataset, get_dataloaders
from src.utils.metrics import evaluate_metrics, plot_confusion_matrix, plot_learning_curves


# ==============================================================================
# PATHS
# ==============================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# ==============================================================================
# CLASS NAME MAPS — dùng cho confusion matrix và logging
# ==============================================================================
CLASS_NAMES = {
    "valence": ["Low Valence", "High Valence"],
    "arousal": ["Low Arousal", "High Arousal"],
    # y = valence * 2 + arousal
    # 0: LowV-LowA (V=0,A=0), 1: LowV-HighA (V=0,A=1),
    # 2: HighV-LowA (V=1,A=0), 3: HighV-HighA (V=1,A=1)
    "4class": ["LowV-LowA", "LowV-HighA", "HighV-LowA", "HighV-HighA"],
}


# ==============================================================================
# ADAPTIVE LR SCHEDULER — Tự động điều chỉnh LR dựa trên Val F1
# ==============================================================================
class AdaptiveLRScheduler:
    """
    Adaptive Learning Rate Scheduler dựa trên xu hướng Val F1-macro.

    Rules:
        1. Val F1 không tăng >= patience epoch → LR × decay_factor (0.5)
        2. Val F1 tụt liên tục >= drop_patience epoch → LR × boost_factor (1.1)
           (thử thoát local minimum)
        3. Val F1 tăng đều → giữ nguyên LR

    Args:
        optimizer: Torch optimizer
        patience (int): Số epoch chờ trước khi giảm LR (rule 1). Default: 5
        decay_factor (float): Hệ số giảm LR khi plateau. Default: 0.5
        drop_patience (int): Số epoch F1 tụt liên tục trước khi boost LR. Default: 3
        boost_factor (float): Hệ số tăng LR khi tụt liên tục. Default: 1.1
        min_lr (float): LR tối thiểu, không giảm dưới mức này. Default: 1e-6
        max_lr (float): LR tối đa, không tăng vượt mức này. Default: 1e-2
        min_delta (float): Ngưỡng cải thiện tối thiểu để coi là "tăng". Default: 1e-4
    """

    def __init__(self, optimizer, patience=5, decay_factor=0.5,
                 drop_patience=3, boost_factor=1.1,
                 min_lr=1e-6, max_lr=1e-2, min_delta=1e-4):
        self.optimizer = optimizer
        self.patience = patience
        self.decay_factor = decay_factor
        self.drop_patience = drop_patience
        self.boost_factor = boost_factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_delta = min_delta

        # Internal state
        self.best_f1 = -1.0
        self.epochs_no_improve = 0   # Đếm epoch không cải thiện (rule 1)
        self.consecutive_drops = 0   # Đếm epoch F1 tụt liên tục (rule 2)
        self.prev_f1 = -1.0
        self.history = []            # Lịch sử LR để debug

    def step(self, val_f1):
        """
        Gọi sau mỗi epoch với val_f1 hiện tại.
        Returns:
            str: Mô tả action đã thực hiện ('decay', 'boost', 'hold')
        """
        current_lr = self.optimizer.param_groups[0]['lr']
        action = 'hold'

        # --- Kiểm tra xu hướng ---
        f1_improved = val_f1 >= self.best_f1 + self.min_delta
        f1_dropped = val_f1 < self.prev_f1 - self.min_delta

        if f1_improved:
            # Rule 3: F1 tăng → giữ nguyên, reset counters
            self.best_f1 = val_f1
            self.epochs_no_improve = 0
            self.consecutive_drops = 0
            action = 'hold'
        else:
            # F1 không cải thiện so với best
            self.epochs_no_improve += 1

            if f1_dropped:
                # F1 tụt so với epoch trước
                self.consecutive_drops += 1
            else:
                # F1 không tụt (chỉ đứng yên) → reset drop counter
                self.consecutive_drops = 0

            # Rule 2: F1 tụt liên tục → boost LR để thoát local minimum
            #         (ưu tiên rule 2 trước rule 1)
            if self.consecutive_drops >= self.drop_patience:
                new_lr = min(current_lr * self.boost_factor, self.max_lr)
                if new_lr != current_lr:
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = new_lr
                    action = 'boost'
                self.consecutive_drops = 0  # Reset sau khi boost

            # Rule 1: F1 không tăng đủ lâu → giảm LR
            elif self.epochs_no_improve >= self.patience:
                new_lr = max(current_lr * self.decay_factor, self.min_lr)
                if new_lr != current_lr:
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = new_lr
                    action = 'decay'
                self.epochs_no_improve = 0  # Reset sau khi giảm

        self.prev_f1 = val_f1
        new_lr = self.optimizer.param_groups[0]['lr']
        self.history.append({
            'val_f1': val_f1, 'lr': new_lr, 'action': action
        })

        return action

    def get_last_lr(self):
        """Trả về LR hiện tại (tương thích API)."""
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ==============================================================================
# HELPER: train_one_epoch
# ==============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm=1.0):
    """
    Train model qua 1 epoch.

    Args:
        max_grad_norm (float): Ngưỡng clip gradient để tránh explosion (EEG hay có outliers)

    Returns:
        avg_loss (float): loss trung bình trên toàn epoch
        all_preds (list): dự đoán (argmax) của toàn bộ batch
        all_labels (list): nhãn thực tế tương ứng
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # unsqueeze(1): (N, 32, 128) → (N, 1, 32, 128) cho EEGNet2D
        X_batch = X_batch.unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        # Gradient clipping — ngăn gradient explosion từ EEG outliers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        running_loss += loss.item() * y_batch.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, all_preds, all_labels


# ==============================================================================
# HELPER: evaluate
# ==============================================================================
def evaluate(model, loader, criterion, device):
    """
    Đánh giá model trên tập validation / test.

    Returns:
        avg_loss (float): loss trung bình
        metrics (dict): {'accuracy', 'precision', 'recall', 'f1_score'}
        all_preds (list): dự đoán
        all_labels (list): nhãn thực tế
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # unsqueeze(1): (N, 32, 128) → (N, 1, 32, 128)
            X_batch = X_batch.unsqueeze(1)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            running_loss += loss.item() * y_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    metrics = evaluate_metrics(all_labels, all_preds)
    return avg_loss, metrics, all_preds, all_labels


# ==============================================================================
# MAIN: run_experiment
# ==============================================================================
def run_experiment(config: dict):
    """
    Chạy toàn bộ pipeline cho 1 experiment.

    Args:
        config (dict): Cấu hình experiment gồm các key:
            - exp_name    (str): Tên experiment, dùng để lưu kết quả
            - num_classes  (int): 2 hoặc 4
            - label        (str): "valence", "arousal", hoặc "4class"
            - cv           (str): "stratified_kfold" hoặc "leave_one_group_out"
            - n_splits     (int): Số folds (chỉ dùng cho stratified_kfold)
            - batch_size   (int): Batch size (default: 64)
            - num_epochs   (int): Số epoch tối đa (default: 50)
            - lr          (float): Learning rate (default: 1e-3)
            - weight_decay(float): L2 regularization (default: 1e-4)
            - patience_lr  (int): Số epoch chờ trước khi giảm LR (default: 5)
            - patience_es  (int): Early stopping patience (default: 10)
            - max_grad_norm(float): Ngưỡng clip gradient (default: 1.0)
    """
    # ── 0. Parse config ──────────────────────────────────────────────────────
    exp_name    = config["exp_name"]
    num_classes = config["num_classes"]
    label_type  = config["label"]
    cv_method   = config["cv"]
    n_splits    = config.get("n_splits", 5)

    # ── Hyper-parameters (config-driven, có default) ─────────────────────────
    BATCH_SIZE     = config.get("batch_size", 256)
    NUM_EPOCHS     = config.get("num_epochs", 50)
    LR             = config.get("lr", 3e-4)
    WEIGHT_DECAY   = config.get("weight_decay", 1e-4)
    PATIENCE_LR    = config.get("patience_lr", 5)
    PATIENCE_ES    = config.get("patience_es", 15)
    MAX_GRAD_NORM  = config.get("max_grad_norm", 1.0)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"🚀 EXPERIMENT: {exp_name}")
    print(f"   Classes   : {num_classes}  |  Label: {label_type}")
    print(f"   CV Method : {cv_method}  |  Folds: {n_splits}")
    print(f"   Device    : {device}")
    print("=" * 80)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    X = np.load(os.path.join(DATA_DIR, "X_epochs.npy"))
    y_valence = np.load(os.path.join(DATA_DIR, "y_valence.npy"))
    y_arousal = np.load(os.path.join(DATA_DIR, "y_arousal.npy"))
    subject_groups = np.load(os.path.join(DATA_DIR, "subject_groups.npy"))

    print(f"📦 Loaded data: X={X.shape}, y_val={y_valence.shape}, y_aro={y_arousal.shape}")

    # ── 2. Tạo target label ──────────────────────────────────────────────────
    if label_type == "valence":
        y = y_valence
    elif label_type == "arousal":
        y = y_arousal
    elif label_type == "4class":
        # y = valence * 2 + arousal → 4 nhóm: {0, 1, 2, 3}
        y = y_valence * 2 + y_arousal
    else:
        raise ValueError(f"❌ label_type không hợp lệ: {label_type}")

    print(f"🏷️  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ── 3. Cross-validation splitter ─────────────────────────────────────────
    if cv_method == "stratified_kfold":
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=42
        )
        splits = list(splitter.split(X, y))
    elif cv_method in ("leave_one_group_out", "loso"):
        splitter = LeaveOneGroupOut()
        splits = list(splitter.split(X, y, groups=subject_groups))
        n_splits = len(splits)
        print(f"   LOSO: {n_splits} folds (1 subject/fold)")

    else:
        raise ValueError(f"❌ cv_method không hợp lệ: {cv_method}")

    # ── 4. Tạo thư mục output ────────────────────────────────────────────────
    exp_dir = os.path.join(RESULTS_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # ── 5. Vòng lặp K-Fold ──────────────────────────────────────────────────
    fold_results = []           # list of metrics dict per fold
    best_fold_f1 = -1.0         # track fold có F1 cao nhất
    best_fold_idx = 0
    best_fold_curves = None     # learning curves của best fold

    class_names = CLASS_NAMES.get(label_type, [str(i) for i in range(num_classes)])

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # Seed mỗi fold để đảm bảo reproducibility xuyên suốt LOSO/KFold
        set_seed(42 + fold_idx)

        fold_start = time.time()
        print(f"\n{'─' * 70}")
        if cv_method in ("leave_one_group_out", "loso"):
            subject_id = subject_groups[val_idx[0]] + 1
            print(f"📂 FOLD {fold_idx+1}/{n_splits} — Test Subject: S{subject_id:02d}")
        else:
            print(f"📂 FOLD {fold_idx + 1}/{n_splits}")
        print(f"{'─' * 70}")
        print(f"   Train: {len(train_idx)} samples  |  Val: {len(val_idx)} samples")

        # ── 5a. Split & Normalize ────────────────────────────────────────────
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_scaled, X_val_scaled, _ = normalize_after_split(
            X_train_raw, X_val_raw, mode='channel'
        )

        # ── 5b. Class weights ────────────────────────────────────────────────
        class_weights = get_dynamic_class_weights(
            y_train, n_classes=num_classes
        ).to(device)
        print(f"   Class weights: {class_weights.cpu().numpy()}")

        # ── 5c. DataLoaders ──────────────────────────────────────────────────
        train_loader, val_loader = get_dataloaders(
            X_train_scaled, X_val_scaled, y_train, y_val,
            batch_size=BATCH_SIZE
        )

        # ── 5d. Model, Optimizer, Scheduler, Criterion ──────────────────────
        model = EEGNet2D(num_classes=num_classes).to(device)
        optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = AdaptiveLRScheduler(
            optimizer,
            patience=PATIENCE_LR,       # N epoch không tăng → LR × 0.5
            decay_factor=0.5,
            drop_patience=3,            # F1 tụt 3 epoch liên tục → LR × 1.1
            boost_factor=1.1,
            min_lr=1e-6,
            max_lr=1e-2,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # ── 5e. Training loop ────────────────────────────────────────────────
        best_val_f1 = -1.0
        best_model_state = None
        epochs_no_improve = 0       # Counter cho early stopping

        # Tracking learning curves
        hist_train_loss = []
        hist_val_loss = []
        hist_train_acc = []
        hist_val_acc = []
        hist_train_f1 = []
        hist_val_f1   = []

        for epoch in range(NUM_EPOCHS):
            # --- Train (có gradient clipping) ---
            train_loss, train_preds, train_labels = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                max_grad_norm=MAX_GRAD_NORM
            )
            train_metrics = evaluate_metrics(train_labels, train_preds)

            # --- Validate ---
            val_loss, val_metrics, _, _ = evaluate(
                model, val_loader, criterion, device
            )

            # --- Adaptive LR step (theo val F1-macro) ---
            lr_action = scheduler.step(val_metrics['f1_score'])

            # --- Save learning curves ---
            hist_train_loss.append(train_loss)
            hist_val_loss.append(val_loss)
            hist_train_acc.append(train_metrics['accuracy'])
            hist_val_acc.append(val_metrics['accuracy'])
            hist_train_f1.append(train_metrics['f1_score'])
            hist_val_f1.append(val_metrics['f1_score'])

            # --- Best model checkpoint (theo val F1-macro) ---
            min_delta = 1e-4 
            if val_metrics['f1_score'] >= best_val_f1 + min_delta:
                best_val_f1 = val_metrics['f1_score']
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # --- Log LR action khi thay đổi ---
            current_lr = optimizer.param_groups[0]['lr']
            if lr_action != 'hold':
                action_emoji = '📉' if lr_action == 'decay' else '📈'
                print(
                    f"   {action_emoji} Epoch {epoch + 1}: LR {lr_action} → {current_lr:.1e}"
                )

            # --- Log mỗi 10 epoch hoặc epoch cuối ---
            if (epoch + 1) % 10 == 0 or epoch == NUM_EPOCHS - 1:
                print(
                    f"   Epoch {epoch + 1:3d}/{NUM_EPOCHS}  |  "
                    f"Train Loss: {train_loss:.4f}  Acc: {train_metrics['accuracy']:.4f}  |  "
                    f"Val Loss: {val_loss:.4f}  F1: {val_metrics['f1_score']:.4f}  "
                    f"Acc: {val_metrics['accuracy']:.4f}  |  "
                    f"LR: {current_lr:.1e}"
                )

            # --- Early stopping ---
            if epochs_no_improve >= PATIENCE_ES:
                print(f"   ⏹ Early stopping tại epoch {epoch + 1} "
                      f"(không cải thiện {PATIENCE_ES} epoch liên tiếp)")
                break

        # ── 4f. Load best model → evaluate lại trên val ─────────────────────
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        else:
            print("   ⚠️ Không tìm được best model, giữ nguyên model cuối cùng")

        _, final_metrics, final_preds, final_labels = evaluate(
            model, val_loader, criterion, device
        )

        fold_time = time.time() - fold_start
        print(f"\n   ✅ Fold {fold_idx + 1} hoàn tất ({fold_time:.1f}s)")
        print(f"   📊 Best Val F1: {final_metrics['f1_score']:.4f}  "
              f"Acc: {final_metrics['accuracy']:.4f}  "
              f"Prec: {final_metrics['precision']:.4f}  "
              f"Rec: {final_metrics['recall']:.4f}")

        fold_results.append(final_metrics)

        # ── 4g-extra. Lưu best model weights ra disk ─────────────────────────
        if best_model_state is not None:
            model_path = os.path.join(exp_dir, f"best_model_fold{fold_idx + 1}.pt")
            torch.save(best_model_state, model_path)
            print(f"   💾 Model saved: {model_path}")

        # ── 4g. Confusion matrix cho fold này ────────────────────────────────
        plot_confusion_matrix(
            final_labels, final_preds,
            classes=class_names,
            title=f"{exp_name} — Fold {fold_idx + 1} Confusion Matrix",
            save_path=os.path.join(exp_dir, f"cm_fold{fold_idx + 1}.png")
        )

        # ── 4h. Track best fold cho learning curves ──────────────────────────
        if final_metrics['f1_score'] > best_fold_f1:
            best_fold_f1 = final_metrics['f1_score']
            best_fold_idx = fold_idx
            best_fold_curves = {
                'train_losses': hist_train_loss,
                'val_losses':   hist_val_loss,
                'train_accs':   hist_train_acc,
                'val_accs':     hist_val_acc,
                'train_f1s':    hist_train_f1,   # dang thừa 
                'val_f1s':      hist_val_f1,   # dang thừa 
            }

    # ══════════════════════════════════════════════════════════════════════════
    # 5. Tổng kết cross-validation
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print(f"📊 TỔNG KẾT: {exp_name}")
    print(f"{'=' * 80}")

    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score']
    summary_lines = []
    summary_lines.append(f"Experiment : {exp_name}")
    summary_lines.append(f"Classes    : {num_classes}  |  Label: {label_type}")
    summary_lines.append(f"CV Method  : {cv_method}  |  Folds: {n_splits}")
    summary_lines.append(f"Epochs     : {NUM_EPOCHS}  |  Batch: {BATCH_SIZE}")
    summary_lines.append(f"LR         : {LR}  |  Weight Decay: {WEIGHT_DECAY}")
    summary_lines.append(f"Device     : {device}")
    summary_lines.append("")
    summary_lines.append(f"{'Metric':<15} {'Mean':>10} {'Std':>10}")
    summary_lines.append("─" * 38)

    for key in metric_keys:
        values = [r[key] for r in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        line = f"{key:<15} {mean_val:>10.4f} {std_val:>10.4f}"
        summary_lines.append(line)
        print(f"   {key:<15}: {mean_val:.4f} ± {std_val:.4f}")

    summary_lines.append("")
    summary_lines.append(f"Best Fold  : {best_fold_idx + 1}  (F1 = {best_fold_f1:.4f})")

    # ── Lưu summary text ─────────────────────────────────────────────────────
    summary_path = os.path.join(RESULTS_DIR, f"{exp_name}_summary.txt")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))
    print(f"\n💾 Summary đã lưu tại: {summary_path}")

    # ── 6. Learning curves cho fold có F1 cao nhất ───────────────────────────
    if best_fold_curves is not None:
        plot_learning_curves(
            train_losses=best_fold_curves['train_losses'],
            val_losses=best_fold_curves['val_losses'],
            train_accs=best_fold_curves['train_accs'],
            val_accs=best_fold_curves['val_accs'],
            title=f"{exp_name} — Best Fold {best_fold_idx + 1} Learning Curves",
            save_path=os.path.join(exp_dir, f"learning_curves_fold{best_fold_idx + 1}.png")
        )

    print(f"\n🏁 Experiment [{exp_name}] hoàn tất!\n")
    return fold_results

