import os
import sys
import copy
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import LeaveOneGroupOut, train_test_split

# ── Thêm project root vào sys.path ──────────────────────────────────────────
current_file = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.eegnet import EEGnet
from src.data_pipeline.preprocess import normalize_after_split, get_dynamic_class_weights
from src.utils.dataset import set_seed, get_dataloaders
from src.utils.metrics import evaluate_metrics, plot_confusion_matrix, plot_learning_curves


# ==============================================================================
# CONFIGURATION
# ==============================================================================
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR    = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "fine_tuning_experiment")
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = {
    "valence" : ["Low Valence",  "High Valence"],
    "arousal" : ["Low Arousal",  "High Arousal"],
    "4class"  : ["LowV-LowA",   "LowV-HighA", "HighV-LowA", "HighV-HighA"],
}


# ==============================================================================
# ADAPTIVE LR SCHEDULER — đầy đủ 3 rules, đồng bộ với train_engine.py
# ==============================================================================
class AdaptiveLRScheduler:
    """
    Adaptive Learning Rate Scheduler dựa trên xu hướng Val F1-macro.

    Rules:
        1. Val F1 không tăng >= patience epoch  → LR × decay_factor (0.5)
        2. Val F1 tụt liên tục >= drop_patience  → LR × boost_factor (1.1)
           (thử thoát local minimum)
        3. Val F1 tăng đều                       → giữ nguyên LR

    Args:
        optimizer     : Torch optimizer
        patience      : Số epoch chờ trước khi giảm LR (rule 1). Default: 5
        decay_factor  : Hệ số giảm LR khi plateau. Default: 0.5
        drop_patience : Số epoch F1 tụt liên tục trước khi boost. Default: 3
        boost_factor  : Hệ số tăng LR khi tụt liên tục. Default: 1.1
        min_lr        : LR tối thiểu. Default: 1e-6
        max_lr        : LR tối đa. Default: 1e-2
        min_delta     : Ngưỡng cải thiện tối thiểu. Default: 1e-4
    """

    def __init__(self, optimizer, patience=5, decay_factor=0.5,
                 drop_patience=3, boost_factor=1.1,
                 min_lr=1e-6, max_lr=1e-2, min_delta=1e-4):
        self.optimizer     = optimizer
        self.patience      = patience
        self.decay_factor  = decay_factor
        self.drop_patience = drop_patience
        self.boost_factor  = boost_factor
        self.min_lr        = min_lr
        self.max_lr        = max_lr
        self.min_delta     = min_delta

        self.best_f1           = -1.0
        self.epochs_no_improve = 0
        self.consecutive_drops = 0
        self.prev_f1           = -1.0
        self.history           = []

    def step(self, val_f1):
        """
        Gọi sau mỗi epoch với val_f1 hiện tại.
        Returns:
            str: 'decay' | 'boost' | 'hold'
        """
        current_lr  = self.optimizer.param_groups[0]['lr']
        action      = 'hold'

        f1_improved = val_f1 >= self.best_f1 + self.min_delta
        f1_dropped  = val_f1 <  self.prev_f1 - self.min_delta

        if f1_improved:
            self.best_f1           = val_f1
            self.epochs_no_improve = 0
            self.consecutive_drops = 0
        else:
            self.epochs_no_improve += 1

            if f1_dropped:
                self.consecutive_drops += 1
            else:
                self.consecutive_drops = 0

            # Rule 2: boost LR để thoát local minimum (ưu tiên trước rule 1)
            if self.consecutive_drops >= self.drop_patience:
                new_lr = min(current_lr * self.boost_factor, self.max_lr)
                if new_lr != current_lr:
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = new_lr
                    action = 'boost'
                self.consecutive_drops = 0

            # Rule 1: giảm LR khi plateau
            elif self.epochs_no_improve >= self.patience:
                new_lr = max(current_lr * self.decay_factor, self.min_lr)
                if new_lr != current_lr:
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = new_lr
                    action = 'decay'
                self.epochs_no_improve = 0

        self.prev_f1 = val_f1
        self.history.append({
            'val_f1': val_f1,
            'lr'    : self.optimizer.param_groups[0]['lr'],
            'action': action,
        })
        return action

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ==============================================================================
# HELPER: train_one_epoch
# — trả về (loss, preds, labels) để tính train metrics & vẽ learning curves
# ==============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm=1.0):
    """
    Train model qua 1 epoch.

    Returns:
        avg_loss  (float): loss trung bình
        all_preds (list) : dự đoán argmax toàn epoch
        all_labels(list) : nhãn thực tế tương ứng
    """
    model.train()
    running_loss = 0.0
    all_preds    = []
    all_labels   = []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # (N, 32, 128) → (N, 1, 32, 128)
        if X_batch.dim() == 3:
            X_batch = X_batch.unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        if hasattr(model, 'apply_max_norm'):
            model.apply_max_norm()

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
    Đánh giá model trên tập val / test.

    Returns:
        avg_loss (float) : loss trung bình
        metrics  (dict)  : accuracy, precision, recall, f1_score
        all_preds (list) : dự đoán
        all_labels(list) : nhãn thực tế
    """
    model.eval()
    all_preds    = []
    all_labels   = []
    running_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if X_batch.dim() == 3:
                X_batch = X_batch.unsqueeze(1)

            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)

            running_loss += loss.item() * y_batch.size(0)
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    metrics  = evaluate_metrics(all_labels, all_preds)
    return avg_loss, metrics, all_preds, all_labels


# ==============================================================================
# MAIN FINE-TUNING ENGINE
# ==============================================================================
def run_subject_specific_finetuning(
    label_type   = "valence",
    num_classes  = 2,
    # Phase 1 hyper-params
    p1_epochs    = 40,
    p1_lr        = 1e-3,
    p1_patience_es = 10,
    # Phase 2 hyper-params
    p2_epochs    = 20,
    p2_lr        = 5e-5,
    p2_patience_es = 7,
    # Shared
    batch_size   = 64,
    weight_decay = 1e-4,
    calib_ratio  = 0.2,        # tỉ lệ data target dùng để calibrate
):
    """
    Chạy toàn bộ fine-tuning pipeline cho tất cả subjects.

    Args:
        label_type     : "valence" | "arousal" | "4class"
        num_classes    : 2 hoặc 4
        p1_epochs      : Số epoch tối đa Phase 1 (pre-train)
        p1_lr          : Learning rate Phase 1
        p1_patience_es : Early stopping patience Phase 1
        p2_epochs      : Số epoch tối đa Phase 2 (fine-tune)
        p2_lr          : Learning rate Phase 2
        p2_patience_es : Early stopping patience Phase 2
        batch_size     : Batch size cho cả 2 phase
        weight_decay   : L2 regularization
        calib_ratio    : Tỉ lệ data target subject dùng để fine-tune (default 20%)
    """
    set_seed(42)

    exp_dir = os.path.join(RESULTS_DIR, label_type)
    os.makedirs(exp_dir, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X         = np.load(os.path.join(DATA_DIR, "X_epochs.npy"))
    y_valence = np.load(os.path.join(DATA_DIR, "y_valence.npy"))
    y_arousal = np.load(os.path.join(DATA_DIR, "y_arousal.npy"))
    groups    = np.load(os.path.join(DATA_DIR, "subject_groups.npy"))

    if label_type == "valence":
        y = y_valence
    elif label_type == "arousal":
        y = y_arousal
    elif label_type == "4class":
        y = y_valence * 2 + y_arousal
    else:
        raise ValueError(f"label_type không hợp lệ: {label_type}")

    class_names = CLASS_NAMES.get(label_type, [str(i) for i in range(num_classes)])

    print("=" * 70)
    print(f"🚀 FINE-TUNING EXPERIMENT | Label: {label_type.upper()} | Classes: {num_classes}")
    print(f"   Device    : {DEVICE}")
    print(f"   Phase 1   : {p1_epochs} epochs, LR={p1_lr}, ES patience={p1_patience_es}")
    print(f"   Phase 2   : {p2_epochs} epochs, LR={p2_lr}, ES patience={p2_patience_es}")
    print(f"   Calib     : {int(calib_ratio*100)}% target data | Test: {int((1-calib_ratio)*100)}%")
    print("=" * 70)

    logo               = LeaveOneGroupOut()
    all_subject_results = []
    n_subjects         = len(np.unique(groups))

    # ── 2. Vòng lặp subject ──────────────────────────────────────────────────
    for sub_idx, (train_idx, target_idx) in enumerate(logo.split(X, y, groups=groups)):
        set_seed(42 + sub_idx)
        sub_id     = groups[target_idx[0]] + 1
        sub_start  = time.time()

        print(f"\n{'─' * 70}")
        print(f"👤 SUBJECT S{sub_id:02d}  ({sub_idx + 1}/{n_subjects})")
        print(f"{'─' * 70}")

        # ── 2a. Chia dữ liệu ────────────────────────────────────────────────
        X_target, y_target = X[target_idx], y[target_idx]
        c_idx, t_idx = train_test_split(
            np.arange(len(y_target)),
            test_size  = 1 - calib_ratio,
            stratify   = y_target,
            random_state = 42,
        )
        X_source, y_source = X[train_idx],     y[train_idx]
        X_calib,  y_calib  = X_target[c_idx],  y_target[c_idx]
        X_test,   y_test   = X_target[t_idx],  y_target[t_idx]

        print(f"   Source : {len(y_source)} samples (31 subjects)")
        print(f"   Calib  : {len(y_calib)} samples ({int(calib_ratio*100)}% of S{sub_id:02d})")
        print(f"   Test   : {len(y_test)} samples ({int((1-calib_ratio)*100)}% of S{sub_id:02d})")

        # ── 2b. Normalize ────────────────────────────────────────────────────
        # Fit scaler trên source, transform calib & test riêng
        X_src_s, X_cal_s, _ = normalize_after_split(X_source, X_calib, mode='channel')
        _,       X_tst_s, _ = normalize_after_split(X_source, X_test,  mode='channel')

        # ── 2c. DataLoaders ──────────────────────────────────────────────────
        source_loader, _    = get_dataloaders(X_src_s, X_cal_s, y_source, y_calib, batch_size=batch_size)
        calib_train_loader, calib_val_loader = get_dataloaders(
            X_cal_s, X_tst_s, y_calib, y_test, batch_size=batch_size
        )
        test_loader = calib_val_loader  # alias cho rõ nghĩa

        # ── 2d. Class weights (tính trên source) ────────────────────────────
        class_weights = get_dynamic_class_weights(y_source, num_classes).to(DEVICE)
        criterion     = nn.CrossEntropyLoss(weight=class_weights)

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 1: PRE-TRAINING trên 31 subjects còn lại
        # ══════════════════════════════════════════════════════════════════════
        print(f"\n   ▶ Phase 1 — Pre-training on {n_subjects - 1} source subjects...")

        model     = EEGnet(num_classes=num_classes).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=p1_lr, weight_decay=weight_decay)
        scheduler = AdaptiveLRScheduler(
            optimizer,
            patience      = 5,
            decay_factor  = 0.5,
            drop_patience = 3,
            boost_factor  = 1.1,
            min_lr        = 1e-6,
            max_lr        = 1e-2,
        )

        best_p1_f1     = -1.0
        best_p1_state  = None
        p1_no_improve  = 0

        # Tracking learning curves Phase 1
        p1_train_losses, p1_val_losses = [], []
        p1_train_accs,   p1_val_accs   = [], []

        for epoch in range(p1_epochs):
            train_loss, train_preds, train_labels = train_one_epoch(
                model, source_loader, optimizer, criterion, DEVICE
            )
            val_loss, val_m, _, _ = evaluate(model, calib_val_loader, criterion, DEVICE)
            lr_action = scheduler.step(val_m['f1_score'])

            train_m = evaluate_metrics(train_labels, train_preds)

            p1_train_losses.append(train_loss)
            p1_val_losses.append(val_loss)
            p1_train_accs.append(train_m['accuracy'])
            p1_val_accs.append(val_m['accuracy'])

            if lr_action != 'hold':
                current_lr = optimizer.param_groups[0]['lr']
                emoji = '📉' if lr_action == 'decay' else '📈'
                print(f"      {emoji} Epoch {epoch+1}: LR {lr_action} → {current_lr:.1e}")

            if val_m['f1_score'] >= best_p1_f1 + 1e-4:
                best_p1_f1    = val_m['f1_score']
                best_p1_state = copy.deepcopy(model.state_dict())
                p1_no_improve = 0
            else:
                p1_no_improve += 1

            if (epoch + 1) % 10 == 0 or epoch == p1_epochs - 1:
                print(
                    f"      Epoch {epoch+1:3d}/{p1_epochs}  |  "
                    f"Train Loss: {train_loss:.4f}  Acc: {train_m['accuracy']:.4f}  |  "
                    f"Val Loss: {val_loss:.4f}  F1: {val_m['f1_score']:.4f}  "
                    f"Acc: {val_m['accuracy']:.4f}"
                )

            if p1_no_improve >= p1_patience_es:
                print(f"      ⏹ Early stopping Phase 1 tại epoch {epoch + 1}")
                break

        # Load best Phase 1 weights
        if best_p1_state is not None:
            model.load_state_dict(best_p1_state)
        print(f"   ✅ Phase 1 hoàn tất — Best Val F1: {best_p1_f1:.4f}")

        # Vẽ learning curves Phase 1
        plot_learning_curves(
            train_losses = p1_train_losses,
            val_losses   = p1_val_losses,
            train_accs   = p1_train_accs,
            val_accs     = p1_val_accs,
            title        = f"S{sub_id:02d} ({label_type}) — Phase 1 Pre-training",
            save_path    = os.path.join(exp_dir, f"lc_phase1_S{sub_id:02d}.png"),
        )

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 2: FINE-TUNING trên calib data của target subject
        # ══════════════════════════════════════════════════════════════════════
        print(f"\n   ▶ Phase 2 — Fine-tuning on S{sub_id:02d} calib data ({len(y_calib)} samples)...")

        # Đóng băng Block 1 — giữ lại kiến thức chung từ Phase 1
        frozen_layers = ["conv1", "depthwise", "bn1", "bn2"]
        for name, param in model.named_parameters():
            if any(x in name for x in frozen_layers):
                param.requires_grad = False

        frozen_count   = sum(1 for p in model.parameters() if not p.requires_grad)
        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"      Frozen params : {frozen_count}  |  Trainable params: {trainable_count}")

        optimizer_ft = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr           = p2_lr,
            weight_decay = weight_decay,
        )
        scheduler_ft = AdaptiveLRScheduler(
            optimizer_ft,
            patience      = 3,
            decay_factor  = 0.5,
            drop_patience = 2,
            boost_factor  = 1.05,
            min_lr        = 1e-7,
            max_lr        = 1e-4,
        )

        best_p2_f1    = -1.0
        best_p2_state = None
        p2_no_improve = 0

        # Tracking learning curves Phase 2
        p2_train_losses, p2_val_losses = [], []
        p2_train_accs,   p2_val_accs   = [], []

        for epoch in range(p2_epochs):
            train_loss, train_preds, train_labels = train_one_epoch(
                model, calib_train_loader, optimizer_ft, criterion, DEVICE
            )
            val_loss, val_m, _, _ = evaluate(model, test_loader, criterion, DEVICE)
            lr_action = scheduler_ft.step(val_m['f1_score'])

            train_m = evaluate_metrics(train_labels, train_preds)

            p2_train_losses.append(train_loss)
            p2_val_losses.append(val_loss)
            p2_train_accs.append(train_m['accuracy'])
            p2_val_accs.append(val_m['accuracy'])

            if lr_action != 'hold':
                current_lr = optimizer_ft.param_groups[0]['lr']
                emoji = '📉' if lr_action == 'decay' else '📈'
                print(f"      {emoji} Epoch {epoch+1}: LR {lr_action} → {current_lr:.1e}")

            if val_m['f1_score'] >= best_p2_f1 + 1e-4:
                best_p2_f1    = val_m['f1_score']
                best_p2_state = copy.deepcopy(model.state_dict())
                p2_no_improve = 0
            else:
                p2_no_improve += 1

            if (epoch + 1) % 5 == 0 or epoch == p2_epochs - 1:
                print(
                    f"      Epoch {epoch+1:3d}/{p2_epochs}  |  "
                    f"Train Loss: {train_loss:.4f}  Acc: {train_m['accuracy']:.4f}  |  "
                    f"Val Loss: {val_loss:.4f}  F1: {val_m['f1_score']:.4f}  "
                    f"Acc: {val_m['accuracy']:.4f}"
                )

            if p2_no_improve >= p2_patience_es:
                print(f"      ⏹ Early stopping Phase 2 tại epoch {epoch + 1}")
                break

        # Load best Phase 2 weights
        if best_p2_state is not None:
            model.load_state_dict(best_p2_state)
        print(f"   ✅ Phase 2 hoàn tất — Best Val F1: {best_p2_f1:.4f}")

        # Vẽ learning curves Phase 2
        plot_learning_curves(
            train_losses = p2_train_losses,
            val_losses   = p2_val_losses,
            train_accs   = p2_train_accs,
            val_accs     = p2_val_accs,
            title        = f"S{sub_id:02d} ({label_type}) — Phase 2 Fine-tuning",
            save_path    = os.path.join(exp_dir, f"lc_phase2_S{sub_id:02d}.png"),
        )

        # ══════════════════════════════════════════════════════════════════════
        # EVALUATION: đánh giá final trên test set
        # ══════════════════════════════════════════════════════════════════════
        _, final_metrics, final_preds, final_labels = evaluate(
            model, test_loader, criterion, DEVICE
        )
        all_subject_results.append(final_metrics)

        sub_time = time.time() - sub_start
        print(
            f"\n   📊 S{sub_id:02d} Final  →  "
            f"Acc: {final_metrics['accuracy']:.4f}  |  "
            f"F1: {final_metrics['f1_score']:.4f}  |  "
            f"Prec: {final_metrics['precision']:.4f}  |  "
            f"Rec: {final_metrics['recall']:.4f}  "
            f"({sub_time:.1f}s)"
        )

        # Lưu confusion matrix
        plot_confusion_matrix(
            final_labels, final_preds,
            classes   = class_names,
            title     = f"Fine-tuned S{sub_id:02d} ({label_type})",
            save_path = os.path.join(exp_dir, f"cm_S{sub_id:02d}.png"),
        )

        # Lưu best model weights
        model_path = os.path.join(exp_dir, f"best_model_S{sub_id:02d}.pt")
        torch.save(best_p2_state if best_p2_state is not None else model.state_dict(), model_path)
        print(f"   💾 Model saved: {model_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # TỔNG KẾT
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"🏆 TỔNG KẾT FINE-TUNING — {label_type.upper()}")
    print(f"{'=' * 70}")

    metric_keys   = ['accuracy', 'precision', 'recall', 'f1_score']
    summary_lines = [
        f"Experiment   : Fine-tuning EEGNet — {label_type.upper()}",
        f"Classes      : {num_classes}  |  Label: {label_type}",
        f"Subjects     : {n_subjects}",
        f"Phase 1      : {p1_epochs} epochs, LR={p1_lr}, ES={p1_patience_es}",
        f"Phase 2      : {p2_epochs} epochs, LR={p2_lr}, ES={p2_patience_es}",
        f"Calib ratio  : {int(calib_ratio*100)}%  |  Batch: {batch_size}",
        f"Device       : {DEVICE}",
        "",
        f"{'Metric':<15} {'Mean':>10} {'Std':>10}",
        "─" * 38,
    ]

    for key in metric_keys:
        values   = [r[key] for r in all_subject_results]
        mean_val = np.mean(values)
        std_val  = np.std(values)
        summary_lines.append(f"{key:<15} {mean_val:>10.4f} {std_val:>10.4f}")
        print(f"   {key:<15}: {mean_val:.4f} ± {std_val:.4f}")

    # Per-subject breakdown
    summary_lines += ["", "Per-subject F1:", "─" * 38]
    for i, r in enumerate(all_subject_results):
        line = f"  S{i+1:02d}  F1={r['f1_score']:.4f}  Acc={r['accuracy']:.4f}"
        summary_lines.append(line)

    # Lưu summary .txt
    summary_path = os.path.join(RESULTS_DIR, f"summary_{label_type}.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))
    print(f"\n💾 Summary đã lưu tại: {summary_path}")
    print(f"🏁 Fine-tuning [{label_type}] hoàn tất!\n")

    return all_subject_results


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    run_subject_specific_finetuning(
        label_type     = "valence",
        num_classes    = 2,
        p1_epochs      = 40,
        p1_lr          = 1e-3,
        p1_patience_es = 10,
        p2_epochs      = 20,
        p2_lr          = 5e-5,
        p2_patience_es = 7,
        batch_size     = 64,
        weight_decay   = 1e-4,
        calib_ratio    = 0.2,
    )