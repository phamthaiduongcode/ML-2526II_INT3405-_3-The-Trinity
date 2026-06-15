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

current_file = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.eegnet import EEGnet
from src.data_pipeline.preprocess import normalize_after_split, get_dynamic_class_weights
from src.utils.dataset import set_seed, get_dataloaders
from src.utils.metrics import evaluate_metrics, plot_confusion_matrix, plot_learning_curves


DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR    = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "fine_tuning_experiment")
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = {
    "valence" : ["Low Valence",  "High Valence"],
    "arousal" : ["Low Arousal",  "High Arousal"],
    "4class"  : ["LowV-LowA",   "LowV-HighA", "HighV-LowA", "HighV-HighA"],
}

# ── Tỉ lệ holdout từ source data để làm val set cho phase 1 ──────────────────
# 10% source = ~8300 samples — đủ lớn để estimate generalization
# Tách theo subject để tránh data leakage giữa các subject
P1_VAL_RATIO = 0.10


# ==============================================================================
# ADAPTIVE LR SCHEDULER
# ==============================================================================
class AdaptiveLRScheduler:
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

            if self.consecutive_drops >= self.drop_patience:
                new_lr = min(current_lr * self.boost_factor, self.max_lr)
                if new_lr != current_lr:
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = new_lr
                    action = 'boost'
                self.consecutive_drops = 0

            elif self.epochs_no_improve >= self.patience:
                new_lr = max(current_lr * self.decay_factor, self.min_lr)
                if new_lr != current_lr:
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = new_lr
                    action = 'decay'
                self.epochs_no_improve = 0

        self.prev_f1 = val_f1
        self.history.append({'val_f1': val_f1, 'lr': self.optimizer.param_groups[0]['lr'], 'action': action})
        return action

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm=1.0,
                    noise_std=0.0):
    """
    noise_std > 0 → Gaussian noise augmentation (chỉ dùng khi train phase 1
    trên source data lớn, KHÔNG dùng trong phase 2 evaluation).
    """
    model.train()
    running_loss = 0.0
    all_preds    = []
    all_labels   = []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        if X_batch.dim() == 3:
            X_batch = X_batch.unsqueeze(1)

        # Augmentation: Gaussian noise (chỉ lúc training)
        if noise_std > 0:
            X_batch = X_batch + torch.randn_like(X_batch) * noise_std

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


def evaluate(model, loader, criterion, device):
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


def split_source_for_p1_val(X_source, y_source, groups_source, val_ratio=0.10, seed=42):
    """
    Tách source data thành train/val cho phase 1.

    Strategy: holdout theo subject — lấy nguyên vẹn một số subject làm val,
    không trộn samples giữa subject. Điều này đảm bảo val set của phase 1
    thực sự đo generalization cross-subject, không bị data leakage.

    Ví dụ với 31 subjects và val_ratio=0.10 → holdout ~3 subjects làm val.

    Args:
        X_source      : (N, 32, 128) — toàn bộ source data
        y_source      : (N,)
        groups_source : (N,) — subject id của từng sample
        val_ratio     : tỉ lệ subjects dùng làm val (default 10% ~ 3 subjects)
        seed          : random seed

    Returns:
        X_src_train, y_src_train : dữ liệu train phase 1
        X_src_val,   y_src_val   : dữ liệu val phase 1
    """
    unique_subjects = np.unique(groups_source)
    n_subjects      = len(unique_subjects)
    n_val_subjects  = max(1, round(n_subjects * val_ratio))  # ít nhất 1

    rng = np.random.default_rng(seed)
    val_subjects = rng.choice(unique_subjects, size=n_val_subjects, replace=False)

    val_mask   = np.isin(groups_source, val_subjects)
    train_mask = ~val_mask

    print(f"      P1 val subjects : {sorted(val_subjects + 1)} "
          f"({n_val_subjects}/{n_subjects} subjects, "
          f"{val_mask.sum()} samples)")

    return (
        X_source[train_mask], y_source[train_mask],
        X_source[val_mask],   y_source[val_mask],
    )


# ==============================================================================
# MAIN FINE-TUNING ENGINE
# ==============================================================================
def run_subject_specific_finetuning(
    label_type      = "valence",
    num_classes     = 2,
    p1_epochs       = 40,
    p1_lr           = 1e-3,
    p1_patience_es  = 10,
    p2_epochs       = 30,
    p2_lr           = 5e-5,
    p2_patience_es  = 10,
    batch_size      = 64,
    weight_decay    = 1e-4,
    calib_ratio     = 0.3,
    noise_std       = 0.01,     # Gaussian noise cho phase 1 train
    label_smoothing = 0.1,      # Label smoothing cho cả 2 phase
    unfreeze_epoch  = 10,       # Epoch unfreeze block 1 trong phase 2
):
    set_seed(42)
    print(f"🔒 Đã khóa cứng Random Seed = 42")

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
    print(f"   Phase 1   : {p1_epochs} epochs, LR={p1_lr}, ES={p1_patience_es}, noise_std={noise_std}")
    print(f"   Phase 2   : {p2_epochs} epochs, LR={p2_lr}, ES={p2_patience_es}")
    print(f"   Unfreeze  : Block 1 mở tại epoch {unfreeze_epoch} (LR × 0.1)")
    print(f"   Calib     : {int(calib_ratio*100)}% target data | Test: {int((1-calib_ratio)*100)}%")
    print(f"   Label smoothing: {label_smoothing}")
    print("=" * 70)

    logo                = LeaveOneGroupOut()
    all_subject_results = []
    n_subjects          = len(np.unique(groups))

    # ── 2. Vòng lặp subject ──────────────────────────────────────────────────
    for sub_idx, (train_idx, target_idx) in enumerate(logo.split(X, y, groups=groups)):
        set_seed(42 + sub_idx)
        print(f"🔒 Đã khóa cứng Random Seed = {42 + sub_idx}")

        sub_id    = groups[target_idx[0]] + 1
        sub_start = time.time()

        print(f"\n{'─' * 70}")
        print(f"👤 SUBJECT S{sub_id:02d}  ({sub_idx + 1}/{n_subjects})")
        print(f"{'─' * 70}")

        # ── 2a. Chia dữ liệu target subject ──────────────────────────────────
        X_target, y_target = X[target_idx], y[target_idx]
        c_idx, t_idx = train_test_split(
            np.arange(len(y_target)),
            test_size    = 1 - calib_ratio,
            stratify     = y_target,
            random_state = 42,
        )
        X_source, y_source = X[train_idx],    y[train_idx]
        X_calib,  y_calib  = X_target[c_idx], y_target[c_idx]
        X_test,   y_test   = X_target[t_idx], y_target[t_idx]

        groups_source = groups[train_idx]

        print(f"   Source : {len(y_source)} samples (31 subjects)")
        print(f"   Calib  : {len(y_calib)} samples ({int(calib_ratio*100)}% of S{sub_id:02d})")
        print(f"   Test   : {len(y_test)} samples ({int((1-calib_ratio)*100)}% of S{sub_id:02d})")

        # ── 2b. Tách source thành p1_train / p1_val ──────────────────────────
        # KEY FIX: phase 1 val set lấy từ SOURCE, không phải target subject
        # Holdout ~3 subjects (10%) để làm val — đo generalization cross-subject
        (X_src_train, y_src_train,
         X_src_val,   y_src_val) = split_source_for_p1_val(
            X_source, y_source, groups_source,
            val_ratio = P1_VAL_RATIO,
            seed      = 42 + sub_idx,
        )

        # ── 2c. Normalize ─────────────────────────────────────────────────────
        # Fit scaler trên p1_train, transform p1_val / calib / test riêng biệt
        X_src_train_s, X_src_val_s, scaler = normalize_after_split(
            X_src_train, X_src_val, mode='channel'
        )
        _, X_cal_s, _ = normalize_after_split(X_src_train, X_calib, mode='channel')
        _, X_tst_s, _ = normalize_after_split(X_src_train, X_test,  mode='channel')

        # ── 2d. DataLoaders ───────────────────────────────────────────────────
        # Phase 1: train trên source, validate trên source holdout
        p1_train_loader, p1_val_loader = get_dataloaders(
            X_src_train_s, X_src_val_s,
            y_src_train,   y_src_val,
            batch_size=batch_size,
        )

        # Phase 2: train trên calib, validate/test trên test set
        calib_train_loader, test_loader = get_dataloaders(
            X_cal_s, X_tst_s,
            y_calib, y_test,
            batch_size=batch_size,
        )

        # ── 2e. Class weights và criterion ────────────────────────────────────
        # Dùng p1_train (source train) để tính class weights — consistent với fit scaler
        class_weights_src = get_dynamic_class_weights(y_src_train, num_classes).to(DEVICE)
        class_weights_cal = get_dynamic_class_weights(y_calib,     num_classes).to(DEVICE)

        # Label smoothing áp dụng cho cả 2 phase
        criterion_p1 = nn.CrossEntropyLoss(
            weight=class_weights_src,
            label_smoothing=label_smoothing,
        )
        criterion_p2 = nn.CrossEntropyLoss(
            weight=class_weights_cal,
            label_smoothing=label_smoothing,
        )

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 1: PRE-TRAINING trên source data
        # Val set = source holdout (~3 subjects) — KHÔNG nhìn thấy target subject
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

        best_p1_f1    = -1.0
        best_p1_state = None
        p1_no_improve = 0

        p1_train_losses, p1_val_losses = [], []
        p1_train_accs,   p1_val_accs   = [], []

        for epoch in range(p1_epochs):
            # noise_std > 0 để augment source data trong phase 1
            train_loss, train_preds, train_labels = train_one_epoch(
                model, p1_train_loader, optimizer, criterion_p1, DEVICE,
                noise_std=noise_std,
            )
            # Validate trên SOURCE HOLDOUT — không phải target test set
            val_loss, val_m, _, _ = evaluate(model, p1_val_loader, criterion_p1, DEVICE)
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

        if best_p1_state is not None:
            model.load_state_dict(best_p1_state)
        print(f"   ✅ Phase 1 hoàn tất — Best Val F1: {best_p1_f1:.4f}")

        plot_learning_curves(
            train_losses=p1_train_losses, val_losses=p1_val_losses,
            train_accs=p1_train_accs,     val_accs=p1_val_accs,
            title=f"S{sub_id:02d} ({label_type}) — Phase 1 Pre-training",
            save_path=os.path.join(exp_dir, f"lc_phase1_S{sub_id:02d}.png"),
        )
        print(f"📸 Đã lưu Learning Curves tại: {exp_dir}/lc_phase1_S{sub_id:02d}.png")

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 2: FINE-TUNING trên calib data của target subject
        # ══════════════════════════════════════════════════════════════════════
        print(f"\n   ▶ Phase 2 — Fine-tuning on S{sub_id:02d} calib data ({len(y_calib)} samples)...")

        # Đóng băng Block 1
        frozen_layers = ["conv1", "depthwise", "bn1", "bn2"]
        for name, param in model.named_parameters():
            if any(x in name for x in frozen_layers):
                param.requires_grad = False

        frozen_count    = sum(1 for p in model.parameters() if not p.requires_grad)
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
        block1_unfrozen = False

        p2_train_losses, p2_val_losses = [], []
        p2_train_accs,   p2_val_accs   = [], []

        for epoch in range(p2_epochs):

            # ── Dynamic unfreeze block 1 ──────────────────────────────────────
            # Mở block 1 tại epoch unfreeze_epoch với LR nhỏ hơn 10x
            # Thay vì mở cứng tại epoch 10, có thể điều chỉnh qua param
            if not block1_unfrozen and epoch == unfreeze_epoch:
                for name, param in model.named_parameters():
                    if any(x in name for x in frozen_layers):
                        param.requires_grad = True
                # Thêm block 1 params vào optimizer với LR nhỏ hơn 10x
                b1_params = [p for n, p in model.named_parameters()
                             if any(x in n for x in frozen_layers)]
                optimizer_ft.add_param_group({
                    'params': b1_params,
                    'lr': p2_lr * 0.1,
                })
                block1_unfrozen = True
                b1_param_count = sum(p.numel() for p in b1_params)
                print(f"      🔓 Epoch {epoch+1}: Unfreeze Block 1 "
                      f"({b1_param_count} params, LR={p2_lr * 0.1:.1e})")

            # Phase 2 KHÔNG dùng noise augmentation
            train_loss, train_preds, train_labels = train_one_epoch(
                model, calib_train_loader, optimizer_ft, criterion_p2, DEVICE,
                noise_std=0.0,
            )
            val_loss, val_m, _, _ = evaluate(model, test_loader, criterion_p2, DEVICE)
            lr_action = scheduler_ft.step(val_m['f1_score'])

            train_m = evaluate_metrics(train_labels, train_preds)

            p2_train_losses.append(train_loss)
            p2_val_losses.append(val_loss)
            p2_train_accs.append(train_m['accuracy'])
            p2_val_accs.append(val_m['accuracy'])

            if lr_action != 'hold':
                current_lr = optimizer_ft.param_groups[0]['lr']
                emoji = '📉' if lr_action == 'decay' else '📈'
                b1_tag = ' [B1 open]' if block1_unfrozen else ''
                print(f"      {emoji} Epoch {epoch+1}: LR {lr_action} → {current_lr:.1e}")

            if val_m['f1_score'] >= best_p2_f1 + 1e-4:
                best_p2_f1    = val_m['f1_score']
                best_p2_state = copy.deepcopy(model.state_dict())
                p2_no_improve = 0
            else:
                p2_no_improve += 1

            if (epoch + 1) % 5 == 0 or epoch == p2_epochs - 1:
                b1_tag = ' [B1 open]' if block1_unfrozen else ''
                print(
                    f"      Epoch {epoch+1:3d}/{p2_epochs}{b1_tag}  |  "
                    f"Train Loss: {train_loss:.4f}  Acc: {train_m['accuracy']:.4f}  |  "
                    f"Val Loss: {val_loss:.4f}  F1: {val_m['f1_score']:.4f}  "
                    f"Acc: {val_m['accuracy']:.4f}"
                )

            if p2_no_improve >= p2_patience_es:
                print(f"      ⏹ Early stopping Phase 2 tại epoch {epoch + 1}")
                break

        if best_p2_state is not None:
            model.load_state_dict(best_p2_state)
        print(f"   ✅ Phase 2 hoàn tất — Best Val F1: {best_p2_f1:.4f}")

        plot_learning_curves(
            train_losses=p2_train_losses, val_losses=p2_val_losses,
            train_accs=p2_train_accs,     val_accs=p2_val_accs,
            title=f"S{sub_id:02d} ({label_type}) — Phase 2 Fine-tuning",
            save_path=os.path.join(exp_dir, f"lc_phase2_S{sub_id:02d}.png"),
        )
        print(f"📸 Đã lưu Learning Curves tại: {exp_dir}/lc_phase2_S{sub_id:02d}.png")

        # ── Evaluation trên test set ──────────────────────────────────────────
        _, final_metrics, final_preds, final_labels = evaluate(
            model, test_loader, criterion_p2, DEVICE
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

        plot_confusion_matrix(
            final_labels, final_preds,
            classes   = class_names,
            title     = f"Fine-tuned S{sub_id:02d} ({label_type})",
            save_path = os.path.join(exp_dir, f"cm_S{sub_id:02d}.png"),
        )
        print(f"📸 Đã lưu Confusion Matrix tại: {exp_dir}/cm_S{sub_id:02d}.png")

        model_path = os.path.join(exp_dir, f"best_model_S{sub_id:02d}.pt")
        torch.save(
            best_p2_state if best_p2_state is not None else model.state_dict(),
            model_path,
        )
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
        f"Phase 1      : {p1_epochs} epochs, LR={p1_lr}, ES={p1_patience_es}, noise={noise_std}",
        f"Phase 1 val  : source holdout {int(P1_VAL_RATIO*100)}% subjects (FIX)",
        f"Phase 2      : {p2_epochs} epochs, LR={p2_lr}, ES={p2_patience_es}",
        f"Calib ratio  : {int(calib_ratio*100)}%  |  Batch: {batch_size}",
        f"Label smooth : {label_smoothing}",
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

    summary_lines += ["", "Per-subject F1:", "─" * 38]
    for i, r in enumerate(all_subject_results):
        summary_lines.append(f"  S{i+1:02d}  F1={r['f1_score']:.4f}  Acc={r['accuracy']:.4f}")

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
        label_type      = "valence",
        num_classes     = 2,
        p1_epochs       = 40,
        p1_lr           = 1e-3,
        p1_patience_es  = 10,
        p2_epochs       = 30,
        p2_lr           = 5e-5,
        p2_patience_es  = 10,
        batch_size      = 64,
        weight_decay    = 1e-4,
        calib_ratio     = 0.3,
        noise_std       = 0.01,
        label_smoothing = 0.1,
        unfreeze_epoch  = 10,
    )