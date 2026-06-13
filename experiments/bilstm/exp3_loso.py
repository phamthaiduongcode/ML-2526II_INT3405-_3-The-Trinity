"""
experiments/exp3_loso.py
Exp 3: Leave-One-Subject-Out (LOSO) Cross-Validation, 2-class (Valence & Arousal)
TỐI ƯU HÓA SONG SONG DUAL-GPU (T4 x2) + AMP FP16 (PyTorch 2.x) + FULL LOGS

Chạy từ thư mục root:
    python -m experiments.exp3_loso
"""

import os
import sys
import json
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

# Lùi 2 bước từ experiments -> root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.models.eeg_bilstm import BiLSTM_Model
from src.data_pipeline.preprocess import normalize_after_split, get_dynamic_class_weights
from src.utils.metrics import evaluate_metrics, plot_confusion_matrix

class AdaptiveLRScheduler:
    def __init__(self, optimizer, patience=5, decay_factor=0.5, drop_patience=3, boost_factor=1.1, min_lr=1e-6, max_lr=1e-2, min_delta=1e-4):
        self.optimizer = optimizer
        self.patience = patience
        self.decay_factor = decay_factor
        self.drop_patience = drop_patience
        self.boost_factor = boost_factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_delta = min_delta
        self.best_f1 = -1.0
        self.epochs_no_improve = 0   
        self.consecutive_drops = 0   
        self.prev_f1 = -1.0

    def step(self, val_f1):
        current_lr = self.optimizer.param_groups[0]['lr']
        action = 'hold'
        f1_improved = val_f1 >= self.best_f1 + self.min_delta
        f1_dropped = val_f1 < self.prev_f1 - self.min_delta

        if f1_improved:
            self.best_f1 = val_f1
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
                    for pg in self.optimizer.param_groups: pg['lr'] = new_lr
                    action = 'boost'
                self.consecutive_drops = 0  

            elif self.epochs_no_improve >= self.patience:
                new_lr = max(current_lr * self.decay_factor, self.min_lr)
                if new_lr != current_lr:
                    for pg in self.optimizer.param_groups: pg['lr'] = new_lr
                    action = 'decay'
                self.epochs_no_improve = 0  

        self.prev_f1 = val_f1
        return action

# ── SIÊU THAM SỐ ──
BATCH_SIZE = 1024
MAX_EPOCHS = 60
PATIENCE   = 15
LR         = 0.0003

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ══════════════════════════════════════════════════════════════════════════════
def run_target_loso(target_name, y_full, X, subject_groups, device_idx):
    set_seed(42)
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [GPU {device_idx}] BẮT ĐẦU CHẠY TARGET: {target_name.upper()}")
    
    # Tiền tố log cho đẹp
    log_prefix_name = "🔵 VAL" if target_name == "Valence" else "🔴 ARO"

    MODEL_NAME = "bilstm" 
    LOG_DIR    = os.path.join(ROOT_DIR, "result", MODEL_NAME, "logs")
    PLOT_DIR   = os.path.join(ROOT_DIR, "result", MODEL_NAME, "plots")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    checkpoint_path = os.path.join(LOG_DIR, f"loso_checkpoint_{target_name.lower()}.json")
    loso_results = {target_name: {}, "Skipped": []}

    for test_subject in range(32):
        start_time = time.time()
        subj_prefix = f"{log_prefix_name} | S{test_subject+1:02d}"
        print(f"\n{'='*65}\n▶️  BẮT ĐẦU {subj_prefix} \n{'='*65}")

        test_idx  = subject_groups == test_subject
        train_idx = subject_groups != test_subject

        X_train, y_train = X[train_idx], y_full[train_idx]
        X_test,  y_test  = X[test_idx],  y_full[test_idx]

        if len(np.unique(y_test)) == 1:
            print(f"⚠️  {subj_prefix} BỎ QUA — Tập test chỉ chứa 1 class!")
            loso_results["Skipped"].append(f"S{test_subject+1}")
            continue

        X_train, X_test = normalize_after_split(X_train, X_test, mode='channel')

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
        )

        class_weights = get_dynamic_class_weights(y_train, n_classes=2, max_weight=5.0).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model     = BiLSTM_Model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = AdaptiveLRScheduler(optimizer, patience=5, decay_factor=0.5, min_lr=1e-6, max_lr=1e-2)
        
        scaler = torch.amp.GradScaler('cuda')

        best_f1          = -1.0
        best_acc         = 0.0
        best_val_loss    = float('inf')
        patience_counter = 0
        best_preds, best_trues = [], []

        for epoch in range(MAX_EPOCHS):
            # --- Phase Train ---
            model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    loss = criterion(model(bx), by)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)

            # --- Phase Eval ---
            model.eval()
            val_loss, all_preds, all_trues = 0.0, [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx, by = bx.to(device), by.to(device)
                    with torch.amp.autocast('cuda'):
                        logits = model(bx)
                        loss_val = criterion(logits, by)
                    val_loss += loss_val.item()
                    
                    all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    all_trues.extend(by.cpu().numpy())

            val_loss /= len(test_loader)
            epoch_metrics = evaluate_metrics(all_trues, all_preds)
            current_f1  = epoch_metrics['f1_score']
            current_acc = epoch_metrics['accuracy']

            # --- LR Scheduler & Logging LR ---
            lr_action = scheduler.step(current_f1)
            if lr_action != 'hold':
                current_lr = optimizer.param_groups[0]['lr']
                action_emoji = '📉 Giảm' if lr_action == 'decay' else '📈 Tăng'
                print(f"   ↳ ⚙️ {subj_prefix} | CẬP NHẬT LR: {action_emoji} xuống {current_lr:.1e}")

            # --- Early Stopping Logic ---
            is_best = False
            if current_f1 >= best_f1 + 1e-4:
                best_f1          = current_f1
                best_acc         = current_acc
                best_val_loss    = val_loss
                best_preds       = all_preds.copy()
                best_trues       = all_trues.copy()
                patience_counter = 0
                is_best = True
            else:
                patience_counter += 1

            # --- In Log 1 dòng tóm tắt Epoch ---
            marker = "🌟 BEST" if is_best else f"⏳ P:{patience_counter}/{PATIENCE}"
            print(f"[{subj_prefix}] Ep {epoch+1:02d}/{MAX_EPOCHS} | "
                  f"TrLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
                  f"Acc: {current_acc:.4f} | F1: {current_f1:.4f} | {marker}")

            if patience_counter >= PATIENCE:
                print(f"   🛑 [{subj_prefix}] EARLY STOPPING KÍCH HOẠT TẠI EPOCH {epoch+1}!")
                break

        exec_time = round((time.time() - start_time) / 60, 2)
        print(f"✅ TỔNG KẾT {subj_prefix} ({exec_time} m) | Chốt Best Acc: {best_acc:.4f} | Chốt Best F1: {best_f1:.4f}")

        plot_confusion_matrix(
            best_trues, best_preds, classes=('Low', 'High'),
            title=f"{target_name} — Subject {test_subject + 1}",
            save_path=os.path.join(PLOT_DIR, f"cm_{target_name.lower()}_S{test_subject + 1}.png")
        )

        loso_results[target_name][f"Subject_{test_subject+1}"] = {
            "Acc"          : float(best_acc),
            "F1_Macro"     : float(best_f1),
            "Best_Val_Loss": float(best_val_loss),
            "Epochs_Ran"   : epoch + 1,
        }

        with open(checkpoint_path, "w") as f:
            json.dump(loso_results, f, indent=4)

        del model, X_train, X_test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"🎉 HOÀN TẤT TARGET {target_name.upper()} TRÊN GPU {device_idx}!")

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
    X              = np.load(os.path.join(DATA_DIR, 'X_epochs.npy'))
    y_val          = np.load(os.path.join(DATA_DIR, 'y_valence.npy'))
    y_aro          = np.load(os.path.join(DATA_DIR, 'y_arousal.npy'))
    subject_groups = np.load(os.path.join(DATA_DIR, 'subject_groups.npy'))
    print(f"📊 Dữ liệu nạp thành công: X = {X.shape}")

    n_gpus = torch.cuda.device_count()
    print(f"🖥️  Số lượng GPU nhận diện được: {n_gpus}")

    if n_gpus >= 2:
        print("🔥 ĐÃ KÍCH HOẠT CHẾ ĐỘ SONG SONG: GPU 0 gánh Valence | GPU 1 gánh Arousal 🔥")
        p_valence = mp.Process(target=run_target_loso, args=("Valence", y_val, X, subject_groups, 0))
        p_arousal = mp.Process(target=run_target_loso, args=("Arousal", y_aro, X, subject_groups, 1))

        p_valence.start()
        p_arousal.start()

        p_valence.join()
        p_arousal.join()
        
        print("\n🎉 XUẤT SẮC! Cả hai tiến trình song song đã hoàn tất toàn bộ kịch bản LOSO!")
    else:
        print("⚠️ Chỉ tìm thấy 1 GPU. Tiến hành chạy tuần tự truyền thống...")
        run_target_loso("Valence", y_val, X, subject_groups, 0)
        run_target_loso("Arousal", y_aro, X, subject_groups, 0)