"""
experiments/exp3_loso.py
Exp 3: Leave-One-Subject-Out (LOSO) Cross-Validation, 2-class (Valence & Arousal)

Chạy từ thư mục root:
    python -m experiments.exp3_loso

Output:
    result/bilstm/logs/loso_checkpoint.json
    result/bilstm/plots/cm_valence_S*.png
    result/bilstm/plots/cm_arousal_S*.png
"""

import os
import sys
import json
import time
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Đảm bảo nhận diện đúng thư mục root của project
# Đảm bảo nhận diện đúng thư mục root của project (lùi 3 bước: bilstm -> experiments -> root)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# 1. IMPORT CÁC HÀM TIỆN ÍCH TỪ PIPELINE CHUẨN (KHÔNG CODE TAY LẠI)
# ══════════════════════════════════════════════════════════════════════════════
from src.models.eeg_bilstm import BiLSTM_Model
from src.data_pipeline.preprocess import normalize_after_split, get_dynamic_class_weights
from src.utils.metrics import evaluate_metrics, plot_confusion_matrix

# ── ĐƯỜNG DẪN ĐẦU RA ──
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_NAME = "bilstm" 
LOG_DIR    = os.path.join(ROOT_DIR, "result", MODEL_NAME, "logs")
PLOT_DIR   = os.path.join(ROOT_DIR, "result", MODEL_NAME, "plots")

for d in [LOG_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── SIÊU THAM SỐ (HYPERPARAMS) ──
BATCH_SIZE = 256
MAX_EPOCHS = 60
PATIENCE   = 15
LR         = 0.001
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def set_seed(seed=42):
    """Khóa cứng tất cả các nguồn random để đảm bảo kết quả đồng nhất."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
# 2. TIẾN TRÌNH HUẤN LUYỆN CHÍNH (MAIN PROCESS)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Khóa seed ngay khi khởi động exp
    set_seed(42)
    print(f"🖥️  Thiết bị đang huấn luyện: {DEVICE}")
    
    # Nạp toàn bộ dữ liệu đã đóng gói từ preprocess.py
    X              = np.load(os.path.join(DATA_DIR, 'X_epochs.npy'))
    y_val          = np.load(os.path.join(DATA_DIR, 'y_valence.npy'))
    y_aro          = np.load(os.path.join(DATA_DIR, 'y_arousal.npy'))
    subject_groups = np.load(os.path.join(DATA_DIR, 'subject_groups.npy'))
    print(f"✅ Đã nạp dữ liệu thành công: X={X.shape}")

    # Cấu trúc lưu log y hệt bản gốc của ông
    loso_results = {"Valence": {}, "Arousal": {}, "Skipped": []}

    # VÒNG LẶP QUA CẢ 2 ĐÍCH: VALENCE VÀ AROUSAL
    for target_name, y_full in [("Valence", y_val), ("Arousal", y_aro)]:
        print("\n" + "="*70)
        print(f"🚀 BẮT ĐẦU CHẠY KIỂM THỬ LEAVE-ONE-SUBJECT-OUT (LOSO): {target_name.upper()}")
        print("="*70)

        # VÒNG LẶP LOSO QUA 32 SUBJECTS
        for test_subject in range(32):
            start_time = time.time()
            
            # Phân tách tập Train (31 người) và tập Test (1 người)
            test_idx  = subject_groups == test_subject
            train_idx = subject_groups != test_subject

            X_train, y_train = X[train_idx], y_full[train_idx]
            X_test,  y_test  = X[test_idx],  y_full[test_idx]

            # Phòng thủ: Bỏ qua nếu tập test xui xẻo chỉ chứa đúng 1 class nhãn
            if len(np.unique(y_test)) == 1:
                print(f"⚠️  BỎ QUA Subject {test_subject+1} — Tập kiểm thử chỉ xuất hiện 1 class duy nhất!")
                loso_results["Skipped"].append(f"{target_name}_S{test_subject+1}")
                continue

            # CHUẨN HÓA DỮ LIỆU CHANNEL-WISE CHỐNG LEAKAGE BẰNG HÀM CHUẨN PIPELINE
            X_train, X_test, _ = normalize_after_split(X_train, X_test, mode='channel')

            # Đóng gói DataLoader bằng PyTorch
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

            # TÍNH TRỌNG SỐ LỚP ĐỘNG ĐÃ ĐƯỢC CHẶN BÙNG NỔ (MAX_WEIGHT=5.0) TỪ PREPROCESS
            class_weights = get_dynamic_class_weights(y_train, n_classes=2, max_weight=5.0).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            # Khởi tạo mạng mạng thần kinh và bộ tối ưu hóa
            model     = BiLSTM_Model().to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            # Bộ theo dõi để áp dụng Early Stopping
            best_val_loss    = float('inf')
            best_f1          = 0.0
            best_acc         = 0.0
            patience_counter = 0
            best_preds       = []
            best_trues       = []

            # MẠNG TRAIN QUA CÁC EPOCH
            for epoch in range(MAX_EPOCHS):
                # Phase Train
                model.train()
                for bx, by in train_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()

                # Phase Eval (Kiểm thử trên Subject bị bỏ lại)
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

                # Đánh giá nhanh hiệu năng epoch bằng hàm dùng chung metrics.py
                epoch_metrics = evaluate_metrics(all_trues, all_preds)
                current_f1  = epoch_metrics['f1_score']
                current_acc = epoch_metrics['accuracy']

                # Lưu vết checkpoint tốt nhất theo Validation Loss
                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    best_f1          = current_f1
                    best_acc         = current_acc
                    best_preds       = all_preds.copy()
                    best_trues       = all_trues.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Kích hoạt Early Stopping nếu loss không giảm sau chuỗi số PATIENCE epoch
                if patience_counter >= PATIENCE:
                    print(f"    🛑 Early stopping kích hoạt tại epoch {epoch+1}")
                    break

            exec_time = round((time.time() - start_time) / 60, 2)
            print(f"✅ S{test_subject+1} ({exec_time} phút) | Best Acc: {best_acc:.4f} | Best F1-Macro: {best_f1:.4f}")

            # TRỰC QUAN HÓA BẰNG HÀM PLOT MA TRẬN NHẦM LẪN DÙNG CHUNG SẮC NÉT
            plot_confusion_matrix(
                best_trues, best_preds, 
                classes=('Low', 'High'),
                title=f"{target_name} — Subject {test_subject + 1}",
                save_path=os.path.join(PLOT_DIR, f"cm_{target_name.lower()}_S{test_subject + 1}.png")
            )

            # Cập nhật kết quả vào bộ log tổng
            loso_results[target_name][f"Subject_{test_subject+1}"] = {
                "Acc"          : float(best_acc),
                "F1_Macro"     : float(best_f1),
                "Best_Val_Loss": float(best_val_loss),
                "Epochs_Ran"   : epoch + 1,
            }

            # Lưu dự phòng Checkpoint liên tục sau mỗi Subject để tránh mất dữ liệu log
            with open(os.path.join(LOG_DIR, "loso_checkpoint.json"), "w") as f:
                json.dump(loso_results, f, indent=4)

            # GIẢI PHÓNG RAM & VRAM: Triệt tiêu rác bộ nhớ sau khi huấn luyện xong 1 Subject
            del model, X_train, X_test, train_loader, test_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n🎉 HOÀN TẤT TOÀN BỘ EXP 3 (LOSO). Nhật ký cấu trúc đã lưu tại: result/{MODEL_NAME}/logs/loso_checkpoint.json")