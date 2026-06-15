"""
experiments/svm/svm_exp1_2class.py
Exp 1 (SVM): Subject-Dependent, GroupKFold (k=5), 2-class (Valence & Arousal)
- Trích xuất 312 đặc trưng (PSD log, Asymmetry, DE)
- LDS Smoothing SAU khi chia train/test (chống Data Leakage)
- GroupKFold chia theo video để tránh rò rỉ thời gian
- Class weight = 'balanced'

Chạy từ thư mục root:
    python -m experiments.svm.svm_exp1_2class

Output:
    result/svm/logs/svm_subject-dependent_valence.json
    result/svm/logs/svm_subject-dependent_arousal.json
    result/svm/plots/cm_svm_subject-dependent_*_sub*_avg5folds.png
    result/svm/plots/cm_svm_subject-dependent_*_Overall.png
"""

import os
import sys
import gc
import time
import numpy as np

from sklearn.model_selection import GroupKFold

# ── Đưa root project vào sys.path ──
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from src.data_pipeline.feature_extraction import extract_features_dataset
from src.utils.dataset import set_seed
from experiments.svm.svm_utils import (
    SEED, USE_LDS, EPOCHS_PER_VIDEO,
    apply_lds_safe, evaluate_and_plot, plot_aggregated_cm,
    train_svm_pipeline, summarize_results,
)

# ── ĐƯỜNG DẪN ──
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_NAME = "svm"
LOG_DIR    = os.path.join(ROOT_DIR, "result", MODEL_NAME, "logs")
PLOT_DIR   = os.path.join(ROOT_DIR, "result", MODEL_NAME, "plots")

for d in [LOG_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── HYPERPARAMS ──
N_SPLITS = 5
MODE = "subject-dependent"


# ══════════════════════════════════════════════════════════════════════════════
# RUN SVM EXP 1 — Subject-Dependent 2-class
# ══════════════════════════════════════════════════════════════════════════════

def run_svm_exp1(X_feat, y_all, groups, label_name):
    print(f"\n{'='*60}\n  SVM Exp1 — {label_name.upper()} — Subject-Dependent, {N_SPLITS}-Fold\n{'='*60}")
    
    unique_groups = np.unique(groups)
    n_subjects = len(unique_groups)
    n_classes = 2
    
    subject_weights = []
    subject_results = []
    t_start = time.time()
    
    print(f"Bắt đầu huấn luyện mô hình riêng rẽ cho {n_subjects} subjects...")
    for sg in unique_groups:
        sub_num = int(sg) + 1
        idx = (groups == sg)
        X_sub, y_sub = X_feat[idx], y_all[idx]
        
        # Kiểm tra Edge Case
        sub_class_counts = np.bincount(y_sub, minlength=n_classes)
        min_samples = np.min(sub_class_counts[sub_class_counts > 0])
        if np.count_nonzero(sub_class_counts) < 2 or min_samples < 6:
            print(f"  [WARNING] Bỏ qua Subject {sub_num:02d}: Thiếu dữ liệu phân bố nhãn tối thiểu.")
            continue
        
        # GroupKFold theo video
        gkf = GroupKFold(n_splits=N_SPLITS)
        num_videos = len(X_sub) // EPOCHS_PER_VIDEO
        valid_len = num_videos * EPOCHS_PER_VIDEO
        
        if valid_len == 0:
            print(f"  [WARNING] Bỏ qua Subject {sub_num:02d}: Không đủ 1 video nguyên vẹn.")
            continue
        
        if valid_len != len(X_sub):
            print(f"  [WARNING] Subject {sub_num:02d}: Có {len(X_sub)} epochs, kỳ vọng bội của {EPOCHS_PER_VIDEO}. Truncate về {valid_len}.")
        
        X_sub = X_sub[:valid_len]
        y_sub = y_sub[:valid_len]
        video_groups = np.repeat(np.arange(num_videos), EPOCHS_PER_VIDEO)
        
        sub_accs, sub_f1s, sub_cms, sub_best_cs = [], [], [], []
        
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_sub, y_sub, groups=video_groups)):
            X_train, X_test = X_sub[train_idx], X_sub[test_idx]
            y_train, y_test = y_sub[train_idx], y_sub[test_idx]
            
            # LDS smooth riêng biệt trên train/test
            if USE_LDS:
                X_train = apply_lds_safe(X_train)
                X_test = apply_lds_safe(X_test)
            
            
            model, best_c = train_svm_pipeline(X_train, y_train, fold_seed=SEED + fold_idx)
            
            svm_step = model.named_steps['svm']
            if hasattr(svm_step, 'coef_'):
                subject_weights.append(svm_step.coef_[0])
            
            y_pred = model.predict(X_test)
            acc, f1, cm = evaluate_and_plot(y_test, y_pred, 
                f"{label_name} - Sub {sub_num:02d} - Fold {fold_idx}", MODE, None)
            
            sub_accs.append(acc)
            sub_f1s.append(f1)
            sub_cms.append(cm)
            sub_best_cs.append(best_c)
        
        # Tổng hợp 5 folds → 1 biểu đồ
        mean_acc = float(np.mean(sub_accs))
        mean_f1 = float(np.mean(sub_f1s))
        sum_cm = np.sum(sub_cms, axis=0)
        
        cm_path_avg = os.path.join(PLOT_DIR, f"cm_svm_{MODE}_{label_name}_sub{sub_num:02d}_avg5folds.png")
        plot_aggregated_cm(sum_cm, mean_acc, mean_f1, 
            f"{label_name} - Sub {sub_num:02d} (Avg {N_SPLITS} Folds)", MODE, cm_path_avg)
        
        subject_results.append({
            'subject': sub_num,
            'acc': mean_acc,
            'f1_macro': mean_f1,
            'best_C': float(np.median(sub_best_cs)),
            'confusion_matrix': sum_cm.tolist(),
            'n_test': len(y_sub)
        })
        
        if sub_num % 5 == 0 or sub_num == n_subjects:
            print(f"  Đã xong subject {sub_num}/{n_subjects}...")
    
    print(f"Đã huấn luyện xong! Mất {time.time()-t_start:.1f}s.")
    return summarize_results(subject_results, subject_weights, label_name, MODE, 
                             is_multi=False, log_dir=LOG_DIR, plot_dir=PLOT_DIR)


if __name__ == "__main__":
    set_seed(SEED)
    print("Đang load dữ liệu tiền xử lý (raw epochs)...")
    X_epochs       = np.load(os.path.join(DATA_DIR, "X_epochs.npy"))
    y_valence      = np.load(os.path.join(DATA_DIR, "y_valence.npy")).astype(np.int64)
    y_arousal      = np.load(os.path.join(DATA_DIR, "y_arousal.npy")).astype(np.int64)
    subject_groups = np.load(os.path.join(DATA_DIR, "subject_groups.npy"))
    
    print("\nBắt đầu trích xuất đặc trưng (Feature Extraction)...")
    X_feat = extract_features_dataset(X_epochs)
    del X_epochs; gc.collect()

    results = {}
    for label_name, y in [("valence", y_valence), ("arousal", y_arousal)]:
        results[label_name] = run_svm_exp1(X_feat, y, subject_groups, label_name)

    print("\n✅ HOÀN TẤT SVM EXP 1 (Subject-Dependent, 2-class)")
