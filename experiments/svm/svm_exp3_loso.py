"""
experiments/svm/svm_exp3_loso.py
Exp 3 (SVM): Leave-One-Subject-Out (LOSO), 2-class (Valence & Arousal)
- Trích xuất 312 đặc trưng (PSD log, Asymmetry, DE)
- LDS Smoothing SAU khi chia train/test (chống Data Leakage)
- Per-subject Z-score normalization SAU khi chia (chống Data Leakage)
- Class weight = 'balanced'

Chạy từ thư mục root:
    python -m experiments.svm.svm_exp3_loso

Output:
    result/svm/logs/svm_subject-independent_valence.json
    result/svm/logs/svm_subject-independent_arousal.json
    result/svm/plots/cm_svm_subject-independent_*_sub*.png
    result/svm/plots/cm_svm_subject-independent_*_Overall.png
"""

import os
import sys
import gc
import time
import numpy as np

# ── Đưa root project vào sys.path ──
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from src.data_pipeline.feature_extraction import extract_features_dataset
from src.utils.dataset import set_seed
from experiments.svm.svm_utils import (
    SEED, USE_LDS,
    apply_lds_safe, apply_per_subject_normalization_safe,
    evaluate_and_plot, train_svm_pipeline, summarize_results,
)

# ── ĐƯỜNG DẪN ──
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_NAME = "svm"
LOG_DIR    = os.path.join(ROOT_DIR, "result", MODEL_NAME, "logs")
PLOT_DIR   = os.path.join(ROOT_DIR, "result", MODEL_NAME, "plots")

for d in [LOG_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

MODE = "subject-independent"


# ══════════════════════════════════════════════════════════════════════════════
# RUN SVM EXP 3 — LOSO 2-class
# ══════════════════════════════════════════════════════════════════════════════

def run_svm_exp3(X_feat, y_all, groups, label_name):
    print(f"\n{'='*60}\n  SVM Exp3 — {label_name.upper()} — LOSO (Subject-Independent)\n{'='*60}")
    
    unique_groups = np.unique(groups)
    n_subjects = len(unique_groups)
    n_classes = 2
    
    subject_weights = []
    subject_results = []
    t_start = time.time()
    
    print(f"Bắt đầu LOSO CV trên {n_subjects} subjects...")
    for fold_i, test_group in enumerate(unique_groups):
        sub_num = int(test_group) + 1
        train_idx = (groups != test_group)
        test_idx = (groups == test_group)
        
        X_train, y_train = X_feat[train_idx], y_all[train_idx]
        X_test, y_test   = X_feat[test_idx], y_all[test_idx]
        groups_train = groups[train_idx]
        groups_test = groups[test_idx]
        
        # Kiểm tra Edge Case
        train_class_counts = np.bincount(y_train, minlength=n_classes)
        test_class_counts = np.bincount(y_test, minlength=n_classes)
        
        if np.count_nonzero(test_class_counts) < 2:
            print(f"  [WARNING] Bỏ qua Subject {sub_num:02d}: Tập test không đủ 2 lớp.")
            continue
            
        min_samples_train = np.min(train_class_counts[train_class_counts > 0])
        if np.count_nonzero(train_class_counts) < 2 or min_samples_train < 5:
            print(f"  [WARNING] Bỏ qua Subject {sub_num:02d}: Tập train quá thiếu mẫu.")
            continue
        
        # LDS smooth riêng biệt — truyền original_indices để nhóm đúng theo video
        if USE_LDS:
            train_orig_idx = np.where(train_idx)[0]
            test_orig_idx = np.where(test_idx)[0]
            X_train = apply_lds_safe(X_train, original_indices=train_orig_idx)
            X_test = apply_lds_safe(X_test, original_indices=test_orig_idx)
        
        # Per-subject normalization SAU khi chia
        X_train, X_test = apply_per_subject_normalization_safe(
            X_train, X_test, groups_train, groups_test
        )
            
        model, best_c = train_svm_pipeline(X_train, y_train, fold_seed=SEED + fold_i)
        
        svm_step = model.named_steps['svm']
        if hasattr(svm_step, 'coef_'):
            subject_weights.append(svm_step.coef_[0])
            
        y_pred = model.predict(X_test)
        
        cm_path = os.path.join(PLOT_DIR, f"cm_svm_{MODE}_{label_name}_sub{sub_num:02d}.png")
        acc, f1, cm = evaluate_and_plot(y_test, y_pred, 
            f"{label_name} - Sub {sub_num:02d}", MODE, cm_path)
        
        subject_results.append({
            'subject': sub_num,
            'acc': float(acc),
            'f1_macro': float(f1),
            'best_C': float(best_c),
            'confusion_matrix': cm.tolist(),
            'n_test': len(y_test)
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
        results[label_name] = run_svm_exp3(X_feat, y, subject_groups, label_name)

    print("\n✅ HOÀN TẤT SVM EXP 3 (LOSO, 2-class)")
