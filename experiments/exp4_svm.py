"""
experiments/exp4_svm.py
Exp 4: Machine Learning truyền thống (SVM) với Feature Extraction
- Hỗ trợ trích xuất 312 đặc trưng (PSD, Asymmetry, DE)
- Hỗ trợ LDS Smoothing
- Class weight = 'balanced'
- Hỗ trợ Subject-Dependent và Subject-Independent (LOSO)
- Trích xuất Feature Importance (Trọng số lớn nhất) cho cả 2-class và 4-class

Chạy từ thư mục root:
    python -m experiments.exp4_svm
"""

import os
import sys
import json
import time
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# ── Đưa root project vào sys.path để import src ──
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.data_pipeline.feature_extraction import extract_features_dataset, apply_lds, FREQ_BANDS
from src.utils.dataset import set_seed

# ── ĐƯỜNG DẪN & CẤU HÌNH ──
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_NAME = "svm"
LOG_DIR    = os.path.join(ROOT_DIR, "result", MODEL_NAME, "logs")
PLOT_DIR   = os.path.join(ROOT_DIR, "result", MODEL_NAME, "plots")

for d in [LOG_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── HYPERPARAMS ──
SEED = 42
USE_LDS = True
USE_GRIDSEARCH = True
MODE = "subject-independent"  # "subject-dependent" hoặc "subject-independent"

def evaluate_and_plot(y_test, y_pred, label_name, mode_name, cm_path):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    if "4class" in label_name.lower():
        names = ["Buon", "So hai", "Thu gian", "Vui ve"]
        cm_labels = [0, 1, 2, 3]
    else:
        names = ["Low", "High"]
        cm_labels = [0, 1]
        
    cm = confusion_matrix(y_test, y_pred, labels=cm_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'SVM Confusion Matrix ({label_name.upper()} - {mode_name})\nAcc: {acc*100:.2f}% | F1: {f1*100:.2f}%')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    
    return acc, f1, cm.tolist()

def print_feature_importance(global_weights_mean, is_multiclass=False):
    """Tính toán và in ra top 10 đặc trưng mạnh nhất từ trọng số trung bình."""
    bands = list(FREQ_BANDS.keys())
    n_bands = len(bands)
    channels = [
        'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
        'Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2', 'FCz', 'Cz'
    ]
    asym_pairs = [
        ('Fp1', 'Fp2'), ('AF3', 'AF4'), ('F3', 'F4'), ('F7', 'F8'), 
        ('FC5', 'FC6'), ('FC1', 'FC2'), ('C3', 'C4'), ('T7', 'T8'), 
        ('CP5', 'CP6'), ('CP1', 'CP2'), ('P3', 'P4'), ('P7', 'P8'), 
        ('PO3', 'PO4'), ('O1', 'O2')
    ]
    
    feature_names = []
    for ch in channels:
        for b in bands:
            feature_names.append(f"PSD_{ch}_{b}")
    for left, right in asym_pairs:
        for b in bands:
            feature_names.append(f"ASYM_{left}-{right}_{b}")
    for ch in channels:
        for b in bands:
            feature_names.append(f"DE_{ch}_{b}")

    if len(global_weights_mean) != len(feature_names):
        print(f"WARNING: Độ dài trọng số ({len(global_weights_mean)}) khác với feature names ({len(feature_names)})")
        return
        
    print("=" * 60)
    print("  ĐỘ QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG EEG (FEATURE IMPORTANCE)")
    print("=" * 60)
    print(f"  Tổng số features: {len(feature_names)}")
    print(f"  PSD: {len(channels) * n_bands} | Asymmetry: {len(asym_pairs) * n_bands} | DE: {len(channels) * n_bands}")
    
    if is_multiclass:
        # Đối với 4-class: Sắp xếp theo trị tuyệt đối biên độ đóng góp chung
        sorted_idx_desc = np.argsort(global_weights_mean)[::-1]
        print("\n[+] Top 10 features quan trọng nhất (Đóng góp phân loại chung cho 4-class):")
        for idx in sorted_idx_desc[:10]:
            print(f"  {feature_names[idx]:<30}: Độ lớn trọng số = {global_weights_mean[idx]:.4f}")
    else:
        # Đối với nhị phân: Giữ nguyên logic chia hướng âm/dương
        sorted_idx = np.argsort(global_weights_mean)
        print("\n[+] Top 10 đặc trưng ủng hộ nhãn Positive/High:")
        for idx in sorted_idx[-10:][::-1]:
            print(f"  {feature_names[idx]:<30}: Trọng số = {global_weights_mean[idx]:.4f}")

        print("\n[-] Top 10 đặc trưng ủng hộ nhãn Negative/Low:")
        for idx in sorted_idx[:10]:
            print(f"  {feature_names[idx]:<30}: Trọng số = {global_weights_mean[idx]:.4f}")
    print("=" * 60 + "\n")

def train_svm_pipeline(X_train, y_train):
    steps = [
        ('scaler', StandardScaler()),
        ('svm', LinearSVC(class_weight='balanced', max_iter=10000, random_state=SEED, dual='auto'))
    ]
    pipe = Pipeline(steps)
    
    if USE_GRIDSEARCH:
        # Giới hạn dải tìm kiếm C nhỏ để tối ưu hóa tốc độ và lọc nhiễu tốt hơn cho EEG
        grid = GridSearchCV(
            pipe, {'svm__C': [0.001, 0.01, 0.1, 1.0]},
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            scoring='f1_macro', n_jobs=-1, verbose=0, refit=True
        )
        grid.fit(X_train, y_train)
        best_c = grid.best_params_['svm__C']
        return grid.best_estimator_, best_c
    else:
        pipe.fit(X_train, y_train)
        return pipe, pipe.named_steps['svm'].C

def run_svm_pipeline(X_feat, y_all, groups, label_name):
    print(f"\n{'='*60}\n  SVM Pipeline — {label_name.upper()} — Mode: {MODE}\n{'='*60}")
    
    unique_groups = np.unique(groups)
    n_subjects = len(unique_groups)
    
    y_test_all, y_pred_all = [], []
    subject_weights = []
    subject_results = []
    
    t_start = time.time()
    is_multi = "4class" in label_name.lower()
    
    if MODE == "subject-independent":
        print(f"Bắt đầu LOSO CV trên {n_subjects} subjects...")
        for fold_i, test_group in enumerate(unique_groups):
            train_idx = (groups != test_group)
            test_idx = (groups == test_group)
            
            X_train, y_train = X_feat[train_idx], y_all[train_idx]
            X_test, y_test   = X_feat[test_idx], y_all[test_idx]
            
            # Kiểm tra Edge Case an toàn cho cả nhị phân và đa lớp bằng bincount
            train_class_counts = np.bincount(y_train)
            test_class_counts = np.bincount(y_test)
            
            if np.count_nonzero(test_class_counts) < 2:
                print(f"  [WARNING] Bỏ qua Subject {test_group + 1:02d}: Tập test không đủ 2 lớp để đánh giá.")
                continue
                
            min_samples_train = np.min(train_class_counts[train_class_counts > 0])
            if np.count_nonzero(train_class_counts) < 2 or min_samples_train < 5:
                print(f"  [WARNING] Bỏ qua Subject {test_group + 1:02d}: Tập train quá thiếu mẫu để chạy K-Fold nội bộ.")
                continue
                
            model, best_c = train_svm_pipeline(X_train, y_train)
            
            svm_step = model.named_steps['svm']
            if hasattr(svm_step, 'coef_'):
                if is_multi:
                    # Đa lớp OvR: Lấy trung bình trị tuyệt đối cường độ các mặt phẳng cắt
                    subject_weights.append(np.mean(np.abs(svm_step.coef_), axis=0))
                else:
                    subject_weights.append(svm_step.coef_[0])
                
            y_pred = model.predict(X_test)
            
            sub_num = int(test_group) + 1
            cm_path = os.path.join(PLOT_DIR, f"cm_svm_{MODE}_{label_name}_sub{sub_num:02d}.png")
            acc, f1, cm = evaluate_and_plot(y_test, y_pred, f"{label_name} - Sub {sub_num:02d}", MODE, cm_path)
            
            y_test_all.append(y_test)
            y_pred_all.append(y_pred)
            
            subject_results.append({
                'subject': sub_num,
                'acc': float(acc),
                'f1_macro': float(f1),
                'best_C': float(best_c),
                'confusion_matrix': cm,
                'n_test': len(y_test)
            })
            
            if sub_num % 5 == 0 or sub_num == n_subjects:
                print(f"  Đã xong subject {sub_num}/{n_subjects}...")
                
    else: # subject-dependent
        print(f"Bắt đầu huấn luyện mô hình riêng rẽ cho {n_subjects} subjects...")
        for sg in unique_groups:
            idx = (groups == sg)
            X_sub, y_sub = X_feat[idx], y_all[idx]
            
            # Kiểm tra Edge Case an toàn cho chế độ phụ thuộc người dùng
            sub_class_counts = np.bincount(y_sub)
            min_samples = np.min(sub_class_counts[sub_class_counts > 0])
            if np.count_nonzero(sub_class_counts) < 2 or min_samples < 6:
                print(f"  [WARNING] Bỏ qua Subject {sg + 1:02d}: Thiếu dữ liệu phân bố nhãn tối thiểu.")
                continue
                
            X_train, X_test, y_train, y_test = train_test_split(
                X_sub, y_sub, test_size=0.2, random_state=SEED, stratify=y_sub
            )
            
            model, best_c = train_svm_pipeline(X_train, y_train)
            
            svm_step = model.named_steps['svm']
            if hasattr(svm_step, 'coef_'):
                if is_multi:
                    subject_weights.append(np.mean(np.abs(svm_step.coef_), axis=0))
                else:
                    subject_weights.append(svm_step.coef_[0])
                
            y_pred = model.predict(X_test)
            
            sub_num = int(sg) + 1
            cm_path = os.path.join(PLOT_DIR, f"cm_svm_{MODE}_{label_name}_sub{sub_num:02d}.png")
            acc, f1, cm = evaluate_and_plot(y_test, y_pred, f"{label_name} - Sub {sub_num:02d}", MODE, cm_path)
            
            y_test_all.append(y_test)
            y_pred_all.append(y_pred)
            
            subject_results.append({
                'subject': sub_num,
                'acc': float(acc),
                'f1_macro': float(f1),
                'best_C': float(best_c),
                'confusion_matrix': cm,
                'n_test': len(y_test)
            })
            
            if sub_num % 5 == 0 or sub_num == n_subjects:
                print(f"  Đã xong subject {sub_num}/{n_subjects}...")
            
    print(f"Đã huấn luyện xong! Mất {time.time()-t_start:.1f}s.")
    
    y_test_combined = np.concatenate(y_test_all)
    y_pred_combined = np.concatenate(y_pred_all)
    
    cm_path = os.path.join(PLOT_DIR, f"cm_svm_{MODE}_{label_name}.png")
    acc_overall, f1_overall, cm_list = evaluate_and_plot(y_test_combined, y_pred_combined, label_name, MODE, cm_path)
    
    print(f"\n  📊 KẾT QUẢ TỔNG HỢP ({MODE}):")
    print(f"     Accuracy : {acc_overall:.4f}")
    print(f"     F1-macro : {f1_overall:.4f}")
    
    if subject_weights:
        global_weights_mean = np.mean(subject_weights, axis=0)
        print_feature_importance(global_weights_mean, is_multiclass=is_multi)
        
    result_data = {
        "experiment": f"Exp4_SVM_{MODE}",
        "model": "SVM_Linear",
        "label": label_name,
        "acc": acc_overall,
        "f1": f1_overall,
        "confusion_matrix": cm_list,
        "subject_results": subject_results
    }
    
    log_path = os.path.join(LOG_DIR, f"svm_{MODE}_{label_name}.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

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
    
    if USE_LDS:
        print("Đang áp dụng LDS Smoothing...")
        X_feat = apply_lds(X_feat)
        
    # Chạy kịch bản nhị phân chuẩn hóa
    #run_svm_pipeline(X_feat, y_valence, subject_groups, "valence")
    #run_svm_pipeline(X_feat, y_arousal, subject_groups, "arousal")
    
    # Kiểm tra kiểm định phân bổ nhãn cứng ngắc trước khi map 4-class
    assert set(np.unique(y_valence)) <= {0, 1}, f"y_valence sai định dạng: {np.unique(y_valence)}"
    assert set(np.unique(y_arousal)) <= {0, 1}, f"y_arousal sai định dạng: {np.unique(y_arousal)}"
    
    y_4class = y_valence * 2 + y_arousal
    print(f"\nPhân bố nhãn toàn tập 4-class: {np.bincount(y_4class)}")
    
    # Chạy kịch bản đa lớp (4 góc phần tư cảm xúc)
    run_svm_pipeline(X_feat, y_4class, subject_groups, "4class")
    
    print("\n✅ HOÀN TẤT EXPERIMENT 4 (SVM)")