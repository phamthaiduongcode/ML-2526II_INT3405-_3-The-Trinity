"""
experiments/svm/svm_utils.py
Các hàm dùng chung cho pipeline SVM: plotting, feature importance, LDS, normalization, training.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.data_pipeline.feature_extraction import apply_lds, FREQ_BANDS

# ── HYPERPARAMS MẶC ĐỊNH ──
SEED = 42
USE_LDS = True
USE_GRIDSEARCH = True
EPOCHS_PER_VIDEO = 60


# ══════════════════════════════════════════════════════════════════════════════
# LDS & NORMALIZATION (Chống Data Leakage)
# ══════════════════════════════════════════════════════════════════════════════

def apply_lds_safe(X_partition, epochs_per_video=EPOCHS_PER_VIDEO, alpha=0.3, original_indices=None):
    """Áp dụng LDS smoothing CHỈ trên 1 partition (train hoặc test) riêng biệt.
    
    Smooth theo từng video (mỗi 60 epochs liên tiếp), không smooth xuyên biên train/test.
    Nếu original_indices được cung cấp, nhóm theo video dựa trên index gốc (cần cho LOSO).
    """
    return apply_lds(X_partition, epochs_per_trial=epochs_per_video, alpha=alpha, original_indices=original_indices)


def apply_per_subject_normalization_safe(X_train, X_test, groups_train, groups_test):
    """Chuẩn hóa Z-score (StandardScaler) riêng cho từng subject.
    
    - Fit trên X_train, transform cả X_train và X_test → tránh Data Leakage.
    - Dùng StandardScaler thay MinMaxScaler → bảo toàn margin geometry cho SVM.
    """
    X_train_norm = np.zeros_like(X_train)
    X_test_norm = np.zeros_like(X_test)
    
    unique_train = np.unique(groups_train)
    unique_test = np.unique(groups_test)
    
    # Fit scaler cho mỗi subject trong train
    subject_scalers = {}
    for g in unique_train:
        idx = (groups_train == g)
        scaler = StandardScaler()
        X_train_norm[idx] = scaler.fit_transform(X_train[idx])
        subject_scalers[g] = scaler
    
    # Transform test
    for g in unique_test:
        idx = (groups_test == g)
        if g in subject_scalers:
            X_test_norm[idx] = subject_scalers[g].transform(X_test[idx])
        else:
            # Subject mới (LOSO): fit scaler mới trên test data
            # (mục đích là triệt tiêu subject-level shift, không phải học từ data)
            scaler = StandardScaler()
            X_test_norm[idx] = scaler.fit_transform(X_test[idx])
    
    return X_train_norm, X_test_norm


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_and_plot(y_test, y_pred, label_name, mode_name, cm_path):
    """Tính metrics và vẽ confusion matrix."""
    acc = accuracy_score(y_test, y_pred)
    
    if "4class" in label_name.lower():
        # Mapping theo Russell's Circumplex Model:
        # 0 = LV+LA (Sad) | 1 = LV+HA (Fear) | 2 = HV+LA (Relaxed) | 3 = HV+HA (Happy)
        names = ["Buon", "So hai", "Thu gian", "Vui ve"]
        cm_labels = [0, 1, 2, 3]
    else:
        names = ["Low", "High"]
        cm_labels = [0, 1]
        
    f1 = f1_score(y_test, y_pred, labels=cm_labels, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=cm_labels)
    
    if cm_path is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=names, yticklabels=names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'SVM Confusion Matrix ({label_name.upper()} - {mode_name})\nAcc: {acc*100:.2f}% | F1: {f1*100:.2f}%')
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        plt.close()
    
    return acc, f1, cm


def plot_aggregated_cm(cm, acc, f1, label_name, mode_name, cm_path):
    """Vẽ confusion matrix tổng hợp."""
    if cm_path is None:
        return
    if "4class" in label_name.lower():
        names = ["Buon", "So hai", "Thu gian", "Vui ve"]
    else:
        names = ["Low", "High"]
        
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'SVM Confusion Matrix ({label_name.upper()} - {mode_name})\nAcc: {acc*100:.2f}% | F1: {f1*100:.2f}%')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def get_feature_names():
    """Tạo danh sách 312 tên features theo thứ tự: PSD (128) + Asymmetry (56) + DE (128)."""
    bands = list(FREQ_BANDS.keys())
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
    return feature_names, channels, asym_pairs, bands


def print_feature_importance(subject_weights, is_multiclass=False):
    """Tính toán và in ra top 10 đặc trưng mạnh nhất."""
    feature_names, channels, asym_pairs, bands = get_feature_names()
    n_bands = len(bands)
    
    abs_weights_mean = np.mean(np.abs(subject_weights), axis=0)
    mean_weights = np.mean(subject_weights, axis=0)

    if len(abs_weights_mean) != len(feature_names):
        print(f"WARNING: Độ dài trọng số ({len(abs_weights_mean)}) khác với feature names ({len(feature_names)})")
        return
        
    print("=" * 60)
    print("  ĐỘ QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG EEG (DỰA TRÊN TRUNG BÌNH TRỊ TUYỆT ĐỐI)")
    print("=" * 60)
    print(f"  Tổng số features: {len(feature_names)}")
    print(f"  PSD: {len(channels) * n_bands} | Asymmetry: {len(asym_pairs) * n_bands} | DE: {len(channels) * n_bands}")
    
    top_features = {}
    if is_multiclass:
        sorted_idx_desc = np.argsort(abs_weights_mean)[::-1]
        print("\n[+] Top 10 features quan trọng nhất (Đóng góp phân loại chung cho 4-class):")
        top_features['top_10_overall'] = []
        for idx in sorted_idx_desc[:10]:
            print(f"  {feature_names[idx]:<30}: Độ lớn trọng số (Abs) = {abs_weights_mean[idx]:.4f}")
            top_features['top_10_overall'].append({"feature": feature_names[idx], "weight": float(abs_weights_mean[idx])})
    else:
        sorted_idx_desc = np.argsort(abs_weights_mean)[::-1]
        
        pos_features = []
        neg_features = []
        for idx in sorted_idx_desc:
            if mean_weights[idx] > 0:
                pos_features.append(idx)
            else:
                neg_features.append(idx)

        print("\n[+] Top 10 đặc trưng quan trọng NHẤT (Ủng hộ nhãn Positive/High):")
        top_features['top_10_positive'] = []
        for idx in pos_features[:10]:
            print(f"  {feature_names[idx]:<30}: Độ lớn (Abs) = {abs_weights_mean[idx]:.4f} | Xu hướng: Dương")
            top_features['top_10_positive'].append({"feature": feature_names[idx], "weight": float(abs_weights_mean[idx])})

        print("\n[-] Top 10 đặc trưng quan trọng NHẤT (Ủng hộ nhãn Negative/Low):")
        top_features['top_10_negative'] = []
        for idx in neg_features[:10]:
            print(f"  {feature_names[idx]:<30}: Độ lớn (Abs) = {abs_weights_mean[idx]:.4f} | Xu hướng: Âm")
            top_features['top_10_negative'].append({"feature": feature_names[idx], "weight": float(-abs_weights_mean[idx])})
    print("=" * 60 + "\n")
    return top_features


# ══════════════════════════════════════════════════════════════════════════════
# SVM TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def train_svm_pipeline(X_train, y_train, fold_seed=SEED, use_gridsearch=USE_GRIDSEARCH):
    """Huấn luyện SVM Linear với GridSearchCV hoặc default C."""
    steps = [
        ('global_scaler', StandardScaler()),
        ('svm', LinearSVC(penalty='l2', dual=False, class_weight='balanced', max_iter=20000, random_state=fold_seed))
    ]
    pipe = Pipeline(steps)
    
    if use_gridsearch:
        grid = GridSearchCV(
            pipe, {'svm__C': [0.001, 0.01, 0.1, 1.0]},
            cv=StratifiedKFold(5, shuffle=True, random_state=fold_seed),
            scoring='f1_macro', n_jobs=-1, verbose=0, refit=True
        )
        grid.fit(X_train, y_train)
        best_c = grid.best_params_['svm__C']
        return grid.best_estimator_, best_c
    else:
        pipe.fit(X_train, y_train)
        return pipe, pipe.named_steps['svm'].C


def summarize_results(subject_results, subject_weights, label_name, mode_name, 
                      is_multi, log_dir, plot_dir):
    """Tổng hợp kết quả per-subject, vẽ Overall CM, in feature importance, lưu JSON."""
    sub_accs = [res['acc'] for res in subject_results]
    sub_f1s = [res['f1_macro'] for res in subject_results]
    
    mean_acc = np.mean(sub_accs)
    std_acc = np.std(sub_accs)
    mean_f1 = np.mean(sub_f1s)
    std_f1 = np.std(sub_f1s)
    
    cm_list = [res['confusion_matrix'] for res in subject_results]
    sum_cm = np.sum(cm_list, axis=0)
    
    cm_path = os.path.join(plot_dir, f"cm_svm_{mode_name}_{label_name}_Overall.png")
    plot_aggregated_cm(sum_cm, mean_acc, mean_f1, f"{label_name} - Overall", mode_name, cm_path)
    
    print(f"\n  📊 KẾT QUẢ TRUNG BÌNH PER-SUBJECT ({mode_name}):")
    print(f"     Accuracy : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"     F1-macro : {mean_f1:.4f} ± {std_f1:.4f}")
    
    top_features = {}
    if subject_weights:
        top_features = print_feature_importance(subject_weights, is_multiclass=is_multi)
    
    result_data = {
        "experiment": f"SVM_{mode_name}",
        "model": "SVM_Linear",
        "label": label_name,
        "acc": float(mean_acc),
        "acc_std": float(std_acc),
        "f1": float(mean_f1),
        "f1_std": float(std_f1),
        "confusion_matrix": cm_list,
        "top_features": top_features,
        "subject_results": subject_results
    }
    
    log_path = os.path.join(log_dir, f"svm_{mode_name}_{label_name}.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    return result_data
