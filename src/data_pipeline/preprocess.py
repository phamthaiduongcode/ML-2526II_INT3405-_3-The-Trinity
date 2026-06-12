# File: src/data_pipeline/preprocess.py
import os
import pickle
import gc
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

def verify_epoch_integrity(eeg_normalized, X):
    """[UNIT TEST] Đảm bảo phép biến đổi không làm xáo trộn thời gian/kênh."""
    assert np.allclose(eeg_normalized[0, 0, :128],  X[0,  0, :]), "❌ Epoch 0 Trial 0 sai!"
    assert np.allclose(eeg_normalized[0, 0, 128:256], X[1, 0, :]), "❌ Epoch 1 Trial 0 sai!"
    assert np.allclose(eeg_normalized[1, 0, :128],   X[60, 0, :]), "❌ Ranh giới Trial 0->1 trộn!"
    assert np.allclose(eeg_normalized[0, 5, :128],   X[0,  5, :]), "❌ Channel 5 bị trộn!"

def preprocess_subject(file_path):
    """Tiền xử lý 1 Subject với ngưỡng Subject-Dependent Median (HỢP LỆ LOSO)."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    raw_eeg    = data['data'][:, :32, :]
    raw_labels = data['labels']

    # ✅ Dùng độc lập Subject-dependent median để gán nhãn
    subject_median_val = np.median(raw_labels[:, 0])
    subject_median_aro = np.median(raw_labels[:, 1])

    y_valence = (raw_labels[:, 0] >= subject_median_val).astype(int)
    y_arousal = (raw_labels[:, 1] >= subject_median_aro).astype(int)

    # Baseline Subtraction
    baseline      = raw_eeg[:, :, :384]
    stimulus      = raw_eeg[:, :, 384:]
    baseline_mean = np.mean(baseline, axis=2, keepdims=True)
    eeg_normalized = (stimulus - baseline_mean).astype(np.float32)

    # Phân đoạn 1 giây không chồng lấp
    epochs = eeg_normalized.reshape(40, 32, 60, 128)
    epochs = epochs.transpose(0, 2, 1, 3).copy()
    X      = epochs.reshape(-1, 32, 128)

    verify_epoch_integrity(eeg_normalized, X)

    y_val_expanded = np.repeat(y_valence, 60)
    y_aro_expanded = np.repeat(y_arousal, 60)

    return X, y_val_expanded, y_aro_expanded


def normalize_after_split(X_train, X_test, mode='channel'):
    """
    Chuẩn hóa dữ liệu SAU KHI chia Train/Test — tránh Data Leakage.
    mode: 'channel'| 'flatten'
    """
    n_train, n_ch, n_t = X_train.shape
    n_test = X_test.shape[0]
    scaler = StandardScaler()

    if mode == 'flatten':
        X_tr_2d = X_train.reshape(n_train, -1)
        X_te_2d = X_test.reshape(n_test, -1)
        scaler.fit(X_tr_2d)
        
        X_tr_scaled = scaler.transform(X_tr_2d).reshape(n_train, n_ch, n_t)
        X_te_scaled = scaler.transform(X_te_2d).reshape(n_test, n_ch, n_t)
        
        if X_val is not None:
            X_va_2d = X_val.reshape(X_val.shape[0], -1)
            X_va_scaled = scaler.transform(X_va_2d).reshape(X_val.shape[0], n_ch, n_t)
            return X_tr_scaled.astype(np.float32), X_te_scaled.astype(np.float32), X_va_scaled.astype(np.float32)
        return X_tr_scaled.astype(np.float32), X_te_scaled.astype(np.float32)

    elif mode == 'channel':
        X_tr_2d = X_train.transpose(0, 2, 1).reshape(-1, n_ch)
        X_te_2d = X_test.transpose(0, 2, 1).reshape(-1, n_ch)
        scaler.fit(X_tr_2d)
        
        X_tr_scaled = scaler.transform(X_tr_2d).reshape(n_train, n_t, n_ch).transpose(0, 2, 1).copy()
        X_te_scaled = scaler.transform(X_te_2d).reshape(n_test, n_t, n_ch).transpose(0, 2, 1).copy()
        
        if X_val is not None:
            X_va_2d = X_val.transpose(0, 2, 1).reshape(-1, n_ch)
            X_va_scaled = scaler.transform(X_va_2d).reshape(X_val.shape[0], n_t, n_ch).transpose(0, 2, 1).copy()
            return X_tr_scaled.astype(np.float32), X_te_scaled.astype(np.float32), X_va_scaled.astype(np.float32)
        return X_tr_scaled.astype(np.float32), X_te_scaled.astype(np.float32)
    else:
        raise ValueError("mode phải là 'flatten' hoặc 'channel'")

def get_dynamic_class_weights(y_train_fold, n_classes=2, max_weight=2.0):
    """Tính trọng số động chống mất cân bằng lớp khi train fold."""
    classes = np.unique(y_train_fold)
    if len(classes) == 1:
        return torch.ones(n_classes, dtype=torch.float32)

    weights_partial = compute_class_weight('balanced', classes=classes, y=y_train_fold)
    weights_partial = np.clip(weights_partial, 0, max_weight)

    weights = torch.zeros(n_classes, dtype=torch.float32)
    for i, c in enumerate(classes):
        weights[c] = weights_partial[i]
    return weights

if __name__ == "__main__":
    ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR   = os.path.join(ROOT_DIR, "data", "raw")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🚀 ĐANG TIỀN XỬ LÝ (SUBJECT-DEPENDENT MEDIAN)...")
    all_X, all_y_val, all_y_aro, all_groups = [], [], [], []

    for sub_id in range(1, 33):
        file_path = os.path.join(DATA_DIR, f"s{sub_id:02d}.dat")
        if not os.path.exists(file_path): continue
        
        X_sub, y_val_sub, y_aro_sub = preprocess_subject(file_path)
        
        all_X.append(X_sub)
        all_y_val.append(y_val_sub)
        all_y_aro.append(y_aro_sub)
        all_groups.append(np.full(len(y_val_sub), sub_id - 1))

    print("\n⏳ Đang ghép mảng và lưu file...")
    final_X = np.concatenate(all_X, axis=0, dtype=np.float32)
    del all_X; gc.collect()
    
    np.save(os.path.join(OUTPUT_DIR, "X_epochs.npy"),       final_X)
    np.save(os.path.join(OUTPUT_DIR, "y_valence.npy"),      np.concatenate(all_y_val))
    np.save(os.path.join(OUTPUT_DIR, "y_arousal.npy"),      np.concatenate(all_y_aro))
    np.save(os.path.join(OUTPUT_DIR, "subject_groups.npy"), np.concatenate(all_groups))
    print(f"✅ Hoàn tất! Shape X={final_X.shape}")