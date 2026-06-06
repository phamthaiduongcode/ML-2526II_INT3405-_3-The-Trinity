import os
import pickle
import warnings
import gc
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

def verify_epoch_integrity(eeg_normalized, X):
    """[UNIT TEST] Đảm bảo phép biến đổi không làm xáo trộn thời gian hoặc kênh tín hiệu."""
    assert np.allclose(eeg_normalized[0, 0, :128], X[0, 0, :]), "❌ Epoch 0 Trial 0 bị sai!"
    assert np.allclose(eeg_normalized[0, 0, 128:256], X[1, 0, :]), "❌ Epoch 1 Trial 0 bị sai!"
    assert np.allclose(eeg_normalized[1, 0, :128], X[60, 0, :]), "❌ Ranh giới Trial 0->1 bị trộn!"
    assert np.allclose(eeg_normalized[0, 5, :128], X[0, 5, :]), "❌ Channel 5 bị trộn sang channel khác!"

def preprocess_subject(file_path, global_median_val, global_median_aro):
    """Tiền xử lý toàn diện dữ liệu của 1 Subject với ngưỡng Global Median."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    raw_eeg = data['data'][:, :32, :] 
    raw_labels = data['labels']       
    
    # 1. Binarize nhãn cảm xúc theo GLOBAL MEDIAN
    y_valence = (raw_labels[:, 0] >= global_median_val).astype(int)
    y_arousal = (raw_labels[:, 1] >= global_median_aro).astype(int)
    
    # 2. Baseline Subtraction
    baseline = raw_eeg[:, :, :384] 
    stimulus = raw_eeg[:, :, 384:] 
    baseline_mean = np.mean(baseline, axis=2, keepdims=True)
    
    # ÉP KIỂU FLOAT32 ĐỂ CỨU RAM
    eeg_normalized = (stimulus - baseline_mean).astype(np.float32)
    
    # 3. Phân đoạn Epochs (1 giây không chồng lấp)
    epochs = eeg_normalized.reshape(40, 32, 60, 128)
    epochs = epochs.transpose(0, 2, 1, 3).copy() 
    X = epochs.reshape(-1, 32, 128)              
    
    verify_epoch_integrity(eeg_normalized, X)
    
    # 4. Nhân bản nhãn
    y_val_expanded = np.repeat(y_valence, 60)
    y_aro_expanded = np.repeat(y_arousal, 60)
    
    return X, y_val_expanded, y_aro_expanded, y_valence, y_arousal

def normalize_after_split(X_train, X_test, mode='channel'):
    """Gọi SAU KHI băm Train/Test."""
    n_train, n_ch, n_t = X_train.shape
    n_test = X_test.shape[0]
    scaler = StandardScaler()

    if mode == 'flatten':
        X_tr_2d = X_train.reshape(n_train, -1)
        X_te_2d = X_test.reshape(n_test, -1)
        X_tr_scaled = scaler.fit_transform(X_tr_2d).reshape(n_train, n_ch, n_t)
        X_te_scaled = scaler.transform(X_te_2d).reshape(n_test, n_ch, n_t)
    elif mode == 'channel':
        X_tr_2d = X_train.transpose(0, 2, 1).reshape(-1, n_ch)
        X_te_2d = X_test.transpose(0, 2, 1).reshape(-1, n_ch)
        X_tr_scaled = scaler.fit_transform(X_tr_2d).reshape(n_train, n_t, n_ch).transpose(0, 2, 1).copy()
        # Đã FIX lỗi thiếu "scaler."
        X_te_scaled = scaler.transform(X_te_2d).reshape(n_test, n_t, n_ch).transpose(0, 2, 1).copy()
    else:
        raise ValueError("mode='flatten' hoặc 'channel'")
        
    return X_tr_scaled.astype(np.float32), X_te_scaled.astype(np.float32), scaler

def get_dynamic_class_weights(y_train_fold, n_classes=2, max_weight=5.0):
    """Tính trọng số động, hỗ trợ n_classes và chặn bùng nổ trọng số."""
    classes = np.unique(y_train_fold)
    
    # Nếu fold hiện tại vô tình chỉ có 1 class
    if len(classes) == 1:
        return torch.ones(n_classes, dtype=torch.float32)
        
    # Tính toán weight nguyên bản
    weights_partial = compute_class_weight(
        class_weight='balanced', classes=classes, y=y_train_fold
    )
    
    # Cắt ngọn (clip) để tránh weight vọt lên mười mấy lần gây nhiễu gradient
    weights_partial = np.clip(weights_partial, 0, max_weight)
    
    # Map vào tensor độ dài chuẩn (n_classes)
    weights = torch.zeros(n_classes, dtype=torch.float32)
    for i, c in enumerate(classes):
        weights[c] = weights_partial[i]
        
    return weights

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "raw")
    OUTPUT_DIR = os.path.join(BASE_DIR, "processed")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("🚀 ĐANG XỬ LÝ DỮ LIỆU DEAP (TWO-PASS GLOBAL MEDIAN)...")
    
    # ==========================================
    # PASS 1: Đọc nhãn toàn cục để tìm Global Median
    # ==========================================
    all_val_scores, all_aro_scores = [], []
    for sub_id in range(1, 33):
        file_path = os.path.join(DATA_DIR, f"s{sub_id:02d}.dat")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            all_val_scores.extend(data['labels'][:, 0])
            all_aro_scores.extend(data['labels'][:, 1])
            
    global_median_val = np.median(all_val_scores)
    global_median_aro = np.median(all_aro_scores)
    
    print(f"📊 Global Median Valence: {global_median_val:.2f}")
    print(f"📊 Global Median Arousal: {global_median_aro:.2f}")
    print("-" * 88)
    
    # ==========================================
    # PASS 2: Tiền xử lý với ngưỡng Global
    # ==========================================
    all_X, all_y_val, all_y_aro, all_groups = [], [], [], []
    processed_count = 0
    
    for sub_id in range(1, 33):
        file_path = os.path.join(DATA_DIR, f"s{sub_id:02d}.dat")
        if not os.path.exists(file_path):
            continue
            
        X_sub, y_val_sub, y_aro_sub, y_val_raw, y_aro_raw = preprocess_subject(
            file_path, global_median_val, global_median_aro
        )
        
        all_X.append(X_sub)
        all_y_val.append(y_val_sub)
        all_y_aro.append(y_aro_sub)
        all_groups.append(np.full(len(y_val_sub), sub_id - 1)) 
        processed_count += 1
        
    if processed_count > 0:
        print("\n⏳ Đang ghép mảng và lưu file...")
        final_X = np.concatenate(all_X, axis=0, dtype=np.float32)
        del all_X; gc.collect() 
        
        final_y_val = np.concatenate(all_y_val)
        final_y_aro = np.concatenate(all_y_aro)
        final_groups = np.concatenate(all_groups)
        
        np.save(os.path.join(OUTPUT_DIR, "X_epochs.npy"), final_X)
        np.save(os.path.join(OUTPUT_DIR, "y_valence.npy"), final_y_val)
        np.save(os.path.join(OUTPUT_DIR, "y_arousal.npy"), final_y_aro)
        np.save(os.path.join(OUTPUT_DIR, "subject_groups.npy"), final_groups)
        
        print(f"✅ Đã đóng gói xong: X={final_X.shape}. Đã lưu tại: {OUTPUT_DIR}")