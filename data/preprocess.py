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

def preprocess_subject(file_path):
    """Tiền xử lý toàn diện dữ liệu của 1 Subject."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    raw_eeg = data['data'][:, :32, :] 
    raw_labels = data['labels']       
    
    # 1. Binarize nhãn cảm xúc theo ngưỡng 5.0
    y_valence = (raw_labels[:, 0] > 5.0).astype(int)
    y_arousal = (raw_labels[:, 1] > 5.0).astype(int)
    
    # 2. Baseline Subtraction
    baseline = raw_eeg[:, :, :384] 
    stimulus = raw_eeg[:, :, 384:] 
    baseline_mean = np.mean(baseline, axis=2, keepdims=True)
    
    # ÉP KIỂU FLOAT32 ĐỂ CỨU RAM CỦA BẠN (Từ 2.5GB xuống 1.2GB)
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
    """Gọi SAU KHI băm Train/Test. 'flatten' cho SVM, 'channel' cho DL."""
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
        X_te_scaled = scaler.transform(X_te_2d).reshape(n_test, n_t, n_ch).transpose(0, 2, 1).copy()
    else:
        raise ValueError("mode='flatten' hoặc 'channel'")
        
    # Ép kiểu lại float32 sau khi scaler (sklearn thường trả về float64)
    return X_tr_scaled.astype(np.float32), X_te_scaled.astype(np.float32), scaler

def get_dynamic_class_weights(y_train_fold):
    """Tính trọng số động dựa trên tập train."""
    classes = np.unique(y_train_fold)
    if len(classes) == 1:
        warnings.warn(f"⚠️ Fold chỉ có 1 class (class={classes[0]}).", RuntimeWarning)
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
        
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_fold)
    return torch.tensor(weights, dtype=torch.float32)

if __name__ == "__main__":
    # Tự động trỏ đến đúng thư mục chứa script này
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "raw")
    OUTPUT_DIR = os.path.join(BASE_DIR, "processed")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("🚀 ĐANG XỬ LÝ DỮ LIỆU DEAP THẬT...")
    print("-" * 88)
    print(f"{'Sub':<6}{'Valence (Trials)':<18}{'Valence (Epochs)':<22}{'Arousal (Trials)':<18}{'Arousal (Epochs)':<20}")
    print("-" * 88)
    
    all_X, all_y_val, all_y_aro, all_groups = [], [], [], []
    processed_count = 0
    
    for sub_id in range(1, 33):
        file_path = os.path.join(DATA_DIR, f"s{sub_id:02d}.dat")
        if not os.path.exists(file_path):
            print(f"❌ Không tìm thấy {file_path}. Vui lòng chạy script tải dữ liệu trước!")
            continue
            
        X_sub, y_val_sub, y_aro_sub, y_val_raw, y_aro_raw = preprocess_subject(file_path)
        
        v_trial = np.bincount(y_val_raw, minlength=2)
        v_epoch = np.bincount(y_val_sub, minlength=2)
        a_trial = np.bincount(y_aro_raw, minlength=2)
        a_epoch = np.bincount(y_aro_sub, minlength=2)
        
        print(f"S{sub_id:02d}  {str(v_trial):<18}{str(v_epoch):<22}{str(a_trial):<18}{str(a_epoch):<20}")
        
        all_X.append(X_sub)
        all_y_val.append(y_val_sub)
        all_y_aro.append(y_aro_sub)
        all_groups.append(np.full(len(y_val_sub), sub_id - 1)) 
        processed_count += 1
        
    if processed_count > 0:
        print("\n⏳ Đang ghép mảng và giải phóng bộ nhớ (tránh OOM)...")
        final_X = np.concatenate(all_X, axis=0, dtype=np.float32)
        
        # Dọn sạch RAM ngay lập tức
        del all_X 
        gc.collect() 
        
        final_y_val = np.concatenate(all_y_val)
        final_y_aro = np.concatenate(all_y_aro)
        final_groups = np.concatenate(all_groups)
        
        print(f"✅ Đã đóng gói xong X_shape={final_X.shape}. Đang lưu ra file .npy...")
        np.save(os.path.join(OUTPUT_DIR, "X_epochs.npy"), final_X)
        np.save(os.path.join(OUTPUT_DIR, "y_valence.npy"), final_y_val)
        np.save(os.path.join(OUTPUT_DIR, "y_arousal.npy"), final_y_aro)
        np.save(os.path.join(OUTPUT_DIR, "subject_groups.npy"), final_groups)
        
        print(f"🎉 HOÀN TẤT! Dữ liệu đã lưu tại: {OUTPUT_DIR}")
        print("Bây giờ bạn có thể ném cho TV1, TV2, TV3 xài được rồi!")