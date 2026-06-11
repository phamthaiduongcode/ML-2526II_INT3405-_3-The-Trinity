import numpy as np
from scipy.linalg import fractional_matrix_power
from sklearn.preprocessing import StandardScaler

def apply_euclidean_alignment(X, groups):
    """
    Áp dụng Euclidean Alignment (EA) để đồng bộ hóa không gian hình học giữa các Subject.
    X shape: (N, n_ch, T) - Tín hiệu EEG thô
    groups shape: (N,) - Mảng chứa ID của Subject dạng số nguyên
    """
    print("   [EA] Tiến hành loại bỏ phông nền sinh lý (Fingerprint) bằng Euclidean Alignment...")
    X_aligned = np.zeros_like(X, dtype=np.float32)
    unique_subs = np.unique(groups)
    
    for sub in unique_subs:
        idx = np.where(groups == sub)[0]
        X_sub = X[idx]
        
        # Tính ma trận hiệp phương sai trung bình của cá thể hiện tại
        cov_sum = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
        for i in range(len(X_sub)):
            cov_sum += np.cov(X_sub[i])
        R_mean = cov_sum / len(X_sub)
        
        # Tính ma trận làm phẳng (Whitening Matrix) R^(-0.5)
        R_inv_sqrt = fractional_matrix_power(R_mean, -0.5).real
        
        # Biến đổi tuyến tính đưa ma trận hiệp phương sai về Identity Matrix (I)
        for count, original_idx in enumerate(idx):
            X_aligned[original_idx] = R_inv_sqrt @ X_sub[count]
            
    print("   [EA] Hoàn tất căn chỉnh hình học không gian per-subject.")
    return X_aligned

def normalize_after_split(X_train, X_test, X_val=None, mode='channel'):
    """
    Chuẩn hóa dữ liệu sau khi chia tập Train/Test/Val.
    mode='channel': Transpose về (N * T, C) để fit per-channel chuẩn baseline gốc.
    """
    N_tr, C, T = X_train.shape
    N_te, _, _ = X_test.shape
    
    # Ép phẳng đưa về dạng (N * T, C) giúp tính toán chuẩn hóa trên từng kênh (Channel-wise)
    X_train_flat = X_train.transpose(0, 2, 1).reshape(-1, C)
    X_test_flat = X_test.transpose(0, 2, 1).reshape(-1, C)
    
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    # Khôi phục lại cấu trúc Tensor gốc (N, C, T)
    X_train_norm = X_train_flat.reshape(N_tr, T, C).transpose(0, 2, 1)
    X_test_norm = X_test_flat.reshape(N_te, T, C).transpose(0, 2, 1)
    
    if X_val is not None:
        N_va = X_val.shape[0]
        X_val_flat = X_val.transpose(0, 2, 1).reshape(-1, C)
        X_val_flat = scaler.transform(X_val_flat)
        X_val_norm = X_val_flat.reshape(N_va, T, C).transpose(0, 2, 1)
        return X_train_norm, X_test_norm, X_val_norm
        
    return X_train_norm, X_test_norm

def get_dynamic_class_weights(y):
    """
    Tính toán trọng số động cho các lớp dựa trên phân phối nhãn của tập dữ liệu hiện tại.
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = total / (len(classes) * counts)
    return weights.astype(np.float32)