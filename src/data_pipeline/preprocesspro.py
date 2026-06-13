import numpy as np
from scipy.linalg import fractional_matrix_power
from sklearn.preprocessing import StandardScaler

def apply_euclidean_alignment(X, groups):
    """Làm phẳng không gian Riemann để triệt tiêu vân tay sinh lý nền."""
    print("   [EA] Tiến hành làm phẳng hình học (Euclidean Alignment) per-subject...")
    X_aligned = np.zeros_like(X, dtype=np.float32)
    unique_subs = np.unique(groups)
    
    for sub in unique_subs:
        idx = np.where(groups == sub)[0]
        X_sub = X[idx]
        
        # FIX: Tính hiệp phương sai trên toàn bộ dữ liệu của subject thay vì trung bình các epoch
        # Reshape X_sub: (N, C, T) -> (C, N*T) để np.cov tính covariance chuẩn xác
        C = X_sub.shape[1]
        X_sub_flat = X_sub.transpose(1, 0, 2).reshape(C, -1)
        R_mean = np.cov(X_sub_flat)
        
        R_inv_sqrt = fractional_matrix_power(R_mean, -0.5).real
        for count, original_idx in enumerate(idx):
            X_aligned[original_idx] = R_inv_sqrt @ X_sub[count]
            
    return X_aligned

def normalize_after_split(X_train, X_test, X_val=None, mode='channel'):
    """Chuẩn hóa Z-score tín hiệu thô dọc theo trục thời gian (Time-axis)."""
    N_tr, C, T = X_train.shape
    N_te, _, _ = X_test.shape
    
    X_train_flat = X_train.transpose(0, 2, 1).reshape(-1, C)
    X_test_flat = X_test.transpose(0, 2, 1).reshape(-1, C)
    
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    X_train_norm = X_train_flat.reshape(N_tr, T, C).transpose(0, 2, 1)
    X_test_norm = X_test_flat.reshape(N_te, T, C).transpose(0, 2, 1)
    
    # FIX: Trả về Dictionary để chống nhầm lẫn unpack (Data Leakage)
    result = {'train': X_train_norm, 'test': X_test_norm}
    
    if X_val is not None:
        N_va = X_val.shape[0]
        X_val_flat = X_val.transpose(0, 2, 1).reshape(-1, C)
        X_val_flat = scaler.transform(X_val_flat)
        X_val_norm = X_val_flat.reshape(N_va, T, C).transpose(0, 2, 1)
        result['val'] = X_val_norm
        
    return result


# Thêm vào cuối src/data_pipeline/preprocess.py

class EEGAugmentor:
    """
    Augmentation nhẹ cho EEG raw signal.
    Chỉ áp dụng lúc train, KHÔNG dùng cho val/test.
    
    Thiết kế cho DEAP: 32 channels, 128Hz, epoch 4s (512 samples).
    """
    def __init__(self,
                 noise_std: float = 0.01,    # σ Gaussian noise — 1% của std signal
                 mask_ratio: float = 0.10,   # 10% time steps bị zero-out
                 channel_drop_p: float = 0.05, # 5% xác suất drop 1 channel
                 apply_p: float = 0.80):     # 80% sample được augment
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio
        self.channel_drop_p = channel_drop_p
        self.apply_p = apply_p

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, C, T) float32
        Returns: (N, C, T) float32, augmented
        """
        X = X.copy()
        N, C, T = X.shape
        
        for i in range(N):
            if np.random.rand() > self.apply_p:
                continue
            
            # 1. Gaussian noise — tỉ lệ với std của chính sample đó
            if self.noise_std > 0:
                sample_std = X[i].std() + 1e-8
                noise = np.random.randn(C, T).astype(np.float32)
                X[i] += noise * self.noise_std * sample_std
            
            # 2. Time masking — zero-out một đoạn liên tục ngẫu nhiên
            if self.mask_ratio > 0:
                mask_len = int(T * self.mask_ratio)
                start = np.random.randint(0, T - mask_len)
                X[i, :, start:start + mask_len] = 0.0
            
            # 3. Channel dropout — zero-out 1 channel ngẫu nhiên (xác suất thấp)
            if self.channel_drop_p > 0 and np.random.rand() < self.channel_drop_p:
                ch = np.random.randint(0, C)
                X[i, ch, :] = 0.0
        
        return X