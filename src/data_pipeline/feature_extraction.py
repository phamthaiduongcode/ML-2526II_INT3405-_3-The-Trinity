import numpy as np
from scipy.signal import welch, butter, sosfilt

FREQ_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45),
}

ASYMMETRY_PAIRS = [
    (0, 16), (1, 17), (2, 18), (3, 19), (4, 20), (5, 21), (6, 22),
    (7, 23), (8, 24), (9, 25), (10, 26), (11, 27), (12, 28), (13, 29),
]

def extract_features_single(epoch, sfreq=128):
    """Trích xuất features từ 1 epoch (32 channels, 128 timestamps).
    
    Trả về vector gồm 312 features:
      - PSD (log):    32 ch x 4 bands = 128 features (log-transformed band power)
      - Asymmetry:    14 pairs x 4 bands = 56 features (log ratio)
      - DE:           32 ch x 4 bands = 128 features (Differential Entropy từ variance thời gian)
    """
    n_channels = epoch.shape[0]
    bands = list(FREQ_BANDS.values())
    band_names = list(FREQ_BANDS.keys())
    
    # --- PSD (Welch) với nperseg=64 để có ≥2 segments averaging ---
    # FIX #13: nperseg=128 trên epoch 128 samples suy biến thành periodogram (chỉ 1 segment).
    # Dùng nperseg=64 với overlap 50% mặc định → 3 segments → giảm variance ước lượng phổ.
    freqs, psd = welch(epoch, fs=sfreq, nperseg=min(64, epoch.shape[1]))
    
    # FIX #5: Cache PSD band power cho từng (channel, band) để reuse cho Asymmetry
    # Tránh tính PSD hai lần.
    psd_band_power = np.zeros((n_channels, len(bands)))
    for ch in range(n_channels):
        for b_i, (fmin, fmax) in enumerate(bands):
            idx = (freqs >= fmin) & (freqs <= fmax)
            psd_band_power[ch, b_i] = np.mean(psd[ch, idx]) if np.any(idx) else 1e-10
    
    # FIX #15: Log-transform PSD features để đồng nhất scale với Asymmetry (log-scale) và DE (log-scale)
    # PSD raw có dải rất rộng O(1e-6) → O(1), log nén về dải hợp lý hơn.
    psd_features = []
    for ch in range(n_channels):
        for b_i in range(len(bands)):
            psd_features.append(np.log(max(psd_band_power[ch, b_i], 1e-10)))
    
    # --- Differential Entropy (DE) ---
    # FIX #3: DE = 0.5 * ln(2πeσ²) trong đó σ² là PHƯƠNG SAI tín hiệu đã lọc bandpass
    # trong miền thời gian, KHÔNG phải mean PSD (mật độ phổ).
    # Dùng bandpass filter (Butterworth bậc 5) + np.var() cho mỗi (channel, band).
    de_features = []
    for ch in range(n_channels):
        for (fmin, fmax) in bands:
            # Bandpass filter Butterworth bậc 5, output SOS cho numerical stability
            nyquist = sfreq / 2.0
            low = max(fmin / nyquist, 0.01)   # Tránh 0 cho low cutoff
            high = min(fmax / nyquist, 0.99)   # Tránh 1 cho high cutoff
            sos = butter(5, [low, high], btype='band', output='sos')
            filtered_signal = sosfilt(sos, epoch[ch])
            variance = np.var(filtered_signal)
            de = 0.5 * np.log(2 * np.pi * np.e * max(variance, 1e-10))
            de_features.append(de)
    
    # --- Asymmetry (Left - Right) log ratio ---
    # FIX #5: Reuse psd_band_power đã cache thay vì tính lại PSD
    asym_features = []
    for (left_ch, right_ch) in ASYMMETRY_PAIRS:
        for b_i in range(len(bands)):
            left_power = max(psd_band_power[left_ch, b_i], 1e-10)
            right_power = max(psd_band_power[right_ch, b_i], 1e-10)
            asym_features.append(np.log(left_power) - np.log(right_power))
    
    return np.concatenate([psd_features, asym_features, de_features])

def extract_features_dataset(X_epochs):
    """Trích xuất features cho toàn bộ dataset."""
    n_epochs = X_epochs.shape[0]
    sample = extract_features_single(X_epochs[0])
    n_features = len(sample)
    print(f"    Trích xuất {n_features} features/epoch x {n_epochs} epochs...")
    
    features = np.zeros((n_epochs, n_features), dtype=np.float32)
    features[0] = sample
    
    for i in range(1, n_epochs):
        features[i] = extract_features_single(X_epochs[i])
        if (i + 1) % 10000 == 0:
            print(f"    ...{i+1}/{n_epochs} epochs")
    
    print(f"    Hoàn tất! Feature matrix shape: {features.shape}")
    return features

def lds_smoothing(features_seq, alpha=0.3):
    """Áp dụng Linear Dynamic System smoothing trên một sequence (vd: 1 video 60s)."""
    if len(features_seq) == 0:
        return features_seq
    smoothed = np.zeros_like(features_seq)
    smoothed[0] = features_seq[0]
    for t in range(1, len(features_seq)):
        smoothed[t] = alpha * features_seq[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed

def apply_lds(X_features, epochs_per_trial=60, alpha=0.3, original_indices=None):
    """
    Áp dụng LDS cho toàn bộ dataset.
    Nếu original_indices được cung cấp (index tuyệt đối của dataset gốc),
    hàm sẽ nhóm theo video và sắp xếp theo thời gian để smooth chuẩn xác kể cả khi epochs không liên tiếp.
    """
    X_smoothed = np.copy(X_features)
    
    if original_indices is None:
        n_trials = len(X_features) // epochs_per_trial
        for t in range(n_trials):
            s, e = t * epochs_per_trial, (t + 1) * epochs_per_trial
            X_smoothed[s:e] = lds_smoothing(X_features[s:e], alpha)
    else:
        original_indices = np.asarray(original_indices)
        video_ids = original_indices // epochs_per_trial
        unique_videos = np.unique(video_ids)
        
        for vid in unique_videos:
            # Lấy vị trí các epoch thuộc video này trong mảng hiện tại
            mask = (video_ids == vid)
            idx_in_X = np.where(mask)[0]
            
            # Sắp xếp các index nội bộ này theo đúng thứ tự thời gian gốc
            abs_indices = original_indices[mask]
            sort_order = np.argsort(abs_indices)
            sorted_idx_in_X = idx_in_X[sort_order]
            
            # Rút trích, smooth và gán lại theo đúng thứ tự
            seq = X_features[sorted_idx_in_X]
            smoothed_seq = lds_smoothing(seq, alpha)
            X_smoothed[sorted_idx_in_X] = smoothed_seq
            
    return X_smoothed
