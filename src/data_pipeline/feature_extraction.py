import numpy as np

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
      - PSD:        32 ch x 4 bands = 128 features
      - Asymmetry:  14 pairs x 4 bands = 56 features
      - DE:         32 ch x 4 bands = 128 features (Differential Entropy)
    """
    from scipy.signal import welch
    
    n_channels = epoch.shape[0]
    bands = list(FREQ_BANDS.values())
    
    # --- PSD (Welch) ---
    freqs, psd = welch(epoch, fs=sfreq, nperseg=min(128, epoch.shape[1]))
    psd_features = []
    de_features = []
    
    for ch in range(n_channels):
        for (fmin, fmax) in bands:
            idx = (freqs >= fmin) & (freqs <= fmax)
            band_power = np.mean(psd[ch, idx]) if np.any(idx) else 1e-10
            psd_features.append(band_power)
            
            # --- Differential Entropy ---
            # DE = 0.5 * log(2 * pi * e * variance_of_band)
            band_var = np.var(psd[ch, idx]) if np.any(idx) else 1e-10
            de = 0.5 * np.log(2 * np.pi * np.e * max(band_var, 1e-10))
            de_features.append(de)
    
    # --- Asymmetry (Left - Right) ---
    asym_features = []
    for (left_ch, right_ch) in ASYMMETRY_PAIRS:
        for b_i, (fmin, fmax) in enumerate(bands):
            idx = (freqs >= fmin) & (freqs <= fmax)
            left_power = np.mean(psd[left_ch, idx]) if np.any(idx) else 1e-10
            right_power = np.mean(psd[right_ch, idx]) if np.any(idx) else 1e-10
            asym_features.append(np.log(max(left_power, 1e-10)) - np.log(max(right_power, 1e-10)))
    
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
    smoothed = np.zeros_like(features_seq)
    smoothed[0] = features_seq[0]
    for t in range(1, len(features_seq)):
        smoothed[t] = alpha * features_seq[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed

def apply_lds(X_features, epochs_per_trial=60, alpha=0.3):
    """Áp dụng LDS cho toàn bộ dataset."""
    X_smoothed = np.copy(X_features)
    n_trials = len(X_features) // epochs_per_trial
    for t in range(n_trials):
        s, e = t * epochs_per_trial, (t + 1) * epochs_per_trial
        X_smoothed[s:e] = lds_smoothing(X_features[s:e], alpha)
    return X_smoothed
