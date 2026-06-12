import numpy as np
from scipy.signal import butter, filtfilt

class EEGFeatureExtractor:
    def __init__(self, sfreq: int = 128, window_len: int = 64, stride: int = 16):
        self.sfreq = sfreq
        self.window_len = window_len
        self.stride = stride

        self.bands = {
            'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 14),
            'beta': (14, 30), 'gamma': (30, 45)
        }
        self.band_names = list(self.bands.keys())
        self.filter_bank = {}
        for name, (low, high) in self.bands.items():
            self.filter_bank[name] = butter(3, [low, high], btype='band', fs=self.sfreq)

    def _apply_filter(self, X: np.ndarray, band_name: str) -> np.ndarray:
        b, a = self.filter_bank[band_name]
        # FIX: Dùng filtfilt (Zero-phase filtering) để tránh phase distortion
        return filtfilt(b, a, X, axis=-1)

    @staticmethod
    def _de(variance: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        return 0.5 * np.log(2.0 * np.pi * np.e * variance + eps)

    def extract_features_pipeline(self, X_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        N, num_ch, T = X_raw.shape
        filtered = {name: self._apply_filter(X_raw, name) for name in self.band_names}

        # 1. SLIDING WINDOW (X_freq_seq) - Dành cho BiLSTM
        seq_len = (T - self.window_len) // self.stride + 1
        X_freq_seq = np.zeros((N, seq_len, num_ch * len(self.bands)), dtype=np.float32)

        for i in range(seq_len):
            s = i * self.stride
            step_de = [self._de(np.var(filtered[name][:, :, s:s+self.window_len], axis=-1)) for name in self.band_names]
            X_freq_seq[:, i, :] = np.concatenate(step_de, axis=-1)

        # 2. XÂY DỰNG NODE FEATURES CHO GRAPH (X_graph)
        bp = {name: np.var(filtered[name], axis=-1) + 1e-8 for name in self.band_names}
        de = {name: self._de(bp[name]) for name in self.band_names}
        
        de_list = [de[name] for name in self.band_names] 
        X_graph = np.stack(de_list, axis=-1).astype(np.float32) 

        return X_freq_seq, X_graph