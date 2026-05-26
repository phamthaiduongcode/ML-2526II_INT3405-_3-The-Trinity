"""
EmoWave Project - DEAP Dataset Loader & Preprocessor
=====================================================
Đọc, tiền xử lý, và gán nhãn dữ liệu DEAP.
Dùng chung cho cả 3 model: SVM, CNN, LSTM.

DEAP format:
  - 32 file: s01.dat → s32.dat
  - Mỗi file: dict với keys 'data' và 'labels'
    + data:   (40 trials, 40 channels, 8064 samples)  @ 128Hz
    + labels: (40 trials, 4) → [valence, arousal, dominance, liking]
  - 32 kênh EEG đầu tiên (index 0-31), còn lại là peripheral
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ─── Cấu hình ───────────────────────────────────────────────────────────────

DATA_DIR   = "./data/deap"   # Thư mục chứa s01.dat → s32.dat
N_SUBJECTS = 32
N_TRIALS   = 40
N_CHANNELS = 32              # Chỉ lấy 32 kênh EEG
SFREQ      = 128             # Hz (đã preprocessed)
EPOCH_SEC  = 1               # Cửa sổ thời gian 1 giây (giống bài báo gốc)
SAMPLES_PER_EPOCH = SFREQ * EPOCH_SEC   # = 128 samples

# Bỏ 3 giây đầu (baseline) → bắt đầu từ sample 384
BASELINE_SAMPLES = 3 * SFREQ  # = 384


# ─── 1. Đọc 1 subject ────────────────────────────────────────────────────────

def load_subject(subject_id: int, data_dir: str = DATA_DIR) -> dict:
    """
    Đọc file .dat của 1 subject.
    subject_id: 1 → 32
    Return: dict {'data': np.array, 'labels': np.array}
    """
    filename = os.path.join(data_dir, f"s{subject_id:02d}.dat")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Không tìm thấy file: {filename}")

    with open(filename, "rb") as f:
        subject = pickle.load(f, encoding="latin1")

    return subject


# ─── 2. Tạo nhãn ─────────────────────────────────────────────────────────────

def make_labels_2class(labels_raw: np.ndarray) -> np.ndarray:
    """
    Phân loại nhị phân dựa trên Valence.
    valence < 5  → 0 (Negative)
    valence >= 5 → 1 (Positive)
    labels_raw shape: (40, 4) → [valence, arousal, dominance, liking]
    """
    valence = labels_raw[:, 0]   # cột 0 = valence
    return (valence >= 5).astype(int)


def make_labels_4class(labels_raw: np.ndarray) -> np.ndarray:
    """
    Phân loại 4 lớp theo mô hình Valence-Arousal:
      Q1 (V+, A+) = 0 → Vui vẻ / Excited
      Q2 (V-, A+) = 1 → Sợ hãi / Stressed
      Q3 (V-, A-) = 2 → Buồn   / Sad
      Q4 (V+, A-) = 3 → Thư giãn / Calm
    """
    valence = labels_raw[:, 0]
    arousal = labels_raw[:, 1]

    classes = np.zeros(len(valence), dtype=int)
    classes[(valence >= 5) & (arousal >= 5)] = 0   # Q1: Vui vẻ
    classes[(valence <  5) & (arousal >= 5)] = 1   # Q2: Sợ hãi
    classes[(valence <  5) & (arousal <  5)] = 2   # Q3: Buồn
    classes[(valence >= 5) & (arousal <  5)] = 3   # Q4: Thư giãn
    return classes


# ─── 3. Cắt epoch ────────────────────────────────────────────────────────────

def segment_trial(trial_data: np.ndarray,
                  baseline: int = BASELINE_SAMPLES,
                  epoch_len: int = SAMPLES_PER_EPOCH) -> np.ndarray:
    """
    Cắt 1 trial thành nhiều epoch không chồng lấp.
    trial_data shape: (n_channels, n_samples)
    Return: (n_epochs, n_channels, epoch_len)
    """
    signal = trial_data[:, baseline:]          # bỏ baseline
    n_samples = signal.shape[1]
    n_epochs  = n_samples // epoch_len

    epochs = np.array([
        signal[:, i*epoch_len : (i+1)*epoch_len]
        for i in range(n_epochs)
    ])
    return epochs   # (n_epochs, 32, 128)


# ─── 4. Load toàn bộ dataset ─────────────────────────────────────────────────

def load_all_subjects(data_dir: str = DATA_DIR,
                      n_subjects: int = N_SUBJECTS,
                      label_type: str = "2class") -> tuple:
    """
    Đọc và xử lý toàn bộ subjects.

    label_type: "2class" hoặc "4class"

    Return:
        X : (total_epochs, n_channels, epoch_len)  → dùng cho CNN/LSTM
        y : (total_epochs,)
        groups : (total_epochs,) → subject id, dùng cho cross-subject split
    """
    X_list, y_list, g_list = [], [], []

    for sid in range(1, n_subjects + 1):
        print(f"  Loading subject {sid:02d}/{n_subjects}...", end="\r")
        try:
            subject = load_subject(sid, data_dir)
        except FileNotFoundError as e:
            print(f"\n  [WARN] {e} — bỏ qua")
            continue

        data_raw   = subject["data"][:, :N_CHANNELS, :]   # (40, 32, 8064)
        labels_raw = subject["labels"]                     # (40, 4)

        # Tạo nhãn
        if label_type == "2class":
            trial_labels = make_labels_2class(labels_raw)
        elif label_type == "4class":
            trial_labels = make_labels_4class(labels_raw)
        else:
            raise ValueError("label_type phải là '2class' hoặc '4class'")

        # Cắt epoch từng trial
        for t in range(N_TRIALS):
            epochs = segment_trial(data_raw[t])    # (n_epochs, 32, 128)
            n_ep   = len(epochs)
            X_list.append(epochs)
            y_list.append(np.full(n_ep, trial_labels[t]))
            g_list.append(np.full(n_ep, sid))

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    g = np.concatenate(g_list, axis=0).astype(np.int64)

    print(f"\n  Xong! X: {X.shape}, y: {y.shape}")
    print(f"  Phân bố nhãn: { {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))} }")
    return X, y, g


# ─── 5. Chuẩn bị dữ liệu cho từng model ─────────────────────────────────────

def prepare_for_svm(X: np.ndarray, y: np.ndarray,
                    test_size: float = 0.2,
                    random_state: int = 42) -> tuple:
    """
    Flatten epoch → vector features cho SVM.
    X shape: (n, 32, 128) → (n, 32*128=4096)
    Return: X_train, X_test, y_train, y_test (đã scale)
    """
    X_flat = X.reshape(len(X), -1)   # (n, 4096)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_flat, y, test_size=test_size,
        random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)
    return X_tr, X_te, y_tr, y_te, scaler


def prepare_for_cnn(X: np.ndarray, y: np.ndarray,
                    test_size: float = 0.2,
                    random_state: int = 42) -> tuple:
    """
    CNN dùng trực tiếp (n, 32, 128) — normalize theo channel.
    Return: X_train, X_test, y_train, y_test
    """
    # Normalize từng channel: mean=0, std=1
    mean = X.mean(axis=2, keepdims=True)
    std  = X.std(axis=2, keepdims=True) + 1e-8
    X_norm = (X - mean) / std

    return train_test_split(
        X_norm, y, test_size=test_size,
        random_state=random_state, stratify=y
    )


def prepare_for_lstm(X: np.ndarray, y: np.ndarray,
                     test_size: float = 0.2,
                     random_state: int = 42) -> tuple:
    """
    LSTM dùng (n, timesteps=128, features=32) — transpose của CNN input.
    Return: X_train, X_test, y_train, y_test
    """
    # Normalize
    mean   = X.mean(axis=2, keepdims=True)
    std    = X.std(axis=2, keepdims=True) + 1e-8
    X_norm = (X - mean) / std

    # Transpose: (n, 32, 128) → (n, 128, 32)
    X_lstm = X_norm.transpose(0, 2, 1)

    return train_test_split(
        X_lstm, y, test_size=test_size,
        random_state=random_state, stratify=y
    )


# ─── 6. Quick test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("EmoWave — DEAP Loader Test")
    print("=" * 50)

    # Test với 1 subject trước
    print("\n[1] Load subject 01...")
    try:
        s = load_subject(1)
        print(f"  data shape  : {s['data'].shape}")
        print(f"  labels shape: {s['labels'].shape}")
        print(f"  valence range: {s['labels'][:,0].min():.1f} – {s['labels'][:,0].max():.1f}")
    except FileNotFoundError:
        print(f"  [!] Chưa có data. Đặt file vào: {DATA_DIR}")
        print(f"      Ví dụ: {DATA_DIR}/s01.dat")
        exit()

    print("\n[2] Load toàn bộ (2-class)...")
    X, y, groups = load_all_subjects(label_type="2class")

    print("\n[3] Chuẩn bị cho SVM...")
    X_tr, X_te, y_tr, y_te, _ = prepare_for_svm(X, y)
    print(f"  Train: {X_tr.shape}, Test: {X_te.shape}")

    print("\n[4] Chuẩn bị cho CNN...")
    X_tr, X_te, y_tr, y_te = prepare_for_cnn(X, y)
    print(f"  Train: {X_tr.shape}, Test: {X_te.shape}")

    print("\n[5] Chuẩn bị cho LSTM...")
    X_tr, X_te, y_tr, y_te = prepare_for_lstm(X, y)
    print(f"  Train: {X_tr.shape}, Test: {X_te.shape}")

    print("\n✓ Loader hoạt động tốt!")
    print("  Tiếp theo: chạy models/svm_model.py, cnn_model.py, lstm_model.py")
