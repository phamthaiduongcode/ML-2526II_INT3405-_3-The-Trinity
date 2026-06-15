# File: src/data_pipeline/preprocess.py
import os
import pickle
import gc
import warnings
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
N_SUBJECTS        = 32
N_CHANNELS        = 32
N_TRIALS          = 40
N_EPOCHS_PER_TRIAL = 60
SFREQ             = 128          # Hz
BASELINE_SAMPLES  = 384          # 3 giây * 128 Hz
STIMULUS_SAMPLES  = 7680         # 60 giây * 128 Hz


# ─────────────────────────────────────────────
#  UNIT TEST
# ─────────────────────────────────────────────
def verify_epoch_integrity(eeg_normalized: np.ndarray, X: np.ndarray) -> None:
    """
    [UNIT TEST] Đảm bảo phép reshape không làm xáo trộn
    thứ tự thời gian hoặc kênh tín hiệu.
    """
    assert np.allclose(
        eeg_normalized[0, 0, :SFREQ], X[0, 0, :]
    ), "❌ Epoch 0 Trial 0 bị sai!"

    assert np.allclose(
        eeg_normalized[0, 0, SFREQ:2*SFREQ], X[1, 0, :]
    ), "❌ Epoch 1 Trial 0 bị sai!"

    assert np.allclose(
        eeg_normalized[1, 0, :SFREQ], X[N_EPOCHS_PER_TRIAL, 0, :]
    ), "❌ Ranh giới Trial 0->1 bị trộn!"

    assert np.allclose(
        eeg_normalized[0, 5, :SFREQ], X[0, 5, :]
    ), "❌ Channel 5 bị trộn sang channel khác!"


# ─────────────────────────────────────────────
#  CORE PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_subject(file_path: str) -> tuple:
    """
    Tiền xử lý toàn diện dữ liệu của 1 Subject.

    Thay đổi so với phiên bản cũ:
    - Ngưỡng binarize nhãn được tính riêng theo MEDIAN CỦA TỪNG SUBJECT
      thay vì dùng global median chung → đảm bảo mỗi subject luôn có
      phân phối nhãn ~50/50, tránh lệch class.

    Returns
    -------
    X              : (N_TRIALS * N_EPOCHS_PER_TRIAL, N_CHANNELS, SFREQ)
    y_val_expanded : nhãn valence đã nhân bản theo epoch
    y_aro_expanded : nhãn arousal đã nhân bản theo epoch
    y_valence      : nhãn valence gốc cấp trial (40,)
    y_arousal      : nhãn arousal gốc cấp trial (40,)
    sub_median_val : ngưỡng median valence của subject này
    sub_median_aro : ngưỡng median arousal của subject này
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        raise RuntimeError(f"❌ Không thể đọc file {file_path}: {e}")

    raw_eeg    = data['data'][:, :N_CHANNELS, :]   # (40, 32, 8064)
    raw_labels = data['labels']                     # (40, 4)

    # 1. Binarize nhãn theo PER-SUBJECT MEDIAN
    sub_median_val = np.median(raw_labels[:, 0])
    sub_median_aro = np.median(raw_labels[:, 1])
    y_valence = (raw_labels[:, 0] >= sub_median_val).astype(int)
    y_arousal = (raw_labels[:, 1] >= sub_median_aro).astype(int)

    # 2. Baseline Subtraction
    baseline      = raw_eeg[:, :, :BASELINE_SAMPLES]
    stimulus      = raw_eeg[:, :, BASELINE_SAMPLES:]
    baseline_mean = np.mean(baseline, axis=2, keepdims=True)
    eeg_normalized = (stimulus - baseline_mean).astype(np.float32)

    # 3. Phân đoạn Epochs (1 giây, không chồng lấp)
    epochs = eeg_normalized.reshape(N_TRIALS, N_CHANNELS, N_EPOCHS_PER_TRIAL, SFREQ)
    epochs = epochs.transpose(0, 2, 1, 3).copy()           # (40, 60, 32, 128)
    X      = epochs.reshape(-1, N_CHANNELS, SFREQ)          # (2400, 32, 128)

    verify_epoch_integrity(eeg_normalized, X)

    # 4. Nhân bản nhãn cấp trial → cấp epoch
    y_val_expanded = np.repeat(y_valence, N_EPOCHS_PER_TRIAL)
    y_aro_expanded = np.repeat(y_arousal, N_EPOCHS_PER_TRIAL)

    return X, y_val_expanded, y_aro_expanded, y_valence, y_arousal, sub_median_val, sub_median_aro


# ─────────────────────────────────────────────
#  NORMALIZATION
# ─────────────────────────────────────────────
def normalize_after_split(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    mode:    str = 'channel'
) -> tuple:
    """
    Chuẩn hóa dữ liệu SAU KHI chia Train/Test — tránh Data Leakage.

    Parameters
    ----------
    mode : 'channel' | 'flatten'
        'channel' : chuẩn hóa per-channel (khuyến nghị cho EEG)
        'flatten' : chuẩn hóa toàn bộ flatten

    Returns
    -------
    X_tr_scaled, X_te_scaled, scaler
    """
    n_train, n_ch, n_t = X_train.shape
    n_test  = X_test.shape[0]
    scaler  = StandardScaler()

    if mode == 'flatten':
        X_tr_2d     = X_train.reshape(n_train, -1)
        X_te_2d     = X_test.reshape(n_test, -1)
        X_tr_scaled = scaler.fit_transform(X_tr_2d).reshape(n_train, n_ch, n_t)
        X_te_scaled = scaler.transform(X_te_2d).reshape(n_test, n_ch, n_t)

    elif mode == 'channel':
        X_tr_2d     = X_train.transpose(0, 2, 1).reshape(-1, n_ch)
        X_te_2d     = X_test.transpose(0, 2, 1).reshape(-1, n_ch)
        X_tr_scaled = scaler.fit_transform(X_tr_2d).reshape(n_train, n_t, n_ch).transpose(0, 2, 1).copy()
        X_te_scaled = scaler.transform(X_te_2d).reshape(n_test, n_t, n_ch).transpose(0, 2, 1).copy()

    else:
        raise ValueError("mode phải là 'flatten' hoặc 'channel'")

    return X_tr_scaled.astype(np.float32), X_te_scaled.astype(np.float32), scaler


# ─────────────────────────────────────────────
#  CLASS WEIGHTS
# ─────────────────────────────────────────────
def get_dynamic_class_weights(
    y_train_fold: np.ndarray,
    n_classes:    int   = 2,
    max_weight:   float = 5.0
) -> torch.Tensor:
    """
    Tính trọng số động cho từng class, hỗ trợ n_classes tùy ý
    và chặn bùng nổ trọng số bằng max_weight.
    """
    classes = np.unique(y_train_fold)

    if len(classes) == 1:
        warnings.warn(
            f"⚠️  Fold chỉ có 1 class ({classes[0]}) — fold bị degenerate, "
            "trả về trọng số đồng đều.", RuntimeWarning
        )
        return torch.ones(n_classes, dtype=torch.float32)

    weights_partial = compute_class_weight(
        class_weight='balanced', classes=classes, y=y_train_fold
    )
    weights_partial = np.clip(weights_partial, 0, max_weight)

    weights = torch.zeros(n_classes, dtype=torch.float32)
    for i, c in enumerate(classes):
        weights[c] = weights_partial[i]

    return weights


# ─────────────────────────────────────────────
#  MAIN — XỬ LÝ TOÀN BỘ DATASET
# ─────────────────────────────────────────────
if __name__ == "__main__":
    ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR   = os.path.join(ROOT_DIR, "data", "raw")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed_1")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🚀 ĐANG XỬ LÝ DỮ LIỆU DEAP (PER-SUBJECT MEDIAN)...")
    print("=" * 60)

    all_X, all_y_val, all_y_aro, all_groups = [], [], [], []
    processed_count  = 0
    skipped_subjects = []

    # Chỉ cần 1 pass duy nhất — không cần tính global median trước
    for sub_id in range(1, N_SUBJECTS + 1):
        file_path = os.path.join(DATA_DIR, f"s{sub_id:02d}.dat")

        if not os.path.exists(file_path):
            skipped_subjects.append(sub_id)
            continue

        try:
            X_sub, y_val_sub, y_aro_sub, _, _, med_val, med_aro = preprocess_subject(file_path)
        except RuntimeError as e:
            print(e)
            skipped_subjects.append(sub_id)
            continue

        # Kiểm tra cân bằng nhãn per-subject
        val_ratio = y_val_sub.mean()
        aro_ratio = y_aro_sub.mean()
        print(
            f"  ✅ Subject {sub_id:02d} | "
            f"Median Val={med_val:.2f}, Aro={med_aro:.2f} | "
            f"Val ratio={val_ratio:.2f}, Aro ratio={aro_ratio:.2f}"
        )

        all_X.append(X_sub)
        all_y_val.append(y_val_sub)
        all_y_aro.append(y_aro_sub)
        all_groups.append(np.full(len(y_val_sub), sub_id - 1, dtype=np.int32))
        processed_count += 1

    if skipped_subjects:
        print(f"\n⚠️  Bỏ qua {len(skipped_subjects)} subject(s): {skipped_subjects}")

    if processed_count == 0:
        print("❌ Không tìm thấy file dữ liệu nào. Kiểm tra lại DATA_DIR.")
    else:
        print(f"\n⏳ Đang ghép mảng và lưu file ({processed_count} subjects)...")
        final_X      = np.concatenate(all_X,      axis=0).astype(np.float32)
        del all_X; gc.collect()

        final_y_val  = np.concatenate(all_y_val)
        final_y_aro  = np.concatenate(all_y_aro)
        final_groups = np.concatenate(all_groups)

        #np.save(os.path.join(OUTPUT_DIR, "X_epochs.npy"),       final_X)
        #np.save(os.path.join(OUTPUT_DIR, "y_valence.npy"),       final_y_val)
        #np.save(os.path.join(OUTPUT_DIR, "y_arousal.npy"),       final_y_aro)
        #np.save(os.path.join(OUTPUT_DIR, "subject_groups.npy"),  final_groups)

        print(f"\n✅ Hoàn tất!")
        print(f"   X shape      : {final_X.shape}")
        print(f"   y_valence    : {final_y_val.shape} | mean={final_y_val.mean():.3f}")
        print(f"   y_arousal    : {final_y_aro.shape} | mean={final_y_aro.mean():.3f}")
        print(f"   Lưu tại      : {OUTPUT_DIR}")