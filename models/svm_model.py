"""
EmoWave — Người 1: SVM Model (Hoàn chỉnh)
=============================================
Tái hiện bài báo gốc (Wang et al., 2014):
  Power Spectrum + Asymmetry Features + LDS Smoothing + SVM

Pipeline:
  1. Load DEAP data → cắt epoch 1s
  2. Trích xuất Power Spectrum features (32 ch × 5 bands = 160)
  3. Trích xuất Differential Asymmetry features (14 pairs × 5 bands = 70)
  4. (Tùy chọn) LDS smoothing
  5. StandardScaler → SVM (Linear / RBF)
  6. GridSearchCV + 10-Fold Cross-Validation
  7. Confusion Matrix + Classification Report

Tối ưu cho máy local (24GB RAM):
  - Giải phóng raw data sau khi extract features
  - Dùng LinearSVC (nhanh gấp 50x so với SVC linear)
  - Subsample cho GridSearch RBF (tránh kernel matrix O(n²))
"""

import numpy as np
import pickle
import os
import sys
import json
import time
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)


# ─── Cấu hình ───────────────────────────────────────────────────────────────
DATA_DIR    = "./data/deap"
N_SUBJECTS  = 32
N_TRIALS    = 40
N_CHANNELS  = 32
SFREQ       = 128
EPOCH_SEC   = 1
SAMPLES_PER_EPOCH = SFREQ * EPOCH_SEC   # 128
BASELINE_SAMPLES  = 3 * SFREQ           # 384
RESULTS_DIR = "./results"

# 5 dải tần EEG theo chuẩn y khoa
FREQ_BANDS = {
    'delta': (0.5, 4),    # Ngủ sâu — ít liên quan cảm xúc
    'theta': (4, 8),      # Buồn ngủ, thiền
    'alpha': (8, 13),     # Thư giãn — QUAN TRỌNG cho cảm xúc
    'beta':  (13, 30),    # Tập trung, căng thẳng — QUAN TRỌNG
    'gamma': (30, 45),    # Xử lý thông tin cao — QUAN TRỌNG
}

# 14 cặp điện cực đối xứng trái-phải trong hệ thống 10-20
# Cặp (left_idx, right_idx) theo thứ tự kênh DEAP
ASYMMETRY_PAIRS = [
    (0, 16),   # Fp1 - Fp2
    (1, 17),   # AF3 - AF4
    (2, 18),   # F3  - F4
    (3, 19),   # F7  - F8
    (4, 20),   # FC5 - FC6
    (5, 21),   # FC1 - FC2
    (6, 22),   # C3  - C4
    (7, 23),   # T7  - T8
    (8, 24),   # CP5 - CP6
    (9, 25),   # CP1 - CP2
    (10, 26),  # P3  - P4
    (11, 27),  # P7  - P8
    (12, 28),  # PO3 - PO4
    (13, 29),  # O1  - O2
]


# ─── DEAP Loader (tự chứa) ──────────────────────────────────────────────────

def load_subject(subject_id, data_dir=DATA_DIR):
    """Đọc 1 file .dat"""
    filename = os.path.join(data_dir, f"s{subject_id:02d}.dat")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Không tìm thấy: {filename}")
    with open(filename, "rb") as f:
        return pickle.load(f, encoding="latin1")


def make_labels_2class(labels_raw):
    return (labels_raw[:, 0] >= 5).astype(int)


def make_labels_4class(labels_raw):
    valence = labels_raw[:, 0]
    arousal = labels_raw[:, 1]
    classes = np.zeros(len(valence), dtype=int)
    classes[(valence >= 5) & (arousal >= 5)] = 0
    classes[(valence <  5) & (arousal >= 5)] = 1
    classes[(valence <  5) & (arousal <  5)] = 2
    classes[(valence >= 5) & (arousal <  5)] = 3
    return classes


def segment_trial(trial_data):
    signal = trial_data[:, BASELINE_SAMPLES:]
    n_epochs = signal.shape[1] // SAMPLES_PER_EPOCH
    return np.array([
        signal[:, i * SAMPLES_PER_EPOCH : (i + 1) * SAMPLES_PER_EPOCH]
        for i in range(n_epochs)
    ])


def load_all_subjects(data_dir=DATA_DIR, n_subjects=N_SUBJECTS, label_type="2class"):
    X_list, y_list, g_list = [], [], []
    for sid in range(1, n_subjects + 1):
        print(f"  Loading subject {sid:02d}/{n_subjects}...", end="\r")
        try:
            subject = load_subject(sid, data_dir)
        except FileNotFoundError as e:
            print(f"\n  [WARN] {e} — bỏ qua")
            continue

        data_raw   = subject["data"][:, :N_CHANNELS, :]
        labels_raw = subject["labels"]

        if label_type == "2class":
            trial_labels = make_labels_2class(labels_raw)
        else:
            trial_labels = make_labels_4class(labels_raw)

        for t in range(N_TRIALS):
            epochs = segment_trial(data_raw[t])
            n_ep   = len(epochs)
            X_list.append(epochs)
            y_list.append(np.full(n_ep, trial_labels[t]))
            g_list.append(np.full(n_ep, sid))

        # Giải phóng bộ nhớ sau mỗi subject
        del subject, data_raw, labels_raw
        gc.collect()

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    g = np.concatenate(g_list, axis=0).astype(np.int64)

    del X_list, y_list, g_list
    gc.collect()

    print(f"\n  Xong! X: {X.shape}, y: {y.shape}")
    print(f"  Phân bố nhãn: { {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))} }")
    return X, y, g


# ─── Feature Extraction ─────────────────────────────────────────────────────

def extract_power_spectrum(epoch, sfreq=SFREQ):
    """
    Tính năng lượng phổ cho 1 epoch.

    Input:  epoch shape (32, 128) — 32 channels × 128 samples (1 giây)
    Output: features shape (32, 5) — 32 channels × 5 bands

    Phương pháp: FFT → |FFT|² → mean trong mỗi dải tần → log transform
    """
    fft_vals = np.abs(np.fft.rfft(epoch, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(epoch.shape[-1], 1.0 / sfreq)

    band_powers = []
    for (f_low, f_high) in FREQ_BANDS.values():
        idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
        if len(idx) == 0:
            band_powers.append(np.zeros(epoch.shape[0]))
        else:
            power = np.log1p(fft_vals[:, idx].mean(axis=-1))
            band_powers.append(power)

    return np.stack(band_powers, axis=-1)  # (32, 5)


def extract_asymmetry(power_features):
    """
    Tính chỉ số bất đối xứng trái - phải.

    Input:  power_features shape (32, 5) từ extract_power_spectrum()
    Output: asymmetry shape (14, 5) — 14 cặp × 5 bands

    DASI = log(Power_Left) - log(Power_Right)
    Vì power_features đã log rồi → chỉ cần trừ.
    """
    asymmetry = []
    for left_idx, right_idx in ASYMMETRY_PAIRS:
        diff = power_features[left_idx] - power_features[right_idx]
        asymmetry.append(diff)
    return np.array(asymmetry)  # (14, 5)


def extract_all_features(epoch, sfreq=SFREQ):
    """
    Trích xuất toàn bộ features cho 1 epoch.

    Input:  epoch (32, 128)
    Output: feature_vector (230,)
      - Power Spectrum:  32 channels × 5 bands = 160 features
      - Asymmetry:       14 pairs   × 5 bands = 70  features
    """
    power = extract_power_spectrum(epoch, sfreq)   # (32, 5)
    asym = extract_asymmetry(power)                 # (14, 5)
    return np.concatenate([power.flatten(), asym.flatten()])  # (230,)


def extract_features_dataset(X_epochs, sfreq=SFREQ):
    """
    Trích xuất features cho toàn bộ dataset.

    Input:  X_epochs shape (n_epochs, 32, 128)
    Output: X_features shape (n_epochs, 230)
    """
    n_total = len(X_epochs)
    print(f"  Trích xuất {n_total} epochs...")
    start = time.time()

    features = []
    for i in range(n_total):
        features.append(extract_all_features(X_epochs[i], sfreq))
        if (i + 1) % 10000 == 0 or (i + 1) == n_total:
            elapsed = time.time() - start
            speed = (i + 1) / elapsed
            eta = (n_total - i - 1) / speed if speed > 0 else 0
            print(f"    {i+1}/{n_total}  ({speed:.0f} eps/s, ETA: {eta:.0f}s)")

    result = np.array(features, dtype=np.float32)
    print(f"  Feature shape: {result.shape} ({time.time()-start:.1f}s)")
    return result


# ─── LDS Smoothing ──────────────────────────────────────────────────────────

def lds_smoothing(features_seq, alpha=0.3):
    """
    LDS đơn giản bằng exponential moving average.

    Input:  features_seq (n_epochs_per_trial, n_features)
    Output: smoothed (n_epochs_per_trial, n_features)

    alpha nhỏ → mượt nhiều, alpha lớn → mượt ít
    """
    smoothed = np.zeros_like(features_seq)
    smoothed[0] = features_seq[0]
    for t in range(1, len(features_seq)):
        smoothed[t] = alpha * features_seq[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def apply_lds(X_features, epochs_per_trial=60, alpha=0.3):
    """
    Áp dụng LDS cho toàn bộ dataset.
    Giả sử mỗi 60 epochs liên tiếp = 1 trial.
    """
    print(f"  Áp dụng LDS smoothing (alpha={alpha})...")
    X_smoothed = np.copy(X_features)
    n_trials = len(X_features) // epochs_per_trial
    for t in range(n_trials):
        s, e = t * epochs_per_trial, (t + 1) * epochs_per_trial
        X_smoothed[s:e] = lds_smoothing(X_features[s:e], alpha)
    print("  LDS xong!")
    return X_smoothed


# ─── SVM Training ───────────────────────────────────────────────────────────

def train_svm_simple(X_train, y_train, kernel="linear", C=1.0):
    """Pipeline đơn giản: StandardScaler → SVM."""
    if kernel == "linear":
        # LinearSVC nhanh gấp 50x so với SVC(kernel='linear')
        # max_iter cao hơn để đảm bảo hội tụ với data lớn
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', LinearSVC(C=C, max_iter=5000, random_state=42,
                              dual='auto'))
        ])
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel=kernel, C=C, random_state=42))
        ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_svm_gridsearch(X_train, y_train):
    """
    Tìm hyperparameter tốt nhất bằng GridSearchCV.

    Tối ưu:
    - Dùng LinearSVC cho linear kernel (nhanh hơn nhiều)
    - Subsample cho RBF kernel (tránh kernel matrix O(n²) quá lớn)
    """
    n_train = len(X_train)

    # ── Phần 1: GridSearch Linear (nhanh — dùng LinearSVC) ──
    print("  [1/2] GridSearch Linear kernel...")
    linear_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', LinearSVC(max_iter=5000, random_state=42, dual='auto'))
    ])
    linear_grid = GridSearchCV(
        linear_pipeline,
        {'svm__C': [0.01, 0.1, 1, 10, 100]},
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='accuracy', n_jobs=-1, verbose=0, refit=True
    )
    start = time.time()
    linear_grid.fit(X_train, y_train)
    print(f"    Linear best: C={linear_grid.best_params_['svm__C']}"
          f" → {linear_grid.best_score_*100:.2f}%  ({time.time()-start:.1f}s)")

    # ── Phần 2: GridSearch RBF (chậm — subsample nếu data lớn) ──
    print("  [2/2] GridSearch RBF kernel...")
    MAX_SAMPLES_RBF = 15000  # Giới hạn samples cho RBF GridSearch

    if n_train > MAX_SAMPLES_RBF:
        print(f"    Subsample {MAX_SAMPLES_RBF}/{n_train} cho RBF (tránh quá tải)...")
        rng = np.random.RandomState(42)
        idx = rng.choice(n_train, MAX_SAMPLES_RBF, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]
    else:
        X_sub, y_sub = X_train, y_train

    rbf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', random_state=42))
    ])
    rbf_grid = GridSearchCV(
        rbf_pipeline,
        {'svm__C': [1, 10, 100], 'svm__gamma': ['scale', 0.01]},
        cv=StratifiedKFold(3, shuffle=True, random_state=42),
        scoring='accuracy', n_jobs=-1, verbose=0, refit=True
    )
    start = time.time()
    rbf_grid.fit(X_sub, y_sub)
    print(f"    RBF best: C={rbf_grid.best_params_['svm__C']}, "
          f"gamma={rbf_grid.best_params_['svm__gamma']}"
          f" → {rbf_grid.best_score_*100:.2f}%  ({time.time()-start:.1f}s)")

    # ── So sánh và chọn tốt nhất ──
    if linear_grid.best_score_ >= rbf_grid.best_score_:
        print(f"\n  >> Chọn: Linear (CV={linear_grid.best_score_*100:.2f}%)")
        best = linear_grid.best_estimator_
        best_params = {'kernel': 'linear', **linear_grid.best_params_}
        best_score = linear_grid.best_score_
    else:
        print(f"\n  >> Chọn: RBF (CV={rbf_grid.best_score_*100:.2f}%)")
        # Train RBF trên TOÀN BỘ data với best params (nếu đã subsample)
        if n_train > MAX_SAMPLES_RBF:
            print(f"    Re-training RBF trên toàn bộ {n_train} samples...")
            best = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf',
                            C=rbf_grid.best_params_['svm__C'],
                            gamma=rbf_grid.best_params_['svm__gamma'],
                            random_state=42))
            ])
            best.fit(X_train, y_train)
        else:
            best = rbf_grid.best_estimator_
        best_params = {'kernel': 'rbf', **rbf_grid.best_params_}
        best_score = rbf_grid.best_score_

    return best, best_params, best_score


# ─── Cross-Validation ───────────────────────────────────────────────────────

def cross_validate_svm(X, y, kernel='linear', C=1.0, n_folds=10):
    """
    10-fold stratified cross-validation.
    Stratified = giữ tỷ lệ classes giống nhau trong mỗi fold.
    """
    if kernel == 'linear':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', LinearSVC(C=C, max_iter=5000, random_state=42,
                              dual='auto'))
        ])
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel=kernel, C=C, random_state=42))
        ])

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    scores = cross_val_score(pipeline, X, y, cv=cv,
                              scoring='accuracy', n_jobs=-1)

    print(f"\n  10-Fold CV Results:")
    print(f"  {'Fold':<6} {'Accuracy':>10}")
    print(f"  {'---':<6} {'---':>10}")
    for i, s in enumerate(scores, 1):
        print(f"  {i:<6} {s*100:>9.2f}%")
    print(f"  {'---':<6} {'---':>10}")
    print(f"  {'Mean':<6} {scores.mean()*100:>9.2f}%")
    print(f"  {'Std':<6} {scores.std()*100:>9.2f}%")

    return scores


# ─── Evaluation + Visualization ─────────────────────────────────────────────

def evaluate_and_save(model, X_test, y_test, label_type="2class",
                      best_params=None, cv_score=None):
    """Đánh giá mô hình + vẽ confusion matrix + lưu kết quả JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    names = (["Negative", "Positive"] if label_type == "2class"
             else ["Vui ve", "So hai", "Buon", "Thu gian"])

    # ── In báo cáo ──
    print(f"\n  {'='*40}")
    print(f"  SVM --- {label_type}")
    print(f"  {'='*40}")
    print(f"  Accuracy:      {acc*100:.2f}%")
    print(f"  F1 (weighted): {f1*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=names))

    # ── Vẽ Confusion Matrix ──
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'SVM Confusion Matrix ({label_type})\n'
                 f'Accuracy: {acc*100:.2f}%', fontsize=14)
    plt.tight_layout()

    fig_path = os.path.join(RESULTS_DIR,
                            f"confusion_matrix_svm_{label_type}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # ── Lưu kết quả JSON ──
    results = {
        "model": "SVM",
        "label_type": label_type,
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1, 4),
        "best_params": str(best_params) if best_params else None,
        "cv_score": round(cv_score, 4) if cv_score else None,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=names, output_dict=True
        ),
    }

    json_path = os.path.join(RESULTS_DIR, f"svm_results_{label_type}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  -> Saved: {fig_path}")
    print(f"  -> Saved: {json_path}")
    return results


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    USE_LDS = True           # Bật/tắt LDS smoothing
    USE_GRIDSEARCH = True    # Bật/tắt GridSearch (tắt để chạy nhanh hơn)

    all_results = {}

    for label_type in ["2class", "4class"]:
        print(f"\n{'='*60}")
        print(f"  EmoWave SVM Pipeline --- {label_type}")
        print(f"{'='*60}")

        # 1. Load data
        print("\n[1] Loading DEAP data...")
        X, y, groups = load_all_subjects(label_type=label_type)
        # X shape: (n_epochs, 32, 128)

        # 2. Extract features
        print("\n[2] Feature extraction (Power Spectrum + Asymmetry)...")
        X_feat = extract_features_dataset(X)
        # X_feat shape: (n_epochs, 230)

        # Giải phóng raw data — không cần nữa
        del X
        gc.collect()
        print("  (Freed raw EEG data from memory)")

        # 3. LDS smoothing (optional)
        if USE_LDS:
            print("\n[3] LDS smoothing...")
            X_feat = apply_lds(X_feat, epochs_per_trial=60, alpha=0.3)
        else:
            print("\n[3] LDS smoothing --- SKIPPED")

        # 4. Train/test split
        print("\n[4] Train/test split (80/20, stratified)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_feat, y, test_size=0.2, random_state=42, stratify=y
        )
        del X_feat  # Giải phóng
        gc.collect()
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

        # 5. Train SVM
        if USE_GRIDSEARCH:
            print("\n[5] GridSearchCV (Linear + RBF)...")
            model, best_params, cv_score = train_svm_gridsearch(
                X_train, y_train
            )
        else:
            print("\n[5] Training SVM (linear, C=1.0)...")
            model = train_svm_simple(X_train, y_train,
                                     kernel='linear', C=1.0)
            best_params, cv_score = None, None

        # 6. Evaluate
        print("\n[6] Evaluation...")
        results = evaluate_and_save(
            model, X_test, y_test, label_type, best_params, cv_score
        )
        all_results[label_type] = results

        # Giải phóng cho vòng tiếp theo
        del model, X_train, X_test, y_train, y_test
        gc.collect()

    # ── Tổng kết ──
    print(f"\n{'='*60}")
    print("  SVM Pipeline hoan tat!")
    print(f"{'='*60}")
    print(f"\n  {'Label Type':<12} {'Accuracy':>10} {'F1':>10}")
    print(f"  {'---':<12} {'---':>10} {'---':>10}")
    for lt, r in all_results.items():
        print(f"  {lt:<12} {r['accuracy']*100:>9.2f}% {r['f1_weighted']*100:>9.2f}%")
    print(f"\n  Ket qua luu tai: {os.path.abspath(RESULTS_DIR)}/")
