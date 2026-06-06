# Kiến Thức Nền Tảng để Làm Dự Án EEG Emotion Recognition

> Tài liệu này bao gồm tất cả kiến thức cần thiết để hiểu và thực hiện dự án **EmoWave**.  
> Đọc theo thứ tự từ trên xuống — mỗi phần xây dựng trên phần trước.

---

## MỤC LỤC

1. [EEG là gì?](#1-eeg-là-gì)
2. [Cảm xúc và não bộ](#2-cảm-xúc-và-não-bộ)
3. [Bộ dữ liệu DEAP](#3-bộ-dữ-liệu-deap)
4. [Tiền xử lý tín hiệu EEG](#4-tiền-xử-lý-tín-hiệu-eeg)
5. [Trích xuất đặc trưng (Feature Extraction)](#5-trích-xuất-đặc-trưng)
6. [Các mô hình Machine Learning trong dự án](#6-các-mô-hình-machine-learning)
7. [Đánh giá mô hình](#7-đánh-giá-mô-hình)
8. [Kiến thức Python cần thiết](#8-kiến-thức-python-cần-thiết)
9. [Luồng dữ liệu trong dự án](#9-luồng-dữ-liệu-trong-dự-án)
10. [Câu hỏi thường gặp](#10-câu-hỏi-thường-gặp)

---

## 1. EEG là gì?

### 1.1 Định nghĩa

**EEG (Electroencephalography — Điện não đồ)** là kỹ thuật đo hoạt động điện của não bộ thông qua các điện cực đặt trên da đầu. Khi hàng triệu tế bào thần kinh (neuron) cùng hoạt động, chúng tạo ra các xung điện nhỏ — EEG ghi lại những xung này.

Hình dung đơn giản:
```
Não bộ hoạt động → Tạo ra điện → Điện cực trên đầu thu tín hiệu → Máy tính lưu lại
```

### 1.2 Tại sao dùng EEG?

So sánh với các phương pháp khác để nhận biết cảm xúc:

| Phương pháp | Ưu điểm | Nhược điểm |
|---|---|---|
| Nhận diện khuôn mặt | Không tiếp xúc, dễ dùng | Dễ giả vờ, không phản ánh cảm xúc thật |
| Nhịp tim, da (ECG, SC) | Khách quan hơn | Chậm, ít thông tin |
| **EEG** | **Trực tiếp từ não, khó giả vờ, phản ứng nhanh** | Cần đội mũ điện cực |
| fMRI | Độ phân giải không gian cao | Đắt, cồng kềnh, không thực tế |

### 1.3 Các dải tần EEG

Tín hiệu EEG được phân tích theo tần số (Hz). Mỗi dải tần liên quan đến trạng thái não khác nhau:

| Dải tần | Tần số | Trạng thái não | Liên quan cảm xúc |
|---|---|---|---|
| **Delta** | 0.5 – 4 Hz | Ngủ sâu | Ít liên quan |
| **Theta** | 4 – 8 Hz | Buồn ngủ, thiền | Ít liên quan |
| **Alpha** | 8 – 13 Hz | Thư giãn, nhắm mắt | **Liên quan nhiều** |
| **Beta** | 13 – 30 Hz | Tập trung, căng thẳng | **Liên quan nhiều** |
| **Gamma** | 30 – 50 Hz | Xử lý thông tin cao | **Liên quan nhiều** |

> **Kết quả từ bài báo gốc:** Alpha, Beta, Gamma đóng vai trò quan trọng hơn Delta và Theta trong nhận diện cảm xúc.

### 1.4 Hệ thống điện cực 10-20

Các điện cực được đặt theo tiêu chuẩn quốc tế **10-20 system** — đây là cách đặt tên vị trí điện cực:

```
Tiền tố:  F = Frontal (trán)      → liên quan xử lý cảm xúc
          C = Central (trung tâm)
          P = Parietal (đỉnh)     → liên quan nhận thức không gian
          T = Temporal (thái dương)
          O = Occipital (chẩm)    → liên quan thị giác

Số lẻ = bán cầu TRÁI, Số chẵn = bán cầu PHẢI
Ví dụ: F3 = điện cực số 3 ở vùng trán trái
        C4 = điện cực số 4 ở vùng trung tâm phải
```

Trong DEAP: 32 kênh EEG đầu tiên theo hệ thống này + 8 kênh peripheral (nhịp tim, da...).

---

## 2. Cảm xúc và Não bộ

### 2.1 Mô hình Valence-Arousal (2D)

Bài báo gốc và dự án này dùng mô hình 2 chiều để biểu diễn cảm xúc:

```
         HIGH AROUSAL (kích động cao)
              |
    Sợ hãi   |   Vui vẻ / Hào hứng
    Tức giận |   Phấn khích
             |
NEGATIVE ----+---- POSITIVE  (VALENCE)
(tiêu cực)   |     (tích cực)
             |
    Buồn bã  |   Thư giãn
    Chán nản |   Bình yên
              |
          LOW AROUSAL (kích động thấp)
```

**Valence** = chiều ngang = cảm xúc dễ chịu hay khó chịu (1-9)  
**Arousal** = chiều dọc = mức độ kích thích, năng lượng (1-9)

### 2.2 Phân loại trong dự án

**Bài toán 2 lớp (Binary):**
```
Valence >= 5  →  Positive (tích cực)  →  nhãn = 1
Valence <  5  →  Negative (tiêu cực)  →  nhãn = 0
```

**Bài toán 4 lớp (Multi-class):**
```
Valence >= 5 AND Arousal >= 5  →  Q1: Vui vẻ     →  nhãn = 0
Valence <  5 AND Arousal >= 5  →  Q2: Sợ hãi     →  nhãn = 1
Valence <  5 AND Arousal <  5  →  Q3: Buồn bã    →  nhãn = 2
Valence >= 5 AND Arousal <  5  →  Q4: Thư giãn   →  nhãn = 3
```

### 2.3 Bất đối xứng EEG và Cảm xúc

Một phát hiện quan trọng từ khoa học thần kinh:

- **Bán cầu trái** hoạt động mạnh hơn → cảm xúc **tích cực**, tiếp cận
- **Bán cầu phải** hoạt động mạnh hơn → cảm xúc **tiêu cực**, né tránh

Chỉ số bất đối xứng = `Power(kênh trái) - Power(kênh phải)`  
Ví dụ: F3 - F4, C3 - C4 (dùng trong phần trích xuất đặc trưng)

---

## 3. Bộ Dữ Liệu DEAP

### 3.1 Tổng quan

**DEAP** = Database for Emotion Analysis using Physiological signals  
Tác giả: Koelstra et al. (2012), Queen Mary University of London

| Thông số | Giá trị |
|---|---|
| Số người tham gia | 32 (50% nam, 50% nữ) |
| Số video kích thích | 40 đoạn nhạc 1 phút |
| Số kênh EEG | 32 |
| Số kênh peripheral | 8 (EMG, EOG, GSR, hô hấp...) |
| Tần số lấy mẫu gốc | 512 Hz |
| Tần số sau xử lý | 128 Hz |
| Nhãn | Valence, Arousal, Dominance, Liking (1-9) |

### 3.2 Cấu trúc file

Mỗi file `s01.dat` → `s32.dat` chứa dữ liệu của 1 người:

```python
subject = pickle.load(open("s01.dat", "rb"), encoding="latin1")

subject["data"]    # shape: (40, 40, 8064)
#                            │   │   └── 8064 samples = 63 giây × 128Hz
#                            │   └────── 40 channels (32 EEG + 8 peripheral)
#                            └────────── 40 trials (40 video clips)

subject["labels"]  # shape: (40, 4)
#                            │   └── [valence, arousal, dominance, liking]
#                            └────── 40 trials
```

### 3.3 Tại sao 63 giây, không phải 60?

Mỗi video dài 60 giây + 3 giây baseline trước đó = 63 giây.  
**Baseline** là 3 giây im lặng trước khi video bắt đầu — dùng để đo trạng thái não "bình thường".  
Ta phải **bỏ 3 giây đầu** (384 samples) trước khi phân tích.

```python
# Bỏ baseline
signal = data[trial_idx, :32, 384:]  # còn lại 7680 samples = 60 giây
```

### 3.4 Preprocessing đã có trong DEAP

DEAP đã thực hiện sẵn các bước xử lý sau:
- Downsample từ 512Hz → 128Hz
- Lọc bandpass 4-45 Hz (bỏ tần số quá thấp và quá cao)
- EOG artifact removal (loại bỏ nhiễu do chớp mắt)
- Chuẩn hóa dữ liệu

→ **Nhóm không cần làm lại những bước này**, chỉ cần cắt epoch và trích xuất features.

---

## 4. Tiền Xử Lý Tín Hiệu EEG

### 4.1 Cắt Epoch (Windowing)

Thay vì phân tích toàn bộ 60 giây, ta cắt thành các cửa sổ nhỏ:

```
|---60 giây tín hiệu EEG---|
 [1s][1s][1s][1s]...[1s]
  ↑ mỗi đoạn 1s = 128 samples là 1 "epoch"
  ↑ 60 giây → 60 epochs
```

Tại sao 1 giây? Vì bài báo gốc thử nghiệm 0.5s, 1s, 1.5s, 2s → 1s cho kết quả tốt nhất.

```python
# Cắt epoch không chồng lấp (non-overlapping)
n_epochs = n_samples // epoch_len   # 7680 // 128 = 60 epochs
epochs = signal.reshape(-1, n_channels, epoch_len)  # (60, 32, 128)
```

### 4.2 Chuẩn hóa (Normalization)

Tín hiệu EEG giữa các người có biên độ rất khác nhau → cần chuẩn hóa:

```python
# Z-score normalization cho mỗi channel
mean = signal.mean(axis=-1, keepdims=True)  # mean theo thời gian
std  = signal.std(axis=-1, keepdims=True)
signal_norm = (signal - mean) / (std + 1e-8)
```

### 4.3 LDS Feature Smoothing (từ bài báo gốc)

**Vấn đề:** Tín hiệu EEG thay đổi rất nhanh (noisy), nhưng cảm xúc thay đổi chậm.  
**Giải pháp:** Dùng Linear Dynamical System (LDS) để làm mượt features.

Hình dung: LDS giống như lấy trung bình trượt nhưng thông minh hơn — nó học được mẫu biến đổi theo thời gian.

```
Features gốc:  [1.2, 0.3, 1.8, 0.1, 1.5, ...]  ← lộn xộn, nhiễu nhiều
Sau LDS:       [1.0, 0.8, 1.2, 1.1, 1.3, ...]  ← mượt hơn, xu hướng rõ hơn
```

Trong code, LDS được implement bằng Kalman Filter (sklearn hoặc tự viết).

---

## 5. Trích Xuất Đặc Trưng

### 5.1 Power Spectrum (Phổ công suất) — Người 1 chủ yếu dùng

**Câu hỏi:** Trong 1 epoch 1 giây, tần số nào chiếm ưu thế?

**Cách tính:** Dùng Fast Fourier Transform (FFT)

```python
import numpy as np

def compute_power_spectrum(epoch, sfreq=128):
    """
    epoch: (n_channels, n_samples) = (32, 128)
    return: (n_channels, n_bands) = (32, 5)
    """
    # FFT
    fft_vals = np.abs(np.fft.rfft(epoch, axis=-1)) ** 2
    freqs    = np.fft.rfftfreq(epoch.shape[-1], 1/sfreq)

    # Tính năng lượng trong từng dải tần
    bands = {
        'delta': (0.5, 4),
        'theta': (4,   8),
        'alpha': (8,  13),
        'beta':  (13, 30),
        'gamma': (30, 50),
    }

    features = []
    for band_name, (lo, hi) in bands.items():
        idx  = np.where((freqs >= lo) & (freqs <= hi))[0]
        power = np.log(fft_vals[:, idx].mean(axis=-1) + 1e-8)  # log transform
        features.append(power)  # (32,)

    return np.stack(features, axis=-1)  # (32, 5)
```

**Ý nghĩa:** Nếu alpha power bên trái cao hơn bên phải → có thể là cảm xúc tích cực.

### 5.2 Wavelet Features

FFT cho biết "tần số nào" nhưng không biết "ở thời điểm nào". Wavelet giải quyết vấn đề này:

```python
import pywt

def compute_wavelet_energy(epoch, wavelet='db4', level=5):
    """
    epoch: (n_channels, n_samples)
    Daubechies wavelet bậc 4, 5 levels tương ứng 5 dải tần
    """
    energies = []
    for ch in range(epoch.shape[0]):
        coeffs = pywt.wavedec(epoch[ch], wavelet, level=level)
        energy = [np.sum(c**2) for c in coeffs]
        energies.append(energy)
    return np.array(energies)  # (32, 6)
```

### 5.3 Differential Asymmetry Features

```python
# Các cặp điện cực đối xứng
ASYMMETRY_PAIRS = [
    (0, 1),   # FP1-FP2
    (2, 3),   # AF3-AF4
    # ... (27 cặp tổng cộng)
]

def compute_asymmetry(power_features):
    """power_features: (32, 5) — power của 32 channels, 5 bands"""
    asymmetry = []
    for left, right in ASYMMETRY_PAIRS:
        diff = power_features[left] - power_features[right]  # (5,)
        asymmetry.append(diff)
    return np.array(asymmetry)  # (27, 5)
```

---

## 6. Các Mô Hình Machine Learning

### 6.1 SVM (Support Vector Machine) — Người 1

**Ý tưởng cốt lõi:** Tìm đường thẳng (hyperplane) phân chia 2 lớp với lề lớn nhất.

```
Cảm xúc Negative: ●  ●  ●
                         |
      ← Negative lề      | lề Positive →
                         |
Cảm xúc Positive: ▲  ▲  ▲
```

**Kernel** = cách biến đổi data trước khi tìm hyperplane:
- `linear`: đường thẳng đơn giản → dùng khi data gần tuyến tính
- `rbf`: đường cong → dùng khi data phức tạp hơn
- `poly`: đa thức

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# QUAN TRỌNG: SVM cần scale data trước
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train_scaled, y_train)

# Predict
y_pred = svm.predict(X_test_scaled)
```

**C** = tham số regularization:
- C nhỏ → cho phép sai nhiều nhưng lề rộng (generalize tốt hơn)
- C lớn → ít sai trên train nhưng dễ overfit

### 6.2 CNN 1D (Convolutional Neural Network) — Người 2

**Ý tưởng:** Thay vì trích xuất features thủ công, để CNN tự học features từ raw EEG.

```
Input: (batch, 32 channels, 128 timesteps)
         │
    ┌────▼────────────────────────────────┐
    │  Conv1d(32→64, kernel=3)            │  ← học patterns ngắn
    │  BatchNorm + ReLU                   │
    │  MaxPool(2) → 64 timesteps          │
    ├────────────────────────────────────┤
    │  Conv1d(64→128, kernel=3)           │  ← học patterns phức tạp hơn
    │  BatchNorm + ReLU                   │
    │  MaxPool(2) → 32 timesteps          │
    ├────────────────────────────────────┤
    │  Conv1d(128→256, kernel=3)          │
    │  AdaptiveAvgPool → (256, 1)         │
    └────▼────────────────────────────────┘
         │
    ┌────▼────────────────────────────────┐
    │  Linear(256→64) + ReLU + Dropout    │  ← Fully connected
    │  Linear(64→n_classes)               │
    └────▼────────────────────────────────┘
         │
    Output: (batch, n_classes)
```

**Conv1d khác Conv2d như thế nào?**
- Conv2d: dùng cho ảnh (2D)
- Conv1d: dùng cho chuỗi thời gian (1D) — phù hợp với EEG

**BatchNorm:** Chuẩn hóa activations sau mỗi layer → training ổn định hơn  
**Dropout:** Ngẫu nhiên tắt một số neuron → tránh overfitting  

```python
import torch.nn as nn

class EEG_CNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # ... thêm layers
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        # x: (batch, 32, 128)
        out = self.net(x)        # (batch, 256, 1)
        out = out.squeeze(-1)    # (batch, 256)
        return self.fc(out)      # (batch, n_classes)
```

### 6.3 LSTM (Long Short-Term Memory) — Người 3

**Vấn đề của RNN thường:** Khi chuỗi dài, thông tin cũ bị "quên" dần.  
**LSTM giải quyết:** Có cơ chế "cửa" (gate) để quyết định nhớ hay quên thông tin.

```
Input EEG:  timestep 1 → timestep 2 → timestep 3 → ... → timestep 128
                │              │              │                  │
            ┌───▼──┐       ┌───▼──┐       ┌───▼──┐         ┌───▼──┐
            │ LSTM │──────▶│ LSTM │──────▶│ LSTM │──────▶ │ LSTM │
            │ cell │       │ cell │       │ cell │         │ cell │
            └──────┘       └──────┘       └──────┘         └───┬──┘
                                                               │
                                                          Output → Classifier
```

**Input shape cho LSTM:** `(batch, timesteps, features)` = `(batch, 128, 32)`  
Lưu ý: CNN dùng `(batch, 32, 128)`, LSTM dùng `(batch, 128, 32)` — ngược nhau!

```python
class EEG_LSTM(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=32,      # 32 channels là features tại mỗi timestep
            hidden_size=128,
            num_layers=2,
            batch_first=True,   # (batch, seq, features)
            bidirectional=False,
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, 128, 32)
        out, _ = self.lstm(x)   # out: (batch, 128, 128)
        out = out[:, -1, :]     # lấy output của timestep cuối: (batch, 128)
        return self.fc(out)
```

**BiLSTM (Bidirectional):** Đọc chuỗi cả thuận và ngược → nắm thông tin tốt hơn.  
Chỉ cần thêm `bidirectional=True` và nhân đôi hidden_size trong fc layer.

---

## 7. Đánh Giá Mô Hình

### 7.1 Các chỉ số quan trọng

**Accuracy** = tỷ lệ dự đoán đúng tổng thể:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision** = trong những gì mô hình nói là "Positive", bao nhiêu % thật sự là Positive:
```
Precision = TP / (TP + FP)
```

**Recall** = trong tất cả các "Positive" thật, mô hình tìm ra được bao nhiêu %:
```
Recall = TP / (TP + FN)
```

**F1-Score** = trung bình điều hòa của Precision và Recall:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### 7.2 Confusion Matrix

Với bài toán 4 lớp:

```
                    Dự đoán
                Vui  Sợ  Buồn  Thư giãn
Thực tế  Vui  [ 85   5    3      7  ]
         Sợ   [  8  78    9      5  ]
         Buồn [  4   7   82      7  ]
       Thư g. [  6   4    5     85  ]

Diagonal = dự đoán đúng
Off-diagonal = nhầm lẫn giữa các lớp
```

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Vẽ confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['Vui', 'Sợ', 'Buồn', 'Thư giãn'],
            yticklabels=['Vui', 'Sợ', 'Buồn', 'Thư giãn'])
plt.savefig('results/confusion_matrix.png')
```

### 7.3 Cross-validation (10-fold)

Thay vì split train/test 1 lần, làm 10 lần với các phần data khác nhau:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(svm_model, X, y, cv=10, scoring='accuracy')
print(f"Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
```

### 7.4 Overfitting và cách phát hiện

**Overfitting** = mô hình học thuộc data train nhưng dự đoán tệ trên data mới.

```
Dấu hiệu:
  Train accuracy: 99%
  Test accuracy:  65%   ← chênh lệch lớn = overfitting

Giải pháp:
  - Thêm Dropout
  - Giảm model complexity
  - Thêm data (data augmentation)
  - Early stopping
```

---

## 8. Kiến Thức Python Cần Thiết

### 8.1 NumPy — xử lý mảng số

```python
import numpy as np

# Tạo mảng
arr = np.array([[1, 2, 3], [4, 5, 6]])  # shape: (2, 3)

# Shape và indexing
print(arr.shape)     # (2, 3)
arr[0, :]            # hàng đầu: [1, 2, 3]
arr[:, 2]            # cột thứ 3: [3, 6]
arr[:, :32, :]       # 32 cột đầu (dùng cho lấy 32 kênh EEG)

# Phép toán
arr.mean(axis=0)     # mean theo cột
arr.std(axis=-1)     # std theo chiều cuối
np.concatenate([a, b], axis=0)  # ghép 2 mảng
```

### 8.2 PyTorch — deep learning

```python
import torch
import torch.nn as nn

# Tensor (giống numpy array nhưng chạy trên GPU)
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
x = x.cuda()    # chuyển lên GPU
x = x.cpu()     # chuyển về CPU

# DataLoader — đọc data theo batch
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X_tensor, y_tensor)
loader  = DataLoader(dataset, batch_size=64, shuffle=True)

for X_batch, y_batch in loader:
    # train model...
    pass

# Training loop cơ bản
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model.train()
for X_batch, y_batch in train_loader:
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    optimizer.zero_grad()          # xóa gradient cũ
    output = model(X_batch)        # forward pass
    loss   = criterion(output, y_batch)
    loss.backward()                # tính gradient
    optimizer.step()               # cập nhật weights
```

### 8.3 Scikit-learn — ML truyền thống

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Pipeline chuẩn
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit + transform
X_test  = scaler.transform(X_test)       # chỉ transform (KHÔNG fit lại)

model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### 8.4 Matplotlib / Seaborn — vẽ biểu đồ

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Vẽ loss curve
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses,   label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train')
plt.plot(val_accs,   label='Validation')
plt.title('Accuracy Curve')
plt.legend()
plt.tight_layout()
plt.savefig('results/training_curves.png', dpi=150)
```

---

## 9. Luồng Dữ Liệu Trong Dự Án

### 9.1 Tổng quan pipeline

```
s01.dat → s32.dat     (32 files pickle)
      │
      ▼
  deap_loader.py
      │
      ├─ load_all_subjects()
      │     └─ Đọc file → bỏ baseline → cắt epoch → gán nhãn
      │
      ├─ prepare_for_svm()     → X: (N, 4096)      y: (N,)
      │     └─ flatten epoch + StandardScaler
      │
      ├─ prepare_for_cnn()     → X: (N, 32, 128)   y: (N,)
      │     └─ normalize theo channel
      │
      └─ prepare_for_lstm()    → X: (N, 128, 32)   y: (N,)
            └─ normalize + transpose
                  │
      ┌───────────┼───────────┐
      ▼           ▼           ▼
  svm_model   cnn_model   lstm_model
      │           │           │
      └───────────┴───────────┘
                  │
            So sánh kết quả
            Confusion matrix
            Báo cáo LaTeX
```

### 9.2 Kích thước dữ liệu qua từng bước

```
32 subjects × 40 trials × 60 epochs = 76,800 epochs tổng
Mỗi epoch: (32 channels × 128 samples)

Sau prepare_for_svm:
  X_train: (61,440, 4096)  — 80% × 76,800 = 61,440
  X_test:  (15,360, 4096)  — 20%

Sau prepare_for_cnn:
  X_train: (61,440, 32, 128)
  X_test:  (15,360, 32, 128)

Sau prepare_for_lstm:
  X_train: (61,440, 128, 32)
  X_test:  (15,360, 128, 32)
```

### 9.3 Cấu trúc thư mục cuối cùng

```
emowave/
├── data/
│   ├── deap/
│   │   ├── s01.dat → s32.dat   ← raw data
│   └── cache/
│       ├── X_2class.npy        ← lưu lại sau khi process
│       └── y_2class.npy
│
├── utils/
│   └── deap_loader.py          ← đã xong ✓
│
├── models/
│   ├── svm_model.py            ← Người 1
│   ├── cnn_model.py            ← Người 2
│   └── lstm_model.py           ← Người 3
│
├── results/
│   ├── confusion_matrix_svm.png
│   ├── confusion_matrix_cnn.png
│   ├── confusion_matrix_lstm.png
│   ├── training_curves.png
│   └── comparison_table.csv    ← bảng so sánh accuracy
│
├── notebooks/
│   └── exploration.ipynb       ← khám phá data ban đầu
│
├── report/
│   ├── main.tex
│   └── figures/
│
├── requirements.txt            ← đã xong ✓
└── README.md                   ← đã xong ✓
```

---

## 10. Câu Hỏi Thường Gặp

**Q: Tại sao accuracy của mô hình chênh lệch nhiều giữa các subject?**  
A: Tín hiệu EEG rất khác nhau giữa người với người (inter-subject variability). Đây là thách thức lớn nhất trong EEG research. Mô hình train trên người A chưa chắc dự đoán tốt cho người B.

**Q: Bài toán 4 lớp khó hơn 2 lớp bao nhiêu?**  
A: Đáng kể. Với 2 lớp, random guess = 50%. Với 4 lớp, random guess = 25%. Accuracy tốt với 4 lớp thường thấp hơn 2 lớp khoảng 10-15%.

**Q: CNN và LSTM cái nào tốt hơn cho EEG?**  
A: Phụ thuộc vào bài toán. CNN tốt ở việc học patterns không gian (spatial patterns giữa các channels). LSTM tốt ở việc học patterns thời gian (temporal patterns). Kết quả thực nghiệm thường cho thấy CNN nhỉnh hơn một chút nhưng LSTM hiểu "câu chuyện theo thời gian" tốt hơn.

**Q: Nếu train rất lâu mà loss không giảm thì làm gì?**  
A: Thử: giảm learning rate, kiểm tra data có bị normalize chưa, tăng batch size, hoặc thay optimizer (Adam → SGD với momentum).

**Q: Cần bao nhiêu epochs để train?**  
A: Với dataset này, CNN/LSTM thường hội tụ sau 20-50 epochs. Dùng early stopping để tránh overfit:
```python
# Dừng train nếu val_loss không giảm sau 5 epochs liên tiếp
best_val_loss = float('inf')
patience = 5
counter  = 0
for epoch in range(100):
    val_loss = evaluate(...)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            break
```

**Q: Kết quả trong bài báo gốc là bao nhiêu?**  
A: ~87.53% với SVM linear trên 6 subjects. Với DEAP (32 subjects), accuracy thường thấp hơn vì data đa dạng hơn — expect khoảng 70-85% cho 2 lớp là bình thường.

---

## Tài Liệu Tham Khảo

1. Wang et al. (2014). *Emotional state classification from EEG data using machine learning approach*. Neurocomputing, 129, 94–106.
2. Koelstra et al. (2012). *DEAP: A database for emotion analysis using physiological signals*. IEEE Trans. Affective Computing, 3(1), 18–31.
3. PyTorch Documentation: https://pytorch.org/docs/stable
4. Scikit-learn Documentation: https://scikit-learn.org/stable

---

*Tài liệu này được tạo cho dự án EmoWave — bài tập lớn môn Machine Learning.*
