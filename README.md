# EmoWave 🧠
**EEG-Based Emotion Recognition — From SVM to Deep Learning**

Bài tập lớn môn Machine Learning | Nhóm 3 người | 3 tuần

---

## Cấu trúc project

```
emowave/
│
├── data/
│   └── deap/          ← Đặt s01.dat → s32.dat vào đây
│
├── utils/
│   └── deap_loader.py ← Load & tiền xử lý data (DÙNG CHUNG)
│
├── models/
│   ├── svm_model.py   ← Người 1
│   ├── cnn_model.py   ← Người 2
│   └── lstm_model.py  ← Người 3
│
├── results/           ← Kết quả, confusion matrix, biểu đồ
├── notebooks/         ← Jupyter notebooks thử nghiệm
│
├── requirements.txt
└── README.md
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## Chạy nhanh

```python
from utils.deap_loader import load_all_subjects, prepare_for_svm

# Load data (2 lớp: Positive / Negative)
X, y, groups = load_all_subjects(label_type="2class")

# Hoặc 4 lớp: Vui vẻ / Sợ hãi / Buồn / Thư giãn
X, y, groups = load_all_subjects(label_type="4class")

# Chuẩn bị cho từng model
X_tr, X_te, y_tr, y_te, scaler = prepare_for_svm(X, y)    # Người 1
X_tr, X_te, y_tr, y_te         = prepare_for_cnn(X, y)    # Người 2
X_tr, X_te, y_tr, y_te         = prepare_for_lstm(X, y)   # Người 3
```

---

## Dataset

**DEAP** (Koelstra et al., 2012): 32 subjects, 40 trials/subject, 32 kênh EEG @ 128Hz.
- Nhãn: Valence, Arousal, Dominance, Liking (thang 1–9)
- Download: đặt file vào `data/deap/`

---

## Phân công

| Người | Model | File |
|-------|-------|------|
| Người 1 | SVM + LDS baseline | `models/svm_model.py` |
| Người 2 | CNN 1D | `models/cnn_model.py` |
| Người 3 | LSTM / BiLSTM | `models/lstm_model.py` |

---

## Tài liệu tham khảo

- Wang et al. (2014). *Emotional state classification from EEG data using machine learning approach.* Neurocomputing, 129, 94–106.
- Koelstra et al. (2012). *DEAP: A database for emotion analysis using physiological signals.* IEEE Transactions on Affective Computing.
