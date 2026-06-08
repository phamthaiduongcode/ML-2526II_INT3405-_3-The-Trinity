# EEG Emotion Recognition — BiLSTM

Phân loại cảm xúc từ tín hiệu EEG dùng BiLSTM trên bộ dữ liệu DEAP.

---

## Cấu trúc dự án

```
EEG_Project/
├── data/
│   ├── raw/            ← File .dat gốc (không push lên Git)
│   └── processed/      ← File .npy sau khi chạy preprocess (không push lên Git)
│
├── src/
│   ├── models/
│   │   └── eeg_bilstm.py   ← EEG_BiLSTM (Exp1/2) + BiLSTM_Model (Exp3)
│   ├── utils/
│   │   ├── dataset.py      ← set_seed, EEGDataset, get_dataloaders
│   │   └── metrics.py      ← evaluate_metrics, plot_confusion_matrix, plot_learning_curves
│   └── data_pipeline/
│       ├── load_raw.py     ← Tải DEAP từ Kaggle
│       └── preprocess.py   ← Tiền xử lý, tạo file .npy
│
├── experiments/
│   ├── exp1_2class.py  ← Train Exp 1 (K-Fold, 2-class)
│   ├── exp2_4class.py  ← Train Exp 2 (K-Fold, 4-class)
│   ├── exp3_loso.py    ← Train Exp 3 (LOSO)
│   └── run_tsne.py     ← Vẽ t-SNE từ weights Exp 1
│
├── result/
│   ├── logs/           ← File .json kết quả
│   ├── plots/          ← Ảnh Confusion Matrix, Learning Curve, t-SNE
│   └── checkpoints/    ← Weights .pth và index .npy
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## Quy trình chạy

### Bước 1 — Tải dữ liệu

```bash
python -m src.data_pipeline.load_raw
```

### Bước 2 — Tiền xử lý

```bash
python -m src.data_pipeline.preprocess
```

Tạo ra 4 file trong `data/processed/`: `X_epochs.npy`, `y_valence.npy`, `y_arousal.npy`, `subject_groups.npy`.

### Bước 3 — Chạy thực nghiệm

```bash
# Exp 1: K-Fold, 2-class
python -m experiments.exp1_2class

# Exp 2: K-Fold, 4-class
python -m experiments.exp2_4class

# Exp 3: LOSO
python -m experiments.exp3_loso

# Vẽ t-SNE (sau khi có kết quả Exp 1)
python -m experiments.run_tsne
```

---

## Dataset

**DEAP** (Koelstra et al., 2012): 32 subjects, 40 trials/subject, 32 kênh EEG @ 128 Hz.
- Nhãn: Valence, Arousal, Dominance, Liking (thang 1–9)
- Download: đặt file `.dat` vào `data/raw/` hoặc dùng script `load_raw.py`

---

## Tài liệu tham khảo

- Koelstra et al. (2012). *DEAP: A database for emotion analysis using physiological signals.* IEEE Transactions on Affective Computing.
- Wang et al. (2014). *Emotional state classification from EEG data using machine learning approach.* Neurocomputing, 129, 94–106.
