"""
experiments/run_tsne.py
Trích xuất features từ model Exp 1 đã train và vẽ t-SNE.
Phải chạy experiments/exp1_2class.py trước để có weights.

Chạy từ thư mục root:
    python -m experiments.run_tsne

Output:
    result/plots/tsne_valence.png
    result/plots/tsne_arousal.png
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.models.eeg_bilstm import EEG_BiLSTM
from src.data_pipeline.preprocess import normalize_after_split

# ── ĐƯỜNG DẪN ──
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
CKPT_DIR = os.path.join(ROOT_DIR, "result", "checkpoints")
PLOT_DIR = os.path.join(ROOT_DIR, "result", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_SAMPLES = 3000


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACT FEATURES
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_hidden(model, X, batch_size=512):
    model.eval()
    all_hidden = []
    X_t = torch.tensor(X, dtype=torch.float32)
    for i in range(0, len(X_t), batch_size):
        hidden = model.get_hidden(X_t[i:i+batch_size].to(DEVICE))
        all_hidden.append(hidden.cpu().numpy())
    return np.concatenate(all_hidden, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZE
# ══════════════════════════════════════════════════════════════════════════════

def plot_tsne(hidden_vectors, labels, label_name, save_path):
    print(f"  🎨 Đang chạy t-SNE ({len(hidden_vectors)} samples)...")
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42, n_jobs=-1)
    X_2d = tsne.fit_transform(hidden_vectors)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    colors      = {0: '#4fc3f7', 1: '#ef5350'}
    class_names = {0: 'Low',    1: 'High'}

    for cls in [0, 1]:
        mask = labels == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=colors[cls], label=class_names[cls],
                   alpha=0.6, s=20, linewidths=0)

    patches = [mpatches.Patch(color=colors[0], label=f'Low {label_name}'),
               mpatches.Patch(color=colors[1], label=f'High {label_name}')]
    ax.legend(handles=patches, fontsize=12,
              facecolor='#1e1e2e', edgecolor='#444', labelcolor='white', loc='upper right')
    ax.set_title(f't-SNE Feature Space\nTarget: {label_name}',
                 fontsize=14, fontweight='bold', color='white', pad=15)
    ax.set_xlabel('t-SNE dim 1', color='#aaa', fontsize=11)
    ax.set_ylabel('t-SNE dim 2', color='#aaa', fontsize=11)
    ax.tick_params(colors='#666')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ Lưu ảnh thành công: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    X     = np.load(os.path.join(DATA_DIR, "X_epochs.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_valence.npy"))
    y_aro = np.load(os.path.join(DATA_DIR, "y_arousal.npy"))

    for label_name, y_full in [("valence", y_val), ("arousal", y_aro)]:
        print(f"\n{'='*55}\n🔬 t-SNE: {label_name.upper()}\n{'='*55}")

        model_path = os.path.join(CKPT_DIR, f"exp1_best_model_{label_name}.pth")
        train_path = os.path.join(CKPT_DIR, f"exp1_best_train_idx_{label_name}.npy")
        test_path  = os.path.join(CKPT_DIR, f"exp1_best_test_idx_{label_name}.npy")

        if not os.path.exists(model_path):
            print(f"❌ Không tìm thấy {model_path}! Chạy exp1_2class.py trước.")
            continue

        model = EEG_BiLSTM(n_classes=2).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("🧠 Nạp model thành công!")

        train_idx = np.load(train_path)
        test_idx  = np.load(test_path)

        X_train_raw = X[train_idx]
        X_test_raw  = X[test_idx]
        y_test      = y_full[test_idx]

        # Lấy mẫu nhỏ nếu quá lớn
        if len(y_test) > N_SAMPLES:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=N_SAMPLES, random_state=42)
            _, sample_idx = next(sss.split(X_test_raw, y_test))
            X_test_raw, y_test = X_test_raw[sample_idx], y_test[sample_idx]

        _, X_test_norm, _ = normalize_after_split(X_train_raw, X_test_raw, mode='channel')
        X_test_lstm       = X_test_norm.transpose(0, 2, 1).copy()

        hidden_features = extract_hidden(model, X_test_lstm)
        save_path       = os.path.join(PLOT_DIR, f"tsne_{label_name}.png")
        plot_tsne(hidden_features, y_test, label_name.capitalize(), save_path)

    print("\n🎉 HOÀN TẤT VẼ t-SNE!")
