# File: src/utils/metrics.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)


def _to_numpy_labels(tensor_or_array):
    """
    Hàm bổ trợ (Private): Chuyển đổi an toàn từ PyTorch Tensor (kể cả trên GPU)
    sang NumPy, đồng thời tự động lấy argmax nếu đầu vào là xác suất/logits.
    """
    if isinstance(tensor_or_array, torch.Tensor):
        tensor_or_array = tensor_or_array.detach().cpu().numpy()
    else:
        tensor_or_array = np.array(tensor_or_array)

    if len(tensor_or_array.shape) > 1 and tensor_or_array.shape[1] > 1:
        tensor_or_array = np.argmax(tensor_or_array, axis=1)

    return tensor_or_array


def evaluate_metrics(y_true, y_pred):
    """
    Chấm điểm mô hình và trả về một Dictionary để tiện cho việc Logging.
    Accepts: List, NumPy Array, hoặc PyTorch Tensors.
    """
    y_true = _to_numpy_labels(y_true)
    y_pred = _to_numpy_labels(y_pred)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return {
        'accuracy' : acc,
        'precision': prec,
        'recall'   : rec,
        'f1_score' : f1,
    }


def plot_confusion_matrix(y_true, y_pred, classes=('Low', 'High'),
                          title='Confusion Matrix', save_path=None):
    """Vẽ Confusion Matrix dạng Heatmap sắc nét."""
    y_true = _to_numpy_labels(y_true)
    y_pred = _to_numpy_labels(y_pred)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 14})
    plt.title(title, fontweight='bold', pad=15)
    plt.ylabel('Thực tế (True Label)',    fontweight='bold')
    plt.xlabel('Dự đoán (Predicted Label)', fontweight='bold')
    plt.tight_layout()

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📸 Đã lưu Confusion Matrix tại: {save_path}")

    plt.close()


def plot_learning_curves(train_losses, val_losses, train_accs, val_accs,
                         title='Learning Curves', save_path=None):
    """Vẽ đồ thị song song: Loss và Accuracy qua các Epochs."""
    train_losses = [x.item() if hasattr(x, 'item') else x for x in train_losses]
    val_losses   = [x.item() if hasattr(x, 'item') else x for x in val_losses]
    train_accs   = [x.item() if hasattr(x, 'item') else x for x in train_accs]
    val_accs     = [x.item() if hasattr(x, 'item') else x for x in val_accs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(train_losses, label='Train Loss', color='#1f77b4', linewidth=1.8)
    ax1.plot(val_losses,   label='Val Loss',   color='#ff7f0e', linewidth=1.8, linestyle='--')
    ax1.set_title('Loss Convergence', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss Value')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    ax2.plot(train_accs, label='Train Acc', color='#2ca02c', linewidth=1.8)
    ax2.plot(val_accs,   label='Val Acc',   color='#d62728', linewidth=1.8, linestyle='--')
    ax2.set_title('Accuracy Growth', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    fig.suptitle(title, fontweight='bold', fontsize=14)
    plt.tight_layout()

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📸 Đã lưu Learning Curves tại: {save_path}")

    plt.close()
