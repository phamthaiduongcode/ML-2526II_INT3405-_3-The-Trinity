"""
EmoWave — Người 1: SVM Model
=============================
Tái hiện bài báo gốc (Wang et al., 2014):
  Power Spectrum + LDS smoothing + SVM

TODO (Người 1):
  [ ] Trích xuất Power Spectrum features từ epoch
  [ ] Implement LDS smoothing
  [ ] Train SVM với linear / RBF kernel
  [ ] So sánh kết quả 2 lớp vs 4 lớp
  [ ] Lưu kết quả vào results/svm_results.json
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.deap_loader import load_all_subjects, prepare_for_svm


def train_svm(X_train, y_train, kernel="linear", C=1.0):
    model = SVC(kernel=kernel, C=C, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, label_type="2class"):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    if label_type == "2class":
        target_names = ["Negative", "Positive"]
    else:
        target_names = ["Vui vẻ", "Sợ hãi", "Buồn", "Thư giãn"]

    print(f"  Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=target_names))
    return acc, confusion_matrix(y_test, y_pred)


if __name__ == "__main__":
    for label_type in ["2class", "4class"]:
        print(f"\n{'='*50}")
        print(f"SVM — {label_type}")
        print("="*50)

        X, y, _ = load_all_subjects(label_type=label_type)
        X_tr, X_te, y_tr, y_te, _ = prepare_for_svm(X, y)

        model = train_svm(X_tr, y_tr, kernel="linear")
        acc, cm = evaluate(model, X_te, y_te, label_type)
