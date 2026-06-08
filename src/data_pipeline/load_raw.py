# File: src/data_pipeline/load_raw.py
"""
Tải DEAP dataset từ Kaggle và đặt file vào data/raw/.
Chạy: python -m src.data_pipeline.load_raw
"""
import os
import shutil
import kagglehub


def download_and_setup_deap():
    # 1. Download từ Kaggle
    print("Đang tải DEAP dataset từ Kaggle (manh123df/deap-dataset)...")
    download_path = kagglehub.dataset_download("manh123df/deap-dataset")
    print(f"Đã tải về cache tại: {download_path}")

    # 2. Xác định thư mục đích data/raw (tính từ root project)
    ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    target_dir = os.path.join(ROOT_DIR, "data", "raw")
    os.makedirs(target_dir, exist_ok=True)
    print(f"Thư mục đích: {target_dir}")

    # 3. Copy 32 file .dat
    print("Đang lọc và copy 32 file .dat vào thư mục dự án...")
    count = 0
    for root, dirs, files in os.walk(download_path):
        for file in files:
            if file.endswith(".dat") and file.startswith("s") and len(file) <= 7:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(target_dir, file)
                shutil.copy2(src_path, dst_path)
                count += 1

    print(f"Thành công! Đã copy {count} file vào {target_dir}")
    print("Tiếp theo chạy: python -m src.data_pipeline.preprocess")


if __name__ == "__main__":
    download_and_setup_deap()
