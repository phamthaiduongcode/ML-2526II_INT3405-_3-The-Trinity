import kagglehub
import os
import shutil

def download_and_setup_deap():
    # 1. Download dataset từ Kaggle
    print("Đang tải DEAP dataset từ Kaggle (manh123df/deap-dataset)...")
    download_path = kagglehub.dataset_download("manh123df/deap-dataset")
    print(f"Đã tải về cache tại: {download_path}")

    # 2. Xác định thư mục đích (data/raw)
    # File này nằm trong data/, nên thư mục con sẽ là data/raw
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, "raw")
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Đã tạo thư mục: {target_dir}")

    # 3. Tìm và copy 32 file .dat (s01.dat -> s32.dat)
    print("Đang lọc và copy 32 file .dat vào thư mục dự án...")
    count = 0
    for root, dirs, files in os.walk(download_path):
        for file in files:
            # Chỉ lấy các file định dạng sXX.dat (32 subjects)
            if file.endswith(".dat") and file.startswith("s") and len(file) <= 7:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(target_dir, file)
                shutil.copy2(src_path, dst_path)
                count += 1
    
    print(f"Thành công! Đã copy {count} file vào {target_dir}")
    print("Bây giờ bạn đã có đủ dữ liệu để chạy các script trong models/.")

if __name__ == "__main__":
    download_and_setup_deap()