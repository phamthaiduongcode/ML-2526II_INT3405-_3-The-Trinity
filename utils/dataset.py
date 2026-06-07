# File: utils/dataset.py
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def set_seed(seed=42):
    """
    Khóa toàn bộ các yếu tố ngẫu nhiên để kết quả chạy luôn giống nhau 100%.
    TV1, TV2, TV3 bắt buộc phải gọi hàm này ở đầu file train.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Đã khóa cứng Random Seed = {seed}")

class EEGDataset(Dataset):
    """
    Biến mảng NumPy thành định dạng PyTorch Dataset.
    Luôn giữ nguyên Shape gốc (Batch, Kênh, Thời gian) để dùng chung cho cả nhóm.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(X_train_scaled, X_test_scaled, y_train, y_test, batch_size=64):
    """
    Hàm nhà máy (Factory): Nhận vào dữ liệu ĐÃ CHUẨN HÓA và trả về các DataLoader.
    Giúp tách biệt trách nhiệm: Khâu Normalize xử lý riêng bên ngoài để TV1 xài chung.
    """
    train_dataset = EEGDataset(X_train_scaled, y_train)
    test_dataset = EEGDataset(X_test_scaled, y_test)

    # pin_memory=True tăng tốc transfer CPU→GPU, num_workers>0 load data song song
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=False, num_workers=2, pin_memory=pin
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=2, pin_memory=pin
    )

    return train_loader, test_loader