# 🧠 EmoWave - Thư mục Dữ liệu & Trực quan hóa (Data & Visualization)

Thư mục này quản lý toàn bộ dữ liệu sóng não EEG thuộc tập dữ liệu **DEAP** cùng các công cụ tải dữ liệu, phân tích cấu trúc và trực quan hóa tín hiệu lên màn hình.

---

## 📂 Các thành phần chính

1. **`deap/`**: Thư mục chứa 32 file dữ liệu đã tiền xử lý từ `s01.dat` đến `s32.dat` (mỗi file tương ứng với 1 đối tượng thử nghiệm, dung lượng khoảng 98 MB/file).
2. **`load_raw.py`**: Script hỗ trợ tải tự động bộ dữ liệu DEAP từ Kaggle về máy và phân phối vào đúng thư mục `deap/`.
3. **`view_dat.py`**: Trình xem nhanh số liệu thống kê tín hiệu sóng não EEG và in bảng dữ liệu nhãn cảm xúc Valence, Arousal, Dominance, Liking dưới dạng text ở Terminal.
4. **`visualize_to_image.py`**: **[Khuyên dùng]** Công cụ trực quan hóa sóng não đa kênh xếp chồng và vẽ mô hình cảm xúc 2D Valence-Arousal ra file ảnh `.png` chất lượng cao để mở xem trực tiếp trong VS Code.

---

## 🚀 Hướng dẫn chạy các công cụ

### 1. Tải và thiết lập bộ dữ liệu DEAP (Nếu chưa có dữ liệu)
Nếu thư mục `deap/` của bạn chưa có đủ 32 file `.dat`, hãy chạy lệnh dưới đây để tự động tải về từ Kaggle:
```bash
python data/load_raw.py
```

---

### 2. Trực quan hóa dữ liệu sóng não & Cảm xúc (Dạng Ảnh PNG)
Khi chạy lệnh này, chương trình sẽ tạo ra một đồ thị trực quan hóa đa kênh cực kỳ chi tiết bao gồm: đồ thị sóng não đa kênh xếp chồng, biểu đồ cột nhãn cảm xúc và điểm tọa độ trên lưới 2D Valence-Arousal.

#### Bước 2.1: Cấu hình đối tượng & lần thử muốn xem
Mở file [data/visualize_to_image.py](visualize_to_image.py) bằng trình soạn thảo và điều chỉnh các thông số cấu hình ở ngay đầu file (dòng 17 - 23):
```python
SUBJECT_ID = 1      # Subject muốn xem (từ 1 đến 32)
TRIAL_ID = 1        # Lần thử muốn xem (từ 1 đến 40)
START_SEC = 0.0     # Giây bắt đầu xem trên đồ thị
DURATION_SEC = 6.0  # Độ dài khoảng thời gian hiển thị (giây)
```

#### Bước 2.2: Chạy lệnh xuất ảnh đồ thị
Chạy lệnh sau trong PowerShell hoặc Terminal tại thư mục gốc của dự án:
```powershell
# Thiết lập hiển thị mã ký tự Unicode không bị lỗi chữ
$env:PYTHONIOENCODING="utf-8"

# Thực thi lệnh vẽ ảnh
python data/visualize_to_image.py
```

#### Bước 2.3: Xem kết quả
Một file ảnh tương ứng (ví dụ: `eeg_plot_s01_t01.png`) sẽ được tạo ra ngay trong thư mục `data/`. Bạn chỉ cần click trực tiếp vào file ảnh này trong cây thư mục bên trái của VS Code để xem đồ thị sắc nét trên màn hình!

---

### 3. Xem thống kê dữ liệu dạng văn bản ở Terminal
Để nhanh chóng xem qua cấu trúc hình dáng của mảng dữ liệu (Data shape) và bảng chỉ số Valence/Arousal của Subject 01, chạy lệnh:
```powershell
$env:PYTHONIOENCODING="utf-8"
python data/view_dat.py
```
