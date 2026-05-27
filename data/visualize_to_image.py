"""
EmoWave - Trực quan hóa dữ liệu EEG ra File Ảnh (Không cần GUI desktop)
========================================================================
Dành cho trường hợp máy bạn gặp lỗi hiển thị cửa sổ GUI hoặc chạy trên VS Code Terminal bị treo.
Chỉ cần chạy script này, nó sẽ tạo ra một file ảnh đồ thị .png rất đẹp,
bạn có thể mở file ảnh đó lên xem trực tiếp ngay trong VS Code!
"""

import os
import pickle
import numpy as np
import matplotlib
# Thiết lập chế độ vẽ không cần mở cửa sổ đồ họa (headless mode)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# CẤU HÌNH THÔNG SỐ XEM DỮ LIỆU TẠI ĐÂY
# ==========================================
SUBJECT_ID = 1      # Subject muốn xem (1 đến 32)
TRIAL_ID = 1        # Trial muốn xem (1 đến 40)
START_SEC = 0.0     # Thời gian bắt đầu xem (giây)
DURATION_SEC = 6.0  # Độ rộng cửa sổ xem (giây)
SPACING = 30        # Khoảng cách dọc giữa các đường sóng EEG

# Danh sách các kênh muốn vẽ (tối đa nên từ 5-10 kênh để nhìn rõ sóng)
CHANNELS_TO_PLOT = ["Fp1", "Fp2", "F3", "F4", "Cz", "O1", "Oz", "O2"]
# ==========================================

# Ánh xạ tên kênh sang index của DEAP
EEG_CHANNELS = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2"
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEAP_DIR = os.path.join(SCRIPT_DIR, "deap")

def get_emotion_quadrant(valence, arousal):
    if valence >= 5.0 and arousal >= 5.0:
        return "Q1: Vui vẻ / Hào hứng (Happy/Excited)"
    elif valence < 5.0 and arousal >= 5.0:
        return "Q2: Lo âu / Căng thẳng (Stressed/Angry)"
    elif valence < 5.0 and arousal < 5.0:
        return "Q3: Buồn bã / Chán nản (Sad/Depressed)"
    else:
        return "Q4: Thư giãn / Bình yên (Calm/Relaxed)"

def main():
    print(f"🔄 Đang tải dữ liệu của Subject {SUBJECT_ID:02d}...")
    file_path = os.path.join(DEAP_DIR, f"s{SUBJECT_ID:02d}.dat")
    
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file dữ liệu tại: {file_path}")
        print("Vui lòng kiểm tra lại đường dẫn hoặc chạy data/load_raw.py trước.")
        return

    with open(file_path, "rb") as f:
        subject_data = pickle.load(f, encoding="latin1")

    # Đọc mảng dữ liệu
    signals = subject_data['data'][TRIAL_ID - 1]     # (40, 8064)
    labels = subject_data['labels'][TRIAL_ID - 1]    # [Valence, Arousal, Dominance, Liking]

    # Tính toán thông số thời gian
    sfreq = 128
    start_sample = int(START_SEC * sfreq)
    end_sample = int((START_SEC + DURATION_SEC) * sfreq)
    
    # Giới hạn số mẫu tối đa
    end_sample = min(end_sample, signals.shape[1])
    t = np.arange(start_sample, end_sample) / sfreq

    # Khởi tạo khung vẽ Matplotlib
    fig = plt.figure(figsize=(14, 9), facecolor="#1e1e1e")
    
    # Chia bố cục đồ thị: 
    # Cột 1 (Trái): Sóng EEG 
    # Cột 2 (Phải): Các chỉ số cảm xúc
    gs = fig.add_gridspec(2, 2, width_ratios=[6.5, 3.5], height_ratios=[5, 5], 
                           left=0.08, right=0.92, top=0.90, bottom=0.10, wspace=0.3, hspace=0.3)
    
    ax_eeg = fig.add_subplot(gs[:, 0])
    ax_bars = fig.add_subplot(gs[0, 1])
    ax_quad = fig.add_subplot(gs[1, 1])

    # 1. Vẽ đồ thị sóng EEG (Trái)
    ax_eeg.set_facecolor("#151515")
    
    # Chuyển đổi danh sách tên kênh sang index thực tế
    indices_to_plot = []
    available_channel_names = []
    for name in CHANNELS_TO_PLOT:
        if name in EEG_CHANNELS:
            indices_to_plot.append(EEG_CHANNELS.index(name))
            available_channel_names.append(name)
            
    if not indices_to_plot:
        ax_eeg.text(0.5, 0.5, "Không có kênh EEG hợp lệ được chọn.", color="orange", ha="center")
    else:
        for i, ch_idx in enumerate(indices_to_plot):
            sig_segment = signals[ch_idx, start_sample:end_sample]
            # Tính toán offset để xếp chồng các kênh sóng
            offset = (len(indices_to_plot) - 1 - i) * SPACING
            ax_eeg.plot(t, sig_segment + offset, color="#00adb5", linewidth=0.9)

        # Cài đặt nhãn và trục tọa độ
        ax_eeg.set_yticks([i * SPACING for i in range(len(indices_to_plot))][::-1])
        ax_eeg.set_yticklabels(available_channel_names, color="#eeeeee", fontsize=10)
        ax_eeg.set_xlim(START_SEC, START_SEC + DURATION_SEC)
        ax_eeg.grid(axis='x', color="#444444", linestyle="--", alpha=0.5)
        ax_eeg.set_xlabel("Thời gian (giây)", color="#eeeeee", labelpad=5)
        ax_eeg.tick_params(axis='x', colors='#eeeeee')
        ax_eeg.set_title(f"SÓNG NÃO EEG ĐA KÊNH\nSubject {SUBJECT_ID:02d} - Trial {TRIAL_ID:02d} ({START_SEC}s - {START_SEC+DURATION_SEC}s)", 
                         color="#eeeeee", pad=10, fontsize=12, fontweight="bold")

    # 2. Vẽ đồ thị cột cảm xúc (Phải - Trên)
    ax_bars.set_facecolor("#222222")
    labels_names = ["Valence", "Arousal", "Dominance", "Liking"]
    bar_colors = ["#4caf50", "#2196f3", "#9c27b0", "#ff9800"]
    y_pos = np.arange(len(labels_names))
    
    bars = ax_bars.barh(y_pos, labels, align='center', color=bar_colors, alpha=0.8, height=0.5)
    for bar in bars:
        width = bar.get_width()
        ax_bars.text(width + 0.2, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                     ha='left', va='center', color='#eeeeee', fontweight='bold', fontsize=9)
        
    ax_bars.set_yticks(y_pos)
    ax_bars.set_yticklabels(labels_names, color="#eeeeee", fontsize=10)
    ax_bars.set_xlim(1, 9)
    ax_bars.set_xticks(range(1, 10))
    ax_bars.tick_params(axis='x', colors='#eeeeee')
    ax_bars.grid(axis='x', color="#555555", linestyle=":", alpha=0.5)
    ax_bars.set_title("Chỉ số đánh giá cảm xúc (1-9)", color="#eeeeee", pad=10, fontsize=12, fontweight="bold")

    # 3. Vẽ góc phần tư cảm xúc 2D (Phải - Dưới)
    ax_quad.set_facecolor("#222222")
    val, aro = labels[0], labels[1]
    
    ax_quad.axhline(5.0, color="#666666", linestyle="-", linewidth=1.2)
    ax_quad.axvline(5.0, color="#666666", linestyle="-", linewidth=1.2)
    
    ax_quad.text(7.0, 7.0, "Q1: Vui vẻ\n(Excited/Happy)", color="#4caf50", ha="center", va="center", fontsize=9, alpha=0.7)
    ax_quad.text(3.0, 7.0, "Q2: Căng thẳng\n(Stressed/Angry)", color="#f44336", ha="center", va="center", fontsize=9, alpha=0.7)
    ax_quad.text(3.0, 3.0, "Q3: Buồn bã\n(Sad/Depressed)", color="#2196f3", ha="center", va="center", fontsize=9, alpha=0.7)
    ax_quad.text(7.0, 3.0, "Q4: Thư giãn\n(Calm/Relaxed)", color="#e91e63", ha="center", va="center", fontsize=9, alpha=0.7)
    
    # Điểm đánh giá thực tế
    ax_quad.scatter(val, aro, color="#00ffcc", s=150, edgecolors="white", zorder=5)
    ax_quad.text(val + 0.2, aro + 0.2, f"({val:.2f}, {aro:.2f})", color="#00ffcc", fontweight="bold", fontsize=10, zorder=6)
    
    ax_quad.set_xlim(1, 9)
    ax_quad.set_ylim(1, 9)
    ax_quad.set_xticks(range(1, 10))
    ax_quad.set_yticks(range(1, 10))
    ax_quad.tick_params(axis='both', colors='#eeeeee', labelsize=9)
    ax_quad.set_xlabel("Valence (Mức độ dễ chịu)", color="#eeeeee", fontsize=10)
    ax_quad.set_ylabel("Arousal (Mức độ kích thích)", color="#eeeeee", fontsize=10)
    
    feeling = get_emotion_quadrant(val, aro)
    ax_quad.set_title(f"Mô hình V-A: {feeling.split(':')[0]}", color="#eeeeee", pad=10, fontsize=11, fontweight="bold")

    # Tạo đường dẫn và lưu ảnh
    output_filename = f"eeg_plot_s{SUBJECT_ID:02d}_t{TRIAL_ID:02d}.png"
    output_path = os.path.join(SCRIPT_DIR, output_filename)
    
    plt.savefig(output_path, dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    print("\n" + "="*70)
    print(f"🎉 THÀNH CÔNG! Đã tạo file ảnh đồ thị tại đường dẫn:")
    print(f"👉 {output_path}")
    print("="*70)
    print("💡 MẸO: Bạn hãy click trực tiếp vào file ảnh này ở cây thư mục bên trái của VS Code")
    print("để xem đầy đủ sóng não và các biểu đồ cảm xúc cực kỳ trực quan ngay trên màn hình!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
