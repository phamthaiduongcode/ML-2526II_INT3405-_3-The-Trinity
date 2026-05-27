"""
Viewer cho DEAP .dat files
Xem dữ liệu EEG một cách dễ hiểu
"""

import pickle
import numpy as np
import pandas as pd
import os
import sys

def view_subject(subject_id):
    """Xem chi tiết 1 subject"""
    # Lấy thư mục hiện tại của script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, f"deap/s{subject_id:02d}.dat")
    
    if not os.path.exists(filename):
        print(f"❌ File không tồn tại: {filename}")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 SUBJECT {subject_id:02d}")
    print(f"{'='*60}\n")
    
    with open(filename, "rb") as f:
        subject = pickle.load(f, encoding="latin1")
    
    # Cấu trúc dữ liệu
    data = subject['data']          # (40 trials, 40 channels, 8064 samples)
    labels = subject['labels']      # (40 trials, 4)
    
    print(f"📈 Data shape: {data.shape}")
    print(f"   - Trials: {data.shape[0]}")
    print(f"   - Channels: {data.shape[1]} (32 EEG + 8 peripheral)")
    print(f"   - Samples: {data.shape[2]} @ 128 Hz = {data.shape[2]/128:.1f}s")
    
    print(f"\n📋 Labels shape: {labels.shape}")
    print(f"   - Columns: Valence | Arousal | Dominance | Liking\n")
    
    # Bảng labels
    df_labels = pd.DataFrame(
        labels,
        columns=['Valence', 'Arousal', 'Dominance', 'Liking']
    )
    print(df_labels.head(10).to_string())
    print(f"\n   ... ({len(df_labels)} trials tổng cộng)\n")
    
    # Thống kê
    print(f"📊 Thống kê Labels:")
    print(df_labels.describe().to_string())
    
    # Dữ liệu EEG
    print(f"\n🧠 EEG Data (32 channels):")
    print(f"   Min: {data[:,:32,:].min():.4f}")
    print(f"   Max: {data[:,:32,:].max():.4f}")
    print(f"   Mean: {data[:,:32,:].mean():.4f}")
    print(f"   Std: {data[:,:32,:].std():.4f}")
    
    return subject

def list_all_subjects():
    """Liệt kê tất cả subject có sẵn"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("\n📂 Available subjects:\n")
    for i in range(1, 33):
        filename = os.path.join(script_dir, f"deap/s{i:02d}.dat")
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024*1024)
            print(f"   ✓ s{i:02d}.dat ({size_mb:.1f} MB)")
    print()

if __name__ == "__main__":
    print("\n🎯 DEAP Dataset Viewer\n")
    
    # Danh sách tất cả subject
    list_all_subjects()
    
    # Xem subject đầu tiên
    print("Example: Viewing Subject 01\n")
    view_subject(1)
    
    print("\n" + "="*60)
    print("✨ Usage:")
    print("   from view_dat import view_subject")
    print("   view_subject(1)   # Xem subject 1")
    print("   view_subject(15)  # Xem subject 15")
    print("="*60 + "\n")
