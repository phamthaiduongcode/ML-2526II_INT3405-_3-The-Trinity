# File: models/cnn.py
import torch
import torch.nn as nn


# ==============================================================================
# HELPER: ConvBlock — tái sử dụng pattern Conv → BN → ReLU
# ==============================================================================
class ConvBlock(nn.Module):
    """
    Unit tái sử dụng: Conv2d → BatchNorm2d → ReLU → (optional) MaxPool2d
    Giúp cnn.py gọn và dễ điều chỉnh từng block độc lập.
    """
    def __init__(self, in_ch, out_ch, kernel, padding, pool=None):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool is not None:
            layers.append(nn.MaxPool2d(kernel_size=pool))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ==============================================================================
# MAIN MODEL: EEGNet2D
# ==============================================================================
class EEGNet2D(nn.Module):
    """
    2D-CNN phân loại cảm xúc từ EEG (DEAP dataset).

    Input : (batch, 1, 32, 128)
              └─ 1   : channel dimension (unsqueeze trong training loop)
              └─ 32  : số kênh EEG
              └─ 128 : số timestep (1 giây @ 128Hz)

    Output: (batch, num_classes)  ← logits, KHÔNG qua Softmax
                                    để dùng trực tiếp với CrossEntropyLoss

    Luồng:
        Input → Block1 → Block2 → Block3 → Block4 → GAP → Dropout → FC → Output
    """

    def __init__(self, num_classes: int = 2):
        """
        Args:
            num_classes: 2 cho Exp1 (Valence High/Low),
                         4 cho Exp2 (Happy / Stressed / Relaxed / Sad)
        """
        super().__init__()

        # ------------------------------------------------------------------
        # Block 1: Quét temporal trước — kernel (1,9) không trộn kênh EEG
        # (batch, 1, 32, 128) → (batch, 16, 32, 64)
        # ------------------------------------------------------------------
        self.block1 = ConvBlock(
            in_ch=1, out_ch=16,
            kernel=(1, 9), padding=(0, 4),
            pool=(1, 2)                     # MaxPool theo trục thời gian
        )

        # ------------------------------------------------------------------
        # Block 2: Bắt đầu học cross-channel — kernel (3,5)
        # (batch, 16, 32, 64) → (batch, 32, 16, 32)
        # ------------------------------------------------------------------
        self.block2 = ConvBlock(
            in_ch=16, out_ch=32,
            kernel=(3, 5), padding=(1, 2),
            pool=(2, 2)                     # Pool cả channel lẫn temporal
        )

        # ------------------------------------------------------------------
        # Block 3: Đặc trưng trừu tượng hơn — kernel (3,3) chuẩn
        # (batch, 32, 16, 32) → (batch, 64, 8, 16)
        # ------------------------------------------------------------------
        self.block3 = ConvBlock(
            in_ch=32, out_ch=64,
            kernel=(3, 3), padding=(1, 1),
            pool=(2, 2)
        )

        # ------------------------------------------------------------------
        # Block 4: Tăng depth, giữ nguyên spatial size — KHÔNG pool
        # padding=1 giữ biên trước GAP
        # (batch, 64, 8, 16) → (batch, 128, 8, 16)
        # ------------------------------------------------------------------
        self.block4 = ConvBlock(
            in_ch=64, out_ch=128,
            kernel=(3, 3), padding=(1, 1),
            pool=None                       # Không pool, giữ biên cho GAP
        )

        # ------------------------------------------------------------------
        # GAP: Global Average Pooling — thay thế FC lớn
        # (batch, 128, 8, 16) → (batch, 128, 1, 1) → flatten → (batch, 128)
        # Lợi ích: giảm params từ ~130K xuống ~8K, tránh overfit trên EEG
        # ------------------------------------------------------------------
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # ------------------------------------------------------------------
        # Classifier: FC nhỏ 128 → 64 → num_classes
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),     # Logits — CrossEntropyLoss tự Softmax
        )

        # Khởi tạo weights chuẩn
        self._initialize_weights()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Kiểm tra shape đầu vào — bắt lỗi sớm
        assert x.dim() == 4, (
            f"[EEGNet2D] Cần input 4D (batch, 1, 32, 128), nhận được {tuple(x.shape)}\n"
            f"Hint: hãy gọi  x = x.unsqueeze(1) trong training loop trước khi forward."
        )
        assert x.shape[1] == 1, (
            f"[EEGNet2D] Channel dimension phải = 1, nhận được {x.shape[1]}."
        )

        x = self.block1(x)   # → (batch, 16, 32, 64)
        x = self.block2(x)   # → (batch, 32, 16, 32)
        x = self.block3(x)   # → (batch, 64,  8, 16)
        x = self.block4(x)   # → (batch, 128, 8, 16)

        x = self.gap(x)      # → (batch, 128, 1,  1)
        x = x.flatten(1)     # → (batch, 128)

        x = self.classifier(x)  # → (batch, num_classes)
        return x

    # ------------------------------------------------------------------
    def _initialize_weights(self):
        """
        Kaiming init cho Conv (phù hợp với ReLU activation),
        constant init cho BatchNorm.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)