"""
EEGNet — Compact CNN for EEG-based BCI (Lawhern et al., 2018).

Kiến trúc theo paper gốc, tối ưu cho DEAP dataset:
    Input  : (batch, 1, 32, 128)  ← 32 kênh EEG, 128 timestep (1s @ 128Hz)
    Output : (batch, num_classes)  ← logits, KHÔNG có Softmax

Luồng:
    [Block 1] Conv2D → BN → DepthwiseConv2D (max_norm=1) → BN → ELU → AvgPool → Dropout
    [Block 2] SeparableConv2D (Depthwise + Pointwise) → BN → ELU → AvgPool → Dropout
    [Classifier] Flatten → Linear (max_norm=0.25)
"""

import torch
import torch.nn as nn


# ==============================================================================
# HELPER: MaxNormConstraint — áp dụng max-norm regularization sau mỗi bước
# ==============================================================================
class MaxNormConstraint:
    """
    Max-norm constraint hook cho weight của nn.Module.

    Gắn vào model qua register_post_forward hoặc gọi thủ công sau optimizer.step().
    Theo paper EEGNet, max_norm=1 cho DepthwiseConv, max_norm=0.25 cho Classifier.
    """

    @staticmethod
    def apply(module: nn.Module, max_norm: float):
        """
        Clip weight norm của module về max_norm.
        Trực tiếp truy cập module.weight — an toàn hơn iterate named_parameters.

        Reshape weight → 2D (out_features, -1) rồi norm theo dim=1
        để tương thích mọi shape (Conv2D 4D, Linear 2D).
        """
        if not hasattr(module, 'weight') or module.weight is None:
            return
        with torch.no_grad():
            p = module.weight.data.clone()          
            original_shape = p.shape
            p_2d = p.view(original_shape[0], -1)

            norms = torch.linalg.vector_norm(p_2d, dim=1, keepdim=True)
            desired = torch.clamp(norms, max=max_norm)
            
            # Out-of-place: tạo tensor mới 
            p_2d_new = p_2d * (desired / (norms + 1e-8))
            
            module.weight.data.copy_(p_2d_new.view(original_shape))


# ==============================================================================
# MAIN MODEL: EEGnet
# ==============================================================================
class EEGnet(nn.Module):
    """
    EEGNet cho phân loại cảm xúc EEG (DEAP dataset).

    Args:
        num_classes (int): Số lớp output (2 cho binary, 4 cho 4-class V×A)
        C (int): Số kênh EEG (default: 32 cho DEAP)
        T (int): Số timestep (default: 128 cho 1s @ 128Hz)
        F1 (int): Số temporal filters ở Block 1 (default: 8)
        D (int): Depth multiplier cho DepthwiseConv (default: 2)
        F2 (int): Số filters ở Block 2 (default: F1 * D = 16)
        dropout_rate (float): Tỷ lệ dropout (default: 0.5)
    """

    def __init__(self, num_classes: int = 2,
                 C: int = 32, T: int = 128,
                 F1: int = 8, D: int = 2, F2: int = None,
                 dropout_rate: float = 0.5):
        super().__init__()

        # F2 mặc định = F1 * D 
        if F2 is None:
            F2 = F1 * D

        self.num_classes = num_classes
        self.C = C
        self.T = T

        # ==================================================================
        # Block 1: Temporal Conv → BN → DepthwiseConv → BN → ELU → Pool → Drop
        # ==================================================================
        # Temporal Conv: quét theo trục thời gian, không mix kênh EEG
        # (batch, 1, 32, 128) → (batch, 8, 32, 128)
        # kernel lẻ 63 + padding 31
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=F1,
            kernel_size=(1, 63), padding=(0, 31),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # DepthwiseConv: quét theo trục kênh EEG, mỗi filter 1 nhóm
        # (batch, 8, 32, 128) → (batch, 16, 1, 128)
        # max_norm=1 sẽ được enforce qua hook
        self.depthwise = nn.Conv2d(
            in_channels=F1, out_channels=F1 * D,
            kernel_size=(C, 1),     # (32, 1) — quét toàn bộ C kênh
            groups=F1,              # depthwise: mỗi filter đọc 1 channel
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))   # T: 128 → 32
        self.drop1 = nn.Dropout(p=dropout_rate)

        # ==================================================================
        # Block 2: SeparableConv2D → BN → ELU → Pool → Drop
        # ==================================================================
        # SeparableConv = DepthwiseConv + PointwiseConv (1×1)
        # Depthwise: (batch, 16, 1, 32) → (batch, 16, 1, 32)
        self.sep_depthwise = nn.Conv2d(
            in_channels=F2, out_channels=F2,
            kernel_size=(1, 15), padding=(0, 7),
            groups=F2,             # depthwise
            bias=False
        )
        # Pointwise: (batch, 16, 1, 32) → (batch, 16, 1, 32)
        self.sep_pointwise = nn.Conv2d(
            in_channels=F2, out_channels=F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))   # T: 32 → 4
        self.drop2 = nn.Dropout(p=dropout_rate)

        # ==================================================================
        # Classifier: Flatten → Linear
        # ==================================================================
        # Dynamic flatten size — robust với mọi giá trị T
        # max_norm=0.25 sẽ được enforce qua apply_max_norm()
        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            dummy = self.pool1(self.elu1(self.bn2(self.depthwise(self.bn1(self.conv1(dummy))))))
            dummy = self.pool2(self.elu2(self.bn3(self.sep_pointwise(self.sep_depthwise(dummy)))))
            _flatten_size = dummy.view(1, -1).shape[1]
        self.classifier = nn.Linear(_flatten_size, num_classes)

        # ==================================================================
        # Max-norm values (lưu để enforce trong forward)
        # ==================================================================
        self._max_norm_depthwise = 1.0
        self._max_norm_classifier = 0.25

        # Khởi tạo weights
        self._initialize_weights()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, 1, 32, 128) — input EEG tensor đã unsqueeze

        Returns:
            logits: (batch, num_classes) — raw logits (không Softmax)
        """
        # Kiểm tra shape đầu vào — dùng if/raise thay assert (không bị skip khi -O)
        if x.dim() != 4:
            raise ValueError(
                f"[EEGnet] Cần input 4D (batch, 1, {self.C}, {self.T}), "
                f"nhận được {tuple(x.shape)}\n"
                f"Hint: gọi x = x.unsqueeze(1) trong training loop."
            )

        # ── Block 1 ──────────────────────────────────────────────────────
        x = self.conv1(x)           # (batch, 8, 32, 128)
        x = self.bn1(x)

        x = self.depthwise(x)       # (batch, 16, 1, 128)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)           # (batch, 16, 1, 32)
        x = self.drop1(x)

        # ── Block 2 ──────────────────────────────────────────────────────
        x = self.sep_depthwise(x)   # (batch, 16, 1, 32)
        x = self.sep_pointwise(x)   # (batch, 16, 1, 32)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)           # (batch, 16, 1, 4)
        x = self.drop2(x)

        # ── Classifier ───────────────────────────────────────────────────
        x = self.flatten(x)         # (batch, 64)
        x = self.classifier(x)     # (batch, num_classes)
        return x

    # ------------------------------------------------------------------
    def apply_max_norm(self):
        """
        Enforce max-norm constraint trên weights (gọi sau optimizer.step()).

        Theo paper EEGNet:
            - DepthwiseConv: max_norm = 1.0
            - Classifier Linear: max_norm = 0.25
        """
        MaxNormConstraint.apply(self.depthwise, self._max_norm_depthwise)
        MaxNormConstraint.apply(self.classifier, self._max_norm_classifier)

    # ------------------------------------------------------------------
    def _initialize_weights(self):
        """
        He (Kaiming) init — phù hợp với ELU activation (non-linear, asymmetric).
        Dùng 'relu' làm approximation vì PyTorch chưa hỗ trợ 'elu' cho nonlinearity.
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
