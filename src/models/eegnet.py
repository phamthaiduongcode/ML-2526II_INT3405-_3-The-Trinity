import torch
import torch.nn as nn

# ==============================================================================
# ATTENTION MODULE: Squeeze-and-Excitation (SE) Block
# Giúp mô hình tập trung vào các đặc trưng (channels) quan trọng nhất
# ==============================================================================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Đảm bảo bottleneck tối thiểu 4 để tránh quá hẹp khi channel nhỏ
        reduced = max(channel // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ==============================================================================
# HELPER: MaxNormConstraint
# ==============================================================================
class MaxNormConstraint:
    @staticmethod
    def apply(module: nn.Module, max_norm: float):
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
# MAIN MODEL: EEGnet (Updated with Attention & Optimized Kernels)
#
# Thay đổi so với phiên bản cũ:
#   1. Classifier sâu hơn: Flatten → Linear(64) → BN1d → ELU → Dropout → Linear(classes)
#      (thay vì Flatten → Linear(classes) trực tiếp)
#   2. SEBlock: thêm clamp max(channel//reduction, 4) tránh bottleneck quá hẹp
#   3. apply_max_norm: bổ sung constraint lên classifier[0] (Linear 64)
# ==============================================================================
class EEGnet(nn.Module):
    def __init__(self, num_classes: int = 2,
                 C: int = 32, T: int = 128,
                 F1: int = 8, D: int = 2, F2: int = None,
                 dropout_rate: float = 0.5):
        super().__init__()

        if F2 is None:
            F2 = F1 * D

        self.num_classes = num_classes
        self.C = C
        self.T = T

        # ==================================================================
        # Block 1: Temporal Conv (Kernel 31) -> Depthwise -> SE Attention
        # ==================================================================
        self.conv1 = nn.Conv2d(
            1, F1, kernel_size=(1, 31), padding=(0, 15), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise = nn.Conv2d(
            F1, F1 * D, kernel_size=(C, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.se1 = SEBlock(F1 * D)      # SE attention trước dropout
        self.drop1 = nn.Dropout(p=dropout_rate)

        # ==================================================================
        # Block 2: Separable Conv -> SE Attention
        # Cấu trúc chuẩn: Depthwise -> BN -> ELU -> Pointwise -> BN -> ELU
        # ==================================================================
        self.sep_depthwise = nn.Conv2d(
            F2, F2, kernel_size=(1, 15), padding=(0, 7), groups=F2, bias=False
        )
        self.bn3_dw = nn.BatchNorm2d(F2)    # BN sau depthwise
        self.elu_dw = nn.ELU()              # Activation sau depthwise

        self.sep_pointwise = nn.Conv2d(
            F2, F2, kernel_size=(1, 1), bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)       # BN sau pointwise
        self.elu2 = nn.ELU()                # Activation sau pointwise

        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.se2 = SEBlock(F2)              # SE attention trước dropout
        self.drop2 = nn.Dropout(p=dropout_rate)

        # ==================================================================
        # Classifier: Flatten → Linear(64) → BN1d → ELU → Dropout → Linear
        # Thêm hidden layer 64 giúp học feature space tốt hơn trước softmax
        # ==================================================================
        self.flatten = nn.Flatten()

        # Tính kích thước flatten tự động
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)

            # Block 1
            x = self.bn1(self.conv1(dummy))
            x = self.pool1(self.elu1(self.bn2(self.depthwise(x))))
            x = self.se1(x)

            # Block 2
            x = self.sep_depthwise(x)
            x = self.elu_dw(self.bn3_dw(x))
            x = self.sep_pointwise(x)
            x = self.pool2(self.elu2(self.bn3(x)))
            x = self.se2(x)

            _flatten_size = x.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(_flatten_size, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, num_classes),
        )

        # Max-norm values
        self._max_norm_depthwise   = 1.0
        self._max_norm_classifier  = 0.5   # áp dụng lên lớp Linear cuối

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x) 
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.se1(x)         # SE attention nhìn feature map đầy đủ
        x = self.drop1(x)       # rồi mới dropout

        # Block 2
        x = self.sep_depthwise(x)
        x = self.bn3_dw(x)
        x = self.elu_dw(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.se2(x)         # SE attention trước dropout
        x = self.drop2(x)

        # Classifier
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def apply_max_norm(self):
        MaxNormConstraint.apply(self.depthwise, self._max_norm_depthwise)
        # Áp lên Linear cuối (classifier[-1]) — đây là lớp dễ phát nổ nhất
        MaxNormConstraint.apply(self.classifier[-1], self._max_norm_classifier)

    def _initialize_weights(self):
        """Xavier init cho Conv & Linear (phù hợp ELU), constant init cho BN."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)