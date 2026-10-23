"""用于七分类 CSI 幅度识别的唯一轻量 1D CNN 基线。"""

import torch
from torch import nn


class LightweightCNN(nn.Module):
    """沿时间轴卷积：输入 (B,52,950)，输出七类原始 logits (B,7)。"""

    def __init__(self):
        """定义两层卷积、时间池化和分类层，不加入额外预处理。"""
        super().__init__()
        self.conv1 = nn.Conv1d(52, 64, kernel_size=3, padding=1)  # (B,52,950) -> (B,64,950)
        self.relu1 = nn.ReLU()  # 保持形状不变
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # -> (B,64,475)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)  # -> (B,128,475)
        self.relu2 = nn.ReLU()  # 保持形状不变
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # -> (B,128,1)
        self.flatten = nn.Flatten(start_dim=1)  # -> (B,128)
        self.classifier = nn.Linear(128, 7)  # -> (B,7)，不使用 Softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """提取时间特征并返回 logits，供 CrossEntropyLoss 直接使用。"""
        if x.ndim != 3 or x.shape[1] != 52 or x.shape[2] < 2:
            raise ValueError(f"期望输入 (B,52,L)，且 L>=2；实际为 {tuple(x.shape)}")
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.relu2(self.conv2(x))
        x = self.flatten(self.avg_pool(x))
        return self.classifier(x)
