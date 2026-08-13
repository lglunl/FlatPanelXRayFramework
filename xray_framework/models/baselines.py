"""对照基线模型"""
import torch
import torch.nn as nn

from .base import BaseImageModel
from .registry import register_model


@register_model("identity")
class IdentityModel(BaseImageModel):
    """恒等映射（基线对照）：输出与输入相同，用于评估「不处理」时的指标基线。"""

    def __init__(self, in_channels=1, out_channels=1, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        if self.in_channels == self.out_channels:
            return x
        return x[:, : self.out_channels]


@register_model("highpass")
class HighpassModel(BaseImageModel):
    """高通滤波（基线对照）：简单保留高频细节，去除低频模糊/混叠分量。"""

    def __init__(self, in_channels=1, out_channels=1, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        kernel = torch.tensor([
            [0., -1., 0.],
            [-1., 5., -1.],
            [0., -1., 0.],
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        # 对每个输入通道应用卷积
        outs = []
        for c in range(x.size(1)):
            xc = x[:, c:c + 1]
            if self.in_channels == self.out_channels or c < self.out_channels:
                outs.append(torch.nn.functional.conv2d(
                    xc, self.kernel, padding=1).clamp(0, 1))
        return torch.cat(outs, dim=1)
