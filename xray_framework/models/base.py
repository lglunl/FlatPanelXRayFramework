"""模型基类"""
import torch.nn as nn


class BaseImageModel(nn.Module):
    """
    所有去混叠模型必须继承的基类。

    约定：
      - 输入 x: (B, C_in, H, W)，范围 [0,1]
      - 输出 y: (B, C_out, H, W)，与输入同尺寸
      - 模型应能处理任意 H,W（全卷积结构），或在配置中固定尺寸

    实现新算法时只需：
      1. 继承本类实现 forward
      2. 用 @register_model 装饰器注册
      3. 将文件放入 xray_framework/models/algorithms/ 目录（自动发现）
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        raise NotImplementedError
