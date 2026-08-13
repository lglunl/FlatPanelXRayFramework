"""UNet 去混叠模型

基于参考代码 data_prep_and_train.py 中的 UNet 结构迁移实现。
"""
import torch
import torch.nn as nn

from .base import BaseImageModel
from .registry import register_model


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


@register_model("unet")
class UNet(BaseImageModel):
    """UNet (参考代码迁移版)：经典编码-解码结构 + 跳跃连接，用于图像去混叠/修复。"""

    def __init__(self, in_channels=1, out_channels=1, features=(64, 128, 256, 512), **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        f1, f2, f3, f4 = features

        self.inc = DoubleConv(in_channels, f1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f1, f2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f2, f3))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f3, f4))
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f4, f4 * 2))

        self.up3 = nn.ConvTranspose2d(f4 * 2, f4, 2, stride=2)
        self.conv3 = DoubleConv(f4 * 2, f4)
        self.up2 = nn.ConvTranspose2d(f4, f3, 2, stride=2)
        self.conv2 = DoubleConv(f3 * 2, f3)
        self.up1 = nn.ConvTranspose2d(f3, f2, 2, stride=2)
        self.conv1 = DoubleConv(f2 * 2, f2)
        self.up0 = nn.ConvTranspose2d(f2, f1, 2, stride=2)
        self.conv0 = DoubleConv(f1 * 2, f1)

        self.outc = nn.Conv2d(f1, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        xb = self.bottleneck(x3)

        x = self.up3(xb)
        x = self.conv3(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.conv1(torch.cat([x, x1], dim=1))
        x = self.up0(x)
        x = self.conv0(torch.cat([x, x0], dim=1))

        return self.sigmoid(self.outc(x))
