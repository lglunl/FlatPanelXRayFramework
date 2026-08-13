"""UNet+（改进版）：残差连接 + 通道注意力（SE Block）

作为「算法可迭代」的演示：在经典 UNet 基础上引入注意力机制，
新文献中的改进思路可按同样方式扩展（注册新类即可）。
"""
import torch
import torch.nn as nn

from .base import BaseImageModel
from .registry import register_model


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, max(ch // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(ch // reduction, 4), ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class DoubleConvSE(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            SEBlock(out_ch),
        )

    def forward(self, x):
        return self.conv(x)


@register_model("unet_plus")
class UNetPlus(BaseImageModel):
    """UNet+ (注意力增强版)：在 UNet 的每个卷积块中加入 SE 通道注意力，提升去混叠细节恢复能力。"""

    def __init__(self, in_channels=1, out_channels=1, features=(64, 128, 256, 512), **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        f1, f2, f3, f4 = features

        self.inc = DoubleConvSE(in_channels, f1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConvSE(f1, f2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConvSE(f2, f3))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConvSE(f3, f4))
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConvSE(f4, f4 * 2))

        self.up3 = nn.ConvTranspose2d(f4 * 2, f4, 2, stride=2)
        self.conv3 = DoubleConvSE(f4 * 2, f4)
        self.up2 = nn.ConvTranspose2d(f4, f3, 2, stride=2)
        self.conv2 = DoubleConvSE(f3 * 2, f3)
        self.up1 = nn.ConvTranspose2d(f3, f2, 2, stride=2)
        self.conv1 = DoubleConvSE(f2 * 2, f2)
        self.up0 = nn.ConvTranspose2d(f2, f1, 2, stride=2)
        self.conv0 = DoubleConvSE(f1 * 2, f1)

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
