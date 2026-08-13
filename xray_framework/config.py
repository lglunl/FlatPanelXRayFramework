"""全局配置与数据类定义"""
from dataclasses import dataclass, field
from typing import Optional

# 支持的图像扩展名
SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".npy"}

# 图像归一化范围（16 位 / 8 位）
NORM_RANGE = 65535.0


@dataclass
class PairingConfig:
    """数据配对配置"""
    data_root: str = ""              # 数据根目录
    input_dir: str = "plate"         # 平板X射线源成像目录（含混叠）
    truth_dir: str = "focal"         # 热阴极点源真值目录
    mode: str = "by_id"              # by_id: 按文件名数字ID配对 / by_index: 按顺序配对
    id_regex: str = r"(\d+)"         # 用于提取文件数字ID的正则


@dataclass
class TrainConfig:
    """训练配置"""
    image_size: tuple = (512, 512)   # 输入图像缩放尺寸 (h, w)
    model_name: str = "unet"         # 模型名（来自注册表）
    epochs: int = 50
    batch_size: int = 1
    lr: float = 1e-4
    val_ratio: float = 0.15          # 验证集比例
    loss_name: str = "mse"           # mse / l1 / combined
    optimizer: str = "adam"          # adam / sgd
    seed: int = 42
    device: str = "auto"             # auto / cpu / cuda
    save_dir: str = ""               # 模型保存目录（默认 outputs/models）
    checkpoint: Optional[str] = None # 断点续训路径
    augment: bool = False            # 是否启用数据增强
    num_workers: int = 0
