"""图像读写与预处理"""
import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from ..config import SUPPORTED_EXTS


def list_images(folder: str) -> list:
    """列出文件夹下所有支持的图像文件（按名称排序）"""
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS and not f.startswith(".")
    )


def read_image(path: str, size: Optional[Tuple[int, int]] = None, normalize: bool = True) -> np.ndarray:
    """
    读取图像为 float32 灰度图，范围 [0,1]。
    size: (h, w) 可选缩放。
    支持 8/16 位 PNG、TIFF 等，自动归一化。
    """
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize((size[1], size[0]), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    max_val = np.iinfo(img.dtype).max if hasattr(img, "dtype") and str(img.dtype).startswith("uint") else 255
    # 上面写法对 PIL 模式不可靠，直接根据数组最大值判断位深
    if arr.max() > 255:
        arr = arr / 65535.0
    else:
        arr = arr / 255.0
    if not normalize:
        return arr * 65535.0 if arr.max() <= 1.0 else arr
    return arr


def save_image(arr: np.ndarray, path: str, bit16: bool = True):
    """保存灰度图。arr 范围 [0,1]；bit16=True 时保存为 16 位 TIFF/PNG。"""
    arr = np.clip(arr, 0, 1)
    if bit16:
        img = Image.fromarray((arr * 65535).astype(np.uint16))
    else:
        img = Image.fromarray((arr * 255).astype(np.uint8))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """float [0,1] -> uint8 [0,255]"""
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def to_uint16(arr: np.ndarray) -> np.ndarray:
    """float [0,1] -> uint16 [0,65535]"""
    return (np.clip(arr, 0, 1) * 65535).astype(np.uint16)
