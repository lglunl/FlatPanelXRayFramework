"""PyTorch 数据集"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .image_io import read_image


class PairedImageDataset(Dataset):
    """
    配对图像数据集：返回 (输入图[含混叠], 真值图) 张量对。
    每项张量 shape: (1, H, W)，范围 [0,1]。
    """
    def __init__(self, pairs: list, image_size=(512, 512), augment: bool = False):
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_path, truth_path = self.pairs[idx]
        x = read_image(input_path, size=self.image_size)
        y = read_image(truth_path, size=self.image_size)

        if self.augment:
            x, y = self._augment(x, y)

        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.from_numpy(y).unsqueeze(0)
        return x, y

    def _augment(self, x: np.ndarray, y: np.ndarray):
        # 简单的随机翻转与 90 度旋转增强
        import random
        if random.random() < 0.5:
            x, y = np.flip(x, 1), np.flip(y, 1)
        if random.random() < 0.5:
            x, y = np.flip(x, 0), np.flip(y, 0)
        k = random.choice([0, 1, 2, 3])
        if k:
            x, y = np.rot90(x, k).copy(), np.rot90(y, k).copy()
        return x, y


def split_pairs(pairs: list, val_ratio: float = 0.15, seed: int = 42):
    """按比例划分训练/验证集"""
    import random
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    n_val = int(len(idx) * val_ratio)
    val_idx = set(idx[:n_val])
    train_pairs = [p for i, p in enumerate(pairs) if i not in val_idx]
    val_pairs = [p for i, p in enumerate(pairs) if i in val_idx]
    return train_pairs, val_pairs
