"""图像质量评估指标: PSNR / SSIM"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(img_true: np.ndarray, img_pred: np.ndarray) -> float:
    """PSNR (dB)。输入为 [0,1] float 灰度图。"""
    img_true = np.clip(img_true, 0, 1)
    img_pred = np.clip(img_pred, 0, 1)
    return float(peak_signal_noise_ratio(img_true, img_pred, data_range=1.0))


def ssim(img_true: np.ndarray, img_pred: np.ndarray) -> float:
    """SSIM [0,1]。输入为 [0,1] float 灰度图。"""
    img_true = np.clip(img_true, 0, 1)
    img_pred = np.clip(img_pred, 0, 1)
    return float(structural_similarity(img_true, img_pred, data_range=1.0))


def evaluate_pair(img_true: np.ndarray, img_pred: np.ndarray) -> dict:
    """同时计算 PSNR 与 SSIM"""
    return {"PSNR (dB)": round(psnr(img_true, img_pred), 3),
            "SSIM": round(ssim(img_true, img_pred), 4)}
