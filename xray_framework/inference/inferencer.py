"""推理器

功能：
  - 加载本地已训练模型（框架生成的 .pth，兼容 dict/state_dict 两种格式）
  - 导入外部模型（用户提供的 .py 模型定义 + 权重，或 PyTorchScript / ONNX）
  - 对单张/批量图片去混叠推理
"""
import importlib.util
import os
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from ..models.base import BaseImageModel
from ..models.registry import discover, get_model
from ..data.image_io import list_images, read_image, save_image, to_uint8


class Inferencer:
    def __init__(self, device: str = "auto", image_size: Optional[tuple] = None):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = None
        self.image_size = image_size or (512, 512)
        self.model_name = None

    # ---------- 模型加载 ----------

    def load_local_model(self, model_path: str, model_name: Optional[str] = None):
        """加载框架训练生成的 .pth 模型文件"""
        ckpt = torch.load(model_path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            mname = ckpt.get("model_name") or model_name or "unet"
            self.model = get_model(mname, in_channels=1, out_channels=1).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            if "image_size" in ckpt:
                self.image_size = tuple(ckpt["image_size"])
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            mname = model_name or "unet"
            self.model = get_model(mname, in_channels=1, out_channels=1).to(self.device)
            self.model.load_state_dict(ckpt["state_dict"])
        else:
            # 直接是 state_dict 或完整模型
            mname = model_name or "unet"
            self.model = get_model(mname, in_channels=1, out_channels=1).to(self.device)
            self.model.load_state_dict(ckpt)
        self.model_name = mname
        self.model.eval()
        return self.model_name

    def load_external_model(self, py_path: str, class_name: Optional[str] = None,
                            weights_path: Optional[str] = None,
                            registry_name: Optional[str] = None):
        """
        导入外部模型：
          - py_path: 用户提供的模型定义 .py 文件（必须包含模型类，最好用 @register_model 注册；
                     若未注册，可提供 class_name 动态构造）
          - weights_path: 权重文件（可选；若带模型名信息则自动匹配）
          - registry_name: 若外部 .py 用 register_model 注册了名字，可指定
        """
        spec = importlib.util.spec_from_file_location("external_model", py_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        cls = None
        if registry_name:
            from ..models.registry import MODEL_REGISTRY
            cls = MODEL_REGISTRY.get(registry_name.lower())
        if cls is None and class_name:
            cls = getattr(mod, class_name, None)
        if cls is None:
            # 自动查找 BaseImageModel 或 nn.Module 子类
            for name in dir(mod):
                obj = getattr(mod, name)
                if isinstance(obj, type) and issubclass(obj, nn.Module) and obj not in (nn.Module, BaseImageModel):
                    cls = obj
                    break
        if cls is None:
            raise ValueError(f"无法从 {py_path} 中找到模型类，请指定 class_name 或 registry_name")

        self.model = cls(in_channels=1, out_channels=1).to(self.device)
        self.model_name = registry_name or (class_name or cls.__name__)

        if weights_path:
            ckpt = torch.load(weights_path, map_location=self.device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                self.model.load_state_dict(ckpt["state_dict"])
            else:
                self.model.load_state_dict(ckpt)

        self.model.eval()
        return self.model_name

    def load_torchscript(self, model_path: str):
        """加载 PyTorchScript 导出模型（TorchScript）"""
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.model_name = "torchscript"
        return self.model_name

    def load_onnx(self, model_path: str):
        """加载 ONNX 模型（需要 onnxruntime）"""
        import onnxruntime as ort
        self.model = ONNXWrapper(ort.InferenceSession(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]))
        self.model_name = "onnx"
        return self.model_name

    # ---------- 推理 ----------

    @torch.no_grad()
    def infer_array(self, arr: np.ndarray) -> np.ndarray:
        """推理单张 float [0,1] 灰度图，返回 float [0,1] 结果。"""
        if self.model is None:
            raise RuntimeError("请先加载模型 (load_local_model / load_external_model)")
        h, w = arr.shape[:2]
        x = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        y = self.model(x)
        out = y[0, 0].cpu().numpy()
        return np.clip(out, 0, 1)

    @torch.no_grad()
    def infer_image(self, img_path: str, save_path: Optional[str] = None,
                    save16: bool = False) -> np.ndarray:
        """推理单张图片文件，返回 uint8 结果；save_path 非空则保存。"""
        arr = read_image(img_path, size=self.image_size)
        out = self.infer_array(arr)
        if save_path:
            save_image(out, save_path, bit16=save16)
        return to_uint8(out)

    def infer_folder(self, in_dir: str, out_dir: str, save16: bool = False) -> dict:
        """批量推理文件夹内所有图片，返回 {文件名: 保存路径}"""
        os.makedirs(out_dir, exist_ok=True)
        results = {}
        for f in list_images(in_dir):
            try:
                save_path = os.path.join(out_dir, f)
                self.infer_image(os.path.join(in_dir, f), save_path, save16)
                results[f] = save_path
            except Exception as e:
                print(f"[infer] {f} 失败: {e}")
        return results


class ONNXWrapper(nn.Module):
    """ONNX 推理封装（仅推理）"""
    def __init__(self, session):
        super().__init__()
        self.session = session
        self.input_name = session.get_inputs()[0].name

    def forward(self, x):
        out = self.session.run(None, {self.input_name: x.cpu().numpy()})[0]
        return torch.from_numpy(out)
