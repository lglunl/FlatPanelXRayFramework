"""端到端冒烟测试：模拟数据 -> 配对 -> 训练 -> 推理"""
import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xray_framework.config import PairingConfig, TrainConfig
from xray_framework.data.pairing import PairingEngine
from xray_framework.models.registry import list_models
from xray_framework.training.trainer import Trainer
from xray_framework.inference.inferencer import Inferencer
from xray_framework.utils.metrics import evaluate_pair

def make_sample(seed):
    rng = np.random.default_rng(seed)
    base = rng.random((128, 128)).astype(np.float32)
    # 模拟"混叠"输入：低通模糊 + 噪声
    inp = np.clip(base + 0.15 * rng.normal(0, 0.1, base.shape), 0, 1)
    # 真值：原始
    truth = base
    return inp, truth

def main():
    tmp = tempfile.mkdtemp(prefix="xray_test_")
    plate_dir = os.path.join(tmp, "plate")
    focal_dir = os.path.join(tmp, "focal")
    os.makedirs(plate_dir, exist_ok=True)
    os.makedirs(focal_dir, exist_ok=True)

    for i in range(8):
        inp, truth = make_sample(i)
        Image.fromarray((inp * 255).astype(np.uint8)).save(os.path.join(plate_dir, f"pos{i+1:03d}.png"))
        Image.fromarray((truth * 255).astype(np.uint8)).save(os.path.join(focal_dir, f"pos{i+1:03d}.png"))

    print("== 1. 模型注册表 ==")
    print("可用模型:", list_models())

    print("\n== 2. 数据配对 ==")
    pcfg = PairingConfig(data_root=tmp, input_dir="plate", truth_dir="focal", mode="by_id")
    engine = PairingEngine(pcfg)
    print(engine.summary())
    print("配对预览:", engine.preview(3))

    print("\n== 3. 训练 (2 epochs) ==")
    tcfg = TrainConfig(image_size=(128, 128), model_name="unet", epochs=2,
                       batch_size=2, lr=1e-3, val_ratio=0.2, loss_name="combined")
    trainer = Trainer(tcfg, engine.pairs)
    best_path = trainer.train(save_dir=os.path.join(tmp, "models"))
    print("模型保存:", best_path)

    print("\n== 4. 推理 ==")
    infer = Inferencer()
    name = infer.load_local_model(best_path)
    print("加载模型:", name)
    test_inp = read_gray(os.path.join(plate_dir, "pos001.png"))
    out = infer.infer_array(test_inp)
    truth_arr = read_gray(os.path.join(focal_dir, "pos001.png"))
    m = evaluate_pair(truth_arr, out)
    print("PSNR/SSIM:", m)

    print("\n== 5. 外部模型导入测试 ==")
    ext_py = os.path.join(tmp, "ext_model.py")
    with open(ext_py, "w", encoding="utf-8") as f:
        f.write("""import torch.nn as nn
from xray_framework.models.base import BaseImageModel
from xray_framework.models.registry import register_model

@register_model("ext_unet")
class ExtUNet(BaseImageModel):
    def __init__(self, in_channels=1, out_channels=1, **kwargs):
        super().__init__(in_channels, out_channels)
        self.net = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 3, padding=1), nn.Sigmoid())
    def forward(self, x):
        return self.net(x)
""")
    infer2 = Inferencer()
    infer2.load_external_model(ext_py, registry_name="ext_unet")
    out2 = infer2.infer_array(test_inp)
    print("外部模型输出 shape:", out2.shape)
    print("新模型已注册:", "ext_unet" in list_models())

    print("\n✅ 全部冒烟测试通过！")

def read_gray(path):
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0

if __name__ == "__main__":
    main()
