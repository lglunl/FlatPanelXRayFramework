# 平板X射线去混叠成像算法框架

解决平板X射线源导致的图像混叠问题的算法框架，提供可视化操作界面（Streamlit）。

## 核心功能

### 1. 模型训练板块
使用你自己的数据库（物体各位置的**平板X射线源成像** ↔ **热阴极点源真值**配对数据）训练去混叠模型。

- 灵活的**数据配对引擎**（按文件名 ID 或按顺序配对），界面可直接预览配对结果
- 多种**可替换算法**（UNet、UNet+ 注意力增强、基线对照），未来可继续扩展
- 可视化训练过程：进度条、损失曲线、实时日志
- 自动保存最优模型（.pth），支持断点续训

### 2. 模型推理板块
加载本地已训练模型或导入外部模型，处理图片解决混叠问题。

- 支持：框架训练输出模型 / 外部 `.py` 模型 + 权重 / TorchScript / ONNX
- 单张图片或文件夹批量处理
- 原图-结果（-真值）三图对比
- 可选 PSNR / SSIM 定量评估

## 环境安装

需要 Python 3.9+。推荐创建虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

> 若使用 GPU 训练，请参考 https://pytorch.org 安装对应 CUDA 版本的 torch。

## 启动

```bash
streamlit run app.py
```

或双击 `run_app.bat`。浏览器自动打开 http://localhost:8501

## 数据目录结构

```
data_root/
├── plate/          # 平板X射线源成像（含混叠，输入）
│   ├── pos001.tif
│   └── pos002.tif
└── focal/          # 热阴极点源真值（标签）
    ├── pos001.tif
    └── pos002.tif
```

- `by_id` 模式：从文件名提取数字 ID 配对（推荐）
- `by_index` 模式：按文件名排序逐对配对

## 算法迭代与扩展

**设计理念：算法可更新、可替换、可选择。**

所有算法通过注册表管理（`xray_framework/models/registry.py`），
新增算法只需三步（详见 `xray_framework/models/algorithms/README.md`）：

1. 在 `xray_framework/models/algorithms/` 新建 `.py` 文件
2. 继承 `BaseImageModel` 并加 `@register_model` 装饰器
3. 保存刷新页面即可在界面中选择

后续加入新文献时，可按文献思路实现新模型类放入该目录，实现持续迭代。

## 项目结构

```
FlatPanelXRayFramework/
├── app.py                        # Streamlit 入口
├── run_app.bat                   # Windows 启动脚本
├── requirements.txt
├── xray_framework/
│   ├── config.py                 # 配置数据类
│   ├── data/                     # 图像IO、配对引擎、数据集
│   ├── models/
│   │   ├── base.py               # 模型基类
│   │   ├── registry.py           # 注册表 + 自动发现
│   │   ├── unet.py               # UNet（参考算法迁移）
│   │   ├── unet_plus.py          # UNet+ 注意力增强
│   │   ├── baselines.py          # 对照基线
│   │   └── algorithms/           # ★ 新算法放这里
│   ├── training/                 # 损失、训练器、进度回调
│   ├── inference/                # 推理器（本地/外部/TorchScript/ONNX）
│   ├── ui/                       # Streamlit 页面组件
│   └── utils/                    # PSNR/SSIM 指标
└── outputs/                      # 训练输出（模型、结果）
```
