# 外部模型目录

通过界面「文献导入」页的「添加外部模型」导入的模型定义文件（.py）会保存到本目录，
并记录在 `../external_registry.json` 中实现持久注册。

- 每个 `.py` 文件应定义一个继承 `xray_framework.models.base.BaseImageModel` 的类
- 注册后重启框架仍可用，并出现在训练/推理的模型下拉框中
- 权重文件（.pth）保存到 `outputs/models/external/`（不入 git）
