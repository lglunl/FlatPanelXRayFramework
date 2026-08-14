# 外部模型目录

通过界面「文献导入」页的「添加外部模型」导入的模型定义文件（.py）会保存到本目录，
并记录在 `../external_registry.json` 中实现持久注册。

- 每个 `.py` 文件应定义一个继承 `xray_framework.models.base.BaseImageModel` 的类
- 注册后重启框架仍可用，并出现在训练/推理的模型下拉框中

## 本地模型存放位置

所有模型权重统一保存在本地模型仓库 `outputs/models/`（不入 git）：

| 来源 | 定义文件 | 权重保存位置 |
| --- | --- | --- |
| 外部上传的模型 | 本目录 `external/xxx.py` | `outputs/models/external/<注册名>.pth` |
| 框架训练产物 | `xray_framework/models/algorithms/` | `outputs/models/<模型>_e<轮数>_<时间戳>.pth` |
| 按论文新建的模型 | 本目录或 `algorithms/` | 训练后自动保存到 `outputs/models/` |

说明：
- 导入外部模型时携带的权重（.pth/.pt）由 `add_external_model(..., weights_path=...)` 统一复制到
  `outputs/models/external/`，并把本地路径记录到 `external_registry.json`（重启后仍可见）。
- 界面「文献导入」页的「本地模型仓库」板块会展示全部已保存模型权重。
