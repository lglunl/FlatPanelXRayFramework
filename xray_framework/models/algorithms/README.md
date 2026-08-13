# 可扩展算法目录

把你新实现的去混叠算法 `.py` 文件放到**这个目录**，框架会自动发现并出现在界面的模型下拉框中。

## 如何新增一个算法（三步）

1. **创建文件**：在 `algorithms/` 下新建 `my_algo.py`

2. **继承基类并注册**：

```python
import torch.nn as nn
from ..base import BaseImageModel
from ..registry import register_model

@register_model("my_algo")          # 界面下拉框显示的名字
class MyAlgoModel(BaseImageModel):
    """算法说明（界面会显示这段文字）"""

    def __init__(self, in_channels=1, out_channels=1, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        # ... 搭建网络 ...

    def forward(self, x):
        # x: (B, 1, H, W) 范围 [0,1] 的含混叠图像
        # 返回: (B, 1, H, W) 范围 [0,1] 的去混叠图像
        return x
```

3. **保存文件**，刷新 Streamlit 页面，模型下拉框即可选择 `my_algo`。

## 迭代记录建议

每实现一个新算法，建议在本目录附带说明文档（如 `my_algo_notes.md`），记录：
- 参考文献/来源
- 网络结构要点
- 预期效果与实验对比
