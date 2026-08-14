"""文献导入与算法迭代请求 - 冒烟测试

验证闭环：导入文献 → 自动提取代码块 → 生成迭代请求 → 状态流转
运行：.venv\\Scripts\\python smoke_test_lit.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from xray_framework.literature import (
    save_literature_text,
    list_literature,
    remove_literature,
    create_iteration_request,
    list_iteration_requests,
    get_iteration_request,
    mark_request_status,
)
from xray_framework.literature.extract import extract_code_blocks, guess_title

MOCK_LIT = """# 基于注意力机制的平板X射线混叠抑制

## 摘要
本文提出一种基于通道注意力的去混叠网络。

## 核心代码
```python
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w
```

## 结论
注意力机制可有效抑制混叠伪影。
"""


def main():
    passed = 0

    # 1. 文本提取
    blocks = extract_code_blocks(MOCK_LIT)
    assert len(blocks) == 1 and "ChannelAttention" in blocks[0], "代码块提取失败"
    assert "注意力机制的平板X射线" in guess_title(MOCK_LIT), "标题猜测失败"
    passed += 1
    print("[OK] 代码块提取与标题猜测")

    # 2. 保存文献
    save_literature_text("test_lit_attention.md", MOCK_LIT)
    records = list_literature()
    rec = next((r for r in records if r["file"] == "test_lit_attention.md"), None)
    assert rec is not None, "文献未入库"
    assert len(rec["code_blocks"]) == 1, "入库文献未提取代码块"
    passed += 1
    print("[OK] 文献入库与索引")

    # 3. 生成迭代请求
    req = create_iteration_request(
        base_model="unet",
        goal="引入文献中的通道注意力，改进 UNet 编码器抑制平板混叠伪影",
        ref_files=["test_lit_attention.md"],
        code_snippets=[blocks[0]],
        notes="冒烟测试",
    )
    assert req["status"] == "pending", "请求初始状态错误"
    assert req["base_model"] == "unet", "基类模型记录错误"
    assert len(req["references"]) == 1, "引用文献缺失"
    assert req["references"][0]["file"] == "test_lit_attention.md", "引用文献路径错误"
    assert len(req["references"][0]["code_blocks"]) == 1, "引用文献代码块缺失"
    passed += 1
    print("[OK] 迭代请求生成")

    # 4. 请求列表与状态流转
    reqs = list_iteration_requests()
    assert any(r["id"] == req["id"] for r in reqs), "请求未出现在列表"
    assert get_iteration_request(req["id"])["id"] == req["id"], "按 id 读取失败"
    mark_request_status(req["id"], "done")
    assert get_iteration_request(req["id"])["status"] == "done", "状态流转失败"
    passed += 1
    print("[OK] 请求列表与状态流转")

    # 5. 清理测试数据
    remove_literature("test_lit_attention.md")
    assert not list_literature() or not any(
        r["file"] == "test_lit_attention.md" for r in list_literature()
    ), "测试文献未清理"
    req_file = os.path.join(ROOT, "requests", f"{req['id']}.json")
    if os.path.exists(req_file):
        os.remove(req_file)
    passed += 1
    print("[OK] 清理测试数据")

    print(f"\n全部通过 ({passed}/5)")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n断言失败: {e}")
        sys.exit(1)
