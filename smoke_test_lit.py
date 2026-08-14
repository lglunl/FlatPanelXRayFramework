"""文献导入与算法迭代请求 - 冒烟测试

验证闭环：导入文献（多语言代码）→ 提取代码块 → 分类/删除 → 生成迭代请求
        → 非 Python 代码转换请求 → 转换回填 → 状态流转
运行：.venv\\Scripts\\python smoke_test_lit.py
"""
import os
import sys
import tempfile

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
    set_literature_category,
    list_categories,
    create_code_conversion_request,
    list_code_conversion_requests,
    get_code_conversion_request,
    mark_conversion_status,
    update_literature_code_blocks,
)
from xray_framework.literature.extract import (
    extract_code_blocks,
    extract_all_code_blocks,
    guess_title,
)
from xray_framework.models.registry import (
    discover,
    list_models,
    add_external_model,
    list_local_models,
    _load_external_records,
    _save_external_records,
    _EXTERNAL_DIR,
)

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

## 参考实现（MATLAB）
```matlab
function y = myfilter(x)
    y = imfilter(x, fspecial('gaussian', 5, 1));
end
```

## 结论
注意力机制可有效抑制混叠伪影。
"""


def main():
    passed = 0

    # 1. 文本提取（含多语言代码识别）
    blocks = extract_code_blocks(MOCK_LIT)
    assert len(blocks) == 1 and "ChannelAttention" in blocks[0], "Python 代码块提取失败"
    assert "注意力机制的平板X射线" in guess_title(MOCK_LIT), "标题猜测失败"
    allb = extract_all_code_blocks(MOCK_LIT)
    assert len(allb) == 2, "多语言代码块提取失败"
    mb = next((x for x in allb if x["lang"] == "matlab"), None)
    assert mb is not None and mb["needs_conversion"], "非 Python 代码未标注需转换"
    passed += 1
    print("[OK] 代码块提取与标题猜测（含多语言识别）")

    # 2. 保存文献
    save_literature_text("test_lit_attention.md", MOCK_LIT)
    records = list_literature()
    rec = next((r for r in records if r["file"] == "test_lit_attention.md"), None)
    assert rec is not None, "文献未入库"
    assert len(rec["code_blocks"]) == 1, "入库文献未提取 Python 代码块"
    assert len(rec["code_blocks_all"]) == 2, "入库文献未记录多语言代码块"
    assert rec["category"] == "未分类", "文献默认分类错误"
    passed += 1
    print("[OK] 文献入库与多语言代码索引")

    # 3. 文献分类
    assert set_literature_category("test_lit_attention.md", "注意力机制"), "分类设置失败"
    rec = next(r for r in list_literature() if r["file"] == "test_lit_attention.md")
    assert rec["category"] == "注意力机制", "分类未生效"
    assert "注意力机制" in list_categories(), "分类列表缺失"
    passed += 1
    print("[OK] 文献分类设置与列表")

    # 4. 生成迭代请求
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
    assert len(req["references"][0]["code_blocks"]) == 1, "引用文献 Python 代码块缺失"
    assert len(req["references"][0]["code_blocks_all"]) == 2, "引用文献多语言代码缺失"
    assert any(
        b["lang"] == "matlab" and b["needs_conversion"]
        for b in req["references"][0]["code_blocks_all"]
    ), "引用文献未标注非 Python 代码"
    passed += 1
    print("[OK] 迭代请求生成（携带多语言代码）")

    # 5. 请求列表与状态流转
    reqs = list_iteration_requests()
    assert any(r["id"] == req["id"] for r in reqs), "请求未出现在列表"
    assert get_iteration_request(req["id"])["id"] == req["id"], "按 id 读取失败"
    mark_request_status(req["id"], "done")
    assert get_iteration_request(req["id"])["status"] == "done", "状态流转失败"
    passed += 1
    print("[OK] 请求列表与状态流转")

    # 6. 新建模型类型的迭代请求
    req_new = create_iteration_request(
        base_model="",
        goal="根据引用文献设计全新的平板X射线去混叠网络",
        ref_files=["test_lit_attention.md"],
        code_snippets=[],
        request_type="new",
    )
    assert req_new["request_type"] == "new", "新建模型类型未记录"
    assert req_new["base_model"] == "", "新建模型基类应为空"
    passed += 1
    print("[OK] 新建模型迭代请求")

    # 7. 非 Python 代码转换：生成转换请求 → 回填转换结果 → 状态流转
    rec = next(r for r in list_literature() if r["file"] == "test_lit_attention.md")
    m_idx = next(
        i for i, b in enumerate(rec["code_blocks_all"]) if b["lang"] == "matlab"
    )
    conv = create_code_conversion_request(
        "test_lit_attention.md", m_idx, "matlab",
        rec["code_blocks_all"][m_idx]["code"], notes="转换为 Python",
    )
    assert conv["status"] == "pending", "转换请求初始状态错误"
    assert conv["type"] == "code_conversion", "转换请求类型错误"
    assert any(
        c["id"] == conv["id"] for c in list_code_conversion_requests()
    ), "转换请求未列出"
    # 模拟 CodeBuddy 转换后回填
    py_code = (
        "import cv2\nimport numpy as np\n\n"
        "def myfilter(x):\n"
        "    k = cv2.getGaussianKernel(5, 1)\n"
        "    return cv2.filter2D(x, -1, k)\n"
    )
    assert update_literature_code_blocks(
        "test_lit_attention.md", m_idx, py_code
    ), "转换结果回填失败"
    rec = next(r for r in list_literature() if r["file"] == "test_lit_attention.md")
    m_new = rec["code_blocks_all"][m_idx]
    assert m_new["is_python"] and not m_new["needs_conversion"], "转换结果未生效"
    assert m_new["converted_from"] == "matlab", "转换来源未记录"
    assert any(py_code.strip() in b for b in rec["code_blocks"]), "转换后的 Python 未进入代码块列表"
    assert mark_conversion_status(conv["id"], "done"), "转换状态流转失败"
    assert get_code_conversion_request(conv["id"])["status"] == "done", "转换状态未生效"
    passed += 1
    print("[OK] 非 Python 代码转换（请求→回填→状态流转）")

    # 8. 外部模型持久化注册（含权重保存到本地模型仓库）
    ext_py = os.path.join(tempfile.gettempdir(), "ext_smoke_model.py")
    with open(ext_py, "w", encoding="utf-8") as f:
        f.write(
            "import torch.nn as nn\n"
            "from xray_framework.models.base import BaseImageModel\n\n"
            "class SimpleResBlock(BaseImageModel):\n"
            "    '''simple external model'''\n"
            "    def __init__(self, in_channels=1, out_channels=1, **kwargs):\n"
            "        super().__init__(in_channels=in_channels, out_channels=out_channels)\n"
            "        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)\n"
            "    def forward(self, x):\n"
            "        return self.conv(x)\n"
        )
    discover()
    w_path = os.path.join(tempfile.gettempdir(), "ext_smoke_weights.pth")
    with open(w_path, "wb") as f:
        f.write(b"\x80\x03}q\x00X\x01\x00\x00\x00aq\x01K\x01s.")  # 占位权重
    name = add_external_model(
        ext_py, registry_name="ext_res", class_name="SimpleResBlock", weights_path=w_path
    )
    assert name == "ext_res", "外部模型注册名错误"
    assert "ext_res" in list_models(), "外部模型未出现在模型列表"
    # 持久化记录存在
    recs = _load_external_records()
    assert any(r["registry_name"] == "ext_res" for r in recs), "外部模型未持久化"
    # 权重已统一复制到本地模型仓库 outputs/models/external/ 并记录路径
    rec_ext = next(r for r in recs if r["registry_name"] == "ext_res")
    weights_saved = rec_ext.get("weights", "")
    assert weights_saved and os.path.isfile(weights_saved), "外部模型权重未保存到本地仓库"
    assert weights_saved.replace("\\", "/").endswith(
        "outputs/models/external/ext_res.pth"
    ), f"权重保存位置错误: {weights_saved}"
    # 本地模型仓库可见该权重
    local = list_local_models()
    assert any(
        m["path"] == os.path.abspath(weights_saved) for m in local
    ), "本地模型仓库未列出外部权重"
    passed += 1
    print("[OK] 外部模型持久化注册（含权重保存到本地模型仓库）")

    # 9. 清理测试数据
    remove_literature("test_lit_attention.md")
    assert not list_literature() or not any(
        r["file"] == "test_lit_attention.md" for r in list_literature()
    ), "测试文献未清理"
    for rid in (req["id"], req_new["id"], conv["id"]):
        req_file = os.path.join(ROOT, "requests", f"{rid}.json")
        if os.path.exists(req_file):
            os.remove(req_file)
    # 清理外部模型
    import xray_framework.models.registry as _reg
    _reg.MODEL_REGISTRY.pop("ext_res", None)
    _save_external_records([r for r in recs if r["registry_name"] != "ext_res"])
    dst = os.path.join(_EXTERNAL_DIR, "ext_smoke_model.py")
    if os.path.exists(dst):
        os.remove(dst)
    if os.path.exists(ext_py):
        os.remove(ext_py)
    if weights_saved and os.path.exists(weights_saved):
        os.remove(weights_saved)
    if os.path.exists(w_path):
        os.remove(w_path)
    passed += 1
    print("[OK] 清理测试数据")

    print(f"\n全部通过 ({passed}/9)")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n断言失败: {e}")
        sys.exit(1)
