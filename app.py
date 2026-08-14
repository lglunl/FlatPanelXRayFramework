"""
平板X射线去混叠成像算法框架 - 可视化操作界面

启动方式：
    streamlit run app.py
"""
import os
import sys

# 保证项目根目录在 sys.path 中（无论从哪里启动）
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import streamlit as st

from xray_framework.models.registry import list_models
from xray_framework.ui.train_page import render_train_page
from xray_framework.ui.infer_page import render_infer_page
from xray_framework.ui.lit_page import render as render_lit_page
from xray_framework import __version__

st.set_page_config(
    page_title="平板X射线去混叠成像框架",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- 侧边栏导航 ----------
with st.sidebar:
    st.title("🩻 平板X射线")
    st.subheader("去混叠成像算法框架")
    st.caption(f"v{__version__}")

    nav = st.radio(
        "导航",
        ["模型训练", "模型推理", "文献导入"],
        label_visibility="collapsed",
    )

    st.divider()
    try:
        models = list_models()
        st.caption(f"可用算法模型 ({len(models)}): {', '.join(models)}")
    except Exception as e:
        st.caption(f"模型加载异常: {e}")

    st.divider()
    st.caption("""
**算法迭代说明**：在「文献导入」页上传文献并生成迭代请求，
CodeBuddy 读取请求后实现新算法；新增的 `.py` 文件放入
`xray_framework/models/algorithms/` 目录即可被自动发现。
""")

# ---------- 主内容 ----------
if nav == "模型训练":
    render_train_page()
elif nav == "模型推理":
    render_infer_page()
else:
    render_lit_page()
