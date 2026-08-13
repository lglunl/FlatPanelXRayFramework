"""训练板块 UI"""
import os
import threading

import pandas as pd
import streamlit as st

from ..config import PairingConfig, TrainConfig
from ..data.pairing import PairingEngine
from ..models.registry import list_models, model_info
from ..training.trainer import Trainer
from .common import UIProgressReporter, get_job_state, render_logs, render_loss_curves


def _scan_data_layouts():
    """提示用户可能的数据目录结构"""
    st.markdown("""
#### 数据目录结构说明
数据根目录下包含两个子目录，文件名中的**数字 ID 必须对应**：

```
data_root/
├── plate/          # 平板X射线源成像（含混叠，作为输入）
│   ├── pos001.tif
│   └── pos002.tif
└── focal/          # 热阴极点源真值（作为标签）
    ├── pos001.tif
    └── pos002.tif
```

- **by_id 模式**：从文件名提取数字 ID 配对（推荐，例如 `scan_0001.png` ↔ `truth_0001.png`）
- **by_index 模式**：两个目录按文件名排序后逐对配对（要求数量与顺序一致）
""")


def render_train_page():
    st.header("模型训练")
    st.caption("使用「平板X射线源成像 ↔ 热阴极点源真值」配对数据训练去混叠模型")

    state = get_job_state()
    if state.running:
        _render_running(state)
        return

    col1, col2 = st.columns([2, 3])

    # ---------- 左侧：数据配置 ----------
    with col1:
        st.subheader("1. 数据配置")
        data_root = st.text_input("数据根目录", value=state.cfg.get("data_root", "") if state.cfg else "",
                                  help="包含输入/真值子目录的文件夹路径")
        input_dir = st.text_input("输入目录名（平板源成像）", value="plate")
        truth_dir = st.text_input("真值目录名（点源成像）", value="focal")
        mode = st.selectbox("配对模式", ["by_id", "by_index"], index=0,
                            help="by_id: 按文件名数字ID配对; by_index: 按顺序配对")
        id_regex = st.text_input("ID 提取正则", value=r"(\d+)",
                                 help="从文件名提取数字ID用的正则，如 (\\d+)")

        if st.button("📋 预览数据配对", use_container_width=True):
            pcfg = PairingConfig(data_root=data_root, input_dir=input_dir,
                                 truth_dir=truth_dir, mode=mode, id_regex=id_regex)
            engine = PairingEngine(pcfg)
            st.session_state["_pair_engine"] = engine
            st.session_state["_pair_cfg"] = pcfg
            st.success(engine.summary())
            if engine.num_pairs > 0:
                st.dataframe(pd.DataFrame(engine.preview(12),
                                          columns=["输入(平板源)", "真值(点源)"]),
                             use_container_width=True)
            if engine.unmatched:
                st.warning(f"以下 {len(engine.unmatched)} 个文件未配对：")
                st.write("、".join(os.path.basename(u) for u in engine.unmatched[:20]))

    # ---------- 右侧：模型与训练参数 ----------
    with col2:
        st.subheader("2. 模型与训练参数")

        models = list_models()
        model_name = st.selectbox("选择算法模型", models)
        info = model_info(model_name)
        if info:
            st.info(info)

        loss_name = st.selectbox("损失函数", ["mse", "l1", "combined"],
                                 help="mse: 均方误差; l1: 平均绝对误差; combined: 两者混合")
        optimizer = st.selectbox("优化器", ["adam", "sgd"])

        c1, c2, c3 = st.columns(3)
        epochs = c1.number_input("训练轮数 epochs", min_value=1, value=50, step=1)
        batch_size = c2.number_input("批量大小 batch", min_value=1, value=1, step=1)
        lr = c3.number_input("学习率 lr", min_value=1e-6, value=1e-4, format="%.6f")

        c4, c5, c6 = st.columns(3)
        val_ratio = c4.slider("验证集比例", 0.0, 0.4, 0.15, 0.05)
        size_h = c5.number_input("图像高 H", min_value=64, value=512, step=64)
        size_w = c6.number_input("图像宽 W", min_value=64, value=512, step=64)

        c7, c8, c9 = st.columns(3)
        augment = c7.checkbox("数据增强", value=False, help="随机翻转/旋转")
        seed = c8.number_input("随机种子", min_value=0, value=42, step=1)
        device = c9.selectbox("计算设备", ["auto", "cpu", "cuda"])

        model_out_dir = st.text_input("模型保存目录", value=os.path.join("outputs", "models"))

    st.divider()

    # ---------- 底部：启动训练 ----------
    st.subheader("3. 启动训练")
    if st.button("🚀 开始训练", type="primary", use_container_width=True):
        # 校验
        if not data_root or not os.path.isdir(data_root):
            st.error("请填写有效的数据根目录路径")
        else:
            pcfg = PairingConfig(data_root=data_root, input_dir=input_dir,
                                 truth_dir=truth_dir, mode=mode, id_regex=id_regex)
            engine = PairingEngine(pcfg)
            if engine.num_pairs == 0:
                st.error(f"配对数量为 0，请检查目录结构：{engine.summary()}")
            else:
                tcfg = TrainConfig(
                    image_size=(int(size_h), int(size_w)),
                    model_name=model_name,
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    lr=float(lr),
                    val_ratio=float(val_ratio),
                    loss_name=loss_name,
                    optimizer=optimizer,
                    seed=int(seed),
                    device=device,
                    save_dir=model_out_dir,
                    augment=augment,
                )
                state.cfg = {"data_root": data_root}
                state.running = True
                state.error = None
                state.best_path = None
                state.progress = 0.0
                state.total_epochs = int(epochs)
                state.logs = [f"配对成功 {engine.num_pairs} 对，开始准备训练..."]
                reporter = UIProgressReporter(state)
                trainer = Trainer(tcfg, engine.pairs, reporter=reporter)

                def _run():
                    try:
                        reporter.on_log(f"设备: {trainer.device} | 模型: {tcfg.model_name}")
                        path = trainer.train(save_dir=model_out_dir)
                        state.best_path = path
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        state.error = str(e)
                        state.running = False

                state.thread = threading.Thread(target=_run, daemon=True)
                state.thread.start()
                st.rerun()

    _render_history(state)


def _render_running(state):
    st.warning("⏳ 训练进行中，本页面将实时刷新...")
    st.progress(state.progress)
    render_loss_curves(state)
    render_logs(state)
    st.button("🔄 刷新状态", on_click=lambda: st.rerun(), use_container_width=True)


def _render_history(state):
    if state.best_path:
        st.success(f"✅ 最新训练完成，模型已保存：`{state.best_path}`")
        # 提供继续训练的提示
        st.caption("训练结束后，可前往「模型推理」页面加载该模型进行去混叠处理。")
