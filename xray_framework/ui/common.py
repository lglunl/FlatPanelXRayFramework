"""UI 公共组件与训练状态管理"""
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import streamlit as st

from ..models.registry import discover, list_models, model_info
from ..training.progress import ProgressReporter, EpochResult


@dataclass
class TrainJobState:
    """训练任务状态（跨 rerun 保存在 session_state 中）"""
    running: bool = False
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
    logs: list = field(default_factory=list)
    best_path: Optional[str] = None
    error: Optional[str] = None
    thread: Optional[threading.Thread] = None
    cfg: Optional[dict] = None


def get_job_state() -> TrainJobState:
    if "train_job" not in st.session_state:
        st.session_state.train_job = TrainJobState()
    return st.session_state.train_job


class UIProgressReporter(ProgressReporter):
    """把训练事件写入 TrainJobState"""

    def __init__(self, state: TrainJobState):
        self.state = state

    def on_log(self, message: str):
        self.state.logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def on_train_start(self, total_epochs: int):
        self.state.total_epochs = total_epochs
        self.state.train_losses = []
        self.state.val_losses = []
        self.state.logs.append(f"[{time.strftime('%H:%M:%S')}] 开始训练，共 {total_epochs} epochs")

    def on_epoch_end(self, result: EpochResult):
        self.state.current_epoch = result.epoch
        self.state.progress = result.epoch / max(result.total_epochs if hasattr(result, "total_epochs") else self.state.total_epochs, 1)
        self.state.train_losses.append(float(result.train_loss))
        self.state.val_losses.append(float(result.val_loss))
        mark = " [最优]" if result.best else ""
        self.state.logs.append(
            f"[{time.strftime('%H:%M:%S')}] epoch {result.epoch}/{self.state.total_epochs}: "
            f"train={result.train_loss:.6f} val={result.val_loss:.6f}{mark}")

    def on_train_end(self, save_path: str):
        self.state.best_path = save_path
        self.state.running = False
        self.state.progress = 1.0
        self.state.logs.append(f"[{time.strftime('%H:%M:%S')}] 训练完成，模型保存: {save_path}")


def render_logs(state: TrainJobState, height: int = 260):
    import streamlit as st
    st.text_area("训练日志", value="\n".join(state.logs[-80:]), height=height, disabled=True)


def render_loss_curves(state: TrainJobState):
    import streamlit as st
    if not state.train_losses:
        st.info("训练开始后此处显示损失曲线")
        return
    chart_data = {"训练损失 (train loss)": state.train_losses,
                  "验证损失 (val loss)": state.val_losses}
    st.line_chart(chart_data)


def init_models():
    """确保模型已发现注册（供 UI 调用）"""
    discover()
    return list_models()
