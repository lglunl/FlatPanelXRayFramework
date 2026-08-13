"""训练进度回调机制

通过回调对象把训练过程事件推送给 UI（Streamlit）或控制台。
UI 侧只需实现 ProgressReporter 接口并注册。
"""
import time
from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    best: bool = False
    lr: float = 0.0
    seconds: float = 0.0


class ProgressReporter:
    """进度事件接收者基类（可在子类中覆写）"""
    def on_train_start(self, total_epochs: int):
        pass

    def on_epoch_end(self, result: EpochResult):
        pass

    def on_train_end(self, save_path: str):
        pass

    def on_log(self, message: str):
        pass


class ConsoleReporter(ProgressReporter):
    def on_train_start(self, total_epochs: int):
        print(f"[train] 开始训练，共 {total_epochs} epochs")

    def on_epoch_end(self, result: EpochResult):
        mark = " *best*" if result.best else ""
        print(f"[train] epoch {result.epoch}: train_loss={result.train_loss:.6f}, "
              f"val_loss={result.val_loss:.6f}{mark} ({result.seconds:.1f}s)")

    def on_train_end(self, save_path: str):
        print(f"[train] 训练完成，模型已保存: {save_path}")

    def on_log(self, message: str):
        print(f"[train] {message}")
