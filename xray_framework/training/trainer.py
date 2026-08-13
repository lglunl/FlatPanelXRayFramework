"""训练器"""
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import TrainConfig
from ..data.dataset import PairedImageDataset, split_pairs
from ..models.registry import get_model
from .losses import get_loss
from .progress import ConsoleReporter, EpochResult, ProgressReporter


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class Trainer:
    def __init__(self, cfg: TrainConfig, pairs: list,
                 reporter: ProgressReporter = None):
        self.cfg = cfg
        self.pairs = pairs
        self.reporter = reporter or ConsoleReporter()

        set_seed(cfg.seed)
        self.device = resolve_device(cfg.device)
        self.reporter.on_log(f"使用设备: {self.device}")

        self.model = get_model(cfg.model_name,
                               in_channels=1, out_channels=1).to(self.device)
        self.criterion = get_loss(cfg.loss_name)
        self.optimizer = self._build_optimizer()
        self.history = {"train_loss": [], "val_loss": []}

    def _build_optimizer(self):
        if self.cfg.optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr, momentum=0.9)
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    def _build_loaders(self):
        train_pairs, val_pairs = split_pairs(self.pairs, self.cfg.val_ratio, self.cfg.seed)
        self.reporter.on_log(
            f"数据划分: 训练 {len(train_pairs)} 对 / 验证 {len(val_pairs)} 对")
        train_ds = PairedImageDataset(train_pairs, self.cfg.image_size, augment=self.cfg.augment)
        val_ds = PairedImageDataset(val_pairs, self.cfg.image_size)
        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size,
                                  shuffle=True, num_workers=self.cfg.num_workers)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                num_workers=self.cfg.num_workers)
        return train_loader, val_loader

    def train(self, save_dir: str = "") -> str:
        save_dir = save_dir or os.path.join("outputs", "models")
        os.makedirs(save_dir, exist_ok=True)
        model_name = self.cfg.model_name
        tag = f"{model_name}_e{self.cfg.epochs}_{time.strftime('%Y%m%d_%H%M%S')}"
        best_path = os.path.join(save_dir, f"{tag}.pth")

        train_loader, val_loader = self._build_loaders()
        best_val_loss = float("inf")
        self.reporter.on_train_start(self.cfg.epochs)

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                torch.save({
                    "model_name": self.cfg.model_name,
                    "model_state_dict": self.model.state_dict(),
                    "image_size": list(self.cfg.image_size),
                    "loss_name": self.cfg.loss_name,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                }, best_path)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            self.reporter.on_epoch_end(EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best=is_best,
                lr=self.optimizer.param_groups[0]["lr"],
                seconds=time.time() - t0,
            ))

        self.reporter.on_train_end(best_path)
        return best_path

    def _run_epoch(self, loader, train: bool) -> float:
        self.model.train(train)
        total, count = 0.0, 0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                if train:
                    self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                if train:
                    loss.backward()
                    self.optimizer.step()
                total += loss.item() * x.size(0)
                count += x.size(0)
        return total / max(count, 1)
