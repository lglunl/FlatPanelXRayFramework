"""数据配对引擎

负责将「平板X射线源成像」（含混叠，作为输入）与「热阴极点源真值」（作为标签）配对。

支持的配对模式：
  - by_id   : 从文件名提取数字 ID，相同 ID 的输入与真值配对
              （例如 plate_scan_pos001.tif <-> focal_pos001.tif，或 scan_0001.png <-> truth_0001.png）
  - by_index: 两个目录按文件名排序后逐对配对（要求数量一致、顺序一致）

数据目录建议结构：
    data_root/
    ├── plate/          # 平板源成像（输入）
    │   ├── pos001.tif
    │   └── pos002.tif
    └── focal/          # 点源真值（标签）
        ├── pos001.tif
        └── pos002.tif
"""
import os
import re

from ..config import PairingConfig
from .image_io import list_images


class PairingEngine:
    def __init__(self, cfg: PairingConfig):
        self.cfg = cfg
        self.input_dir = os.path.join(cfg.data_root, cfg.input_dir)
        self.truth_dir = os.path.join(cfg.data_root, cfg.truth_dir)
        self._input_files = list_images(self.input_dir)
        self._truth_files = list_images(self.truth_dir)
        self.pairs = []   # list[(input_path, truth_path)]
        self.unmatched = []  # list[path] 未配对文件
        self._build()

    def _build(self):
        if self.cfg.mode == "by_id":
            self._build_by_id()
        else:
            self._build_by_index()

    def _build_by_id(self):
        regex = re.compile(self.cfg.id_regex)
        truth_map = {}
        for f in self._truth_files:
            m = regex.search(f)
            if m:
                truth_map.setdefault(m.group(1), os.path.join(self.truth_dir, f))
        for f in self._input_files:
            m = regex.search(f)
            if m and m.group(1) in truth_map:
                self.pairs.append((os.path.join(self.input_dir, f), truth_map[m.group(1)]))
            else:
                self.unmatched.append(os.path.join(self.input_dir, f))
        # 找出没有对应输入的真值
        for f in self._truth_files:
            m = regex.search(f)
            if m and m.group(1) not in {re.search(regex, os.path.basename(p[0])).group(1)
                                         if re.search(regex, os.path.basename(p[0])) else None
                                         for p in self.pairs}:
                if not any(t == os.path.join(self.truth_dir, f) for _, t in self.pairs):
                    self.unmatched.append(os.path.join(self.truth_dir, f))

    def _build_by_index(self):
        n = min(len(self._input_files), len(self._truth_files))
        for i in range(n):
            self.pairs.append((
                os.path.join(self.input_dir, self._input_files[i]),
                os.path.join(self.truth_dir, self._truth_files[i]),
            ))
        for f in self._input_files[n:]:
            self.unmatched.append(os.path.join(self.input_dir, f))
        for f in self._truth_files[n:]:
            self.unmatched.append(os.path.join(self.truth_dir, f))

    @property
    def num_pairs(self) -> int:
        return len(self.pairs)

    def preview(self, limit: int = 10) -> list:
        """返回前 limit 对配对的 (输入文件名, 真值文件名)"""
        return [(os.path.basename(i), os.path.basename(t)) for i, t in self.pairs[:limit]]

    def summary(self) -> str:
        return (f"配对模式: {self.cfg.mode} | 输入 {len(self._input_files)} 张, "
                f"真值 {len(self._truth_files)} 张, 成功配对 {len(self.pairs)} 对, "
                f"未配对 {len(self.unmatched)} 个")
