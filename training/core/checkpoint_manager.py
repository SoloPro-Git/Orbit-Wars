"""Checkpoint管理器，用于自动清理旧的checkpoint。"""
import os
import torch
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class CheckpointInfo:
    """Checkpoint信息。"""
    path: Path
    iteration: int
    win_rate: float = 0.0
    file_size: float = 0.0  # MB


class CheckpointManager:
    """Checkpoint管理器，自动清理旧模型。"""

    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 5,
        keep_best_n: int = 2,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.keep_best_n = keep_best_n
        self.checkpoints: List[CheckpointInfo] = []
        self.best_checkpoints: List[CheckpointInfo] = []

        # 加载已有checkpoint
        self._load_existing_checkpoints()

    def _load_existing_checkpoints(self):
        """加载已存在的checkpoint信息。"""
        if not self.checkpoint_dir.exists():
            return

        for ckpt_file in self.checkpoint_dir.glob("model_iter_*.pt"):
            try:
                # 从文件名提取iteration
                iter_str = ckpt_file.stem.replace("model_iter_", "")
                iteration = int(iter_str)

                # 获取文件大小
                file_size = ckpt_file.stat().st_size / (1024 * 1024)  # MB

                ckpt_info = CheckpointInfo(
                    path=ckpt_file,
                    iteration=iteration,
                    file_size=file_size,
                )
                self.checkpoints.append(ckpt_info)
            except (ValueError, OSError):
                continue

        # 按iteration排序
        self.checkpoints.sort(key=lambda x: x.iteration)

    def add_checkpoint(self, iteration: int, win_rate: float = 0.0) -> CheckpointInfo:
        """添加新的checkpoint信息。"""
        ckpt_path = self.checkpoint_dir / f"model_iter_{iteration}.pt"
        file_size = ckpt_path.stat().st_size / (1024 * 1024)  # MB

        ckpt_info = CheckpointInfo(
            path=ckpt_path,
            iteration=iteration,
            win_rate=win_rate,
            file_size=file_size,
        )

        self.checkpoints.append(ckpt_info)
        return ckpt_info

    def update_best_checkpoint(self, iteration: int, win_rate: float):
        """更新最佳checkpoint列表。"""
        ckpt_info = next(
            (c for c in self.checkpoints if c.iteration == iteration),
            None
        )
        if ckpt_info:
            ckpt_info.win_rate = win_rate
            self.best_checkpoints.append(ckpt_info)
            # 按胜率排序
            self.best_checkpoints.sort(key=lambda x: x.win_rate, reverse=True)

    def cleanup_old_checkpoints(self):
        """清理旧的checkpoint。"""
        if not self.checkpoint_dir.exists():
            return

        # 获取所有checkpoint文件
        all_checkpoints = list(self.checkpoint_dir.glob("model_iter_*.pt"))

        if len(all_checkpoints) <= self.max_checkpoints:
            return  # 不需要清理

        # 按iteration排序，保留最新的
        all_checkpoints.sort(key=lambda p: self._extract_iteration(p))

        # 确定要保留的checkpoint
        to_keep = set()

        # 1. 保留最新的max_checkpoints个
        for ckpt in all_checkpoints[-self.max_checkpoints:]:
            to_keep.add(ckpt)

        # 2. 保留胜率最高的keep_best_n个
        if self.best_checkpoints:
            best_by_winrate = sorted(
                self.best_checkpoints,
                key=lambda x: x.win_rate,
                reverse=True
            )[:self.keep_best_n]
            for ckpt_info in best_by_winrate:
                to_keep.add(ckpt_info.path)

        # 删除不在保留列表中的checkpoint
        removed_count = 0
        removed_size = 0.0
        for ckpt in all_checkpoints:
            if ckpt not in to_keep:
                try:
                    size = ckpt.stat().st_size / (1024 * 1024)  # MB
                    ckpt.unlink()
                    removed_count += 1
                    removed_size += size
                    print(f"  [Checkpoint清理] 删除: {ckpt.name} ({size:.1f}MB)")
                except OSError as e:
                    print(f"  [Checkpoint清理] 删除失败 {ckpt.name}: {e}")

        if removed_count > 0:
            print(f"  [Checkpoint清理] 删除了 {removed_count} 个旧checkpoint，释放 {removed_size:.1f}MB")

        # 更新内部列表
        self._load_existing_checkpoints()

    def _extract_iteration(self, path: Path) -> int:
        """从路径中提取iteration。"""
        try:
            iter_str = path.stem.replace("model_iter_", "")
            return int(iter_str)
        except ValueError:
            return 0

    def get_latest_checkpoint(self) -> Optional[Path]:
        """获取最新的checkpoint。"""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1].path

    def get_best_checkpoint(self) -> Optional[Path]:
        """获取胜率最高的checkpoint。"""
        if not self.best_checkpoints:
            return None
        return self.best_checkpoints[0].path

    def get_stats(self) -> Dict:
        """获取统计信息。"""
        total_size = sum(c.file_size for c in self.checkpoints)
        return {
            "total_checkpoints": len(self.checkpoints),
            "total_size_mb": total_size,
            "latest_iteration": self.checkpoints[-1].iteration if self.checkpoints else 0,
            "best_win_rate": self.best_checkpoints[0].win_rate if self.best_checkpoints else 0.0,
        }
