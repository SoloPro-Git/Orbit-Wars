"""训练监控和指标收集工具。"""
import torch
import time
from typing import Dict, Any


class TrainingMonitor:
    """训练监控器，用于收集和记录训练过程中的各种指标。"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.step_count = 0
        self.start_time = time.time()

    def get_gpu_stats(self) -> Dict[str, float]:
        """获取GPU使用率和显存信息。"""
        if not torch.cuda.is_available():
            return {}

        stats = {}
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
            memory_total = props.total_memory / 1024**3  # GB

            stats[f"gpu_{i}_memory_allocated_gb"] = memory_allocated
            stats[f"gpu_{i}_memory_reserved_gb"] = memory_reserved
            stats[f"gpu_{i}_memory_total_gb"] = memory_total
            stats[f"gpu_{i}_memory_utilization"] = memory_allocated / memory_total if memory_total > 0 else 0

        return stats

    def get_training_speed_stats(
        self,
        num_steps: int,
        elapsed_time: float,
        buffer_size: int,
        num_games: int,
    ) -> Dict[str, float]:
        """计算训练速度相关指标。"""
        if elapsed_time == 0:
            return {}

        return {
            "steps_per_second": num_steps / elapsed_time if elapsed_time > 0 else 0,
            "samples_per_second": buffer_size / elapsed_time if elapsed_time > 0 else 0,
            "games_per_second": num_games / elapsed_time if elapsed_time > 0 else 0,
            "time_per_step": elapsed_time / num_steps if num_steps > 0 else 0,
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统级统计信息。"""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        return {
            "cpu_percent": process.cpu_percent(),
            "memory_rss_gb": process.memory_info().rss / 1024**3,
            "num_threads": process.num_threads(),
        }

    def update_metrics_with_monitoring(
        self,
        metrics: Dict[str, Any],
        num_games: int = 0,
        rollout_time: float = 0,
    ) -> Dict[str, Any]:
        """将监控指标添加到训练指标中。"""
        # 添加GPU统计
        gpu_stats = self.get_gpu_stats()
        metrics.update(gpu_stats)

        # 添加训练速度统计
        if rollout_time > 0 and num_games > 0:
            speed_stats = self.get_training_speed_stats(
                num_steps=1,
                elapsed_time=rollout_time,
                buffer_size=metrics.get("buffer_size", 0),
                num_games=num_games,
            )
            metrics.update(speed_stats)

        # 添加系统统计
        try:
            system_stats = self.get_system_stats()
            metrics.update(system_stats)
        except Exception:
            pass  # 系统统计可选，失败不影响训练

        self.step_count += 1
        elapsed_total = time.time() - self.start_time
        metrics["elapsed_time_total"] = elapsed_total

        return metrics
