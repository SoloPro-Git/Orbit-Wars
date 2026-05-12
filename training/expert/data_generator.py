"""专家数据生成器模块。

使用专家策略生成训练数据，为模型提供冷启动数据。
数据格式: JSONL (每行一个 JSON 对象)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from training.core.env_wrapper import OrbitWarsEnv
from training.expert.kaggle_expert import KaggleExpertAgent


class ExpertDataGenerator:
    """专家数据生成器。

    运行专家策略对战，生成 (observation, action) 训练对。
    """

    def __init__(
        self,
        num_players: int = 4,
        save_dir: str | Path = "data/expert_demonstrations",
        save_format: str = "jsonl",
    ):
        """初始化数据生成器。

        Args:
            num_players: 玩家数量。
            save_dir: 数据保存目录。
            save_format: 保存格式，"jsonl" 或 "pkl"。
        """
        self.num_players = num_players
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_format = save_format

        # 创建专家智能体
        self.experts = [
            KaggleExpertAgent(player_id=i) for i in range(num_players)
        ]

    def generate_episode(
        self,
        seed: int | None = None,
        max_steps: int = 500,
    ) -> list[dict]:
        """生成一局游戏的对局数据。

        Args:
            seed: 随机种子。
            max_steps: 最大步数。

        Returns:
            轨迹列表，每个元素是 {observation, actions, reward} 字典。
        """
        env = OrbitWarsEnv(num_players=self.num_players)
        observations = env.reset(seed=seed)

        trajectory = []

        for step_idx in range(max_steps):
            if env.done:
                break

            # 收集所有玩家的动作
            actions = {}
            obs_dict = {}
            for player_id in range(self.num_players):
                obs = env.get_raw_observation(player_id)
                obs_dict[player_id] = obs

                # 使用专家策略生成动作
                expert_actions = self.experts[player_id].get_actions(obs)
                actions[player_id] = expert_actions

            # 执行一步
            next_obs, rewards, dones, infos = env.step(actions)

            # 记录每个玩家的经验
            for player_id in range(self.num_players):
                trajectory.append({
                    "player_id": player_id,
                    "step": step_idx,
                    "observation": obs_dict[player_id],
                    "actions": actions[player_id],
                    "reward": rewards.get(player_id, 0.0),
                    "done": dones.get(player_id, True),
                })

            observations = next_obs

        return trajectory

    def generate_dataset(
        self,
        num_episodes: int = 100,
        episodes_per_file: int = 10,
        prefix: str = "expert_data",
    ) -> list[Path]:
        """生成完整数据集。

        Args:
            num_episodes: 生成的对局数量。
            episodes_per_file: 每个文件保存的对局数量。
            prefix: 文件名前缀。

        Returns:
            保存的文件路径列表。
        """
        all_trajectories = []
        file_paths = []
        file_idx = 0

        # 根据格式确定文件扩展名
        ext = "jsonl" if self.save_format == "jsonl" else "pkl"

        print(f"开始生成 {num_episodes} 局专家演示数据...")

        for episode_idx in tqdm(range(num_episodes), desc="生成对局"):
            seed = 42 + episode_idx
            trajectory = self.generate_episode(seed=seed)
            all_trajectories.extend(trajectory)

            # 达到指定数量后保存
            if (episode_idx + 1) % episodes_per_file == 0:
                file_path = self.save_dir / f"{prefix}_{file_idx:04d}.{ext}"
                self._save_trajectories(all_trajectories, file_path)
                file_paths.append(file_path)
                all_trajectories = []
                file_idx += 1

        # 保存剩余数据
        if all_trajectories:
            file_path = self.save_dir / f"{prefix}_{file_idx:04d}.{ext}"
            self._save_trajectories(all_trajectories, file_path)
            file_paths.append(file_path)

        print(f"✓ 数据生成完成，共保存 {len(file_paths)} 个文件")
        return file_paths

    def _save_trajectories(
        self,
        trajectories: list[dict],
        file_path: Path,
    ) -> None:
        """保存轨迹到文件。

        Args:
            trajectories: 轨迹列表。
            file_path: 保存路径。
        """
        if self.save_format == "jsonl":
            # JSONL 格式：每行一个 JSON 对象
            with open(file_path, "w") as f:
                for traj in trajectories:
                    # 转换 numpy 类型为 Python 原生类型
                    traj_serializable = self._make_json_serializable(traj)
                    f.write(json.dumps(traj_serializable, ensure_ascii=False) + "\n")
        elif self.save_format == "pkl":
            # Pickle 格式（向后兼容）
            import pickle
            with open(file_path, "wb") as f:
                pickle.dump(trajectories, f)
        else:
            raise ValueError(f"Unsupported save format: {self.save_format}")

    def _make_json_serializable(self, obj: Any) -> Any:
        """将对象转换为 JSON 可序列化格式。

        Args:
            obj: 要转换的对象。

        Returns:
            JSON 可序列化的对象。
        """
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def load_trajectories(self, file_path: Path) -> list[dict]:
        """从文件加载轨迹。

        Args:
            file_path: 文件路径。

        Returns:
            轨迹列表。
        """
        # 根据文件扩展名检测格式
        if file_path.suffix == ".jsonl":
            # JSONL 格式
            trajectories = []
            with open(file_path, "r") as f:
                for line in f:
                    if line.strip():
                        trajectories.append(json.loads(line))
            return trajectories
        elif file_path.suffix == ".pkl":
            # Pickle 格式（向后兼容）
            import pickle
            with open(file_path, "rb") as f:
                return pickle.load(f)
        else:
            # 尝试根据内容检测
            try:
                with open(file_path, "r") as f:
                    first_char = f.read(1)
                    if first_char == "{":
                        # JSONL 格式
                        f.seek(0)
                        trajectories = []
                        for line in f:
                            if line.strip():
                                trajectories.append(json.loads(line))
                        return trajectories
            except:
                pass

            # 默认尝试 pickle
            import pickle
            with open(file_path, "rb") as f:
                return pickle.load(f)


class ExpertDataset:
    """专家数据集类。

    提供数据加载和批处理功能。
    """

    def __init__(
        self,
        data_dir: str | Path = "data/expert_demonstrations",
        extra_data_dirs: list[str | Path] | None = None,
        max_samples: int | None = None,
    ):
        """初始化数据集。

        Args:
            data_dir: 数据目录。
            max_samples: 最大样本数量，None 表示全部加载。
        """
        self.data_dir = Path(data_dir)
        self.extra_data_dirs = [Path(p) for p in (extra_data_dirs or [])]
        self.max_samples = max_samples
        self.data = []
        self._load_data()

    def _load_data(self):
        """加载所有数据文件。"""
        all_dirs = [self.data_dir] + self.extra_data_dirs
        data_files = []
        for d in all_dirs:
            if not d.exists():
                continue
            jsonl_files = list(d.glob("*.jsonl"))
            pkl_files = list(d.glob("*.pkl"))
            data_files.extend(jsonl_files + pkl_files)

        if self.max_samples:
            print(f"发现 {len(data_files)} 个数据文件（限制加载 {self.max_samples} 样本）")
        else:
            print(f"发现 {len(data_files)} 个数据文件")

        generator = ExpertDataGenerator()

        for file_path in tqdm(data_files, desc="加载数据"):
            trajectories = generator.load_trajectories(file_path)
            self.data.extend(trajectories)

            if self.max_samples and len(self.data) >= self.max_samples:
                self.data = self.data[:self.max_samples]
                break

        print(f"✓ 共加载 {len(self.data)} 个样本")

    def __len__(self) -> int:
        """返回数据集大小。"""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """获取单个样本。"""
        return self.data[idx]

    def get_batch(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> list[dict]:
        """获取一个批次的数据。

        Args:
            batch_size: 批次大小。
            shuffle: 是否打乱数据。

        Returns:
            批次数据列表。
        """
        if len(self.data) == 0 or batch_size <= 0:
            return []

        indices = np.random.choice(len(self.data), size=batch_size, replace=True)
        return [self.data[i] for i in indices]

    def split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> tuple["ExpertDataset", "ExpertDataset"]:
        """分割数据集为训练集和验证集。

        Args:
            train_ratio: 训练集比例。
            seed: 随机种子（用于可复现随机切分）。

        Returns:
            (train_dataset, val_dataset)
        """
        if len(self.data) == 0:
            train_data = []
            val_data = []
        else:
            rng = np.random.default_rng(seed)
            indices = np.arange(len(self.data))
            rng.shuffle(indices)
            n_train = int(len(indices) * train_ratio)
            train_idx = indices[:n_train]
            val_idx = indices[n_train:]
            train_data = [self.data[i] for i in train_idx]
            val_data = [self.data[i] for i in val_idx]

        train_ds = ExpertDataset.__new__(ExpertDataset)
        train_ds.data = train_data
        train_ds.max_samples = None
        train_ds.data_dir = self.data_dir
        train_ds.extra_data_dirs = self.extra_data_dirs

        val_ds = ExpertDataset.__new__(ExpertDataset)
        val_ds.data = val_data
        val_ds.max_samples = None
        val_ds.data_dir = self.data_dir
        val_ds.extra_data_dirs = self.extra_data_dirs

        return train_ds, val_ds


def print_statistics(dataset: ExpertDataset) -> None:
    """打印数据集统计信息。

    Args:
        dataset: 数据集对象。
    """
    print(f"\n{'='*60}")
    print(f"数据集统计信息")
    print(f"{'='*60}")
    print(f"总样本数: {len(dataset)}")

    if len(dataset) == 0:
        print(f"{'='*60}\n")
        return

    # 统计每个玩家的样本数
    player_counts = {}
    for sample in dataset.data:
        pid = sample["player_id"]
        player_counts[pid] = player_counts.get(pid, 0) + 1

    print(f"\n玩家样本分布:")
    for pid, count in sorted(player_counts.items()):
        print(f"  玩家 {pid}: {count} 个样本")

    # 统计动作分布
    action_counts = []
    for sample in dataset.data:
        action_counts.append(len(sample["actions"]))

    if action_counts:
        print(f"\n每步动作数量:")
        print(f"  最小: {min(action_counts)}")
        print(f"  最大: {max(action_counts)}")
        print(f"  平均: {np.mean(action_counts):.2f}")
        print(f"  中位数: {np.median(action_counts):.2f}")

    # 统计奖励分布
    rewards = [sample["reward"] for sample in dataset.data]
    if rewards:
        print(f"\n奖励分布:")
        print(f"  最小: {min(rewards):.2f}")
        print(f"  最大: {max(rewards):.2f}")
        print(f"  平均: {np.mean(rewards):.2f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def generate_expert_data(
    num_episodes: int = 100,
    save_dir: str = "data/expert_demonstrations",
) -> list[Path]:
    """生成专家数据的便捷函数。

    Args:
        num_episodes: 生成的对局数量。
        save_dir: 保存目录。

    Returns:
        保存的文件路径列表。
    """
    generator = ExpertDataGenerator(save_dir=save_dir)
    return generator.generate_dataset(num_episodes=num_episodes)


def load_expert_dataset(
    data_dir: str = "data/expert_demonstrations",
    extra_data_dirs: list[str] | None = None,
    max_samples: int | None = None,
) -> ExpertDataset:
    """加载专家数据集的便捷函数。

    Args:
        data_dir: 数据目录。
        max_samples: 最大样本数。

    Returns:
        数据集对象。
    """
    return ExpertDataset(
        data_dir=data_dir,
        extra_data_dirs=extra_data_dirs,
        max_samples=max_samples,
    )
