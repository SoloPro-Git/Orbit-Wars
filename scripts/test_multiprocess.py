"""测试多进程生成器（生成 10 局验证功能）。"""

import subprocess
import sys

def test_multiprocess_generator():
    """测试多进程生成器。"""
    print("=" * 60)
    print("测试多进程生成器（10 局）")
    print("=" * 60)
    print()

    # 运行测试
    cmd = [
        sys.executable, "-m", "training.expert.multiprocess_generator",
        "--num_episodes", "10",
        "--num_processes", "2",
        "--save_dir", "data/test_multiprocess",
        "--format", "jsonl",
        "--validate",
    ]

    print(f"运行命令: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, check=True)

    print()
    print("=" * 60)
    print("✓ 测试通过！")
    print("=" * 60)
    print()
    print("现在可以运行完整生成：")
    print("  ./generate_expert_data.sh")
    print()

if __name__ == "__main__":
    test_multiprocess_generator()
