"""灵活的多GPU多进程训练 — 可以自由指定每张卡的进程数。"""
from __future__ import annotations

import os
import sys
import time
import signal
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

def parse_gpu_config(config_str: str) -> Dict[int, int]:
    """解析GPU配置字符串。

    支持的格式：
    - "0:4,1:4,2:4,3:4" (GPU 0-4个进程, GPU 1-4个进程, etc.)
    - "0,1,2,3" (默认每张卡4个进程)
    - "0:2,1:6,2:4" (GPU 0-2个进程, GPU 1-6个进程, GPU 2-4个进程)
    """
    gpu_config = {}
    parts = config_str.split(',')

    for part in parts:
        part = part.strip()
        if ':' in part:
            # 格式: "gpu_id:processes"
            gpu_id, processes = part.split(':')
            gpu_config[int(gpu_id)] = int(processes)
        else:
            # 格式: "gpu_id" (使用默认值)
            gpu_config[int(part)] = 4  # 默认4个进程

    return gpu_config


def kill_existing_processes():
    """清理已有的训练进程。"""
    import psutil
    current_pid = os.getpid()
    killed = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['pid'] == current_pid:
                continue
            cmdline = proc.info['cmdline']
            if cmdline and any('train_ddp.py' in str(cmd) for cmd in cmdline):
                print(f"杀死已有进程: PID {proc.info['pid']}")
                proc.kill()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if killed > 0:
        print(f"已杀死 {killed} 个旧进程")
        time.sleep(2)


def start_worker_process(gpu_id: int, worker_id: int, config_path: str, output_file: str) -> subprocess.Popen:
    """启动一个worker进程。"""
    cmd = [
        sys.executable,
        "train_ddp.py",
        config_path,
    ]

    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # 打开输出文件
    output = open(output_file, 'w', buffering=1)

    # 启动进程
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=output,
        stderr=subprocess.STDOUT,
        text=True,
        errors='replace'
    )

    return process


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="灵活的多GPU多进程训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # GPU 0,1,2,3 各运行 4 个进程
  %(prog)s --gpu-config "0:4,1:4,2:4,3:4"

  # GPU 0 运行 2 个进程，GPU 1 运行 6 个进程，GPU 2 运行 4 个进程
  %(prog)s --gpu-config "0:2,1:6,2:4"

  # GPU 0,1,2 各运行 4 个进程（使用默认值）
  %(prog)s --gpu-config "0,1,2"

  # 只使用 GPU 0，运行 8 个进程
  %(prog)s --gpu-config "0:8"
        """
    )
    parser.add_argument(
        "--gpu-config",
        type=str,
        required=True,
        help="GPU配置，格式: 'gpu_id:processes,gpu_id:processes' 或 'gpu_id,gpu_id,...'"
    )
    parser.add_argument("--config", type=str, default="config/default.yaml", help="配置文件路径")
    parser.add_argument("--kill-existing", action="store_true", help="启动前杀死已有进程")
    parser.add_argument("--log-dir", type=str, default="logs/multiprocess", help="日志目录")
    args = parser.parse_args()

    # 解析GPU配置
    gpu_config = parse_gpu_config(args.gpu_config)
    config_path = args.config
    log_dir = Path(args.log_dir)

    # 创建日志目录
    log_dir.mkdir(parents=True, exist_ok=True)

    # 计算总进程数
    total_processes = sum(gpu_config.values())

    print(f"========================================")
    print(f"  多GPU多进程训练")
    print(f"========================================")
    print(f"GPU配置:")
    for gpu_id, num_procs in sorted(gpu_config.items()):
        print(f"  GPU {gpu_id}: {num_procs} 个进程")
    print(f"\n总进程数: {total_processes}")
    print(f"配置文件: {config_path}")
    print(f"日志目录: {log_dir}")
    print(f"========================================")
    print()

    # 清理已有进程
    if args.kill_existing:
        kill_existing_processes()

    # 启动多个worker进程
    processes: Dict[int, subprocess.Popen] = {}
    print(f"正在启动 {total_processes} 个worker进程...")

    worker_id = 0
    for gpu_id, num_procs in sorted(gpu_config.items()):
        for i in range(num_procs):
            log_file = log_dir / f"gpu{gpu_id}_worker{i}.log"
            print(f"启动 worker {worker_id+1}/{total_processes} (GPU {gpu_id}, 进程 {i+1}/{num_procs}) -> {log_file}")
            proc = start_worker_process(gpu_id, worker_id, config_path, str(log_file))
            processes[worker_id] = proc
            worker_id += 1
            time.sleep(1)  # 错开启动时间

    print(f"\n所有进程已启动！")
    print()
    print(f"监控命令：")
    print(f"  - GPU使用: watch -n 1 nvidia-smi")
    print(f"  - CPU使用: htop")
    print(f"  - 查看所有日志: tail -f {log_dir}/*.log")
    print(f"  - 查看特定GPU日志: tail -f {log_dir}/gpu0_worker0.log")
    print()
    print(f"实时统计：")
    print(f"  watch -n 5 'tail -n 20 {log_dir}/gpu0_worker0.log | grep Iter'")
    print()
    print(f"按 Ctrl+C 停止所有进程")
    print(f"========================================")

    # 信号处理
    def signal_handler(signum, frame):
        print(f"\n收到停止信号，正在清理 {len(processes)} 个进程...")
        for worker_id, proc in processes.items():
            try:
                proc.terminate()
            except:
                pass
        time.sleep(2)
        for worker_id, proc in processes.items():
            try:
                if proc.poll() is None:
                    proc.kill()
            except:
                pass
        print("所有进程已停止")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 监控循环
    try:
        while True:
            time.sleep(60)

            # 检查进程状态
            alive_count = sum(1 for proc in processes.values() if proc.poll() is None)
            if alive_count == 0:
                print("所有进程已退出")
                break

            # 打印统计
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 存活进程: {alive_count}/{len(processes)}")

    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
