#!/usr/bin/env python3
"""
Multi-GPU training manager for Orbit Wars PPO.
Runs 1v1 and 4-player training in parallel on different GPUs.
"""
import os
import sys
import torch
import json
import subprocess
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def get_gpu_count():
    """Get number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1

def launch_training(gpu_id, mode, exp_name, bc_checkpoint=None):
    """Launch a training process on a specific GPU."""
    cmd = [
        "python3", "-u", "rl/train.py",
        "--total-timesteps", "2000000",
        "--rollout-steps", "4096",
        "--eval-freq", "5",
        "--eval-games", "50",
        "--embed-dim", "64",
        "--opponents", "starter", "random", "aggressive",
        "--device", f"cuda:{gpu_id}",
        "--exp-name", exp_name,
        "--mode", mode,
    ]
    
    if bc_checkpoint and os.path.exists(bc_checkpoint):
        cmd.extend(["--bc-checkpoint", bc_checkpoint])
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    log_file = f"ppo_{mode}.log"
    print(f"Launching {mode} training on GPU {gpu_id} -> {log_file}")
    
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd="/root/lance/orbit-wars",
            env=env
        )
    
    return proc, log_file

def monitor_processes(processes, log_files):
    """Monitor training processes and print status."""
    while any(p.poll() is None for p in processes.values()):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training Status:")
        for mode, proc in processes.items():
            if proc.poll() is None:
                log = log_files[mode]
                lines = subprocess.check_output(
                    ["tail", "-1", f"/root/lance/orbit-wars/{log}"],
                    text=True
                ).strip()
                print(f"  {mode:12s}: PID {proc.pid:6d} | {lines[:80]}")
            else:
                print(f"  {mode:12s}: Completed (exit code {proc.returncode})")
        
        time.sleep(30)
    
    print("\nAll training processes completed!")

def main():
    # Kill existing training
    subprocess.run(["pkill", "-f", "rl/train.py"], check=False)
    time.sleep(2)
    
    # Setup SwanLab
    os.environ["SWANLAB_API_KEY"] = "p35ZRsu65XMhF5rrFw0yw"
    
    gpu_count = get_gpu_count()
    print(f"Available GPUs: {gpu_count}")
    
    # Assign GPUs
    gpu_1v1 = 0
    gpu_4p = 1 if gpu_count > 1 else 0
    
    # BC checkpoints
    bc_1v1 = "/root/lance/orbit-wars/checkpoints/bc_1v1_v2_best.pt"
    bc_4p = "/root/lance/orbit-wars/checkpoints/bc_4p_v2_best.pt"
    
    # Launch training
    processes = {}
    log_files = {}
    
    # 1v1 training
    proc_1v1, log_1v1 = launch_training(
        gpu_1v1, "1v1", "ppo_1v1_lance",
        bc_checkpoint=bc_1v1
    )
    processes["1v1"] = proc_1v1
    log_files["1v1"] = log_1v1
    
    # 4-player training (if enough GPUs or use same GPU)
    if gpu_count > 1 or True:  # Always launch for now
        proc_4p, log_4p = launch_training(
            gpu_4p, "4p", "ppo_4p_lance",
            bc_checkpoint=bc_4p if os.path.exists(bc_4p) else None
        )
        processes["4p"] = proc_4p
        log_files["4p"] = log_4p
    
    # Monitor
    monitor_processes(processes, log_files)

if __name__ == "__main__":
    main()
