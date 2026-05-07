#!/usr/bin/env python
"""测试DDP环境配置是否正确。"""
import torch
import torch.distributed as dist
import os


def test_ddp():
    """测试DDP环境。"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("未检测到DDP环境变量，请使用 torchrun 或单GPU模式")
        return False

    try:
        # 初始化进程组
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=rank,
            world_size=world_size
        )

        # 设置当前进程使用的GPU
        torch.cuda.set_device(local_rank)
        device = torch.cuda.current_device()

        print(f"[Rank {rank}/{world_size}] GPU {device}: {torch.cuda.get_device_name(device)}")
        print(f"[Rank {rank}/{world_size}] 显存: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f}GB")

        # 简单的all_reduce测试
        tensor = torch.ones(1).cuda()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"[Rank {rank}/{world_size}] All-reduce测试: tensor={tensor.item()} (应为{world_size})")

        # 测试通过
        if rank == 0:
            print(f"\n✓ DDP环境配置正确！{world_size}卡训练准备就绪。")

        dist.destroy_process_group()
        return True

    except Exception as e:
        print(f"[Rank {rank}] 错误: {e}")
        return False


if __name__ == "__main__":
    import sys
    success = test_ddp()
    sys.exit(0 if success else 1)
