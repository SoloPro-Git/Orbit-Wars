#!/bin/bash
# 生成 1000 局专家演示数据
# 使用 Kaggle 高分策略生成训练数据

echo "============================================================"
echo "Orbit Wars 专家数据生成 (1000 局)"
echo "============================================================"
echo ""

# 激活虚拟环境
source ../../.venv/bin/activate

# 参数设置
NUM_EPISODES=1000          # 总对局数
NUM_PROCESSES=8            # 并行进程数（建议 CPU 核心数）
EPISODES_PER_FILE=20       # 每个文件保存的对局数
SAVE_DIR="data/expert_demonstrations"  # 保存目录
FORMAT="jsonl"             # 数据格式

echo "配置信息:"
echo "  总对局数: ${NUM_EPISODES}"
echo "  并行进程数: ${NUM_PROCESSES}"
echo "  每文件对局数: ${EPISODES_PER_FILE}"
echo "  保存目录: ${SAVE_DIR}"
echo "  数据格式: ${FORMAT}"
echo ""

# 估算时间和空间
EST_TIME_MIN=$((NUM_EPISODES * 6 / 60 / NUM_PROCESSES))
EST_SIZE_GB=$((NUM_EPISODES * 10 / 1024))
echo "预计资源消耗:"
echo "  时间: ~${EST_TIME_MIN} 分钟"
echo "  磁盘: ~${EST_SIZE_GB} GB"
echo ""

echo "开始生成（自动运行，无需确认）..."
echo ""

# 运行多进程生成脚本
python -m training.expert.multiprocess_generator \
    --num_episodes ${NUM_EPISODES} \
    --num_processes ${NUM_PROCESSES} \
    --episodes_per_file ${EPISODES_PER_FILE} \
    --save_dir ${SAVE_DIR} \
    --format ${FORMAT} \
    --validate

echo ""
echo "============================================================"
echo "✓ 数据生成完成！"
echo "============================================================"
echo ""
echo "数据位置: ${SAVE_DIR}/"
echo "文件数量: $(ls ${SAVE_DIR}/*.jsonl 2>/dev/null | wc -l)"
echo "总大小: $(du -sh ${SAVE_DIR} | cut -f1)"
echo ""
echo "使用方法:"
echo "  from training.expert import load_expert_dataset"
echo "  dataset = load_expert_dataset('${SAVE_DIR}')"
echo "  print(f'总样本数: {len(dataset)}')"
echo ""
