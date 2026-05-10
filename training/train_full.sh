#!/bin/bash
# 完整训练流程脚本：专家预训练 → 评估 → 强化学习（自动化）

echo "============================================================"
echo "Orbit Wars 完整训练流程（自动化）"
echo "============================================================"
echo ""

# 激活虚拟环境
source .venv/bin/activate

echo "训练流程："
echo "  1. 专家数据预训练（Ray 多GPU 分布式）"
echo "  2. 自动评估模型质量（不中断）"
echo "  3. 强化学习训练（Ray 多GPU PPO）"
echo ""

# 检查数据是否存在
if [ ! -d "data/expert_demonstrations" ] || [ -z "$(ls -A data/expert_demonstrations 2>/dev/null)" ]; then
    echo "⚠️  未找到专家数据！"
    echo ""
    echo "正在生成专家数据..."
    ./generate_expert_data.sh
    echo ""
fi

echo "✅ 专家数据就绪"
echo ""
echo "开始自动化训练（无交互）..."
echo ""

# 运行完整训练流程
python training/train_with_expert.py \
    --config training/config/expert_pretraining.yaml \
    --device cuda

echo ""
echo "============================================================"
echo "训练完成"
echo "============================================================"
echo ""
echo "模型保存在: training/checkpoints/"
echo "日志查看: SwanLab"
echo ""
