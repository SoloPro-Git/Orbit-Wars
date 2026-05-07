#!/bin/bash
# 实时监控训练进度

echo "=========================================="
echo "  Orbit Wars 训练实时监控"
echo "=========================================="
echo ""

# 检查训练是否在运行
if ! pgrep -f "train_ddp.py" > /dev/null; then
    echo "训练未运行"
    echo "请先启动训练: ./train_8gpu.sh"
    exit 1
fi

echo "✓ 训练正在运行"
echo ""

# 获取最新的swanlab日志目录
LATEST_LOG=$(ls -td swanlog/run-* 2>/dev/null | head -1)
if [ -z "$LATEST_LOG" ]; then
    echo "等待SwanLab日志..."
    sleep 3
    LATEST_LOG=$(ls -td swanlog/run-* 2>/dev/null | head -1)
fi

if [ -n "$LATEST_LOG" ]; then
    echo "📊 监控数据目录: $LATEST_LOG"
    echo ""
    echo "📈 实时指标（每5秒更新）:"
    echo "----------------------------------------"
    watch -n 5 -c "
        echo '训练进度监控 - ' $(date '+%H:%M:%S')
        echo '=========================================='

        # 显示GPU使用情况
        echo '🎮 GPU状态:'
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | awk -F',' '{printf \"  GPU %s: %s%% 使用率, %s/%s MB\\n\", \$1, \$3, \$4, \$5}'
        echo ''

        # 显示最新的训练指标
        if [ -f '$LATEST_LOG/logs.txt' ]; then
            echo '📊 最新训练指标:'
            tail -20 '$LATEST_LOG/logs.txt' 2>/dev/null | grep -E '(mean_reward|eval_win_rate|iteration|buffer_size)' | tail -10 | while read line; do
                echo \"  \$line\"
            done
        fi

        echo '=========================================='
        echo '按 Ctrl+C 退出监控'
    "
else
    echo "等待日志生成..."
    echo "请稍后再试"
fi