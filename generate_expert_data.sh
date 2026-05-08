#!/bin/bash
# 快捷脚本：生成 1000 局专家演示数据
# 这个脚本位于根目录，方便调用

echo "============================================================"
echo "Orbit Wars 专家数据生成 (1000 局)"
echo "============================================================"
echo ""

# 激活虚拟环境
source .venv/bin/activate

# 调用 expert 目录下的脚本
bash training/expert/generate_1000_episodes.sh
