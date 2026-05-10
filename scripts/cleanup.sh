#!/bin/bash
# 清理脚本 - 清理不需要的文件和目录

echo "============================================================"
echo "Orbit Wars 项目清理工具"
echo "============================================================"
echo ""

# 函数：清理数据目录
clean_data() {
    echo "清理 data 目录..."
    if [ -d "data/test_multiprocess" ]; then
        rm -rf data/test_multiprocess
        echo "  ✓ 删除 data/test_multiprocess"
    fi
    if [ -d "data/test_jsonl" ]; then
        rm -rf data/test_jsonl
        echo "  ✓ 删除 data/test_jsonl"
    fi
    if [ -d "data/test_expert_demonstrations" ]; then
        rm -rf data/test_expert_demonstrations
        echo "  ✓ 删除 data/test_expert_demonstrations"
    fi
    echo ""
}

# 函数：清理检查点
clean_checkpoints() {
    echo "清理旧检查点..."
    if [ -d "training/checkpoints" ]; then
        echo "  检查点位于: training/checkpoints/"
        echo "  大小: $(du -sh training/checkpoints | cut -f1)"
        read -p "  是否清理? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            rm -rf training/checkpoints/*
            echo "  ✓ 已清理所有检查点"
        fi
    fi
    echo ""
}

# 函数：清理日志
clean_logs() {
    echo "清理日志..."
    if [ -d "swanlog" ]; then
        echo "  日志位于: swanlog/"
        echo "  大小: $(du -sh swanlog | cut -f1)"
        read -p "  是否清理? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            rm -rf swanlog/*
            echo "  ✓ 已清理所有日志"
        fi
    fi

    if [ -d "training/logs" ]; then
        echo "  训练日志位于: training/logs/"
        echo "  大小: $(du -sh training/logs | cut -f1)"
        read -p "  是否清理? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            rm -rf training/logs/*
            echo "  ✓ 已清理所有训练日志"
        fi
    fi
    echo ""
}

# 函数：清理 Python 缓存
clean_pycache() {
    echo "清理 Python 缓存..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    find . -type f -name "*.pyc" -delete 2>/dev/null
    echo "  ✓ 已清理所有 Python 缓存"
    echo ""
}

# 函数：显示磁盘使用
show_disk_usage() {
    echo "磁盘使用情况:"
    echo "  data/: $(du -sh data 2>/dev/null | cut -f1)"
    echo "  training/checkpoints/: $(du -sh training/checkpoints 2>/dev/null | cut -f1)"
    echo "  swanlog/: $(du -sh swanlog 2>/dev/null | cut -f1)"
    echo "  总计: $(du -sh . 2>/dev/null | cut -f1)"
    echo ""
}

# 主菜单
main() {
    while true; do
        echo "============================================================"
        echo "清理选项:"
        echo "============================================================"
        echo "1. 清理测试数据 (data/test_*)"
        echo "2. 清理检查点 (training/checkpoints)"
        echo "3. 清理日志 (swanlog, training/logs)"
        echo "4. 清理 Python 缓存"
        echo "5. 全部清理"
        echo "6. 显示磁盘使用"
        echo "0. 退出"
        echo ""

        read -p "请选择 (0-6): " choice

        case $choice in
            1)
                clean_data
                ;;
            2)
                clean_checkpoints
                ;;
            3)
                clean_logs
                ;;
            4)
                clean_pycache
                ;;
            5)
                clean_data
                clean_checkpoints
                clean_logs
                clean_pycache
                echo "✓ 全部清理完成"
                ;;
            6)
                show_disk_usage
                ;;
            0)
                echo "退出"
                break
                ;;
            *)
                echo "无效选择"
                ;;
        esac

        read -p "按 Enter 继续..."
        clear
    done
}

# 如果有参数，直接执行对应的清理
if [ $# -gt 0 ]; then
    case $1 in
        data)
            clean_data
            ;;
        checkpoints)
            clean_checkpoints
            ;;
        logs)
            clean_logs
            ;;
        pycache)
            clean_pycache
            ;;
        all)
            clean_data
            clean_checkpoints
            clean_logs
            clean_pycache
            ;;
        *)
            echo "用法: $0 [data|checkpoints|logs|pycache|all]"
            ;;
    esac
else
    main
fi
