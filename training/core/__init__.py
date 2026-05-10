"""Core training modules.

This module sets up the import path to allow relative imports like `from core.config`
to work from both the training directory and the project root.
"""

import sys
from pathlib import Path

# 获取项目根目录（training/ 的父目录）
training_dir = Path(__file__).parent.parent
project_root = training_dir.parent

# 确保 project_root 在 sys.path 中
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 确保 training 目录也在 sys.path 中（作为 core 的父目录）
if str(training_dir) not in sys.path:
    sys.path.insert(0, str(training_dir))
