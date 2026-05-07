"""训练配置验证和完整性检查。"""
import torch
from pathlib import Path
from typing import List, Tuple


class TrainingConfigValidator:
    """训练配置验证器，确保所有必需的依赖和配置都正确。"""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def check_pytorch_version(self) -> bool:
        """检查PyTorch版本。"""
        try:
            import torch
            version = torch.__version__
            major, minor = map(int, version.split('+')[0].split('.')[:2])

            if major < 2 or (major == 2 and minor < 5):
                self.errors.append(f"PyTorch版本过低: {version}，建议 >= 2.5.0")
                return False
            return True
        except ImportError:
            self.errors.append("PyTorch未安装")
            return False

    def check_cuda_availability(self) -> Tuple[bool, int]:
        """检查CUDA可用性和GPU数量。"""
        if not torch.cuda.is_available():
            self.warnings.append("CUDA不可用，将使用CPU训练（速度较慢）")
            return False, 0

        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            self.warnings.append("检测到CUDA但没有可用的GPU")
            return False, 0

        return True, gpu_count

    def check_dependencies(self) -> bool:
        """检查必需的依赖包。"""
        required_packages = [
            ("numpy", "numpy"),
            ("kaggle_environments", "kaggle-environments"),
            ("yaml", "pyyaml"),
            ("swanlab", "swanlab"),
        ]

        all_ok = True
        for module_name, package_name in required_packages:
            try:
                __import__(module_name)
            except ImportError:
                self.errors.append(f"缺少必需的包: {package_name}")
                all_ok = False

        return all_ok

    def check_config_file(self, config_path: str) -> bool:
        """检查配置文件是否存在。"""
        config_file = Path(config_path)
        if not config_file.exists():
            self.errors.append(f"配置文件不存在: {config_path}")
            return False
        return True

    def check_swanlab_key(self) -> bool:
        """检查SwanLab API key。"""
        key_file = Path(__file__).parent.parent / "config" / "swanlab_key.txt"
        if not key_file.exists():
            self.warnings.append("未找到SwanLab API key，将使用本地模式")
            return False

        key_content = key_file.read_text().strip()
        if len(key_content) < 10:
            self.warnings.append("SwanLab API key可能无效")
            return False

        return True

    def check_disk_space(self, required_gb: float = 10.0) -> bool:
        """检查磁盘空间。"""
        import shutil

        try:
            usage = shutil.disk_usage(".")
            free_gb = usage.free / (1024**3)

            if free_gb < required_gb:
                self.warnings.append(f"磁盘空间不足: {free_gb:.1f}GB 可用，建议至少 {required_gb}GB")
                return False
            return True
        except Exception:
            return True  # 无法检查时跳过

    def validate_all(self, config_path: str = "config/default.yaml") -> Tuple[bool, List[str], List[str]]:
        """执行所有检查。"""
        self.errors.clear()
        self.warnings.clear()

        # 检查PyTorch
        self.check_pytorch_version()

        # 检查CUDA
        cuda_ok, gpu_count = self.check_cuda_availability()
        if cuda_ok and gpu_count > 0:
            print(f"✓ 检测到 {gpu_count} 张GPU")

        # 检查依赖
        self.check_dependencies()

        # 检查配置文件
        self.check_config_file(config_path)

        # 检查SwanLab
        self.check_swanlab_key()

        # 检查磁盘空间
        self.check_disk_space()

        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings

    def print_validation_report(self):
        """打印验证报告。"""
        is_valid, errors, warnings = self.validate_all()

        print("\n" + "="*50)
        print("训练环境检查报告")
        print("="*50)

        if is_valid and len(warnings) == 0:
            print("✓ 所有检查通过，训练环境准备就绪！")
        else:
            if errors:
                print("\n❌ 错误 (必须修复):")
                for error in errors:
                    print(f"  - {error}")

            if warnings:
                print("\n⚠️  警告 (建议处理):")
                for warning in warnings:
                    print(f"  - {warning}")

        print("="*50 + "\n")

        return is_valid


def validate_training_environment(config_path: str = "config/default.yaml") -> bool:
    """验证训练环境的便捷函数。"""
    validator = TrainingConfigValidator()
    return validator.print_validation_report()
