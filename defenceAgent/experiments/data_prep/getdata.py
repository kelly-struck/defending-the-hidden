# 下载 Hugging Face 上的 MME 数据集到指定目录（Windows）
# 目标目录：D:\AgentScope\defenceAgent\statics\MME

import os
import sys
import subprocess
from pathlib import Path

# 使用镜像源（需在导入 huggingface_hub 之前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 禁用 hf_transfer（在 Windows 上更稳定，避免文件锁导致的删除失败）
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

def ensure_pkg(pkg: str):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# 确保依赖存在
ensure_pkg("huggingface_hub")

from huggingface_hub import snapshot_download

def main():
    repo_id = "sinwang/SIUO"  # 数据集仓库
    dest_dir = Path(r"D:\AgentScope\defenceAgent\statics\SIUO")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 首次下载尝试（禁用 hf_transfer，单线程，更稳）
    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest_dir),
            revision="main",
            resume_download=True,  # 断点续传，避免重复下载
            max_workers=1,         # Windows 下避免并发引起的文件占用冲突
        )
    except RuntimeError as e:
        # 如果环境中仍然触发了 hf_transfer 的相关报错，降级重试一次
        if "hf_transfer" in str(e):
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
            local_path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(dest_dir),
                revision="main",
                resume_download=True,
                max_workers=1,
            )
        else:
            raise

    # 打印结果与简单统计
    file_count = sum(1 for _ in Path(local_path).rglob("*") if _.is_file())
    print(f"当前下载源: {os.environ.get('HF_ENDPOINT')}")
    print(f"HF_HUB_ENABLE_HF_TRANSFER: {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}")
    print(f"已下载到: {local_path}")
    print(f"文件总数: {file_count}")

if __name__ == "__main__":
    main()