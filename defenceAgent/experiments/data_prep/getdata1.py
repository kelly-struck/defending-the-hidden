from huggingface_hub import snapshot_download

# 仅下载仓库中的 images 目录
snapshot_download(
    repo_id="sinwang/SIUO",
    repo_type="dataset",
    allow_patterns=["images/**"],             # 只包含 images 文件夹及其子文件
    local_dir="D:/AgentScope/defenceAgent/statics/SIUO",  # 下载到此目录
    local_dir_use_symlinks=False              # 在 Windows 上避免符号链接
)