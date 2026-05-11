from huggingface_hub import snapshot_download
from pathlib import Path
repo_id = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
repo_id = "openai/gpt-oss-20b"
repo_id = "nvidia/Gemma-4-31B-IT-NVFP4"
repo_id = "RedHatAI/gemma-4-26B-A4B-it-NVFP4"
# extract filename
filename = Path(repo_id).name
local_dir = "/home/ankur/dev/models/" + filename


snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    ignore_patterns=["metal/**", "original/**"],
)