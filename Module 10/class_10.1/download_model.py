from huggingface_hub import snapshot_download
from pathlib import Path
repo_id = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
# extract filename
filename = Path(repo_id).name
local_dir = "/home/ankur/dev/models/" + filename

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,

)