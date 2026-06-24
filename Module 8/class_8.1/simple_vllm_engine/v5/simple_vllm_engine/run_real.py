#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def _drop_package_dir_from_sys_path() -> None:
    # When this file is launched directly, Python puts this package directory on
    # sys.path. That makes local `requests.py` shadow the third-party `requests`
    # package used by Transformers/Hugging Face. We import through the version
    # root instead, so the package dir itself should not be a top-level path.
    package_dir = Path(__file__).resolve().parent

    def is_package_dir(entry: str) -> bool:
        path = Path.cwd() if entry == "" else Path(entry)
        return path.resolve() == package_dir

    sys.path[:] = [entry for entry in sys.path if not is_package_dir(entry)]


_drop_package_dir_from_sys_path()

_VERSION_ROOT = Path(__file__).resolve().parents[1]
_TEACHING_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_TEACHING_ROOT), str(_VERSION_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from common.run_real import VersionSettings, main

# This wrapper is intentionally tiny: the duplicated CLI/workload code
# lives in common.run_real, while this file shows only what changes in
# this teaching step.
SETTINGS = VersionSettings(
    version_name='v5_local_model_kernels',
    version_topic='packed projections, cached RoPE, fused local kernels, and slot-mapped prefill writes',
    attention_backend='flash_attn_paged',
)


if __name__ == "__main__":
    main(SETTINGS)
