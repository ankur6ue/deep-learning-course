#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def _prepare_imports() -> None:
    # Direct script execution puts this package directory on sys.path. Remove it
    # so local requests.py does not shadow third-party packages, then import this
    # version through its parent directory as simple_vllm_engine.
    package_dir = Path(__file__).resolve().parent
    version_root = package_dir.parent
    teaching_root = version_root.parent

    def is_package_dir(entry: str) -> bool:
        path = Path.cwd() if entry == "" else Path(entry)
        return path.resolve() == package_dir

    sys.path[:] = [entry for entry in sys.path if not is_package_dir(entry)]
    for path in (teaching_root, version_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_prepare_imports()

from common.synthetic_bench import main


if __name__ == "__main__":
    main("v3")
