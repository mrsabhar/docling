"""Shared helpers for the Intel CPU experiment scripts.

Only the standard library is used here, so that the helpers also work in a
plain interpreter (for example to verify the corpus before installation).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

EXP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXP_DIR.parent.parent
CONFIG_DIR = EXP_DIR / "config"
RESULTS_DIR = EXP_DIR / "results"
CORPUS_FILE = CONFIG_DIR / "corpus.json"

# Environment variables that change threading or device selection. They are
# recorded with every result so that runs on different machines can be compared.
THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OMP_WAIT_POLICY",
    "MKL_NUM_THREADS",
    "MKL_DYNAMIC",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "KMP_AFFINITY",
    "KMP_BLOCKTIME",
    "KMP_HW_SUBSET",
    "DNNL_MAX_CPU_ISA",
    "ONEDNN_MAX_CPU_ISA",
    "ORT_NUM_THREADS",
    "OV_CPU_THREADS_NUM",
    "TBB_NUM_THREADS",
    "TORCH_NUM_THREADS",
    "DOCLING_NUM_THREADS",
    "DOCLING_DEVICE",
    "DOCLING_PERF_PAGE_BATCH_SIZE",
    "DOCLING_PERF_DOC_BATCH_SIZE",
    "DOCLING_PERF_DOC_BATCH_CONCURRENCY",
    "CUDA_VISIBLE_DEVICES",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def thread_env() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in THREAD_ENV_VARS}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def git_info() -> dict[str, str | None]:
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    return {
        "head": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "upstream_main": _git("rev-parse", "upstream/main"),
        "merge_base_upstream_main": _git("merge-base", "HEAD", "upstream/main"),
        "dirty": _git("status", "--porcelain", "--untracked-files=no"),
    }


def machine_tag() -> str:
    """Short, filesystem-safe tag for the current machine."""
    node = platform.node().split(".")[0] or "host"
    return f"{platform.system().lower()}-{node}".replace(" ", "_")
