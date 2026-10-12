"""Collect a machine and dependency report for the Intel CPU experiments.

Works on Linux and Windows. Writes env.json and deps.txt into --out.

    python experiments/intel-cpu/scripts/env_report.py --out <results dir>
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import EXP_DIR, REPO_ROOT, git_info, sha256_file, thread_env, write_json

ISA_FLAGS = [
    "avx",
    "avx2",
    "fma",
    "f16c",
    "avx_vnni",
    "avx_vnni_int8",
    "avx_ifma",
    "avx_ne_convert",
    "avx512f",
    "avx512bw",
    "avx512vl",
    "avx512dq",
    "avx512_vnni",
    "avx512_bf16",
    "avx512_fp16",
    "amx_tile",
    "amx_int8",
    "amx_bf16",
    "amx_fp16",
]


def _run(cmd: list[str]) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def cpu_info_linux() -> dict[str, Any]:
    info: dict[str, Any] = {}
    cpuinfo = _read(Path("/proc/cpuinfo")) or ""
    model = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)
    flags = re.search(r"^flags\s*:\s*(.+)$", cpuinfo, re.M)
    info["model_name"] = model.group(1) if model else None
    for key in ("cpu family", "model", "stepping"):
        match = re.search(rf"^{key}\s*:\s*(.+)$", cpuinfo, re.M)
        info[key.replace(" ", "_")] = match.group(1) if match else None
    flag_set = set(flags.group(1).split()) if flags else set()
    info["isa_flags_os"] = {flag: flag in flag_set for flag in ISA_FLAGS}
    info["hypervisor"] = "hypervisor" in flag_set
    info["lscpu"] = _run(["lscpu"])
    # Hybrid Intel parts expose one PMU per core type.
    hybrid: dict[str, str | None] = {}
    for core_type in ("cpu_core", "cpu_atom"):
        cpus = _read(Path(f"/sys/devices/{core_type}/cpus"))
        if cpus is not None:
            hybrid[core_type] = cpus
    info["hybrid_core_types"] = hybrid or None
    cpu0 = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    info["scaling_governor"] = _read(cpu0 / "scaling_governor")
    info["energy_performance_preference"] = _read(
        cpu0 / "energy_performance_preference"
    )
    info["scaling_driver"] = _read(cpu0 / "scaling_driver")
    info["cgroup_cpu_max"] = _read(Path("/sys/fs/cgroup/cpu.max"))
    info["cgroup_memory_max"] = _read(Path("/sys/fs/cgroup/memory.max"))
    try:
        info["affinity"] = sorted(os.sched_getaffinity(0))
    except AttributeError:
        info["affinity"] = None
    return info


def cpu_info_windows() -> dict[str, Any]:
    ps = (
        "Get-CimInstance Win32_Processor | Select-Object Name,Manufacturer,"
        "NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,L2CacheSize,"
        "L3CacheSize | ConvertTo-Json"
    )
    return {
        "model_name": platform.processor(),
        "win32_processor": _run(["powershell", "-NoProfile", "-Command", ps]),
        "power_scheme": _run(["powercfg", "/getactivescheme"]),
        "os_build": platform.version(),
    }


def numpy_cpu_features() -> dict[str, bool] | None:
    try:
        from numpy._core._multiarray_umath import __cpu_features__  # type: ignore
    except ImportError:
        try:
            from numpy.core._multiarray_umath import __cpu_features__  # type: ignore
        except ImportError:
            return None
    return {k: bool(v) for k, v in __cpu_features__.items()}


def runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "mkldnn_available": torch.backends.mkldnn.is_available(),
            "parallel_info": torch.__config__.parallel_info(),
            "cuda_build": torch.version.cuda,
            "xpu_available": torch.xpu.is_available(),
        }
    except ImportError:
        info["torch"] = None
    try:
        import onnxruntime as ort

        info["onnxruntime"] = {
            "version": ort.__version__,
            "providers": ort.get_available_providers(),
        }
    except ImportError:
        info["onnxruntime"] = None
    try:
        import openvino as ov

        core = ov.Core()
        info["openvino"] = {
            "version": ov.get_version(),
            "devices": core.available_devices,
        }
    except ImportError:
        info["openvino"] = None
    try:
        from docling.datamodel.accelerator_options import AcceleratorOptions

        opts = AcceleratorOptions()
        info["docling_accelerator_defaults"] = {
            "num_threads": opts.num_threads,
            "device": str(opts.device),
        }
    except ImportError:
        info["docling_accelerator_defaults"] = None
    return info


def memory_info() -> dict[str, Any]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "total_bytes": vm.total,
            "available_bytes": vm.available,
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cpus": psutil.cpu_count(logical=True),
        }
    except ImportError:
        return {"logical_cpus": os.cpu_count()}


def compiler_info() -> dict[str, str | None]:
    gcc = _run(["gcc", "--version"])
    clang = _run(["clang", "--version"])
    return {
        "python_compiler": platform.python_compiler(),
        "gcc": gcc.splitlines()[0] if gcc else None,
        "clang": clang.splitlines()[0] if clang else None,
        "msvc_cl": "present" if shutil.which("cl") else None,
    }


def deps_manifest() -> list[str]:
    names = {
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib.metadata.distributions()
    }
    return sorted(names, key=str.lower)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    system = platform.system()
    cpu = cpu_info_linux() if system == "Linux" else cpu_info_windows()
    lock = EXP_DIR / "uvproject" / "uv.lock"
    report = {
        "os": {
            "system": system,
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "os_release": _read(Path("/etc/os-release")),
        },
        "cpu": cpu,
        "cpu_features_numpy": numpy_cpu_features(),
        "memory": memory_info(),
        "python": {"version": sys.version, "executable": sys.executable},
        "compiler": compiler_info(),
        "runtimes": runtime_info(),
        "thread_env": thread_env(),
        "git": git_info(),
        "lockfile": {
            "path": str(lock.relative_to(REPO_ROOT)),
            "sha256": sha256_file(lock) if lock.exists() else None,
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "env.json", report)
    (args.out / "deps.txt").write_text(
        "\n".join(deps_manifest()) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out / 'env.json'} and {args.out / 'deps.txt'}")


if __name__ == "__main__":
    main()
