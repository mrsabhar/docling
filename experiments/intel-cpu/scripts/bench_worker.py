"""Benchmark one document in one configuration, in a fresh process.

The orchestrator (bench.py) starts one worker per (document, mode), so the
first conversion is a true cold start: Python import of docling, pipeline and
model initialization, and the first conversion.

Sequence inside the worker:
  1. cold:    import docling -> initialize_pipeline() -> first convert()
  2. warmup:  N conversions, not used for statistics (default 1)
  3. runs:    M measured conversions (default 5)

Peak RSS method: a background thread samples the process RSS every
--rss-interval-ms (psutil). The max per phase is reported. The OS high-water
mark (Linux ru_maxrss, Windows PeakWorkingSetSize) is reported for the whole
process as a cross-check; it can catch short peaks between samples.
"""

from __future__ import annotations

import time

_T0 = time.perf_counter()  # before any heavy import

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from common import REPO_ROOT, thread_env  # noqa: E402

try:
    import psutil
except ImportError:  # pragma: no cover - psutil ships with docling[standard]
    psutil = None


class RssSampler:
    """Track the maximum RSS of this process, per named phase."""

    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.phase = "startup"
        self.peaks: dict[str, int] = {}
        self._stop = threading.Event()
        self._proc = psutil.Process() if psutil is not None else None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        if self._proc is not None:
            self._thread.start()

    def set_phase(self, phase: str) -> None:
        self._sample()
        self.phase = phase
        self._sample()

    def _sample(self) -> None:
        if self._proc is None:
            return
        rss = self._proc.memory_info().rss
        if rss > self.peaks.get(self.phase, 0):
            self.peaks[self.phase] = rss

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def stop(self) -> None:
        self._sample()
        self._stop.set()


def os_peak_rss_bytes() -> int | None:
    if sys.platform == "win32":
        if psutil is None:
            return None
        return int(psutil.Process().memory_info().peak_wset)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    return int(peak if sys.platform == "darwin" else peak * 1024)


def build_converter(args: argparse.Namespace):
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        OcrAutoOptions,
        PdfPipelineOptions,
        RapidOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    accel_kwargs: dict[str, Any] = {"device": args.device}
    if args.num_threads is not None:
        accel_kwargs["num_threads"] = args.num_threads
    opts = PdfPipelineOptions(accelerator_options=AcceleratorOptions(**accel_kwargs))
    opts.do_ocr = args.ocr == "on"
    opts.do_table_structure = True
    opts.table_structure_options.do_cell_matching = True
    if args.ocr_engine == "rapidocr":
        opts.ocr_options = RapidOcrOptions(backend=args.ocr_backend)
    else:
        opts.ocr_options = OcrAutoOptions()
    if args.layout_engine != "transformers":
        opts.layout_options = build_layout_options(args)
    if args.layout_batch_size is not None:
        opts.layout_batch_size = args.layout_batch_size
    if args.table_batch_size is not None:
        opts.table_batch_size = args.table_batch_size
    if args.ocr_batch_size is not None:
        opts.ocr_batch_size = args.ocr_batch_size
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    return converter, opts


def build_layout_options(args: argparse.Namespace):
    """Heron layout on the ONNX export, run by ONNX Runtime or OpenVINO."""
    from docling.datamodel.object_detection_engine_options import (
        OnnxRuntimeObjectDetectionEngineOptions,
        OpenVinoObjectDetectionEngineOptions,
    )
    from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions

    if args.layout_engine == "onnxruntime":
        engine_options = OnnxRuntimeObjectDetectionEngineOptions()
    else:
        engine_options = OpenVinoObjectDetectionEngineOptions(
            performance_hint=args.ov_hint,
            num_streams=args.ov_streams,
            cache_dir=args.ov_cache_dir,
        )
    return LayoutObjectDetectionOptions.from_preset(
        "layout_heron_default", engine_options=engine_options
    )


def timings_of(conv_res) -> dict[str, dict[str, float]]:
    return {
        key: {
            "total_s": float(sum(item.times)),
            "count": int(item.count),
            "scope": str(item.scope.value),
        }
        for key, item in conv_res.timings.items()
    }


def convert_once(converter, source: Path, sampler: RssSampler, phase: str):
    sampler.set_phase(phase)
    start = time.perf_counter()
    conv_res = converter.convert(source, raises_on_error=False)
    wall = time.perf_counter() - start
    sampler.set_phase(f"{phase}:done")
    record = {
        "wall_s": wall,
        "status": str(conv_res.status.value),
        "pages": len(conv_res.pages),
        "timings": timings_of(conv_res),
        "rss_peak_bytes": sampler.peaks.get(phase),
        "errors": [str(e.error_message) for e in conv_res.errors],
    }
    return conv_res, record


def save_outputs(conv_res, out_dir: Path, stem: str) -> dict[str, str]:
    from docling_core.types.doc import ImageRefMode

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = conv_res.document
    md_path = out_dir / f"{stem}.md"
    gt_md_path = out_dir / f"{stem}.compact.md"
    json_path = out_dir / f"{stem}.json"
    md_path.write_text(doc.export_to_markdown(), encoding="utf-8")
    # Same markdown flavour as the repo ground truth (tests/verify_utils.py).
    gt_md_path.write_text(doc.export_to_markdown(compact_tables=True), encoding="utf-8")
    doc.save_as_json(json_path, image_mode=ImageRefMode.PLACEHOLDER)
    return {"md": md_path.name, "compact_md": gt_md_path.name, "json": json_path.name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--doc", type=Path, required=True)
    parser.add_argument("--ocr", choices=["on", "off"], required=True)
    parser.add_argument(
        "--ocr-engine", choices=["rapidocr", "auto"], default="rapidocr"
    )
    parser.add_argument(
        "--ocr-backend",
        choices=["onnxruntime", "openvino", "torch", "paddle"],
        default="onnxruntime",
    )
    parser.add_argument(
        "--layout-engine",
        choices=["transformers", "onnxruntime", "openvino"],
        default="transformers",
    )
    parser.add_argument(
        "--ov-hint",
        choices=["LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"],
        default="LATENCY",
        help="OpenVINO PERFORMANCE_HINT (layout engine openvino)",
    )
    parser.add_argument("--ov-streams", type=int, help="OpenVINO NUM_STREAMS")
    parser.add_argument(
        "--ov-cache-dir", type=Path, help="OpenVINO compiled-model cache folder"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--layout-batch-size", type=int)
    parser.add_argument("--table-batch-size", type=int)
    parser.add_argument("--ocr-batch-size", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--rss-interval-ms", type=float, default=10.0)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--outputs-dir", type=Path)
    args = parser.parse_args()

    sampler = RssSampler(args.rss_interval_ms / 1000.0)
    sampler.start()
    sampler.set_phase("import")

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.settings import settings

    settings.debug.profile_pipeline_timings = True
    t_import = time.perf_counter()

    sampler.set_phase("init")
    converter, opts = build_converter(args)
    converter.initialize_pipeline(InputFormat.PDF)
    t_init = time.perf_counter()

    source = args.doc if args.doc.is_absolute() else REPO_ROOT / args.doc
    conv_res, cold = convert_once(converter, source, sampler, "cold_convert")
    t_cold = time.perf_counter()
    cold.update(
        {
            "import_s": t_import - _T0,
            "init_pipeline_s": t_init - t_import,
            "first_convert_s": cold["wall_s"],
            "total_s": t_cold - _T0,
            "rss_peak_bytes": max(sampler.peaks.values(), default=None),
        }
    )

    warmups = [
        convert_once(converter, source, sampler, f"warmup{i}")[1]
        for i in range(args.warmup)
    ]
    runs = []
    for i in range(args.runs):
        conv_res, record = convert_once(converter, source, sampler, f"run{i}")
        runs.append(record)

    outputs = None
    if args.outputs_dir is not None:
        outputs = save_outputs(
            conv_res, args.outputs_dir, f"{args.doc_id}.ocr-{args.ocr}"
        )
    sampler.stop()

    import torch

    runtime_versions: dict[str, str | None] = {}
    for dist in ("torch", "onnxruntime", "openvino", "rapidocr"):
        try:
            from importlib.metadata import version

            runtime_versions[dist] = version(dist)
        except Exception:
            runtime_versions[dist] = None

    result = {
        "runtime_versions": runtime_versions,
        "doc_id": args.doc_id,
        "doc": str(args.doc),
        "ocr": args.ocr,
        "config": {
            "device": args.device,
            "num_threads_arg": args.num_threads,
            "num_threads_effective": opts.accelerator_options.num_threads,
            "ocr_engine": args.ocr_engine,
            "ocr_backend": args.ocr_backend if args.ocr_engine == "rapidocr" else None,
            "layout_engine": args.layout_engine,
            "ov_hint": args.ov_hint if args.layout_engine == "openvino" else None,
            "ov_streams": args.ov_streams,
            "ov_cache_dir": str(args.ov_cache_dir) if args.ov_cache_dir else None,
            "layout_batch_size": opts.layout_batch_size,
            "table_batch_size": opts.table_batch_size,
            "ocr_batch_size": opts.ocr_batch_size,
            "page_batch_size": settings.perf.page_batch_size,
            "warmup": args.warmup,
            "runs": args.runs,
            "rss_method": (
                f"psutil RSS sampled every {args.rss_interval_ms} ms; "
                "OS high-water: ru_maxrss (POSIX) / PeakWorkingSetSize (Windows)"
            ),
        },
        "thread_env": thread_env(),
        "torch_threads": {
            "intra_op": torch.get_num_threads(),
            "inter_op": torch.get_num_interop_threads(),
        },
        "platform": platform.platform(),
        "pid_cpu_affinity": (
            sorted(os.sched_getaffinity(0)) if sys.platform == "linux" else None
        ),
        "cold": cold,
        "warmup": warmups,
        "runs": runs,
        "rss_peak_by_phase_bytes": sampler.peaks,
        "process_peak_rss_os_bytes": os_peak_rss_bytes(),
        "outputs": outputs,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
