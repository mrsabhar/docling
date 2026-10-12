"""Aggregate raw worker results into summary.json / summary.csv / stages.csv / summary.md.

    python experiments/intel-cpu/scripts/summarize.py <results dir>

Statistics:
  - latency: median and p95 (numpy linear interpolation) over the measured runs.
    With 5 runs, p95 sits between the 4th and 5th sorted value.
  - pages/sec: pages / median latency.
  - stage time: per run, the sum of all recorded intervals for a stage key
    (docling.utils.profiling); median over the runs.

The standard PDF pipeline runs its stages in parallel threads. Stage times
therefore overlap in wall-clock time, and their sum can exceed the latency.
The "share" column is each stage's part of the summed stage time; it ranks
where compute goes, not how much latency a stage adds on its own.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from common import load_json, write_json

# docling timing key -> report stage name
STAGES = [
    ("page_parse", "parse"),
    ("layout", "layout"),
    ("layout_postprocess", "layout_postprocess"),
    ("ocr", "ocr"),
    ("table_structure", "table"),
    ("page_assemble", "page_assemble"),
    ("reading_order", "reading_order"),
    ("doc_assemble", "doc_assemble"),
]
# doc_assemble contains reading_order, so reading_order is not summed again.
SUMMED = [k for k, _ in STAGES if k != "reading_order"]


def p95(values: list[float]) -> float:
    return float(np.percentile(values, 95)) if values else float("nan")


def stage_median(runs: list[dict[str, Any]], key: str) -> float:
    vals = [r["timings"].get(key, {}).get("total_s", 0.0) for r in runs]
    return float(median(vals)) if vals else 0.0


def summarize_one(res: dict[str, Any]) -> dict[str, Any]:
    runs = [r for r in res["runs"] if r["status"] in ("success", "partial_success")]
    walls = [r["wall_s"] for r in runs]
    pages = runs[0]["pages"] if runs else res["cold"]["pages"]
    med = float(median(walls)) if walls else float("nan")
    stages = {name: stage_median(runs, key) for key, name in STAGES}
    pipeline_total = stage_median(runs, "pipeline_total")
    run_rss = [r["rss_peak_bytes"] for r in res["runs"] if r["rss_peak_bytes"]]
    return {
        "doc_id": res["doc_id"],
        "ocr": res["ocr"],
        "pages": pages,
        "status": sorted({r["status"] for r in res["runs"]}),
        "cold_total_s": res["cold"]["total_s"],
        "cold_import_s": res["cold"]["import_s"],
        "cold_init_pipeline_s": res["cold"]["init_pipeline_s"],
        "cold_first_convert_s": res["cold"]["first_convert_s"],
        "warm_runs": len(walls),
        "warm_median_s": med,
        "warm_p95_s": p95(walls),
        "warm_min_s": min(walls) if walls else float("nan"),
        "warm_max_s": max(walls) if walls else float("nan"),
        "pages_per_s": pages / med if walls and med > 0 else float("nan"),
        "rss_peak_cold_mib": (res["cold"]["rss_peak_bytes"] or 0) / 2**20,
        "rss_peak_warm_mib": max(run_rss) / 2**20 if run_rss else float("nan"),
        "rss_peak_os_mib": (res["process_peak_rss_os_bytes"] or 0) / 2**20,
        "pipeline_total_s": pipeline_total,
        "stages_s": stages,
        "stage_sum_s": sum(stages[name] for key, name in STAGES if key in SUMMED),
        "num_threads": res["config"]["num_threads_effective"],
        "torch_threads": res["torch_threads"],
        "thread_env": {k: v for k, v in res["thread_env"].items() if v is not None},
    }


def corpus_rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, Any] = {}
    for mode in sorted({r["ocr"] for r in rows}):
        sel = [r for r in rows if r["ocr"] == mode]
        totals = {name: sum(r["stages_s"][name] for r in sel) for _, name in STAGES}
        summed = sum(totals[name] for key, name in STAGES if key in SUMMED)
        pages = sum(r["pages"] for r in sel)
        wall = sum(r["warm_median_s"] for r in sel)
        ranking = sorted(
            (
                (name, t)
                for key, name in STAGES
                if key in SUMMED
                for t in [totals[name]]
            ),
            key=lambda x: -x[1],
        )
        by_mode[mode] = {
            "documents": len(sel),
            "pages": pages,
            "sum_warm_median_s": wall,
            "corpus_pages_per_s": pages / wall if wall else float("nan"),
            "stage_totals_s": totals,
            "stage_share": {n: (t / summed if summed else 0.0) for n, t in ranking},
            "bottleneck_ranking": [n for n, _ in ranking],
        }
    return by_mode


def write_md(path: Path, rows: list[dict[str, Any]], rollup: dict[str, Any]) -> None:
    lines = [
        "# Benchmark summary",
        "",
        "| doc | ocr | pages | cold s | warm median s | warm p95 s | pages/s "
        "| peak RSS warm MiB | peak RSS OS MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['doc_id']} | {r['ocr']} | {r['pages']} | {r['cold_total_s']:.2f} "
            f"| {r['warm_median_s']:.2f} | {r['warm_p95_s']:.2f} | {r['pages_per_s']:.2f} "
            f"| {r['rss_peak_warm_mib']:.0f} | {r['rss_peak_os_mib']:.0f} |"
        )
    names = [name for _, name in STAGES]
    lines += [
        "",
        "## Per-stage median time (s, overlapping threads)",
        "",
        "| doc | ocr | " + " | ".join(names) + " | pipeline_total | wall |",
        "|---|---|" + "---:|" * (len(names) + 2),
    ]
    for r in rows:
        cells = " | ".join(f"{r['stages_s'][n]:.2f}" for n in names)
        lines.append(
            f"| {r['doc_id']} | {r['ocr']} | {cells} | {r['pipeline_total_s']:.2f} "
            f"| {r['warm_median_s']:.2f} |"
        )
    lines += ["", "## Corpus rollup", ""]
    for mode, data in rollup.items():
        share = ", ".join(f"{n} {s:.0%}" for n, s in data["stage_share"].items())
        lines.append(
            f"- OCR {mode}: {data['pages']} pages, {data['corpus_pages_per_s']:.2f} "
            f"pages/s (sum of warm medians {data['sum_warm_median_s']:.1f} s). "
            f"Stage share: {share}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out = Path(sys.argv[1])
    raw_files = sorted(
        p for p in (out / "raw").glob("*.json") if not p.name.startswith("_")
    )
    rows = [summarize_one(load_json(p)) for p in raw_files]
    rows.sort(key=lambda r: (r["ocr"], r["doc_id"]))
    rollup = corpus_rollup(rows)
    write_json(out / "summary.json", {"documents": rows, "corpus": rollup})

    flat_keys = (
        [
            k
            for k in rows[0]
            if k not in ("stages_s", "torch_threads", "thread_env", "status")
        ]
        if rows
        else []
    )
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(flat_keys)
        for r in rows:
            writer.writerow([r[k] for k in flat_keys])
    with (out / "stages.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["doc_id", "ocr", "stage", "median_s"])
        for r in rows:
            for name, val in r["stages_s"].items():
                writer.writerow([r["doc_id"], r["ocr"], name, f"{val:.6f}"])
    write_md(out / "summary.md", rows, rollup)
    print(f"wrote summary for {len(rows)} results to {out}")


if __name__ == "__main__":
    main()
