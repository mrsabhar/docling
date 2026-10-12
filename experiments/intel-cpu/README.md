# Intel CPU optimization experiments for Docling

Fork: `mrsabhar/docling`, branch `intel-cpu-exp`.
Upstream base: `docling-project/docling` `main` at
`ad2e0e1a4fd3d5b0e709f450befa64811bff14cb` (2026-10-12, "fix(jats): parse ordered
lists and list item labels (#4770)").

All experiment files are in this folder. Upstream files are not changed in
experiment A.

| Letter | Experiment | Status |
|---|---|---|
| A | Baseline | Harness ready. Measurement blocked: model hosts not reachable from the cloud VM. |
| B | OpenVINO for layout / table models | Plan only ([plan](reports/plan-B-E.md)) |
| C | OCR on OpenVINO | Plan only |
| D | CPU threading / concurrency tuning | Plan only |
| E | oneTBB prototype (optional) | Plan only |

## Layout

```text
experiments/intel-cpu/
  README.md                    this file
  run_bench.sh                 Linux runner (bash)
  run_bench.ps1                Windows runner (PowerShell)
  uvproject/pyproject.toml     experiment environment (editable docling + pinned runtimes)
  uvproject/uv.lock            lockfile for that environment
  config/corpus.json           corpus list with SHA256 and page counts
  scripts/
    common.py                  shared helpers (paths, hashes, thread env vars)
    env_report.py              OS / CPU / ISA / memory / runtimes / deps manifest
    corpus.py                  verify or record corpus hashes
    bench.py                   orchestrator: one fresh process per (doc, OCR mode)
    bench_worker.py            cold start + warmup + measured runs, RSS sampling
    summarize.py               summary.json / summary.csv / stages.csv / summary.md
    correctness.py             diff against repo ground truth or a reference run
  results/<experiment>/<machine label>/
  reports/
    A-baseline.md              baseline report
    plan-B-E.md                plan for experiments B-E
    panther-lake-run-guide.md  how to run on Panther Lake (Windows and Linux)
    cloud-validation-scope.md  what the cloud VM can and cannot validate
```

## Quick start

Linux:

```bash
experiments/intel-cpu/run_bench.sh --experiment A-baseline --label ptl-linux-ac
```

Windows (PowerShell):

```powershell
.\experiments\intel-cpu\run_bench.ps1 -Experiment A-baseline -Label ptl-win-ac
```

Both runners do `uv sync --frozen --project experiments/intel-cpu/uvproject`,
run the benchmark (OCR off, then OCR on), and run the correctness check.
See [the Panther Lake guide](reports/panther-lake-run-guide.md) for power and
pinning settings.

## Pinned runtime versions

Set in `uvproject/pyproject.toml` and locked in `uvproject/uv.lock`:

| Package | Version | Note |
|---|---|---|
| torch | 2.14.1 | Latest release. Upstream `uv.lock` pins 2.11.0 for Python 3.12 on Linux x86_64. |
| torchvision | 0.29.1 | Matches torch 2.14 |
| onnxruntime | 1.31.0 | Latest release. Requires Python >= 3.11. |
| Python | 3.12 | |
| docling-slim | editable checkout | extras `standard`, `feat-ocr-rapidocr-onnx` |
| all other shared packages | as upstream `uv.lock` | enforced with `constraint-dependencies` |

Note: the PyPI torch wheel for Linux x86_64 is the CUDA build (`+cu130`). It
runs on CPU without a GPU. The PyPI wheel for Windows is the CPU build. A
CPU-only Linux wheel from `download.pytorch.org` would cut install size; it is
not used, so that all machines resolve from the same index.
