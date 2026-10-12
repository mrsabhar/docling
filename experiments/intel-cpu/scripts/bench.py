"""Run the Intel CPU benchmark over the corpus.

One fresh worker process per (document, OCR mode). Results go to
experiments/intel-cpu/results/<experiment>/<label>/:

  env.json, deps.txt, corpus.json   environment, dependency and corpus manifests
  raw/<doc>.ocr-<on|off>.json       per-worker raw measurements
  outputs/<doc>.ocr-<on|off>.*      markdown + JSON of the last measured run
  summary.json, summary.csv,        aggregated statistics (see summarize.py)
  stages.csv, summary.md

Example:
  python experiments/intel-cpu/scripts/bench.py --experiment A-baseline \
      --label cloud-functional
Extra arguments after "--" go to every worker, for example:
  ... -- --num-threads 8 --ocr-backend openvino
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from common import CORPUS_FILE, RESULTS_DIR, load_json, machine_tag

SCRIPTS = Path(__file__).resolve().parent


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--experiment", default="A-baseline")
    parser.add_argument("--label", default=None, help="default: <os>-<hostname>")
    parser.add_argument(
        "--ocr", nargs="+", choices=["on", "off"], default=["off", "on"]
    )
    parser.add_argument("--docs", nargs="*", help="subset of corpus ids")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--no-prefetch",
        action="store_true",
        help="skip the untimed model prefetch run",
    )
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra = [a for a in args.worker_args if a != "--"]

    label = args.label or machine_tag()
    out = RESULTS_DIR / args.experiment / label
    raw = out / "raw"
    outputs = out / "outputs"
    raw.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    if run([py, str(SCRIPTS / "env_report.py"), "--out", str(out)]) != 0:
        return 1
    if run([py, str(SCRIPTS / "corpus.py"), "--out", str(out)]) != 0:
        print("corpus hash check failed; stop", file=sys.stderr)
        return 1

    docs = load_json(CORPUS_FILE)["documents"]
    if args.docs:
        docs = [d for d in docs if d["id"] in set(args.docs)]

    if not args.no_prefetch:
        # Untimed: makes sure the default models are in the local cache, so a
        # download never ends up inside a cold-start measurement.
        smallest = min(docs, key=lambda d: d.get("pages", 1))
        cmd = [
            py, str(SCRIPTS / "bench_worker.py"),
            "--doc-id", "prefetch", "--doc", smallest["path"], "--ocr", "on",
            "--warmup", "0", "--runs", "0",
            "--out-json", str(raw / "_prefetch.json"), *extra,
        ]  # fmt: skip
        if run(cmd) != 0:
            print("prefetch failed (model download blocked?); stop", file=sys.stderr)
            return 1

    failures = 0
    started = time.time()
    for mode in args.ocr:
        for doc in docs:
            cmd = [
                py, str(SCRIPTS / "bench_worker.py"),
                "--doc-id", doc["id"], "--doc", doc["path"], "--ocr", mode,
                "--warmup", str(args.warmup), "--runs", str(args.runs),
                "--out-json", str(raw / f"{doc['id']}.ocr-{mode}.json"),
                "--outputs-dir", str(outputs), *extra,
            ]  # fmt: skip
            if run(cmd) != 0:
                failures += 1
    print(f"benchmark wall time: {time.time() - started:.1f} s, failures: {failures}")

    run([py, str(SCRIPTS / "summarize.py"), str(out)])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
