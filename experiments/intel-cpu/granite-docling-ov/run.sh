#!/usr/bin/env bash
# Export Granite-Docling-258M to OpenVINO and compare it with PyTorch on CPU.
#
#   experiments/intel-cpu/granite-docling-ov/run.sh [--label NAME] [--weights fp32|int8]
#       [--threads N] [--pdf PATH] [--page N]
#
# Output: experiments/intel-cpu/results/G-granite-docling-ov/<label>/
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
cd "$REPO"

LABEL="$(hostname)"
WEIGHTS="fp32"
THREADS="$(nproc)"
PDF="tests/data/pdf/sources/2206.01062.pdf"
PAGE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --weights) WEIGHTS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --pdf) PDF="$2"; shift 2 ;;
    --page) PAGE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

uv sync --frozen --project "$HERE"
RUN=(uv run --frozen --no-sync --project "$HERE")

OUT="$HERE/../results/G-granite-docling-ov/$LABEL"
MODEL_DIR="${GD_OV_MODEL_DIR:-$HOME/.cache/docling-intel-cpu/granite-docling-258M-ov-$WEIGHTS}"
mkdir -p "$OUT"

if [[ ! -f "$MODEL_DIR/openvino_language_model.xml" ]]; then
  start=$(date +%s)
  "${RUN[@]}" optimum-cli export openvino --model ibm-granite/granite-docling-258M \
    --task image-text-to-text --weight-format "$WEIGHTS" "$MODEL_DIR"
  echo "export_s=$(( $(date +%s) - start ))" > "$OUT/export-$WEIGHTS.txt"
fi

"${RUN[@]}" python "$HERE/scripts/compare.py" "$PDF" "$PAGE" "$MODEL_DIR" \
  "$OUT/compare-$WEIGHTS-p$PAGE.json" "$THREADS"
echo "results: $OUT"
