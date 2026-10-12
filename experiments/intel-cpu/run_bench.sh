#!/usr/bin/env bash
# Run the Intel CPU benchmark + correctness check (Linux / macOS / WSL).
#
#   experiments/intel-cpu/run_bench.sh [--experiment A-baseline] [--label NAME] \
#       [--ocr off on] [--docs ID ...] [-- worker args]
#
# Thread variables (OMP_NUM_THREADS, DOCLING_NUM_THREADS, ...) are inherited
# from the caller and recorded in every result file.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PROJECT="$HERE/uvproject"
cd "$REPO"

uv sync --frozen --project "$PROJECT"
PY=(uv run --frozen --no-sync --project "$PROJECT" python)

EXPERIMENT="A-baseline"
LABEL=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in
    --experiment) EXPERIMENT="${args[$((i + 1))]}" ;;
    --label) LABEL="${args[$((i + 1))]}" ;;
    --) break ;;
  esac
done

"${PY[@]}" "$HERE/scripts/bench.py" "$@"

if [[ -z "$LABEL" ]]; then
  LABEL="$("${PY[@]}" -c "import sys; sys.path.insert(0, '$HERE/scripts'); from common import machine_tag; print(machine_tag())")"
fi
RESULTS="$HERE/results/$EXPERIMENT/$LABEL"
"${PY[@]}" "$HERE/scripts/correctness.py" groundtruth "$RESULTS" --repo-check
echo "results: $RESULTS"
