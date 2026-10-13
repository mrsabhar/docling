# Panther Lake (PTL) manual test guide: baseline and OpenVINO

Kit: `docling-ptl-kit.zip`. Everything is manual: copy each command, run it,
check the expected result, then go to the next step. Every step has a Linux
(bash) and a Windows (PowerShell) form.

| Part | What | Docling code |
|---|---|---|
| 1 | Set up upstream Docling and the kit | upstream `main` at `ad2e0e1` |
| 2 | Baseline quality: Docling's own e2e test | unchanged upstream |
| 3 | Baseline benchmark | unchanged upstream |
| 4 | Apply the OpenVINO patch and verify accuracy | upstream + patch |
| 5 | OpenVINO benchmark | upstream + patch |
| 6 | Optional: Granite-Docling on OpenVINO | separate environment |
| 7 | Send the results back | - |
| 8 | Make the patch yourself (reference) | - |

Kit content:

```text
docling-ptl-kit/
  GUIDE.md                                    this file
  patches/0001-openvino-layout-engine.patch   Docling change + unit tests
  experiments/intel-cpu/                      benchmark harness (copied into the clone)
    run_bench.sh, run_bench.ps1               runners
    uvproject/                                pinned environment (uv.lock)
    scripts/                                  bench, correctness, compare_layout_engines
    tests/test_e2e_layout_engine.py           runs tests/test_e2e_conversion.py per layout engine
    granite-docling-ov/                       Part 6
  cloud-results/                              results from the 4 vCPU cloud VM (reference)
```

Before every measured run (Parts 3 and 5), prepare the machine:
AC power, best-performance power mode, close other apps, idle 5 minutes, same
place and lid state each time. Details: `experiments/intel-cpu/reports/panther-lake-run-guide.md`
sections 2 and 3.

---

## Part 0: Tools (once)

Linux:

```bash
sudo apt install -y git unzip
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
git config --global core.longpaths true
```

Open a new terminal. Check: `git --version` and `uv --version`.

---

## Part 1: Set up

### 1.1 Folder layout

Use one work folder, for example `~/ptl` (Linux) or `C:\ptl` (Windows).
Put `docling-ptl-kit.zip` there.

Linux:

```bash
mkdir -p ~/ptl && cd ~/ptl
unzip docling-ptl-kit.zip            # creates ./docling-ptl-kit
mkdir -p ptl-results
```

Windows:

```powershell
New-Item -ItemType Directory -Force C:\ptl | Out-Null; Set-Location C:\ptl
Expand-Archive .\docling-ptl-kit.zip -DestinationPath .
New-Item -ItemType Directory -Force ptl-results | Out-Null
```

### 1.2 Clone upstream Docling at the pinned commit

All results (cloud and PTL) use upstream `main` at `ad2e0e1`.

```bash
git clone https://github.com/docling-project/docling docling-upstream
cd docling-upstream
git checkout -b ptl-baseline ad2e0e1a4fd3d5b0e709f450befa64811bff14cb
git log --oneline -1
```

Expected: `ad2e0e1 fix(jats): parse ordered lists and list item labels (#4770)`.
Same commands in PowerShell.

### 1.3 Copy the harness into the clone

The harness lives in its own folder `experiments/intel-cpu/`. It does not
change any Docling file.

Linux:

```bash
cp -r ../docling-ptl-kit/experiments .
chmod +x experiments/intel-cpu/run_bench.sh experiments/intel-cpu/granite-docling-ov/run.sh
git status --short          # expected: ?? experiments/
```

Windows:

```powershell
Copy-Item -Recurse ..\docling-ptl-kit\experiments .
git status --short          # expected: ?? experiments/
```

### 1.4 Create the upstream environment

This uses the upstream lock without change. It installs the standard
Docling stack and the pytest tools.

```bash
uv sync --frozen --group dev --extra standard --extra feat-ocr-rapidocr-onnx
```

Note: upstream `make setup` uses `--all-extras`. That also installs
`tesserocr` (needs the Tesseract C library to build), Whisper, Nemotron OCR
and more, which the tests below do not need.

Record the environment:

Linux:

```bash
uv pip list > ../ptl-results/upstream-deps.txt
lscpu > ../ptl-results/lscpu.txt
cat /sys/devices/cpu_core/cpus /sys/devices/cpu_atom/cpus > ../ptl-results/core-types.txt 2>/dev/null
```

Windows:

```powershell
uv pip list > ..\ptl-results\upstream-deps.txt
Get-CimInstance Win32_Processor | Format-List Name,NumberOfCores,NumberOfLogicalProcessors > ..\ptl-results\cpu.txt
powercfg /getactivescheme > ..\ptl-results\power.txt
```

---

## Part 2: Baseline quality (Docling's own test)

`tests/test_e2e_conversion.py` converts every PDF in
`tests/data/pdf/sources/` (OCR off, tables on) and checks text, DocTags and
JSON against the ground truth in the repo. The first run downloads the models
from Hugging Face.

Linux:

```bash
uv run --no-sync pytest tests/test_e2e_conversion.py -v -s 2>&1 | tee ../ptl-results/A-test_e2e_conversion.txt
```

Windows:

```powershell
uv run --no-sync pytest tests/test_e2e_conversion.py -v -s 2>&1 | Tee-Object ..\ptl-results\A-test_e2e_conversion.txt
```

Expected: a table with 17 rows, all `PASS`, and `1 passed`.
(Cloud: 17/17 PASS.) If a row fails, the table names the document and the
check. Stop and send the file before you continue.

After this first run the models are cached. To forbid more downloads:
`export HF_HUB_OFFLINE=1` (Linux) or `$env:HF_HUB_OFFLINE = "1"` (Windows).
Part 4 downloads one more model (the Heron ONNX export); set the variable
after Part 4.4 Step 1.

---

## Part 3: Baseline benchmark

Upstream Docling has no full-pipeline benchmark (`perfs/` only measures PDF
parsing), so use the harness. For each document it starts a fresh process
(cold start), runs 1 warmup and 5 measured conversions, OCR off and on, and
records per-stage times and peak memory. It then checks the outputs against the
repo ground truth.

The harness has its own environment (`experiments/intel-cpu/uvproject/uv.lock`:
torch 2.14.1, onnxruntime 1.31.0, openvino 2026.4.1, transformers 5.18.0).
It uses the Docling code of this clone (editable), so at this point the code is
unchanged upstream.

Quick check first (about 5 minutes):

```bash
experiments/intel-cpu/run_bench.sh --experiment smoke --label smoke --docs ocr_test normal_4pages --runs 1
```

```powershell
.\experiments\intel-cpu\run_bench.ps1 -Experiment smoke -Label smoke -Docs ocr_test,normal_4pages -Runs 1
```

Expected: `failures: 0` and a `results:` line. Then delete
`experiments/intel-cpu/results/smoke`.

Full baseline. Replace `<sku>` with a short machine name (for example
`ptl-u7-355`). Use the same label in Part 5.

```bash
experiments/intel-cpu/run_bench.sh --experiment A-baseline --label <sku>-linux-ac
```

```powershell
.\experiments\intel-cpu\run_bench.ps1 -Experiment A-baseline -Label <sku>-win-ac
```

Expected: `failures: 0`; correctness table rows `PASS` for OCR off.
Results: `experiments/intel-cpu/results/A-baseline/<label>/summary.md`.
Time on the cloud VM (4 vCPU): about 60 minutes.

Thread variant (optional, input for experiment D), Linux example with
P-cores 0-3:

```bash
DOCLING_NUM_THREADS=4 taskset -c 0-3 experiments/intel-cpu/run_bench.sh --experiment A-baseline --label <sku>-linux-ac-pcores
```

---

## Part 4: OpenVINO patch and accuracy

### 4.1 What the patch does

- Adds the layout engine `ObjectDetectionEngineType.OPENVINO` with
  `OpenVinoObjectDetectionEngineOptions` (`device`, `performance_hint`,
  `num_streams`, `cache_dir`).
- It compiles the Heron ONNX export (`docling-project/docling-layout-heron-onnx`)
  with OpenVINO Runtime. Pre- and post-processing are the same as in the
  existing ONNX Runtime engine.
- Off by default. Without the new option, Docling behaves exactly as before.
- Files: 4 changed, 1 new engine file, 1 new test file.

### 4.2 Apply the patch

Use a new branch, so that the baseline branch stays clean. The untracked
`experiments/` folder stays in place.

```bash
git checkout -b ptl-openvino
git apply --check ../docling-ptl-kit/patches/0001-openvino-layout-engine.patch
git apply ../docling-ptl-kit/patches/0001-openvino-layout-engine.patch
git status --short
```

Windows: same commands with `..\docling-ptl-kit\patches\...`. If
`git apply --check` reports whitespace errors on Windows, add
`--ignore-whitespace` to both commands.

`git apply --check` prints nothing when the patch fits. Expected
`git status --short`:

```text
 M docling/datamodel/object_detection_engine_options.py
 M docling/datamodel/stage_model_specs.py
 M docling/models/inference_engines/object_detection/base.py
 M docling/models/inference_engines/object_detection/factory.py
?? docling/models/inference_engines/object_detection/openvino_engine.py
?? experiments/
?? tests/test_openvino_object_detection_engine.py
```

### 4.3 Add OpenVINO to the upstream environment

```bash
uv pip install openvino==2026.4.1
uv run --no-sync python -c "import openvino as ov; print(ov.__version__, ov.Core().available_devices)"
```

Expected: `2026.4.1-...` and a list that contains `CPU` (also `GPU`, `NPU`
if the drivers are installed).

Important: from now on always use `uv run --no-sync`. A plain `uv run` or
`uv sync` removes `openvino`, because it is not in the upstream lock.

### 4.4 Verify accuracy (before any benchmark)

**Step 1: unit tests and parity** (OpenVINO vs ONNX Runtime on one page:
same labels, scores within 0.001, boxes within 1 px):

```bash
uv run --no-sync pytest tests/test_openvino_object_detection_engine.py -v 2>&1 | tee ../ptl-results/B-parity-test.txt
```

Windows: replace `| tee <file>` with `| Tee-Object <file>` and `/` with `\`
in the result path.

Expected: `3 passed`.

**Step 2: Docling's e2e test with each layout engine.** The wrapper runs
`tests/test_e2e_conversion.py` unchanged and only replaces the layout engine.

Linux:

```bash
for E in transformers onnxruntime openvino; do
  DOCLING_EXP_LAYOUT_ENGINE=$E uv run --no-sync pytest -s \
    experiments/intel-cpu/tests/test_e2e_layout_engine.py 2>&1 | tee ../ptl-results/B-e2e-layout-$E.txt
done
```

Windows:

```powershell
foreach ($E in "transformers","onnxruntime","openvino") {
  $env:DOCLING_EXP_LAYOUT_ENGINE = $E
  uv run --no-sync pytest -s experiments/intel-cpu/tests/test_e2e_layout_engine.py 2>&1 |
    Tee-Object ..\ptl-results\B-e2e-layout-$E.txt
}
Remove-Item Env:DOCLING_EXP_LAYOUT_ENGINE
```

Expected (cloud): 17/17 PASS for all three engines.

**Step 3: strict comparison, ONNX Runtime vs OpenVINO.**

```bash
uv run --no-sync python experiments/intel-cpu/scripts/compare_layout_engines.py \
  onnxruntime openvino --out ../ptl-results/B-strict-ort-vs-ov.json 2>&1 | tee ../ptl-results/B-strict-ort-vs-ov.txt
```

Expected (cloud): every row `same | same | n/n | same`, largest box difference
below 0.001 pt.

How to judge:

| Observation | Meaning |
|---|---|
| `transformers` result differs from Part 2 | machine or environment problem, not OpenVINO |
| `onnxruntime` fails where `transformers` passes | the ONNX export differs from the PyTorch model, not OpenVINO |
| `openvino` differs from `onnxruntime` (Step 2 or 3) | OpenVINO problem: stop and send the files |
| all PASS and Step 3 all `same` | OpenVINO is accurate: go to Part 5 |

---

## Part 5: OpenVINO benchmark

Same machine state and same day as Part 3. The harness environment already
contains openvino 2026.4.1 and uses the patched code of this clone.

Linux:

```bash
# B1: Heron ONNX on ONNX Runtime (separates "ONNX model" from "OpenVINO")
experiments/intel-cpu/run_bench.sh --experiment B1-layout-ort --label <sku>-linux-ac \
  -- --layout-engine onnxruntime
# B2: Heron ONNX on OpenVINO
experiments/intel-cpu/run_bench.sh --experiment B2-layout-ov --label <sku>-linux-ac \
  -- --layout-engine openvino --ov-hint LATENCY
# B2 with the compiled-model cache (shorter cold start from the second process on)
experiments/intel-cpu/run_bench.sh --experiment B2-layout-ov-cache --label <sku>-linux-ac \
  -- --layout-engine openvino --ov-cache-dir ~/.cache/docling-ov
# C1: RapidOCR on OpenVINO (OCR on only; no Docling code change)
experiments/intel-cpu/run_bench.sh --experiment C1-ocr-ov --label <sku>-linux-ac \
  --ocr on -- --ocr-backend openvino
```

Windows:

```powershell
.\experiments\intel-cpu\run_bench.ps1 -Experiment B1-layout-ort -Label <sku>-win-ac -WorkerArgs @('--layout-engine','onnxruntime')
.\experiments\intel-cpu\run_bench.ps1 -Experiment B2-layout-ov -Label <sku>-win-ac -WorkerArgs @('--layout-engine','openvino','--ov-hint','LATENCY')
.\experiments\intel-cpu\run_bench.ps1 -Experiment B2-layout-ov-cache -Label <sku>-win-ac -WorkerArgs @('--layout-engine','openvino','--ov-cache-dir',"$env:LOCALAPPDATA\docling-ov")
.\experiments\intel-cpu\run_bench.ps1 -Experiment C1-ocr-ov -Label <sku>-win-ac -Ocr on -WorkerArgs @('--ocr-backend','openvino')
```

Compare the outputs of B1 and B2 (the harness lists every difference):

```bash
uv run --frozen --no-sync --project experiments/intel-cpu/uvproject python \
  experiments/intel-cpu/scripts/correctness.py reference \
  experiments/intel-cpu/results/B1-layout-ort/<sku>-linux-ac \
  experiments/intel-cpu/results/B2-layout-ov/<sku>-linux-ac
```

Read the numbers in `summary.md` of A-baseline, B1 and B2:

- `warm median s` per document (lower is better), `pages/s` in the corpus
  rollup, `peak RSS`.
- Per-stage table, column `layout`: this is the stage that OpenVINO changes.
  `table` and `ocr` must stay about the same in B1 and B2.
- `cold s` includes the OpenVINO compile; compare B2 with B2-layout-ov-cache.
- Claim a speed-up only when the medians differ by more than the min-max
  spread of both runs.

---

## Part 6 (optional): Granite-Docling on OpenVINO

Speed test only. The OpenVINO export follows transformers 4.57.6, which gives
different (worse) output than transformers 5.18.0 used by Docling. See
`cloud-results/G-granite-docling-ov/README.md`.

```bash
experiments/intel-cpu/granite-docling-ov/run.sh --label <sku>-linux-ac --weights fp32
```

Windows: run the steps of `run.sh` by hand (`uv sync --frozen --project
experiments\intel-cpu\granite-docling-ov`, then `optimum-cli export openvino ...`,
then `scripts\compare.py`).

---

## Part 7: Send the results back

Linux:

```bash
cd ~/ptl
cp -r docling-upstream/experiments/intel-cpu/results ptl-results/harness-results
zip -r ptl-results-<sku>.zip ptl-results
```

Windows:

```powershell
Set-Location C:\ptl
Copy-Item -Recurse docling-upstream\experiments\intel-cpu\results ptl-results\harness-results
Compress-Archive ptl-results ptl-results-<sku>.zip
```

Add a short `notes.txt` (laptop model, BIOS, power mode, AC or battery,
anything unusual).

---

## Part 8: Make the patch yourself (reference)

From a clone of the fork:

```bash
git clone https://github.com/mrsabhar/docling docling-fork
cd docling-fork
git diff ad2e0e1a4fd3d5b0e709f450befa64811bff14cb origin/intel-cpu-openvino -- \
  docling/ tests/test_openvino_object_detection_engine.py \
  > 0001-openvino-layout-engine.patch
git apply --stat 0001-openvino-layout-engine.patch
```

Expected `--stat`: 6 files changed, 296 insertions, 2 deletions.

To make the change by hand without `git apply`: open the patch in an editor.
Each `diff --git a/<file>` block is one file. In each block, lines that start
with `+` are added, lines with `-` are removed, and the other lines are context
that shows the place. `openvino_engine.py` and
`test_openvino_object_detection_engine.py` are new files (all lines `+`); copy
them without the leading `+`.
