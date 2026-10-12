# Plan for experiments B to E

Status: **plan only. Not implemented. Waiting for approval.**

## Rules for every experiment

- Same corpus ([`config/corpus.json`](../config/corpus.json)), same harness
  (`bench.py`: fresh process per document, cold start, 1 warmup, 5 runs, OCR
  off and on), same lockfile plus only the packages the experiment needs.
- Compare an experiment only with experiment A **on the same machine, same
  power state, same day**. Cloud VM results are functional reference only.
- A speed-up is claimed only when the median difference is larger than the
  min-max spread of both runs, for the same document. Otherwise the result is
  "no measurable change".
- Correctness gate: `correctness.py groundtruth --repo-check` for OCR off, and
  `correctness.py reference <A> <X>` for all outputs. Any difference is listed
  and explained in the experiment report; no silent acceptance.
- Each experiment is a separate commit series on `intel-cpu-exp`, with its own
  `results/<letter>-<name>/` folder and `reports/<letter>-<name>.md`.
- New runtime switches are off by default. Default Docling behaviour must not
  change.

## B: OpenVINO for layout and table models

### Approach

**Layout (Heron).** Docling already ships an ONNX Runtime object-detection
engine
([`onnxruntime_engine.py`](../../../docling/models/inference_engines/object_detection/onnxruntime_engine.py))
and Heron already has an ONNX export: `docling-project/docling-layout-heron-onnx`
([`stage_model_specs.py:1017-1026`](../../../docling/datamodel/stage_model_specs.py)).

- B1 (no Docling code change): ONNX Runtime engine with the existing CPU
  provider, to separate "ONNX graph" from "OpenVINO" effects.
- B2 (main): new `OPENVINO` object-detection engine that loads the same ONNX
  file with OpenVINO Runtime (`ov.Core().compile_model(..., "CPU")`), with
  `PERFORMANCE_HINT` (LATENCY / THROUGHPUT), `INFERENCE_NUM_THREADS`,
  `SCHEDULING_CORE_TYPE`, and `CACHE_DIR` for compiled-model caching.
  Pre- and post-processing are shared with the ONNX Runtime engine so outputs
  stay comparable.
- B3 (optional): ONNX Runtime with the OpenVINO execution provider
  (`onnxruntime-openvino`). Only if a wheel matching the pinned ONNX Runtime
  exists; it replaces the `onnxruntime` package, so it needs its own lock.
- Precision: FP32 first. INT8 (NNCF post-training quantization) only as a
  follow-up with its own accuracy gate. BF16 is not useful on Panther Lake
  (no AMX / AVX-512 BF16).

**Table (TableFormer v1).** Torch model with an autoregressive tag decoder;
a static export is harder.

- B4: `torch.compile(model, backend="openvino")` (OpenVINO torch backend) on
  the TableFormer encoder and decoder step, behind an experiment option.
  Watch recompiles from dynamic shapes in the decoder loop.
- B5 (if B4 is unstable): export only the image encoder to OpenVINO, keep the
  decoder in torch.

### Files touched (B2, B4)

- `docling/datamodel/object_detection_engine_options.py` (new
  `OpenVinoObjectDetectionEngineOptions`, new `ObjectDetectionEngineType.OPENVINO`)
- `docling/models/inference_engines/object_detection/openvino_engine.py` (new)
- `docling/models/inference_engines/object_detection/factory.py` (register engine)
- `docling/datamodel/stage_model_specs.py` (engine override for Heron ONNX)
- `docling/models/stages/table_structure/table_structure_model.py` (compile hook)
- `pyproject.toml` new extra `models-openvino` (only if upstreamed; for the
  experiment, `experiments/intel-cpu/uvproject/pyproject.toml` adds `openvino`)
- `experiments/intel-cpu/scripts/bench_worker.py` (`--layout-engine`,
  `--table-compile` flags)
- Tests: engine-selection unit test, plus the existing e2e ground truth.

### Needs approval

- Download of `docling-project/docling-layout-heron-onnx` (Docling-published,
  but not a default model).
- New package `openvino` (latest release, pinned in the experiment lock).

### Risks

- Small numeric differences change layout cluster boundaries and then the
  markdown. Mitigation: compare boxes with the repo bbox tolerance, list every
  text difference.
- OpenVINO model compile adds to cold start. Mitigation: report compile time
  separately; test `CACHE_DIR`.
- Two inference runtimes (torch + OpenVINO) in one process raise RSS.

### Measurement

Layout stage time and table stage time (per-stage profiler), warm latency,
cold start split (import / init incl. compile / first convert), peak RSS. B1,
B2 and B4 each against A.

## C: OCR on OpenVINO

### Approach

RapidOCR already supports an OpenVINO backend in Docling:
`RapidOcrOptions(backend="openvino")`
([`pipeline_options.py:346`](../../../docling/datamodel/pipeline_options.py),
[`rapid_ocr_model.py:328`](../../../docling/models/stages/ocr/rapid_ocr_model.py)),
and Docling passes `num_threads` to `EngineConfig.openvino.inference_num_threads`
([`rapid_ocr_model.py:632`](../../../docling/models/stages/ocr/rapid_ocr_model.py)).
RapidOCR's OpenVINO config also exposes `performance_hint`, `num_streams`,
`enable_cpu_pinning`, `enable_hyper_threading` and `scheduling_core_type`,
which can be set through `RapidOcrOptions.rapidocr_params` without code change.

- C1: `--ocr-backend openvino` (harness flag exists). No Docling code change.
- C2: sweep `performance_hint` (LATENCY / THROUGHPUT), `scheduling_core_type`
  (ANY_CORE / PCORE_ONLY) and thread count via `rapidocr_params`. Needs a small
  harness flag to pass the params.
- C3 (optional): `OcrMode.FULL_PAGE` run on the scanned documents, to measure
  OCR in isolation from layout.

### Files touched

- `experiments/intel-cpu/uvproject/pyproject.toml` (add `openvino`)
- `experiments/intel-cpu/scripts/bench_worker.py` (`--rapidocr-param KEY=VALUE`)
- No change to `docling/` expected.

### Risks

- RapidOCR's OpenVINO path may use different pre-processing or model files
  than the ONNX Runtime path; text can differ slightly. Mitigation: text
  similarity vs A reference outputs per document.
- OCR is only a large share on scanned pages. On born-digital documents C
  may show no change; this is expected and will be reported as such.

### Measurement

OCR stage time on `nemotron_multipage` and `ocr_test` (and the redp5110 cover),
warm latency, peak RSS, text similarity vs A.

## D: CPU threading and concurrency tuning

### Facts from the code

- `AcceleratorOptions.num_threads` defaults to **4**, independent of the core
  count ([`accelerator_options.py`](../../../docling/datamodel/accelerator_options.py)).
- The standard pipeline runs preprocess, layout, OCR, layout postprocess,
  table and assemble as separate Python threads with bounded queues
  ([`standard_pdf_pipeline.py:730-800`](../../../docling/pipeline/standard_pdf_pipeline.py)).
- Layout and TableFormer call `torch.set_num_threads(num_threads)`
  ([`transformers_engine.py:140`](../../../docling/models/inference_engines/object_detection/transformers_engine.py),
  docling-ibm-models `tf_predictor.py`); RapidOCR uses ONNX Runtime
  intra-op threads = `num_threads`. With the OpenMP backend each calling
  thread can get its own thread team, so up to 3 x `num_threads` compute
  threads can be active at once.
- Batch sizes: `layout_batch_size`, `table_batch_size`, `ocr_batch_size`
  (default 4), `queue_max_size` (100), `settings.perf.page_batch_size` (4).

### Approach

- D1 (no code change): sweep `num_threads` / `OMP_NUM_THREADS` in
  {1, 2, 4, P-core count, P+E core count}, plus `OMP_WAIT_POLICY=PASSIVE`
  / `KMP_BLOCKTIME=0` (spin-wait between pipelined stages costs CPU on
  small core counts).
- D2 (no code change): batch size sweep (layout 1/4/8, table 1/4/8).
- D3 (no code change): process-level parallelism for throughput: N worker
  processes, each with `num_threads = cores / N`, over the corpus. Harness
  addition only.
- D4 (code change): per-stage thread budgets (for example layout 4, table 2,
  OCR 2) instead of one global `num_threads`. Touches
  `docling/datamodel/accelerator_options.py` or the per-stage options,
  `transformers_engine.py`, `table_structure_model.py`, `rapid_ocr_model.py`.
- D5: hybrid pinning on Panther Lake: P-cores only vs all cores, via OS
  affinity (see the run guide). No code change.

### Risks

- `torch.set_num_threads` is process-wide in practice; per-stage budgets may
  need thread-local handling or separate processes.
- Results depend strongly on core count and core type; a setting that wins on
  one SKU can lose on another. Report per machine, no universal default claim.

### Measurement

Latency (single document) and throughput (corpus pages/s, D3), CPU time
(user + sys) per page, context switches (psutil), peak RSS. Thread env vars
are already recorded per run.

## E: oneTBB prototype (optional)

### Approach

Do E only if D shows oversubscription or idle cores between stages.

- E1: use one shared OpenVINO `Core` for all OpenVINO stages (after B and C);
  OpenVINO's CPU plugin schedules with oneTBB and CPU streams, so layout and
  OCR requests share one arena instead of separate pools.
- E2: run the Python process under `python -m tbb` (the `tbb` package),
  which makes MKL / numpy / Python thread pools composable through oneTBB.
- PyTorch wheels on PyPI use the OpenMP backend; a TBB-backed ATen build is
  not on PyPI and is out of scope.

### Files touched

- E1: the OpenVINO engines from B/C (shared core factory), new module under
  `docling/models/inference_engines/`.
- E2: `experiments/intel-cpu/uvproject/pyproject.toml` (`tbb`), runner flag.

### Risks

High uncertainty, possible no gain; mixing OpenMP (torch) and TBB (OpenVINO)
runtimes can increase contention. Kept as a prototype; no upstream change
proposed unless the gain is clear.

### Measurement

Same as D, plus total thread count of the process over time.

## Order and effort

1. A on the cloud VM (after unblock) and on Panther Lake (Windows + Linux).
2. D1/D2/D5 (no code, cheap, sets the thread baseline for B and C).
3. C1/C2 (no Docling code).
4. B1, then B2, then B4.
5. D3/D4, then E if justified.
