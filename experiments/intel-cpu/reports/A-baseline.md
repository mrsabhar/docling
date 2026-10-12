# Experiment A: Baseline report

> **Status: measurement blocked.** The cloud VM network policy denies
> `huggingface.co` and `www.modelscope.cn`. Docling downloads its default
> models (Heron layout, TableFormer, RapidOCR) from these hosts. No PDF
> conversion can run without them, so this report has **no latency, RSS or
> per-stage numbers yet**. The harness, corpus, environment report and
> lockfile are complete and committed. The measurement runs when the hosts are
> allowed (see "Unblock" below).
>
> All numbers from this VM, when they exist, are **functional reference only**.
> They are not performance claims for Intel client CPUs.

## 1. Version under test

| Item | Value |
|---|---|
| Upstream repo | `docling-project/docling` |
| Upstream commit | `ad2e0e1a4fd3d5b0e709f450befa64811bff14cb` (main, 2026-10-12) |
| Branch | `intel-cpu-exp` (fork `mrsabhar/docling`) |
| docling-slim | 2.137.0 (editable checkout of the commit above) |
| Upstream files changed | none |

## 2. Environment (cloud VM)

Source: [`results/A-baseline/cloud-functional/env.json`](../results/A-baseline/cloud-functional/env.json).

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.5 LTS, kernel 6.18.44 (KVM guest) |
| CPU | "Intel(R) Xeon(R) Processor @ 2.80GHz", family 6 model 85 stepping 7 (Cascade Lake class server core) |
| Arch | x86_64 |
| Cores / threads | 4 vCPU, 1 thread per core, 1 socket, 1 NUMA node |
| Caches (as reported by the hypervisor) | L1d 4x32 KiB, L2 4x1 MiB, L3 33 MiB |
| ISA (OS flags) | AVX, AVX2, FMA, F16C, AVX-512 F/CD/BW/DQ/VL, AVX-512 VNNI |
| ISA not present | AVX-VNNI (VEX), AVX-512 BF16, AVX-512 FP16, AMX (TILE/INT8/BF16) |
| Hybrid cores | No (no `cpu_core` / `cpu_atom` PMUs) |
| Memory | 15.7 GiB, no swap |
| Python | 3.12.3 (GCC 13.3.0) |
| Compilers | gcc 13.3.0, clang 18.1.3 |
| uv | 0.11.32 |

### Runtime stack

| Package | Version | Note |
|---|---|---|
| torch | 2.14.1+cu130 | CPU capability `AVX512`, oneDNN (MKL-DNN) 3.x, MKL, OpenMP backend |
| torchvision | 0.29.1 | |
| onnxruntime | 1.31.0 | Providers: `CPUExecutionProvider`, `AzureExecutionProvider` |
| openvino | not installed | Needed from experiment B on |
| docling-core | 2.100.0 | |
| docling-parse | 7.22.2 | PDF parser backend |
| docling-ibm-models | 4.0.3 | TableFormer |
| rapidocr | 3.9.2 | |
| transformers | 5.18.0 | Heron layout (RT-DETRv2) |

Full manifest: [`results/A-baseline/cloud-functional/deps.txt`](../results/A-baseline/cloud-functional/deps.txt)
(126 distributions). Lockfile: [`uvproject/uv.lock`](../uvproject/uv.lock).

Version policy: torch 2.14.1, torchvision 0.29.1 and onnxruntime 1.31.0 are
pinned to the latest releases (your request). Every other package shared with
the upstream `uv.lock` is constrained to the upstream version
(`constraint-dependencies` in `uvproject/pyproject.toml`), so output changes
cannot come from drift in docling-core, transformers, rapidocr and so on.
Versus upstream for Python 3.12 / Linux x86_64 the differences are: torch
2.11.0 -> 2.14.1, torchvision -> 0.29.1, onnxruntime 1.30.0 -> 1.31.0.

Install command (same on every machine):

```bash
uv sync --frozen --project experiments/intel-cpu/uvproject
```

### Thread defaults in effect

No thread variables are set on the VM. Effective values:

| Setting | Value | Source |
|---|---|---|
| `AcceleratorOptions.num_threads` | 4 | Docling default (fixed 4, not core count) |
| torch intra-op threads | 4 | = `hardware_concurrency` on this VM |
| torch inter-op threads | 4 | |
| RapidOCR / ONNX Runtime intra-op | 4 | from `num_threads` |

## 2b. Existing support in the code base

| Area | What exists | Files |
|---|---|---|
| Pipeline timings | `settings.debug.profile_pipeline_timings` (env `DOCLING_DEBUG_PROFILE_PIPELINE_TIMINGS=1`) enables `TimeRecorder`; results in `ConversionResult.timings` (`ProfilingItem`: times, count, start timestamps; page or document scope) | `docling/utils/profiling.py`, `docling/datamodel/settings.py:44-54` |
| Timing keys | `page_parse`, `layout`, `layout_postprocess`, `ocr`, `table_structure`, `page_assemble`, `reading_order`, `doc_assemble`, `doc_build`, `pipeline_total`, `doc_enrich` | `page_preprocessing_model.py:67`, `layout_object_detection_model.py:114`, `layout_postprocessing_model.py:57`, `rapid_ocr_model.py:761`, `table_structure_model.py:199`, `page_assemble_model.py:188`, `readingorder_model.py:695`, `standard_pdf_pipeline.py:1048`, `base_pipeline.py:81,303` |
| Pipeline concurrency | `StandardPdfPipeline` = threaded stages (preprocess -> layout -> OCR -> layout postprocess -> table -> assemble) with bounded queues and per-stage batch sizes | `docling/pipeline/standard_pdf_pipeline.py:163-300, 730-800` |
| Batch / queue options | `ocr_batch_size`, `layout_batch_size`, `table_batch_size` (4), `batch_polling_interval_seconds` (0.5), `queue_max_size` (100); `settings.perf.page_batch_size` (4), `doc_batch_size`, `doc_batch_concurrency` (experimental) | `docling/datamodel/pipeline_options.py:2254-2310`, `docling/datamodel/settings.py:32-41` |
| Accelerator / threads | `AcceleratorOptions(num_threads=4, device=auto)`; env `DOCLING_NUM_THREADS`, falls back to `OMP_NUM_THREADS`; devices cpu/cuda/mps/xpu | `docling/datamodel/accelerator_options.py`, `docling/utils/accelerator_utils.py` |
| Where threads are applied | `torch.set_num_threads` in the Transformers layout engine and TableFormer (v1 via docling-ibm-models `TFPredictor`, v2); ONNX Runtime `intra_op_num_threads` in ONNX engines; RapidOCR engine threads per backend | `inference_engines/object_detection/transformers_engine.py:140`, `table_structure_model.py:93`, `table_structure_model_v2.py:76`, `inference_engines/object_detection/onnxruntime_engine.py:122`, `rapid_ocr_model.py:628-651` |
| Layout engines | Transformers (default), ONNX Runtime (CPU / CUDA providers), KServe v2 remote. Heron has an ONNX export repo | `inference_engines/object_detection/factory.py`, `datamodel/stage_model_specs.py:1017-1026` |
| Torch compile | `settings.inference.compile_torch_models` -> `compile_model` default for object-detection and VLM engines | `datamodel/settings.py:57-58,110`, `datamodel/object_detection_engine_options.py:65` |
| OCR backends | Auto (ocrmac -> nemotron -> RapidOCR+onnxruntime -> EasyOCR -> RapidOCR+torch), RapidOCR (onnxruntime / **openvino** / paddle / torch), EasyOCR, Tesseract (lib and CLI), ocrmac, Nemotron, KServe v2 | `models/stages/ocr/auto_ocr_model.py`, `models/stages/ocr/rapid_ocr_model.py:328`, `datamodel/pipeline_options.py:346-390` |
| OpenVINO | Only through RapidOCR (`backend="openvino"`, threads via `EngineConfig.openvino.inference_num_threads`). No OpenVINO path for layout or table models. `openvino` is not a dependency of any extra | `rapid_ocr_model.py:631-632`, `docs/concepts/OCR_native.md` |
| Benchmarks | No PDF pipeline benchmark in the repo. `perfs/` has parser iteration and memory plotting scripts; `docs/examples/run_with_accelerator.py` shows timings usage | `perfs/iterate_pdf_pages.py`, `perfs/plot_memory_metrics.py`, `docs/examples/run_with_accelerator.py` |
| Ground truth | OCR off, tables + cell matching, CPU: `tests/data/pdf/groundtruth/*.{md,json,doctags.txt,pages.meta.json}`; OCR GT per engine in `tests/data/ocr/groundtruth/` | `tests/test_e2e_conversion.py:42-60`, `tests/verify_utils.py` |

## 3. Corpus

9 documents, 59 pages, all from the repository test data. Hashes are checked
before every run ([`config/corpus.json`](../config/corpus.json)).

| id | category | pages | SHA256 | repo ground truth |
|---|---|---:|---|---|
| normal_4pages | text-only | 4 | `ca9c83b1653221a66b30cdb543c1ef7b116827762ae65fca36b87b332532b51e` | yes |
| multi_page | text-only | 5 | `956846c28f9ed9d3b5cbe071865c082bd415f56e3e299c479aeea1f615eefa61` | yes |
| 2206.01062 | tables + multi-column | 9 | `5dfbd8c115a15fd3396b68409124cfee29fc8efac7b5c846634ff924e635e0dc` | yes |
| 2203.01017v2 | tables + multi-column | 16 | `4fa8dff93d74a84069210c84a38d14d62a39ec8f4e4c90bf955fdebdebcf6636` | yes |
| redp5110_sampled | tables | 18 | `2f6bc92861997d1434c069040dc63447c3e415e8e9eadbd1a8129ba2327ddba5` | yes |
| table_misidentified_as_form | tables | 1 | `7e8e895d8b1d97676bbddf4274fe83b0c17ce1f3238078d0d705392ef81256ff` | yes |
| newspaper-00 | multi-column | 1 | `ba5f7a78792369ea36dbf2a554aa1635fa9f25983664ff17b406dca326279019` | yes |
| nemotron_multipage | scanned / OCR | 4 | `b4f6d15c517b5d6b0d674e1b4a04219903528c3601fd026f98eadc62c814f568` | no (baseline output becomes reference) |
| ocr_test | scanned / OCR | 1 | `73f23122e9edbdb0a115b448e03c8064a0ea8bdc21d02917ce220cf032454f31` | no (repo GT is for non-default OCR modes) |

Note: `tests/data/pdf/` itself contains only tiny synthetic PDFs (< 5 KB). The
real test PDFs are in `tests/data/pdf/sources/` and `tests/data/ocr/sources/`.

## 4. Method

### Pipeline configuration

`StandardPdfPipeline` (the default threaded PDF pipeline) with
`PdfPipelineOptions`:

| Option | OCR off | OCR on |
|---|---|---|
| `accelerator_options.device` | `cpu` (forced; `auto` could pick XPU on a machine with an XPU torch build) | `cpu` |
| `accelerator_options.num_threads` | default (4) | default (4) |
| `do_table_structure` / `do_cell_matching` | True / True | True / True |
| `do_ocr` | False | True |
| `ocr_options` | n/a | `RapidOcrOptions(backend="onnxruntime")`, mode `DEFAULT` (bitmap regions only). Pinned so all machines use the same engine; this is also what `OcrAutoOptions` picks when onnxruntime is installed. |
| layout model | Heron (`docling-project/docling-layout-heron`), Transformers engine (torch) | same |
| table model | TableFormer v1 accurate (`docling-project/docling-models`) | same |
| batch sizes | layout 4, table 4, OCR 4 (defaults) | same |

The "OCR off" configuration equals the configuration that produced the repo
ground truth (`tests/test_e2e_conversion.py`).

### Measurement

- One fresh Python process per (document, OCR mode)
  ([`bench_worker.py`](../scripts/bench_worker.py)).
- **Cold start** = Python import of docling + `initialize_pipeline()` (model
  load) + first `convert()`. Each part is reported separately.
- Then **1 warmup** (discarded) and **5 measured** conversions in the same process.
- A model prefetch run happens before all measurements and is not timed, so a
  download never falls inside a cold start.
- **Latency**: wall clock of `DocumentConverter.convert()` (`time.perf_counter`).
  Median and p95 (numpy linear interpolation; with 5 runs p95 is between the
  4th and 5th sorted value).
- **Pages/sec**: pages / median latency.
- **Peak RSS**: background thread samples process RSS every 10 ms (psutil),
  max per phase. Cross-check: OS high-water mark for the whole process
  (`ru_maxrss` on Linux, `PeakWorkingSetSize` on Windows).
- **Per-stage time**: Docling's built-in profiler
  (`settings.debug.profile_pipeline_timings = True`,
  [`docling/utils/profiling.py`](../../../docling/utils/profiling.py)).
  Stage keys: `page_parse` (parse), `layout`, `layout_postprocess`, `ocr`,
  `table_structure` (table), `page_assemble` + `doc_assemble` (assembly;
  `doc_assemble` contains `reading_order`). Per run, intervals are summed per
  key; the report gives the median over runs.
- **Thread env vars** in effect are recorded in every raw result.

Caveat: the pipeline stages run in parallel threads with bounded queues.
Stage times overlap, so their sum can be larger than the wall time. Stage
share shows where compute goes, not the latency that one stage adds alone.

### Correctness

[`correctness.py`](../scripts/correctness.py):

- OCR off vs repo ground truth (`tests/data/pdf/groundtruth/`): markdown exact
  match (same `compact_tables=True` flavour as the repo tests), markdown line
  similarity, JSON text-item sequence similarity, table count / shape / cell
  text equality. `--repo-check` also runs the repo's strict check
  (`tests/verify_utils.py::check_conversion_result_v2`, includes bbox tolerance).
- OCR on and the 2 scanned documents: the baseline outputs in
  `results/A-baseline/<label>/outputs/` become the reference for B-E
  (`correctness.py reference <A dir> <B dir>`).

## 5. Results

**Not available.** Blocked by the network policy (section 7).

Harness status on the VM:

| Check | Result |
|---|---|
| `uv sync --frozen` of the experiment lockfile | pass |
| Environment report | pass |
| Corpus hash verification | pass (9/9) |
| Orchestrator, up to the model prefetch | pass; stops with "prefetch failed (model download blocked?)" |
| `summarize.py` on a synthetic raw file (code path test only, not committed) | pass |
| `correctness.py reference` on repo ground-truth files (code path test) | pass |
| Worker cold / warm measurement on a real PDF | **not run** (needs models) |

## 6. Bottleneck analysis

**Not done.** The task asks to identify the two largest bottlenecks from
per-stage data. There is no per-stage data yet, so no bottleneck is claimed.

What the code shows (structure only, no timing claim):

- The layout stage runs Heron (RT-DETRv2, ResNet-50 class backbone) on every
  page image through torch / Transformers
  ([`layout_object_detection_model.py:114`](../../../docling/models/stages/layout/layout_object_detection_model.py),
  [`transformers_engine.py:140`](../../../docling/models/inference_engines/object_detection/transformers_engine.py)).
  Its cost scales with page count, independent of content.
- The table stage runs TableFormer (encoder + autoregressive tag decoder) once
  per table, through torch
  ([`table_structure_model.py:199`](../../../docling/models/stages/table_structure/table_structure_model.py)).
  Its cost scales with table count and table size.
- OCR (when on) runs only on bitmap regions in `DEFAULT` mode, so it is
  significant only for scanned pages.

These are the candidates to confirm or reject with the per-stage data.

### Conclusions that may not hold on hybrid P/E-core client CPUs

- This VM has 4 identical server cores. Panther Lake mixes P-cores and
  E-cores (and LP E-cores). OpenMP static scheduling waits for the slowest
  thread; a team spread across P and E cores runs at E-core speed.
- The VM has AVX-512 + VNNI. Client Intel CPUs since Alder Lake have no
  AVX-512; they have AVX2 + AVX-VNNI. oneDNN, ONNX Runtime and OpenVINO pick
  different kernels, so the relative cost of layout vs table vs OCR can change.
- The Docling default `num_threads=4` equals the core count here. On a client
  CPU with more cores it uses only part of the CPU; on battery or with E-cores
  only, the effect differs again.
- Each pipeline stage is a Python thread that calls torch or ONNX Runtime with
  its own intra-op pool. With 4 threads per pool, layout + table + OCR can
  start up to 12 compute threads. On 4 vCPUs this oversubscribes; on a
  16-core client CPU it may not. Thread contention results do not transfer.
- The VM has no turbo, no thermal limit and no OS power policy visible. Client
  results depend on power plan, AC vs battery, and thermal state.
- Memory bandwidth and cache sizes differ (33 MiB L3 here as reported by the
  hypervisor; Panther Lake has a smaller shared L3 and LPDDR5X).

## 7. Unblock

Allow these hosts in the cloud environment's network settings (environment
menu -> Edit -> Network access -> Allowed domains, keep "Allow package
managers" ticked), see
<https://code.claude.com/docs/en/cloud-environments#network-access>:

- `huggingface.co` and its file hosts (`cdn-lfs.huggingface.co`,
  `cas-bridge.xethub.hf.co`, `*.hf.co`): Heron layout, TableFormer.
- `www.modelscope.cn` (and any CDN host it redirects to; the proxy log names
  it): RapidOCR PP-OCR ONNX models.

Then run:

```bash
experiments/intel-cpu/run_bench.sh --experiment A-baseline --label cloud-functional
```

and commit `results/A-baseline/cloud-functional/`.

Alternative without network change: copy a populated model cache
(`~/.cache/huggingface/hub` and the docling RapidOCR model folder, or a
`docling-tools models download` output folder passed as `artifacts_path`) to
the VM.
