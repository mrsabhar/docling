# B: OpenVINO layout engine, accuracy (cloud VM, functional reference)

Machine: 4 vCPU `Intel(R) Xeon(R) Processor @ 2.10GHz`, 16 GiB. Code: upstream
`ad2e0e1` + `patches/0001-openvino-layout-engine.patch`. Runtimes: torch
2.14.1, onnxruntime 1.31.0, openvino 2026.4.1, transformers 5.18.0. CPU, FP32.

## 1. Parity test (one Heron page, OpenVINO vs ONNX Runtime)

`tests/test_openvino_object_detection_engine.py`: 3 passed (same labels,
scores within 1e-3, boxes within 1 px). Log: `parity-test.txt`.

## 2. Docling e2e ground-truth test, per layout engine

`tests/test_e2e_conversion.py` unchanged (17 PDFs, OCR off, tables on),
run through `experiments/intel-cpu/tests/test_e2e_layout_engine.py`.

| Layout engine | Model | Result | Test time |
|---|---|---|---:|
| transformers (upstream default) | docling-layout-heron | 17/17 PASS | 307 s |
| onnxruntime | docling-layout-heron-onnx | 17/17 PASS | 185 s |
| openvino | docling-layout-heron-onnx | 17/17 PASS | 178 s |

Logs: `e2e-layout-<engine>.txt`. Test times include model load and table
structure; they are not a benchmark.

## 3. Strict comparison, ONNX Runtime vs OpenVINO (same 17 PDFs)

`scripts/compare_layout_engines.py onnxruntime openvino`:

- Markdown identical: 17/17. DocTags identical: 17/17.
- Layout clusters: 1398 vs 1398, labels identical in order.
- Largest box difference: 0.00024 pt.

Per document: `strict-onnxruntime-vs-openvino.md` / `.json`.
The pipeline was checked to use `OpenVinoObjectDetectionEngine` for the
OpenVINO runs.

## Verdict

OpenVINO gives the same conversion output as the existing ONNX Runtime engine
on all e2e test documents, and both pass Docling's ground truth. Accuracy is
verified on this machine. Repeat Part 2.4 of `reports/ptl-manual-guide.md` on
Panther Lake before benchmarking there.
