# Run notes: A-baseline / cloud-functional

Machine: cloud VM, 4 vCPU `Intel(R) Xeon(R) Processor @ 2.10GHz`, 16 GiB RAM.
Functional reference only (see `reports/cloud-validation-scope.md`).

- The run was split in three parts (a background time limit stopped the first
  part): OCR off for `normal_4pages`, `multi_page`, `2206.01062`,
  `2203.01017v2`; then OCR off for the other five documents (`--no-prefetch`);
  then OCR on for all nine (`--no-prefetch`).
- **CPU contention:** small checks (ruff, two unit tests) ran during the first
  OCR-off part. The whole OCR-on part ran at the same time as the
  Granite-Docling OpenVINO export and the PyTorch/OpenVINO generation test
  (experiment G, 2-4 threads). OCR-on timings are therefore too high and must
  not be used as a baseline. OCR-off timings for the last five documents are
  the cleanest numbers in this run.
- Correctness: OCR off matches the repo ground truth for all 7 documents that
  have ground truth (`correctness-groundtruth.md`).
