# What the cloud VM can and cannot validate

Cloud VM: 4 vCPU KVM guest, Intel Xeon (Cascade Lake class, family 6 model
85), AVX-512 + VNNI, no AMX, 15.7 GiB RAM, Ubuntu 24.04, no GPU. Details:
[`A-baseline.md`](A-baseline.md).

**All performance numbers from this VM are functional reference only.**

## Can validate

| Item | How |
|---|---|
| Install reproducibility | `uv sync --frozen` from `uvproject/uv.lock` |
| Harness correctness | Scripts run end to end; result files are complete |
| Functional correctness of outputs | Diff vs repo ground truth (OCR off) and vs baseline reference (all modes) |
| New code paths run without error | OpenVINO CPU plugin, RapidOCR OpenVINO backend, new engines, option plumbing |
| Determinism | Repeat runs give identical outputs |
| Gross regressions | An order-of-magnitude slowdown or a large RSS increase is visible even here |
| Stage structure | Which stages exist, which run, how many calls per document |
| Unit and e2e tests | `make check`, targeted pytest |

## Cannot validate

| Item | Why |
|---|---|
| Absolute latency, pages/s, RSS on client PCs | Different microarchitecture, clocks, memory (LPDDR5X), caches |
| Relative speed-ups of B-E | The VM uses AVX-512 kernels; Panther Lake uses AVX2 / AVX-VNNI kernels. oneDNN, ONNX Runtime and OpenVINO select different code paths, so ratios do not transfer |
| Hybrid P/E/LP-E scheduling, Thread Director, affinity effects | VM has 4 identical cores |
| Thread scaling above 4 threads | Only 4 vCPUs |
| Power, turbo, thermal behaviour, AC vs battery | Not exposed in the VM |
| Windows-specific behaviour | Linux only (OpenMP runtime, scheduler, Defender, EcoQoS differ on Windows) |
| iGPU / NPU offload | No Intel GPU or NPU; out of scope for CPU work anyway |
| Stable timing | Shared host; noisy neighbours can add variance |
| INT8 / VNNI speed on client CPUs | VM has AVX-512 VNNI, client has AVX-VNNI (256-bit); performance differs |

## Partial checks possible on the VM

- Force a lower ISA in oneDNN to exercise AVX2 code paths:
  `ONEDNN_MAX_CPU_ISA=AVX2` (torch). This checks correctness on those paths,
  not client performance. The VM has no AVX-VNNI (VEX), so the exact
  Panther Lake kernels cannot be exercised.
- Oversubscription can be shown qualitatively with `num_threads` > 4 on 4 vCPUs.
