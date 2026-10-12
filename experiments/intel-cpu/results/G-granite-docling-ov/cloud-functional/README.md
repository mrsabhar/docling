# G: Granite-Docling-258M on OpenVINO, cloud VM (functional reference)

Machine: 4 vCPU `Intel(R) Xeon(R) Processor @ 2.10GHz`, 16 GiB. 4 threads.
No other load during these runs. Page 1 of `tests/data/pdf/sources/2206.01062.pdf`,
rendered at scale 2.0 (1224x1584, 17 image tiles, 1142 prompt tokens).
Greedy decoding, `use_cache=True`, max 8192 new tokens, FP32 everywhere.

| Run | transformers | First token | Tokens | Decode | Total | Output |
|---|---|---:|---:|---:|---:|---|
| PyTorch | 5.18.0 (Docling lock) | 13.2 s | 1220 | 18.8 tok/s | 64.9 s | good page, ends `</doctag>` |
| PyTorch | 4.57.6 | 15.3 s | 8192 | 14.5 tok/s | 563 s | starts like OpenVINO, then loops ("on the other hand") |
| OpenVINO 2026.4.1 (export with 4.57.6) | 4.57.6 processor | 3.8 s | 1380 | 41.0 tok/s | 33.7 s | complete page, ends `</doctag>` |
| OpenVINO 2026.4.1, inputs from 5.18.0 processor | - | - | 1728 | 37.3 tok/s | 46.3 s | complete page, ends `</doctag>` |

## Findings

1. Speed: OpenVINO is about 3.5x faster to the first token (vision encoder +
   prefill) and about 2x faster per token than PyTorch 5.18.0 on this CPU.
2. Output: OpenVINO reproduces the transformers **4.57.6** model (same text,
   same typo "IBM Resecher", same boxes as the 4.57.6 PyTorch run until that
   run starts to loop). It does **not** match transformers **5.18.0**, which
   Docling uses. With identical input tensors (from the 5.18.0 processor) the
   OpenVINO output still differs from PyTorch 5.18.0, so the difference is in
   the model code, not in preprocessing. The 5.18.0 output is better (all
   e-mail addresses, keywords section, no typo).
3. Cause of the version limit: the optimum-intel 2.2.0 Idefics3 exporter
   refuses transformers > 4.57.6 ("Maximum required is 4.57.6").
4. The model's `generation_config.json` sets `use_cache: false`. Docling
   overrides this (`use_kv_cache=True`); without it, generation is unusably
   slow (the full model, vision encoder included, runs for every token).

## Next steps

- Find the Idefics3 / Granite-Docling model-code change between transformers
  4.57.6 and 5.x, and export to OpenVINO with that behaviour (patched export,
  or a newer optimum-intel when it supports transformers 5).
- Then repeat on more pages and documents, compare with DocTags ground truth,
  and test INT8 weights (`--weights int8`).
