"""Granite-Docling on CPU: PyTorch (transformers) vs OpenVINO (optimum-intel).

Usage: python compare.py <pdf> <page_index> <ov_model_dir|-> <out_json> [threads]

With "-" as ov_model_dir, only the PyTorch run is done (for example in the
docling environment uvproject/, which has a newer transformers).
"""

import json
import sys
import time
from pathlib import Path

import pypdfium2
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, StoppingCriteria

REPO = "ibm-granite/granite-docling-258M"
PROMPT = "Convert this page to docling."
MAX_NEW_TOKENS = 8192

pdf_path, page_idx, ov_dir, out_json = sys.argv[1:5]
threads = int(sys.argv[5]) if len(sys.argv) > 5 else 4
torch.set_num_threads(threads)

# Same scale as the docling preset (scale=2.0 -> 144 dpi).
image = pypdfium2.PdfDocument(pdf_path)[int(page_idx)].render(scale=2.0).to_pil()
image = image.convert("RGB")

processor = AutoProcessor.from_pretrained(REPO)
messages = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
n_prompt = inputs["input_ids"].shape[1]
print(f"prompt tokens: {n_prompt}, image {image.size}", flush=True)


class Progress(StoppingCriteria):
    """Print the generated token count every 100 tokens; never stops generation."""

    def __init__(self, name, t0):
        self.name, self.t0, self.first = name, t0, None

    def __call__(self, input_ids, scores, **kwargs):
        n = input_ids.shape[1] - n_prompt
        if n == 1:
            self.first = time.perf_counter() - self.t0
            print(f"{self.name}: first token after {self.first:.1f}s", flush=True)
        elif n % 100 == 0:
            dt = time.perf_counter() - self.t0
            print(f"{self.name}: {n} tokens, {dt:.0f}s", flush=True)
        return torch.zeros(input_ids.shape[0], dtype=torch.bool)


def run(model, name):
    t0 = time.perf_counter()
    progress = Progress(name, t0)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            # The model's generation_config.json sets use_cache=false; Docling
            # passes use_cache=True (VlmEngineOptions.use_kv_cache), so do the same.
            use_cache=True,
            stopping_criteria=[progress],
        )
    dt = time.perf_counter() - t0
    new = out[0, n_prompt:]
    text = processor.decode(new, skip_special_tokens=False)
    print(
        f"{name}: {dt:.1f}s, {len(new)} tokens, {len(new) / dt:.1f} tok/s", flush=True
    )
    return {
        "seconds": dt,
        "first_token_s": progress.first,
        "tokens": len(new),
        "text": text,
    }


results = {"pdf": pdf_path, "page": int(page_idx), "threads": threads}

t0 = time.perf_counter()
pt_model = AutoModelForImageTextToText.from_pretrained(REPO, dtype=torch.float32)
pt_model.eval()
results["pytorch_load_s"] = time.perf_counter() - t0
results["pytorch"] = run(pt_model, "pytorch")
del pt_model

if ov_dir == "-":
    import transformers

    results["transformers"] = transformers.__version__
    Path(out_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    sys.exit(0)

from optimum.intel import OVModelForVisualCausalLM  # noqa: E402

t0 = time.perf_counter()
ov_model = OVModelForVisualCausalLM.from_pretrained(
    ov_dir, ov_config={"INFERENCE_NUM_THREADS": threads, "PERFORMANCE_HINT": "LATENCY"}
)
results["openvino_load_s"] = time.perf_counter() - t0
results["openvino"] = run(ov_model, "openvino")

results["identical_text"] = results["pytorch"]["text"] == results["openvino"]["text"]
Path(out_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
print("identical text:", results["identical_text"])
