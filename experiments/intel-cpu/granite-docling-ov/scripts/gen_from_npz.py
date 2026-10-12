"""Generate from saved processor inputs. argv: npz ov_dir|- out_json threads"""

import json
import sys
import time

import numpy as np
import torch
from transformers import AutoProcessor

npz, ov_dir, out, threads = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
torch.set_num_threads(threads)
inputs = {k: torch.from_numpy(v) for k, v in np.load(npz).items()}
processor = AutoProcessor.from_pretrained("ibm-granite/granite-docling-258M")
if ov_dir == "-":
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        "ibm-granite/granite-docling-258M", dtype=torch.float32
    ).eval()
else:
    from optimum.intel import OVModelForVisualCausalLM

    model = OVModelForVisualCausalLM.from_pretrained(
        ov_dir,
        ov_config={"INFERENCE_NUM_THREADS": threads, "PERFORMANCE_HINT": "LATENCY"},
    )
t0 = time.perf_counter()
with torch.inference_mode():
    o = model.generate(**inputs, max_new_tokens=8192, do_sample=False, use_cache=True)
dt = time.perf_counter() - t0
new = o[0, inputs["input_ids"].shape[1] :]
text = processor.decode(new, skip_special_tokens=False)
print(f"{dt:.1f}s {len(new)} tokens {len(new) / dt:.1f} tok/s", flush=True)
json.dump({"seconds": dt, "tokens": len(new), "text": text}, open(out, "w"))
