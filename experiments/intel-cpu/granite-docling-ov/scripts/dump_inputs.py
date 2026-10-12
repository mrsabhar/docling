"""Save Granite-Docling processor inputs for one page. argv: pdf out_npz"""

import sys

import numpy as np
import pypdfium2
import transformers
from transformers import AutoProcessor

image = pypdfium2.PdfDocument(sys.argv[1])[0].render(scale=2.0).to_pil().convert("RGB")
processor = AutoProcessor.from_pretrained("ibm-granite/granite-docling-258M")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Convert this page to docling."},
        ],
    }
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="np")
print(transformers.__version__, {k: (v.shape, v.dtype) for k, v in inputs.items()})
np.savez(sys.argv[2], **dict(inputs))
