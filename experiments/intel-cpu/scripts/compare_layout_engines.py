"""Strict output comparison of two layout engines on the e2e test PDFs.

Converts every PDF of tests/test_e2e_conversion.py (same pipeline options)
with engine A and engine B and compares markdown, doctags, and layout cluster
boxes. Run from the repository root:

    uv run --frozen --no-sync --project experiments/intel-cpu/uvproject python \\
        experiments/intel-cpu/scripts/compare_layout_engines.py onnxruntime openvino
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from docling.datamodel.base_models import InputFormat
from docling.datamodel.object_detection_engine_options import (
    OnnxRuntimeObjectDetectionEngineOptions,
    OpenVinoObjectDetectionEngineOptions,
)
from docling.datamodel.pipeline_options import (
    LayoutObjectDetectionOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from tests import test_e2e_conversion


def converter_for(engine: str) -> DocumentConverter:
    converter = test_e2e_conversion.get_converter()
    if engine == "transformers":
        return converter
    pdf_option = converter.format_to_options[InputFormat.PDF]
    assert isinstance(pdf_option, PdfFormatOption)
    options = pdf_option.pipeline_options
    engine_options = (
        OpenVinoObjectDetectionEngineOptions()
        if engine == "openvino"
        else OnnxRuntimeObjectDetectionEngineOptions()
    )
    options.layout_options = LayoutObjectDetectionOptions.from_preset(
        "layout_heron_default", engine_options=engine_options
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=options, backend=pdf_option.backend
            )
        }
    )


def snapshot(conv_res) -> dict:
    doc = conv_res.document
    clusters = []
    for page in conv_res.pages:
        if page.predictions.layout is None:
            continue
        for c in page.predictions.layout.clusters:
            clusters.append(
                (page.page_no, str(c.label), c.bbox.l, c.bbox.t, c.bbox.r, c.bbox.b)
            )
    return {
        "md": doc.export_to_markdown(),
        "doctags": doc.export_to_doctags(),
        "clusters": clusters,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine_a")
    parser.add_argument("engine_b")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    pdfs = test_e2e_conversion.get_pdf_paths()
    snaps = {}
    for engine in (args.engine_a, args.engine_b):
        converter = converter_for(engine)
        snaps[engine] = {p.name: snapshot(converter.convert(p)) for p in pdfs}

    rows = []
    for p in pdfs:
        a, b = snaps[args.engine_a][p.name], snaps[args.engine_b][p.name]
        ca, cb = a["clusters"], b["clusters"]
        same_labels = [x[:2] for x in ca] == [x[:2] for x in cb]
        max_dx = (
            max(
                (abs(u - v) for x, y in zip(ca, cb) for u, v in zip(x[2:], y[2:])),
                default=0.0,
            )
            if same_labels
            else None
        )
        rows.append(
            {
                "document": p.name,
                "markdown_identical": a["md"] == b["md"],
                "doctags_identical": a["doctags"] == b["doctags"],
                "clusters": [len(ca), len(cb)],
                "cluster_labels_identical": same_labels,
                "max_box_diff_pt": max_dx,
            }
        )

    header = (
        "| document | markdown | doctags | clusters A/B | labels | max box diff (pt) |"
    )
    print(header)
    print("|---|---|---|---|---|---:|")
    for r in rows:
        diff = "-" if r["max_box_diff_pt"] is None else f"{r['max_box_diff_pt']:.3f}"
        print(
            f"| {r['document']} | {'same' if r['markdown_identical'] else 'DIFF'} "
            f"| {'same' if r['doctags_identical'] else 'DIFF'} "
            f"| {r['clusters'][0]}/{r['clusters'][1]} "
            f"| {'same' if r['cluster_labels_identical'] else 'DIFF'} | {diff} |"
        )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
