"""Run Docling's own e2e ground-truth test with a selected layout engine.

This reuses `tests/test_e2e_conversion.py` unchanged: same PDFs, same pipeline
options, same checks against the repo ground truth. Only the layout engine is
replaced (Heron preset, ONNX export for onnxruntime / openvino).

    DOCLING_EXP_LAYOUT_ENGINE=openvino uv run --frozen --no-sync \\
        --project experiments/intel-cpu/uvproject \\
        python -m pytest experiments/intel-cpu/tests/test_e2e_layout_engine.py -s

DOCLING_EXP_LAYOUT_ENGINE: transformers (default, same as upstream),
onnxruntime, openvino.
"""

import os

import pytest

from docling.datamodel.base_models import InputFormat
from docling.datamodel.object_detection_engine_options import (
    OnnxRuntimeObjectDetectionEngineOptions,
    OpenVinoObjectDetectionEngineOptions,
)
from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from tests import test_e2e_conversion

ENGINE = os.getenv("DOCLING_EXP_LAYOUT_ENGINE", "transformers")

if ENGINE not in {"transformers", "onnxruntime", "openvino"}:
    raise ValueError(f"Unknown DOCLING_EXP_LAYOUT_ENGINE: {ENGINE}")


def _converter_with_layout_engine() -> DocumentConverter:
    converter = _upstream_get_converter()
    if ENGINE == "transformers":
        return converter
    pdf_option = converter.format_to_options[InputFormat.PDF]
    assert isinstance(pdf_option, PdfFormatOption)
    pipeline_options = pdf_option.pipeline_options
    engine_options = (
        OpenVinoObjectDetectionEngineOptions()
        if ENGINE == "openvino"
        else OnnxRuntimeObjectDetectionEngineOptions()
    )
    pipeline_options.layout_options = LayoutObjectDetectionOptions.from_preset(
        "layout_heron_default", engine_options=engine_options
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, backend=pdf_option.backend
            )
        }
    )


_upstream_get_converter = test_e2e_conversion.get_converter


def test_e2e_pdfs_conversions_with_layout_engine(monkeypatch: pytest.MonkeyPatch):
    print(f"\nlayout engine: {ENGINE}")
    monkeypatch.setattr(
        test_e2e_conversion, "get_converter", _converter_with_layout_engine
    )
    test_e2e_conversion.test_e2e_pdfs_conversions()
