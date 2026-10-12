from pathlib import Path

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.object_detection_engine_options import (
    OnnxRuntimeObjectDetectionEngineOptions,
    OpenVinoObjectDetectionEngineOptions,
)
from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions
from docling.datamodel.stage_model_specs import OBJECT_DETECTION_LAYOUT_HERON
from docling.models.inference_engines.object_detection.base import (
    ObjectDetectionEngineInput,
    ObjectDetectionEngineType,
)
from docling.models.inference_engines.object_detection.factory import (
    create_object_detection_engine,
)


def test_openvino_engine_options_resolve_from_dict():
    layout_options = LayoutObjectDetectionOptions.model_validate(
        {
            "engine_options": {
                "engine_type": "openvino",
                "performance_hint": "THROUGHPUT",
                "num_streams": 2,
            }
        }
    )

    engine_options = layout_options.engine_options
    assert isinstance(engine_options, OpenVinoObjectDetectionEngineOptions)
    assert engine_options.performance_hint == "THROUGHPUT"
    assert engine_options.num_streams == 2


def test_openvino_heron_uses_onnx_export():
    config = OBJECT_DETECTION_LAYOUT_HERON.model_spec.get_engine_config(
        ObjectDetectionEngineType.OPENVINO
    )
    assert config.repo_id == "docling-project/docling-layout-heron-onnx"
    assert config.extra_config["model_filename"] == "model.onnx"


def test_openvino_matches_onnxruntime_on_heron():
    pytest.importorskip("openvino")
    pytest.importorskip("onnxruntime")
    import pypdfium2

    pdf = pypdfium2.PdfDocument(Path("tests/data/pdf/sources/2206.01062.pdf"))
    image = pdf[0].render(scale=1.0).to_pil()
    batch = [ObjectDetectionEngineInput(image=image)]
    accelerator_options = AcceleratorOptions(device="cpu", num_threads=2)

    outputs = {}
    for options in (
        OnnxRuntimeObjectDetectionEngineOptions(),
        OpenVinoObjectDetectionEngineOptions(),
    ):
        engine = create_object_detection_engine(
            options=options,
            model_spec=OBJECT_DETECTION_LAYOUT_HERON.model_spec,
            accelerator_options=accelerator_options,
        )
        engine.initialize()
        outputs[options.engine_type] = engine.predict_batch(batch)[0]

    reference = outputs[ObjectDetectionEngineType.ONNXRUNTIME]
    candidate = outputs[ObjectDetectionEngineType.OPENVINO]
    assert reference.label_ids
    assert candidate.label_ids == reference.label_ids
    assert candidate.scores == pytest.approx(reference.scores, abs=1e-3)
    for got, expected in zip(candidate.bboxes, reference.bboxes):
        assert got == pytest.approx(expected, abs=1.0)
