# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""OpenVINO implementation for RT-DETR style object-detection models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import openvino as ov

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.object_detection_engine_options import (
    OpenVinoObjectDetectionEngineOptions,
)
from docling.models.inference_engines.object_detection.base import (
    ObjectDetectionEngineInput,
    ObjectDetectionEngineOutput,
)
from docling.models.inference_engines.object_detection.hf_base import (
    HfObjectDetectionEngineBase,
)

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig

_log = logging.getLogger(__name__)


class OpenVinoObjectDetectionEngine(HfObjectDetectionEngineBase):
    """OpenVINO engine for object detection models.

    Loads the same ONNX export and uses the same HuggingFace preprocessing and
    output handling as the ONNX Runtime engine, so outputs stay comparable.
    """

    def __init__(
        self,
        *,
        options: OpenVinoObjectDetectionEngineOptions,
        model_config: Optional[EngineModelConfig] = None,
        accelerator_options: AcceleratorOptions,
        artifacts_path: Optional[Union[Path, str]] = None,
    ):
        super().__init__(
            options=options,
            model_config=model_config,
            accelerator_options=accelerator_options,
            artifacts_path=artifacts_path,
        )
        self.options: OpenVinoObjectDetectionEngineOptions = options
        self._compiled_model: Optional[ov.CompiledModel] = None
        self._model_path: Optional[Path] = None

    def _resolve_model_artifacts(self) -> tuple[Path, Path]:
        """Resolve model artifacts from artifacts_path or HF download."""
        model_filename = self.options.model_filename
        extra_filename = self._model_config.extra_config.get("model_filename")
        if extra_filename and isinstance(extra_filename, str):
            model_filename = extra_filename

        model_folder = self._resolve_model_folder(
            repo_id=self._repo_id,
            revision=str(self._model_config.revision or "main"),
        )
        model_path = model_folder / model_filename
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model file '{model_filename}' not found: {model_path}"
            )
        return model_folder, model_path

    def _compile_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "PERFORMANCE_HINT": self.options.performance_hint,
            "INFERENCE_NUM_THREADS": self._accelerator_options.num_threads,
        }
        if self.options.num_streams is not None:
            config["NUM_STREAMS"] = self.options.num_streams
        if self.options.cache_dir is not None:
            config["CACHE_DIR"] = str(self.options.cache_dir)
        return config

    def initialize(self) -> None:
        """Compile the ONNX model with OpenVINO and load the preprocessor."""
        import openvino as ov

        _log.info("Initializing OpenVINO object-detection engine")

        model_folder, self._model_path = self._resolve_model_artifacts()
        _log.debug(f"Using ONNX model at {self._model_path}")

        self._processor = self._load_preprocessor(model_folder)
        self._id_to_label = self._load_label_mapping(model_folder)

        core = ov.Core()
        config = self._compile_config()
        self._compiled_model = core.compile_model(
            str(self._model_path), self.options.device, config
        )

        self._initialized = True
        _log.info(
            f"OpenVINO engine ready (device={self.options.device}, config={config})"
        )

    def predict_batch(
        self, input_batch: List[ObjectDetectionEngineInput]
    ) -> List[ObjectDetectionEngineOutput]:
        """Run inference on a batch of inputs."""
        if not input_batch:
            return []
        if self._compiled_model is None or self._processor is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        images = [item.image.convert("RGB") for item in input_batch]
        inputs = self._processor(images=images, return_tensors="np")
        orig_sizes = np.array(
            [[img.width, img.height] for img in images], dtype=np.int64
        )

        # One infer request per call: the layout stage may call from one
        # thread at a time, and a fresh request keeps the engine re-entrant.
        request = self._compiled_model.create_infer_request()
        results = request.infer(
            {
                "images": inputs["pixel_values"],
                "orig_target_sizes": orig_sizes,
            }
        )
        output_tensors = [results[output] for output in self._compiled_model.outputs]

        if len(output_tensors) < 3:
            raise RuntimeError(
                "Expected ONNX model to return at least 3 outputs: "
                "[labels, boxes, scores]"
            )

        labels_batch, boxes_batch, scores_batch = output_tensors[:3]

        return [
            self._build_output(
                input_item=input_item,
                labels=labels_batch[idx],
                scores=scores_batch[idx],
                boxes=boxes_batch[idx],
                apply_score_threshold=True,
            )
            for idx, input_item in enumerate(input_batch)
        ]
