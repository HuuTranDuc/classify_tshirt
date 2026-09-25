"""ONNX Runtime backend for the split defect classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

try:
    from .predict import DefectClassifierBase
except ImportError:
    from predict import DefectClassifierBase


class DefectClassifierONNX(DefectClassifierBase):
    backend_name = "classifier_onnx"

    def __init__(
        self,
        backbone_path: str | Path,
        head_path: str | Path,
        config_path: str | Path | None = None,
        device: str = "cuda",
        batch_size: int = 8,
    ) -> None:
        super().__init__(backbone_path, head_path, config_path, batch_size)

        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda") and "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.backbone = ort.InferenceSession(
            str(self.backbone_path),
            sess_options=options,
            providers=providers,
        )
        self.head = ort.InferenceSession(
            str(self.head_path),
            sess_options=options,
            providers=providers,
        )
        self.providers = {
            "backbone": self.backbone.get_providers(),
            "head": self.head.get_providers(),
        }

    def _encode_batch(self, images: np.ndarray) -> np.ndarray:
        return self.backbone.run(
            [str(self.config["backbone_output"])],
            {
                str(self.config["backbone_input"]): np.ascontiguousarray(
                    images, dtype=np.float32
                )
            },
        )[0]

    def _classify_batch(self, features: np.ndarray) -> np.ndarray:
        return self.head.run(
            [str(self.config["head_output"])],
            {
                str(self.config["head_input"]): np.ascontiguousarray(
                    features, dtype=np.float32
                )
            },
        )[0]


ClassifierONNX = DefectClassifierONNX
