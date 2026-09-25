#!/usr/bin/env python3
"""Common API and CLI for split defect-classifier ONNX/TensorRT inference."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps


def read_image_bgr(path: str | Path) -> np.ndarray | None:
    """Read an EXIF-corrected image as a contiguous HWC BGR uint8 array."""
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
    except (FileNotFoundError, OSError):
        return None


def resize_bgr(
    image_bgr: np.ndarray,
    size: tuple[int, int],
    interpolation: str,
) -> np.ndarray:
    """Resize BGR input with the same Pillow interpolation used for training."""
    pil_interpolation = {
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
    }[interpolation]
    rgb = image_bgr[:, :, ::-1]
    resized = Image.fromarray(rgb).resize(size, pil_interpolation)
    return np.ascontiguousarray(np.asarray(resized)[:, :, ::-1])


@dataclass(frozen=True)
class ClassificationResult:
    method: str
    class_id: int
    class_name: str
    confidence: float


ImageOrImages = np.ndarray | Sequence[np.ndarray]


class DefectClassifierBase:
    """Shared preprocessing, batching and decoding for all backends."""

    backend_name = "classifier"

    def __init__(
        self,
        backbone_path: str | Path,
        head_path: str | Path,
        config_path: str | Path | None = None,
        batch_size: int = 8,
    ) -> None:
        self.backbone_path = Path(backbone_path)
        self.head_path = Path(head_path)
        self.config_path = (
            Path(config_path)
            if config_path is not None
            else self.head_path.parent / "model_config.json"
        )
        for name, path in (
            ("Backbone", self.backbone_path),
            ("Head", self.head_path),
            ("Model config", self.config_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{name} not found: {path}")

        with self.config_path.open(encoding="utf-8") as handle:
            self.config = json.load(handle)

        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.image_size = int(self.config["image_size"])
        self.class_names = tuple(str(name) for name in self.config["class_names"])
        if not self.class_names:
            raise ValueError("class_names must not be empty")

        preprocess = self.config["preprocess"]
        self.mean = np.asarray(preprocess["mean"], dtype=np.float32)
        self.std = np.asarray(preprocess["std"], dtype=np.float32)
        self.interpolation = str(preprocess["interpolation"])

    def _encode_batch(self, images: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _classify_batch(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("expected a BGR image with shape HxWx3")

        height, width = image_bgr.shape[:2]
        scale = self.image_size / min(height, width)
        resized_width = max(self.image_size, round(width * scale))
        resized_height = max(self.image_size, round(height * scale))
        resized = resize_bgr(
            image_bgr,
            (resized_width, resized_height),
            self.interpolation,
        )
        left = (resized_width - self.image_size) // 2
        top = (resized_height - self.image_size) // 2
        crop = resized[top : top + self.image_size, left : left + self.image_size]
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        tensor = ((rgb - self.mean) / self.std).transpose(2, 0, 1)
        return np.ascontiguousarray(tensor, dtype=np.float32)

    @staticmethod
    def _as_image_list(images: ImageOrImages) -> tuple[list[np.ndarray], bool]:
        if isinstance(images, np.ndarray) and images.ndim == 3:
            return [images], True
        if isinstance(images, np.ndarray) and images.ndim == 4:
            return list(images), False
        values = list(images)
        if not values:
            raise ValueError("at least one image is required")
        return values, False

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        values = np.exp(logits - logits.max(axis=1, keepdims=True))
        return values / values.sum(axis=1, keepdims=True)

    def predict(
        self,
        images: ImageOrImages,
    ) -> ClassificationResult | list[ClassificationResult]:
        values, single = self._as_image_list(images)
        tensors = np.stack([self.preprocess(image) for image in values])
        results: list[ClassificationResult] = []

        for start in range(0, len(tensors), self.batch_size):
            image_batch = tensors[start : start + self.batch_size]
            features = self._encode_batch(image_batch)
            logits = self._classify_batch(features)
            probabilities = self._softmax(logits)
            class_ids = probabilities.argmax(axis=1)

            for scores, class_id in zip(probabilities, class_ids):
                index = int(class_id)
                if index >= len(self.class_names):
                    raise ValueError(
                        f"model returned class_id {index}, but config has "
                        f"only {len(self.class_names)} class names"
                    )
                results.append(
                    ClassificationResult(
                        method=self.backend_name,
                        class_id=index,
                        class_name=self.class_names[index],
                        confidence=float(scores[index]),
                    )
                )

        return results[0] if single else results


class DefectClassifier:
    """Facade selecting the ONNX or TensorRT split-model backend."""

    def __init__(
        self,
        backbone_path: str | Path,
        head_path: str | Path,
        backend: str = "auto",
        device: str = "cuda",
        batch_size: int = 8,
        config_path: str | Path | None = None,
    ) -> None:
        backbone_path = Path(backbone_path)
        head_path = Path(head_path)
        if backend == "auto":
            suffixes = {backbone_path.suffix.lower(), head_path.suffix.lower()}
            if suffixes == {".engine"}:
                backend = "tensorrt"
            elif suffixes == {".onnx"}:
                backend = "onnx"
            else:
                raise ValueError(
                    "auto backend requires both paths to use .onnx or both to use .engine"
                )

        if backend == "tensorrt":
            try:
                from .predict_rt import DefectClassifierTRT
            except ImportError:
                from predict_rt import DefectClassifierTRT
            self._backend = DefectClassifierTRT(
                backbone_path, head_path, config_path, batch_size
            )
        elif backend == "onnx":
            try:
                from .predict_onnx import DefectClassifierONNX
            except ImportError:
                from predict_onnx import DefectClassifierONNX

            self._backend = DefectClassifierONNX(
                backbone_path, head_path, config_path, device, batch_size
            )
        else:
            raise ValueError("backend must be auto, onnx, or tensorrt")

    def predict(
        self,
        images: ImageOrImages,
    ) -> ClassificationResult | list[ClassificationResult]:
        return self._backend.predict(images)


Classifier = DefectClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to model_config.json (default: next to head)",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "onnx", "tensorrt"),
        default="auto",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("images", type=Path, nargs="+")
    args = parser.parse_args()

    images = [read_image_bgr(path) for path in args.images]
    if any(image is None for image in images):
        missing = [
            str(path)
            for path, image in zip(args.images, images)
            if image is None
        ]
        raise FileNotFoundError(f"one or more images could not be read: {missing}")

    classifier = DefectClassifier(
        args.backbone,
        args.head,
        args.backend,
        args.device,
        args.batch_size,
        args.config,
    )
    results = classifier.predict(images)
    assert isinstance(results, list)
    for path, result in zip(args.images, results):
        print(
            f"{path}: {result.class_name} "
            f"confidence={result.confidence:.4f} "
            f"class_id={result.class_id} backend={result.method}"
        )


if __name__ == "__main__":
    main()
