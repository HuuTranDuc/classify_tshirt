"""TensorRT backend for the split defect classifier engines."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .predict import DefectClassifierBase
except ImportError:
    from predict import DefectClassifierBase


class _TensorRTEngine:
    def __init__(self, path: Path) -> None:
        try:
            import pycuda.driver as cuda
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError("TensorRT and PyCUDA are required") from exc

        self.cuda = cuda
        self.trt = trt
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {path}")
        self.context = self.engine.create_execution_context()
        names = [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
        ]
        self.inputs = [
            name
            for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.outputs = [
            name
            for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]

    def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        stream = self.cuda.Stream()
        allocations: dict[str, object] = {}
        host_outputs: dict[str, np.ndarray] = {}

        missing = sorted(set(self.inputs).difference(feeds))
        if missing:
            raise KeyError(f"missing TensorRT input(s): {missing}")

        for name in self.inputs:
            data = np.ascontiguousarray(feeds[name])
            if not self.context.set_input_shape(name, data.shape):
                raise ValueError(f"Invalid TensorRT input shape for {name}: {data.shape}")
            device = self.cuda.mem_alloc(data.nbytes)
            allocations[name] = device
            self.context.set_tensor_address(name, int(device))
            self.cuda.memcpy_htod_async(device, data, stream)

        for name in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(
                    f"TensorRT output shape is unresolved for {name}: {shape}"
                )
            dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name)))
            host = np.empty(shape, dtype=dtype)
            device = self.cuda.mem_alloc(host.nbytes)
            allocations[name] = device
            host_outputs[name] = host
            self.context.set_tensor_address(name, int(device))

        if not self.context.execute_async_v3(stream.handle):
            raise RuntimeError("TensorRT execution failed")
        for name, host in host_outputs.items():
            self.cuda.memcpy_dtoh_async(host, allocations[name], stream)
        stream.synchronize()
        return host_outputs


class DefectClassifierTRT(DefectClassifierBase):
    backend_name = "classifier_trt_fp16"

    def __init__(
        self,
        backbone_path: str | Path,
        head_path: str | Path,
        config_path: str | Path | None = None,
        batch_size: int = 8,
    ) -> None:
        super().__init__(backbone_path, head_path, config_path, batch_size)
        import pycuda.driver as cuda

        cuda.init()
        self._cuda_context = cuda.Device(0).retain_primary_context()
        self._cuda_context.push()
        try:
            self.backbone = _TensorRTEngine(self.backbone_path)
            self.head = _TensorRTEngine(self.head_path)
        except Exception:
            self._cuda_context.pop()
            raise

    def _encode_batch(self, images: np.ndarray) -> np.ndarray:
        outputs = self.backbone.run(
            {
                str(self.config["backbone_input"]): np.ascontiguousarray(
                    images, dtype=np.float32
                )
            }
        )
        return outputs[str(self.config["backbone_output"])]

    def _classify_batch(self, features: np.ndarray) -> np.ndarray:
        outputs = self.head.run(
            {
                str(self.config["head_input"]): np.ascontiguousarray(
                    features, dtype=np.float32
                )
            }
        )
        return outputs[str(self.config["head_output"])]

    def __del__(self) -> None:
        context = getattr(self, "_cuda_context", None)
        if context is not None:
            try:
                context.pop()
            except Exception:
                pass


ClassifierTRT = DefectClassifierTRT
