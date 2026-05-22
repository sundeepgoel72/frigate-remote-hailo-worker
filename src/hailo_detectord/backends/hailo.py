import numpy as np
import cv2

from hailo_detectord.backends.base import DetectorBackend
from hailo_detectord.config import Settings
from hailo_detectord.image import letterbox
from hailo_detectord.models import Detection


def _read_labels(settings: Settings, metadata: dict) -> list[str]:
    if settings.labelmap_path:
        with open(settings.labelmap_path, encoding="utf-8") as label_file:
            labels = [line.strip() for line in label_file if line.strip()]
            if labels:
                return labels
    if label_map := metadata.get("labelMap"):
        return [label_map[str(i)] for i in range(len(label_map))]
    return list(settings.labels)


class HailoBackend(DetectorBackend):
    name = "hailo"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.metadata = settings.metadata()
        self.labels = _read_labels(settings, self.metadata)

        if not settings.model_path:
            raise RuntimeError("HAILO_MODEL_PATH must point to the Frigate-trained .hef model")

        try:
            from hailo_platform import FormatType, HailoSchedulingAlgorithm, HEF, VDevice
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "python3-hailort / hailo_platform is not available. "
                "Install the HailoRT Python bindings on the RPi5 or run outside a venv "
                "that hides system site packages."
            ) from exc

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        self.hef = HEF(settings.model_path)
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(settings.model_path)
        self.infer_model.set_batch_size(1)
        self.infer_model.input().set_format_type(FormatType.UINT8)
        self.input_shape = self.hef.get_input_vstream_infos()[0].shape
        self.model_h, self.model_w = self.input_shape[:2]
        self.configured_infer_model = self.infer_model.configure()
        self.configured = self.configured_infer_model.__enter__()

    def _create_bindings(self):
        output_buffers = {}
        for output_info in self.hef.get_output_vstream_infos():
            output = self.infer_model.output(output_info.name)
            dtype_name = str(output_info.format.type).split(".")[-1].lower()
            dtype = getattr(np, dtype_name, np.float32)
            output_buffers[output_info.name] = np.empty(output.shape, dtype=dtype)
        return self.configured.create_bindings(output_buffers=output_buffers)

    def _run_hailo(self, tensor: np.ndarray):
        bindings = self._create_bindings()
        bindings.input().set_buffer(tensor)
        self.configured.wait_for_async_ready(timeout_ms=10000)
        job = self.configured.run_async([bindings], lambda *args, **kwargs: None)
        job.wait(10000)

        output_names = bindings._output_names
        if len(output_names) == 1:
            return bindings.output().get_buffer()
        return {name: bindings.output(name).get_buffer() for name in output_names}

    def _iter_nms_detections(self, result):
        if isinstance(result, dict):
            if len(result) != 1:
                raise RuntimeError(
                    "Multiple raw Hailo outputs were returned. Use a Frigate-trained HEF "
                    "with Hailo NMS output or add model-specific postprocessing."
                )
            result = next(iter(result.values()))

        if isinstance(result, np.ndarray) and result.ndim >= 1 and result.shape[0] == 1:
            result = result[0]

        for class_id, detection_set in enumerate(result):
            detections = np.asarray(detection_set)
            if detections.size == 0:
                continue
            detections = detections.reshape(-1, detections.shape[-1])
            for det in detections:
                if det.shape[0] < 5:
                    continue
                score = float(det[4])
                if score < self.settings.confidence_threshold:
                    continue
                yield class_id, score, det[:4]

    def _to_pixels(
        self,
        box: np.ndarray,
        image_w: int,
        image_h: int,
        scale: float,
        x_offset: int,
        y_offset: int,
    ) -> tuple[int, int, int, int]:
        if self.settings.bbox_order == "xyxy":
            x_min, y_min, x_max, y_max = [float(value) for value in box]
        else:
            y_min, x_min, y_max, x_max = [float(value) for value in box]

        if max(x_min, y_min, x_max, y_max) <= 1.5:
            x_min *= self.model_w
            x_max *= self.model_w
            y_min *= self.model_h
            y_max *= self.model_h

        x_min = (x_min - x_offset) / scale
        x_max = (x_max - x_offset) / scale
        y_min = (y_min - y_offset) / scale
        y_max = (y_max - y_offset) / scale

        return (
            max(0, min(image_w, round(x_min))),
            max(0, min(image_h, round(y_min))),
            max(0, min(image_w, round(x_max))),
            max(0, min(image_h, round(y_max))),
        )

    def detect(self, image: np.ndarray) -> list[Detection]:
        image_h, image_w = image.shape[:2]
        padded, scale, x_offset, y_offset = letterbox(image, self.model_w, self.model_h)
        if self.settings.input_pixel_format == "rgb":
            padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = np.expand_dims(np.ascontiguousarray(padded), axis=0)
        result = self._run_hailo(tensor)

        predictions = []
        for class_id, confidence, box in self._iter_nms_detections(result):
            label = self.labels[class_id] if class_id < len(self.labels) else str(class_id)
            x_min, y_min, x_max, y_max = self._to_pixels(
                box, image_w, image_h, scale, x_offset, y_offset
            )
            if x_max <= x_min or y_max <= y_min:
                continue
            predictions.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                )
            )
        return predictions

    def close(self) -> None:
        if hasattr(self, "configured_infer_model"):
            self.configured_infer_model.__exit__(None, None, None)
        if hasattr(self, "target"):
            self.target.release()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
