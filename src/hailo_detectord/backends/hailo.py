import numpy as np
import cv2
from threading import RLock

from hailo_detectord.backends.base import DetectorBackend
from hailo_detectord.config import Settings
from hailo_detectord.image import letterbox
from hailo_detectord.models import Classification, Detection


def _read_labels(settings: Settings, metadata: dict) -> list[str]:
    if settings.labelmap_path:
        with open(settings.labelmap_path, encoding="utf-8") as label_file:
            labels = [line.strip() for line in label_file if line.strip()]
            if labels:
                return labels
    if label_map := metadata.get("labelMap"):
        return [label_map[str(i)] for i in range(len(label_map))]
    return list(settings.labels)


def _read_label_file(path: str | None) -> list[str]:
    if not path:
        return []
    with open(path, encoding="utf-8") as label_file:
        return [line.strip() for line in label_file if line.strip()]


class HailoConfiguredModel:
    def __init__(self, target, hef, infer_model, configured_infer_model) -> None:
        self.target = target
        self.hef = hef
        self.infer_model = infer_model
        self.configured_infer_model = configured_infer_model
        self.configured = configured_infer_model.__enter__()
        self.input_shape = hef.get_input_vstream_infos()[0].shape
        self.model_h, self.model_w = self.input_shape[:2]

    def create_bindings(self):
        output_buffers = {}
        for output_info in self.hef.get_output_vstream_infos():
            output = self.infer_model.output(output_info.name)
            dtype_name = str(output_info.format.type).split(".")[-1].lower()
            dtype = getattr(np, dtype_name, np.float32)
            output_buffers[output_info.name] = np.empty(output.shape, dtype=dtype)
        return self.configured.create_bindings(output_buffers=output_buffers)

    def run(self, tensor: np.ndarray):
        bindings = self.create_bindings()
        bindings.input().set_buffer(tensor)
        self.configured.wait_for_async_ready(timeout_ms=10000)
        job = self.configured.run_async([bindings], lambda *args, **kwargs: None)
        job.wait(10000)

        output_names = bindings._output_names
        if len(output_names) == 1:
            return bindings.output().get_buffer()
        return {name: bindings.output(name).get_buffer() for name in output_names}

    def close(self) -> None:
        self.configured_infer_model.__exit__(None, None, None)


class HailoBackend(DetectorBackend):
    name = "hailo"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.metadata = settings.metadata()
        self.labels = _read_labels(settings, self.metadata)
        self._lock = RLock()

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
        self.detector = self._configure_model(settings.model_path, FormatType)
        self.input_shape = self.detector.input_shape
        self.model_h, self.model_w = self.detector.model_h, self.detector.model_w
        self.greenhouse = None
        self.greenhouse_labels = _read_label_file(settings.greenhouse_labelmap_path) or list(
            settings.greenhouse_labels
        )

        if settings.greenhouse_backend == "hailo" and settings.greenhouse_autoload:
            self.load_greenhouse()

    def _configure_model(self, path: str, format_type) -> HailoConfiguredModel:
        hef = self.hef if path == self.settings.model_path else self._create_hef(path)
        infer_model = self.target.create_infer_model(path)
        infer_model.set_batch_size(1)
        infer_model.input().set_format_type(format_type.UINT8)
        configured_infer_model = infer_model.configure()
        return HailoConfiguredModel(self.target, hef, infer_model, configured_infer_model)

    def _create_hef(self, path: str):
        from hailo_platform import HEF

        return HEF(path)

    def load_greenhouse(self) -> dict:
        with self._lock:
            if self.greenhouse is not None:
                return {
                    "success": True,
                    "loaded": True,
                    "model_path": self.settings.greenhouse_model_path,
                    "backend": "hailo:greenhouse",
                }

            if self.settings.greenhouse_backend != "hailo":
                raise RuntimeError("Greenhouse Hailo backend is not enabled")
            if not self.settings.greenhouse_model_path:
                raise RuntimeError(
                    "HAILO_GREENHOUSE_MODEL_PATH must point to a classifier .hef when "
                    "HAILO_GREENHOUSE_BACKEND=hailo"
                )

            try:
                from hailo_platform import FormatType
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "python3-hailort / hailo_platform is not available. "
                    "Install the HailoRT Python bindings on the RPi5 or run outside a venv "
                    "that hides system site packages."
                ) from exc

            self.greenhouse = self._configure_model(self.settings.greenhouse_model_path, FormatType)
            return {
                "success": True,
                "loaded": True,
                "model_path": self.settings.greenhouse_model_path,
                "backend": "hailo:greenhouse",
            }

    def unload_greenhouse(self) -> dict:
        with self._lock:
            if self.greenhouse is None:
                return {
                    "success": True,
                    "loaded": False,
                    "model_path": self.settings.greenhouse_model_path,
                    "backend": "hailo:greenhouse",
                }

            self.greenhouse.close()
            self.greenhouse = None
            return {
                "success": True,
                "loaded": False,
                "model_path": self.settings.greenhouse_model_path,
                "backend": "hailo:greenhouse",
            }

    def greenhouse_status(self) -> dict:
        return {
            "success": True,
            "loaded": self.greenhouse is not None,
            "model_path": self.settings.greenhouse_model_path,
            "backend": "hailo:greenhouse" if self.settings.greenhouse_backend == "hailo" else "color",
        }

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
            x_min *= self.detector.model_w
            x_max *= self.detector.model_w
            y_min *= self.detector.model_h
            y_max *= self.detector.model_h

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
        padded, scale, x_offset, y_offset = letterbox(
            image, self.detector.model_w, self.detector.model_h
        )
        if self.settings.input_pixel_format == "rgb":
            padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = np.expand_dims(np.ascontiguousarray(padded), axis=0)
        result = self.detector.run(tensor)

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

    def classify_greenhouse(self, image: np.ndarray) -> list[Classification]:
        with self._lock:
            if self.greenhouse is None:
                raise RuntimeError("Greenhouse Hailo classifier is not loaded")

            resized = cv2.resize(
                image,
                (self.greenhouse.model_w, self.greenhouse.model_h),
                interpolation=cv2.INTER_AREA,
            )
            if self.settings.greenhouse_input_pixel_format == "rgb":
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            tensor = np.expand_dims(np.ascontiguousarray(resized), axis=0)
            result = self.greenhouse.run(tensor)
            scores = self._classification_scores(result)
            count = min(self.settings.greenhouse_top_k, scores.size)
            top_indices = np.argsort(scores)[::-1][:count]

            return [
                Classification(
                    label=self.greenhouse_labels[index]
                    if index < len(self.greenhouse_labels)
                    else str(index),
                    confidence=float(scores[index]),
                )
                for index in top_indices
            ]

    def _classification_scores(self, result) -> np.ndarray:
        if isinstance(result, dict):
            if len(result) != 1:
                raise RuntimeError(
                    "Multiple greenhouse classifier outputs were returned. Add "
                    "model-specific postprocessing for this HEF."
                )
            result = next(iter(result.values()))

        scores = np.asarray(result, dtype=np.float32).reshape(-1)
        if scores.size == 0:
            return scores

        if np.any(scores < 0) or np.any(scores > 1.0) or not np.isclose(
            float(np.sum(scores)), 1.0, atol=0.05
        ):
            scores = scores - np.max(scores)
            exp_scores = np.exp(scores)
            scores = exp_scores / np.sum(exp_scores)

        return scores

    def close(self) -> None:
        if getattr(self, "greenhouse", None) is not None:
            self.greenhouse.close()
        if hasattr(self, "detector"):
            self.detector.close()
        if hasattr(self, "target"):
            self.target.release()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
