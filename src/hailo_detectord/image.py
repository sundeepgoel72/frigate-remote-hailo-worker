import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise ValueError("request body is empty")
    buffer = np.frombuffer(data, dtype=np.uint8)
    try:
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError("request body is not a decodable image") from exc
    if image is None:
        raise ValueError("request body is not a decodable image")
    return image


def letterbox(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, int, int]:
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w, resized_h = int(source_w * scale), int(source_h * scale)
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
    padded = np.full((height, width, 3), 114, dtype=image.dtype)
    x_offset = (width - resized_w) // 2
    y_offset = (height - resized_h) // 2
    padded[y_offset : y_offset + resized_h, x_offset : x_offset + resized_w] = resized
    return padded, scale, x_offset, y_offset
