import cv2
import base64
import numpy as np

from server.config import RENDER_WIDTH, RENDER_HEIGHT

def encode_frame(frame: np.ndarray, quality: int = 75) -> str:
    """
    frame: HxWx3 uint8 RGB array (from dm_control render)
    Returns: base64-encoded JPEG string
    """
    # dm_control returns RGB; OpenCV expects BGR
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (RENDER_WIDTH, RENDER_HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode(".jpg", bgr, encode_params)
    return base64.b64encode(buffer).decode("utf-8")

def encode_all_frames(frames: list[np.ndarray]) -> list[str]:
    return [encode_frame(f) for f in frames]
