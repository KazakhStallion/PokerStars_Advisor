# capture.py
import json
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
from mss import mss

CONFIG_PATH = Path("config_pokerstars.json")


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


class ScreenCapture:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.table_roi = cfg["table_roi"]  # [x, y, w, h]
        self.sct = mss()

    def grab_table_frame(self) -> np.ndarray:
        x, y, w, h = self.table_roi
        monitor = {"top": y, "left": x, "width": w, "height": h}
        sct_img = self.sct.grab(monitor)
        frame = np.array(sct_img)[:, :, :3]  # BGRA → BGR
        return frame


def debug_show_frame(frame: np.ndarray, title: str = "frame"):
    cv2.imshow(title, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        raise KeyboardInterrupt
