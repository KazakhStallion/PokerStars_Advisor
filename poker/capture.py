import json
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
from mss import mss

CONFIG_PATH = Path("config_pokerstars.json")

# Absolute screen coords of the PokerStars table window
TABLE_LEFT = 959     # pixels from left edge of monitor
TABLE_TOP = 603      # pixels from top edge of monitor
TABLE_WIDTH = 1429   # window width in pixels
TABLE_HEIGHT = 985   # window height in pixels


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


class ScreenCapture:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.sct = mss()

    def grab_table_frame(self) -> np.ndarray:
        monitor = {
            "top": TABLE_TOP,
            "left": TABLE_LEFT,
            "width": TABLE_WIDTH,
            "height": TABLE_HEIGHT,
        }
        sct_img = self.sct.grab(monitor)
        frame = np.array(sct_img)[:, :, :3]  # BGRA → BGR
        return frame


def debug_show_frame(frame: np.ndarray, title: str = "frame"):
    cv2.imshow(title, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        raise KeyboardInterrupt
