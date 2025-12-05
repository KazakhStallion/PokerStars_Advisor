# card_templates.py
from pathlib import Path
from typing import Dict, Tuple
import cv2
import numpy as np

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

RANK_LABELS = {
    "A": "A",
    "K": "K",
    "Q": "Q",
    "J": "J",
    "T": "T",
    "9": "9",
    "8": "8",
    "7": "7",
    "6": "6",
    "5": "5",
    "4": "4",
    "3": "3",
    "2": "2",
}

SUIT_LABELS = {
    "h": "h",  # hearts
    "d": "d",  # diamonds
    "c": "c",  # clubs
    "s": "s",  # spades
}


def _load_templates(subdir: str, label_map: Dict[str, str]) -> Dict[str, np.ndarray]:
    base = TEMPLATES_DIR / subdir
    templates: Dict[str, np.ndarray] = {}

    for p in base.glob("*.*"):
        stem = p.stem
        if stem not in label_map:
            continue

        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Normalize size & contrast
        img = cv2.resize(img, (0, 0), fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
        img = cv2.GaussianBlur(img, (3, 3), 0)

        templates[label_map[stem]] = img

    if not templates:
        raise RuntimeError(f"No templates loaded from {base}")
    return templates


def load_all_templates() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    ranks = _load_templates("rank", RANK_LABELS)
    suits = _load_templates("suit", SUIT_LABELS)
    return ranks, suits


def match_best_template(patch_gray: np.ndarray, templates: Dict[str, np.ndarray]) -> Tuple[str, float]:
    """Return (label, score) for best matching template"""
    best_label = "?"
    best_score = -1.0

    for label, tmpl in templates.items():
        ph, pw = patch_gray.shape
        th, tw = tmpl.shape

        if ph < th or pw < tw:
            # Resize template down
            tmpl_resized = cv2.resize(tmpl, (pw, ph), interpolation=cv2.INTER_AREA)
            patch = patch_gray
        else:
            # Resize patch down
            patch = cv2.resize(patch_gray, (tw, th), interpolation=cv2.INTER_AREA)
            tmpl_resized = tmpl

        res = cv2.matchTemplate(patch, tmpl_resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if max_val > best_score:
            best_score = max_val
            best_label = label

    return best_label, best_score
