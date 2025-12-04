# ocr.py
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract

from poker.capture import load_config, ScreenCapture
from poker.models import SeatState, TableState
from poker.card_templates import load_all_templates, match_best_template

# Load rank & suit templates
RANK_TEMPLATES, SUIT_TEMPLATES = load_all_templates()

# Whitelist of lettters for Tesseract
STATUS_WHITELIST = "ABCDEFGHIKLMNORSTUYabcdefghiklmnorstuy,.: "

### ROI helpers

def crop_roi(frame: np.ndarray, roi: List[int]) -> np.ndarray:
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


# Use LSTM engine only
BASE_CONFIG = "--oem 1"

def ocr_text_fast(img, whitelist: str, psm: int = 7) -> str:
    """OCR helper with OEM=1 and a tight whitelist."""
    proc = preprocess_for_ocr(img)
    cfg = f"{BASE_CONFIG} --psm {psm} -c tessedit_char_whitelist={whitelist}"
    txt = pytesseract.image_to_string(proc, config=cfg)
    return txt.strip()


def ocr_amount_fast(img, max_reasonable: float = 100000.0):
    """Digits-only OCR for chip amounts."""
    txt = ocr_text_fast(img, "0123456789.,$", psm=7)
    txt = txt.replace(",", "").replace("$", "")

    m = re.search(r"(\d+(\.\d+)?)", txt)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None

    if val <= 0 or val > max_reasonable:
        return None
    return val


### Game status

def roi_has_card(img: np.ndarray, var_thresh: float = 200.0) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.var()) > var_thresh


def detect_button(img: np.ndarray) -> bool:
    if img is None or img.size == 0:
        return False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([20, 80, 80])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    ratio = mask.mean() / 255.0
    return ratio > 0.08


def normalize_status(text: str) -> Optional[str]:
    t = text.lower().strip()
    if not t:
        return None
    if "fold" in t:
        return "fold"
    if "check" in t:
        return "check"
    if "call" in t:
        return "call"
    if "bet" in t:
        return "bet"
    if "rais" in t:
        return "raise"
    if "sit" in t:
        return "sit_out"
    if "all" in t:
        return "allin"
    return None


def infer_street(board_card_imgs: List[np.ndarray]) -> str:
    present = sum(1 for img in board_card_imgs if roi_has_card(img))
    if present == 0:
        return "preflop"
    if present == 3:
        return "flop"
    if present == 4:
        return "turn"
    if present == 5:
        return "river"
    return "unknown"


def get_seat_status(status_img, stack_img) -> tuple[str | None, str, str]:
    # Primary status in the name/status area
    status_raw = ocr_text_fast(status_img, STATUS_WHITELIST, psm=7)
    status_norm = normalize_status(status_raw)

    if status_norm == "sit_out":
        return status_norm, status_raw, ""

    # 2) Special case: 'Sitting Out' can appear where the stack usually is
    stack_text_raw = ocr_text_fast(stack_img, "SgintuO ", psm=7)  # minimal letters for 'Sitting Out'
    stack_status_norm = normalize_status(stack_text_raw)

    if stack_status_norm == "sit_out":
        return stack_status_norm, status_raw, stack_text_raw

    return status_norm, status_raw, stack_text_raw


### Card template matching

def card_present(img, var_thresh: float = 150.0) -> bool:
    """Heuristic: does this ROI look like it contains a card?"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.var()) > var_thresh


def prep_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return g


def scan_templates(roi_gray: np.ndarray, templates: dict[str, np.ndarray]):
    """
    Slide each template over the ROI and return (best_label, best_score).
    Works even if ROI height differs (hero vs board).
    """
    best_label = "?"
    best_score = -1.0

    for label, tmpl in templates.items():
        th, tw = tmpl.shape[:2]

        # Skip obviously invalid cases
        if roi_gray.shape[0] < th or roi_gray.shape[1] < tw:
            # Template bigger than ROI; resize template down
            scale = min(roi_gray.shape[1] / tw, roi_gray.shape[0] / th)
            if scale <= 0:
                continue
            tmpl_resized = cv2.resize(
                tmpl, (int(tw * scale), int(th * scale)), interpolation=cv2.INTER_AREA
            )
        else:
            tmpl_resized = tmpl

        res = cv2.matchTemplate(roi_gray, tmpl_resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if max_val > best_score:
            best_score = max_val
            best_label = label

    return best_label, best_score


# ---------------------- Table state extraction ----------------------


def extract_table_state(frame: np.ndarray, cfg: Dict[str, Any]) -> TableState:
    hero_cards_imgs = [crop_roi(frame, roi) for roi in cfg["hero_cards"]]
    board_cards_imgs = [crop_roi(frame, roi) for roi in cfg["board_cards"]]

    hero_cards = [recognize_card(img) for img in hero_cards_imgs]
    board_cards = [recognize_card(img) for img in board_cards_imgs]

    street = infer_street(board_cards_imgs)

    pot_img = crop_roi(frame, cfg["pot_text"])
    total_pot_img = crop_roi(frame, cfg["total_pot_text"])

    pot_size = ocr_amount(pot_img) or 0.0
    total_pot = ocr_amount(total_pot_img) or 0.0

    seats: List[SeatState] = []
    button_seat: Optional[int] = None

    for seat_id, seat_cfg in enumerate(cfg["seats"]):
        name = seat_cfg["name"]

        stack_img = crop_roi(frame, seat_cfg["stack"])
        bet_img = crop_roi(frame, seat_cfg["bet"])
        status_img = crop_roi(frame, seat_cfg["status_roi"])
        button_img = crop_roi(frame, seat_cfg["button_roi"])
        card_region_img = crop_roi(frame, seat_cfg["card_region"])

        stack_val = ocr_amount(stack_img)
        bet_val = ocr_amount(bet_img)

        status_raw = ocr_text(
            status_img,
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ",
            psm=7,
        )
        status_norm = normalize_status(status_raw)

        has_button = detect_button(button_img)
        if has_button:
            button_seat = seat_id

        has_cards = roi_has_card(card_region_img)
        is_hero = seat_id == 0
        is_sitting_out = status_norm == "sit_out"
        is_active = has_cards and (status_norm not in ("fold", "sit_out"))

        seat_state = SeatState(
            seat_id=seat_id,
            name=name,
            stack=stack_val,
            bet=bet_val,
            is_hero=is_hero,
            has_cards=has_cards,
            is_active=is_active,
            position=None,
            last_status=status_norm if status_norm not in ("sit_out", None) else None,
            is_sitting_out=is_sitting_out,
        )
        seats.append(seat_state)

    table_state = TableState(
        street=street,
        hero_cards=hero_cards,
        board_cards=board_cards,
        pot_size=pot_size,
        total_pot=total_pot,
        button_seat=button_seat,
        seats=seats,
    )

    return table_state


# ---------------------- Demo / manual test loop ----------------------


def main():
    cfg = load_config()
    cap = ScreenCapture(cfg)

    TARGET_FPS = 10
    FRAME_DELAY = 1.0 / TARGET_FPS
    OCR_EVERY = 2

    frame_idx = 0
    last_state: Optional[TableState] = None

    while True:
        start = time.time()
        frame = cap.grab_table_frame()

        # Run full OCR only every OCR_EVERY frames
        if frame_idx % OCR_EVERY == 0 or last_state is None:
            last_state = extract_table_state(frame, cfg)
            print(last_state)  # only when recomputed

        state = last_state

        debug = frame.copy()
        if state is not None:
            cv2.putText(
                debug,
                f"street: {state.street}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                debug,
                f"pot: {state.pot_size:.2f}  total: {state.total_pot:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            if state.button_seat is not None:
                seat_cfg = cfg["seats"][state.button_seat]
                x, y, w, h = seat_cfg["button_roi"]
                cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("Table + OCR debug", debug)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        frame_idx += 1

        elapsed = time.time() - start
        if elapsed < FRAME_DELAY:
            time.sleep(FRAME_DELAY - elapsed)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
