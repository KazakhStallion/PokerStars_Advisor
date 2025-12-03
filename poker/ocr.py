# ocr.py
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract

from capture import load_config, ScreenCapture
from models import SeatState, TableState
from card_templates import load_all_templates, match_best_template

# Load rank / suit templates once at import
RANK_TEMPLATES, SUIT_TEMPLATES = load_all_templates()


# ---------------------- Generic ROI helpers ----------------------


def crop_roi(frame: np.ndarray, roi: List[int]) -> np.ndarray:
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def ocr_text(img: np.ndarray, whitelist: str, psm: int = 7) -> str:
    proc = preprocess_for_ocr(img)
    config = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
    txt = pytesseract.image_to_string(proc, config=config)
    return txt.strip()


def ocr_amount(img: np.ndarray) -> Optional[float]:
    txt = ocr_text(img, "0123456789.,$", psm=7)
    txt = txt.replace(",", "").replace("$", "")
    m = re.search(r"(\d+(\.\d+)?)", txt)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# ---------------------- Presence / status heuristics ----------------------


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
    if "raise" in t or "rais" in t:
        return "raise"
    if "sit" in t:
        return "sit_out"
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


# ---------------------- Card template matching ----------------------


def split_rank_suit(card_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = card_img.shape[:2]

    x0 = 0
    x1 = int(0.40 * w)

    y0_rank = 0
    y1_rank = int(0.40 * h)

    y0_suit = int(0.35 * h)
    y1_suit = int(0.80 * h)

    rank_patch = card_img[y0_rank:y1_rank, x0:x1]
    suit_patch = card_img[y0_suit:y1_suit, x0:x1]

    return rank_patch, suit_patch


def _prep_gray(patch: np.ndarray) -> np.ndarray:
    if patch.ndim == 3:
        g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        g = patch
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return g


def recognize_card(card_img: np.ndarray, score_thresh: float = 0.6) -> str:
    if card_img is None or card_img.size == 0:
        return "??"

    rank_patch, suit_patch = split_rank_suit(card_img)
    if rank_patch.size == 0 or suit_patch.size == 0:
        return "??"

    rank_gray = _prep_gray(rank_patch)
    suit_gray = _prep_gray(suit_patch)

    rank_label, rank_score = match_best_template(rank_gray, RANK_TEMPLATES)
    suit_label, suit_score = match_best_template(suit_gray, SUIT_TEMPLATES)

    if rank_score < score_thresh or suit_score < score_thresh:
        return "??"

    return f"{rank_label}{suit_label}"


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
