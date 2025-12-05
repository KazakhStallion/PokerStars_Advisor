# ocr.py
from __future__ import annotations

import os
import re
import time
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
import json
import copy

from poker.capture import load_config, ScreenCapture
from poker.models import SeatState, TableState
from poker.card_templates import load_all_templates

# Load rank & suit templates (same as test_card_templates)
RANK_TEMPLATES, SUIT_TEMPLATES = load_all_templates()

# Whitelist of letters for Tesseract (same as your test)
STATUS_WHITELIST = "ABCDEFGHIKLMNORSTUYabcdefghiklmnorstuy "
STACK_STATUS_WHITELIST = "AaIiLlNnOoSsTtUu g-"

# ---------------------------------------------------------------------
# ROI / OCR helpers

def crop_roi(frame: np.ndarray, roi: List[int]) -> np.ndarray:
    x, y, w, h = roi
    return frame[y : y + h, x : x + w].copy()


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


BASE_CONFIG = "--oem 1"  # LSTM-only


def ocr_text_fast(img, whitelist: str, psm: int = 7) -> str:
    proc = preprocess_for_ocr(img)
    cfg = f"{BASE_CONFIG} --psm {psm} -c tessedit_char_whitelist={whitelist}"
    txt = pytesseract.image_to_string(proc, config=cfg)
    return txt.strip()


def ocr_amount_fast(img, max_reasonable: float = 100000.0):
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

# ---------------------------------------------------------------------
# Status / street helpers

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
    if "allin" in t:
        return "allin"
    return None


def infer_street(board_card_imgs: List[np.ndarray]) -> str:
    present = sum(1 for img in board_card_imgs if board_card_present(img))
    if present == 0:
        return "preflop"
    if present == 3:
        return "flop"
    if present == 4:
        return "turn"
    if present == 5:
        return "river"
    return "unknown"


def get_seat_status(status_img, stack_img) -> tuple[Optional[str], str, str]:
    status_raw = ocr_text_fast(status_img, STATUS_WHITELIST, psm=7)
    status_norm = normalize_status(status_raw)

    stack_status_raw = ocr_text_fast(stack_img, STACK_STATUS_WHITELIST, psm=7)
    stack_status_norm = normalize_status(stack_status_raw)

    if stack_status_norm in ("sit_out", "allin"):
        return stack_status_norm, status_raw, stack_status_raw

    return status_norm, status_raw, stack_status_raw


def dealer_button_score(img: np.ndarray) -> float:
    """
    Score how likely this ROI contains the PokerStars dealer button:
    white circle + red spade.

    Returns a value in [0, 1]. Higher = more likely.
    """
    if img is None or img.size == 0:
        return 0.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # White ring (bright / low saturation)
    white_mask = (v >= 200) & (s <= 40)
    white_ratio = white_mask.mean()

    # Red spade (strong red, high sat, decent brightness)
    red_mask = (
        ((h <= 10) | (h >= 170)) &
        (s >= 90) &
        (v >= 80)
    )
    red_ratio = red_mask.mean()

    return float(white_ratio * red_ratio)


def detect_time_bar(img: np.ndarray) -> bool:
    """
    Detect the yellow→green time bar in the given ROI.
    Used only for bottom_right.timebar_roi (for now).
    """
    if img is None or img.size == 0:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask = (
        (h >= 20) & (h <= 90) &   # yellow/green hue
        (s >= 80) &
        (v >= 80)
    )
    return float(mask.mean()) > 0.10


# ---------------------------------------------------------------------
# Card template matching

def card_present(img, var_thresh: float = 150.0) -> bool:
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
    best_label = "?"
    best_score = -1.0

    for label, tmpl in templates.items():
        th, tw = tmpl.shape[:2]

        if roi_gray.shape[0] < th or roi_gray.shape[1] < tw:
            scale = min(roi_gray.shape[1] / tw, roi_gray.shape[0] / th)
            if scale <= 0:
                continue
            tmpl_resized = cv2.resize(
                tmpl,
                (int(tw * scale), int(th * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            tmpl_resized = tmpl

        res = cv2.matchTemplate(roi_gray, tmpl_resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if max_val > best_score:
            best_score = max_val
            best_label = label

    return best_label, best_score


def has_back_cards(img: np.ndarray) -> bool:
    """
    Detects the two red card backs in a seat's card_region.
    Very rough: look for a lot of reddish, fairly bright pixels.
    """
    if img is None or img.size == 0:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    red1 = (h <= 10)
    red2 = (h >= 170)
    red = (red1 | red2) & (s >= 80) & (v >= 60)

    ratio = red.mean()
    return ratio > 0.10  


def board_card_present(img: np.ndarray) -> bool:
    """
    Detect if a *board* ROI actually has a card.
    Board cards are white rectangular tiles; 
    the felt + faded logo alone should not pass this.
    """
    if img is None or img.size == 0:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # "White" pixels: high value, low saturation
    white = (v >= 220) & (s <= 40)
    ratio = white.mean()
    return ratio > 0.12


# ---------------------------------------------------------------------
# Incremental OCR tracking

def gray_simple(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def roi_changed(prev: Optional[np.ndarray],
                curr: np.ndarray,
                diff_thresh: float = 5.0) -> bool:
    g_curr = gray_simple(curr)
    if prev is None or prev.shape != g_curr.shape:
        return True
    diff = cv2.absdiff(prev, g_curr)
    return float(diff.mean()) > diff_thresh


@dataclass
class OcrTask:
    kind: str                # "pot_text", "total_pot_text", "seat_stack", "seat_bet", "seat_status"
    seat_idx: Optional[int]  # None for pot / total_pot
    roi_key: str             # for debugging
    roi: List[int]


class TableTracker:
    """
    Maintains a cached TableState.
    Uses ROI diffing & small OCR task queue.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.prev_rois: Dict[Tuple[str, Optional[int]], Optional[np.ndarray]] = {}
        self.tasks: List[OcrTask] = []
        self.state: TableState = self._init_empty_state()
        self.acting_seat_id: Optional[int] = None

    # init
    def _init_empty_state(self) -> TableState:
        seats: List[SeatState] = []
        for seat_id, seat_cfg in enumerate(self.cfg["seats"]):
            is_hero = seat_id == 0
            seat_state = SeatState(
                seat_id=seat_id,
                name=seat_cfg["name"],
                stack=None,
                bet=None,
                is_hero=is_hero,
                has_cards=False,
                is_active=False,
                position=None,
                last_status=None,
                is_sitting_out=False,
            )
            seats.append(seat_state)

        return TableState(
            street="preflop",
            hero_cards=[],
            board_cards=[],
            pot_size=0.0,
            total_pot=0.0,
            button_seat=None,
            seats=seats,
        )



    # per-frame update

    def update(self, frame: np.ndarray, max_tasks_per_frame: int = 6) -> None:
        self._enqueue_changed_rois(frame)
        self._run_some_tasks(frame, max_tasks_per_frame)
        self._update_cards_and_street(frame)
        self._update_buttons_and_activity(frame)



    # task scheduling

    def _enqueue_changed_rois(self, frame: np.ndarray) -> None:
        # Pot + total pot
        for key in ("pot_text", "total_pot_text"):
            roi = self.cfg[key]
            img = crop_roi(frame, roi)
            prev = self.prev_rois.get((key, None))
            if roi_changed(prev, img):
                self.prev_rois[(key, None)] = gray_simple(img)
                self._add_task_front(key, None, key, roi)

        # Seats
        for seat_idx, seat_cfg in enumerate(self.cfg["seats"]):
            # Stack
            roi = seat_cfg["stack"]
            img = crop_roi(frame, roi)
            prev = self.prev_rois.get(("stack", seat_idx))
            if roi_changed(prev, img):
                self.prev_rois[("stack", seat_idx)] = gray_simple(img)
                self._add_task("seat_stack", seat_idx, "stack", roi)

            # Bet
            roi = seat_cfg["bet"]
            img = crop_roi(frame, roi)
            prev = self.prev_rois.get(("bet", seat_idx))
            if roi_changed(prev, img):
                self.prev_rois[("bet", seat_idx)] = gray_simple(img)
                self._add_task("seat_bet", seat_idx, "bet", roi)

            # Status
            roi = seat_cfg["status_roi"]
            img = crop_roi(frame, roi)
            prev = self.prev_rois.get(("status", seat_idx))
            if roi_changed(prev, img):
                self.prev_rois[("status", seat_idx)] = gray_simple(img)
                self._add_task("seat_status", seat_idx, "status_roi", roi)

    def _add_task(self,
                  kind: str,
                  seat_idx: Optional[int],
                  roi_key: str,
                  roi: List[int]) -> None:
        self.tasks.append(OcrTask(kind=kind, seat_idx=seat_idx,
                                  roi_key=roi_key, roi=roi))
    
    def _add_task_front(self,
                        kind: str,
                        seat_idx: Optional[int],
                        roi_key: str,
                        roi: List[int]) -> None:
        # High-priority task: insert at front of queue
        self.tasks.insert(0, OcrTask(kind=kind, seat_idx=seat_idx,
                                     roi_key=roi_key, roi=roi))


    def _run_some_tasks(self, frame: np.ndarray, max_tasks: int) -> None:
        n = min(max_tasks, len(self.tasks))
        for _ in range(n):
            task = self.tasks.pop(0)
            self._run_task(frame, task)
            
    def drain_all_tasks(self, frame: np.ndarray) -> None:
        """Run all pending OCR tasks on the current frame."""
        while self.tasks:
            task = self.tasks.pop(0)
            self._run_task(frame, task)

    # OCR handlers

    def _run_task(self, frame: np.ndarray, task: OcrTask) -> None:
        img = crop_roi(frame, task.roi)

        if task.kind == "pot_text":
            val = ocr_amount_fast(img)
            if val is not None:
                self.state.pot_size = val

        elif task.kind == "total_pot_text":
            val = ocr_amount_fast(img)
            if val is not None:
                self.state.total_pot = val

        elif task.kind == "seat_stack":
            seat = self.state.seats[task.seat_idx]
            seat.stack = ocr_amount_fast(img)

        elif task.kind == "seat_bet":
            seat = self.state.seats[task.seat_idx]
            g = gray_simple(img)
            if g.mean() < 10:
                seat.bet = None
            else:
                seat.bet = ocr_amount_fast(img)

        elif task.kind == "seat_status":
            seat_cfg = self.cfg["seats"][task.seat_idx]
            seat = self.state.seats[task.seat_idx]

            status_img = img
            stack_img = crop_roi(frame, seat_cfg["stack"])
            status_norm, status_raw, stack_status_raw = get_seat_status(
                status_img, stack_img
            )

            seat.last_status = (
                status_norm if status_norm not in ("sit_out", None) else None
            )
            seat.is_sitting_out = status_norm == "sit_out"



    # cards + street

    def _update_cards_and_street(self, frame: np.ndarray) -> None:
        cfg = self.cfg

        # Hero cards
        hero_cards: List[Optional[str]] = []
        for roi in cfg["hero_cards"]:
            card_img = crop_roi(frame, roi)
            if not card_present(card_img):
                hero_cards.append(None)
                continue
            gray = prep_gray(card_img)
            r_label, r_score = scan_templates(gray, RANK_TEMPLATES)
            s_label, s_score = scan_templates(gray, SUIT_TEMPLATES)
            card_str = f"{r_label}{s_label}"
            hero_cards.append(card_str)

        self.state.hero_cards = [c for c in hero_cards if c is not None]

        # Update hero seat has_cards flag
        hero_seat = next((s for s in self.state.seats if s.is_hero), None)
        if hero_seat is not None:
            hero_seat.has_cards = len(self.state.hero_cards) == 2

        # Board cards (now gate with board_card_present)
        board_cards_imgs: List[np.ndarray] = []
        board_cards: List[Optional[str]] = []
        for roi in cfg["board_cards"]:
            card_img = crop_roi(frame, roi)
            board_cards_imgs.append(card_img)

            if not board_card_present(card_img):
                board_cards.append(None)
                continue

            gray = prep_gray(card_img)
            r_label, r_score = scan_templates(gray, RANK_TEMPLATES)
            s_label, s_score = scan_templates(gray, SUIT_TEMPLATES)
            card_str = f"{r_label}{s_label}"
            board_cards.append(card_str)

        self.state.board_cards = [c for c in board_cards if c is not None]
        self.state.street = infer_street(board_cards_imgs)


    
    # button, has_cards, active

    def _update_buttons_and_activity(self, frame: np.ndarray) -> None:
        self.state.button_seat = None
        self.acting_seat_id = None

        best_button_score = 0.0
        best_button_seat: Optional[int] = None

        for seat_id, (seat, seat_cfg) in enumerate(
            zip(self.state.seats, self.cfg["seats"])
        ):
            card_region_img = crop_roi(frame, seat_cfg["card_region"])
            seat.has_cards = roi_has_card(card_region_img)

            # dealer button detection
            btn_roi = seat_cfg.get("button_roi")
            if btn_roi:
                btn_img = crop_roi(frame, btn_roi)
                score = dealer_button_score(btn_img)
                if score > best_button_score:
                    best_button_score = score
                    best_button_seat = seat_id

            # time bar / acting seat (bottom_right has the only timebar_roi for now)
            tb_roi = seat_cfg.get("timebar_roi")
            if tb_roi:
                tb_img = crop_roi(frame, tb_roi)
                if detect_time_bar(tb_img):
                    self.acting_seat_id = seat_id

            seat.is_active = seat.has_cards and not (
                seat.last_status in ("fold",) or seat.is_sitting_out
            )

        # Finalize dealer seat after scanning all seats
        if best_button_score > 0.01:
            self.state.button_seat = best_button_seat
        else:
            self.state.button_seat = None



    # hero-to-act

    def hero_to_act(self) -> bool:
        hero = next((s for s in self.state.seats if s.is_hero), None)
        if hero is None:
            return False

        # If we have a detected acting_seat_id and it matches hero, that's definitive.
        if self.acting_seat_id is not None:
            return self.acting_seat_id == hero.seat_id

        # Fallback: old logical checks if we couldn't see the time bar.
        if hero.is_sitting_out or not hero.has_cards:
            return False
        if hero.last_status in ("fold", "allin"):
            return False
        return True

    
    # Ssnapshot
    def snapshot(self) -> TableState:
        return self.state
    
    def snapshot_copy(self) -> TableState:
        """Deep copy so future updates don’t mutate the saved snapshot."""
        return copy.deepcopy(self.state)


# ---------------------------------------------------------------------
# Main loop

def write_state_json(state: TableState, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(state), f, indent=2)


def log_game_state(state: TableState,
                   frame: np.ndarray,
                   base_dir: str = "logs",
                   tag: str = "") -> None:
    """
    Save a single 'GTO-ready' snapshot:
      - JSON table state
      - Corresponding frame PNG (for debugging / replay)
    """
    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    prefix = f"{tag}_" if tag else ""

    json_path = os.path.join(base_dir, f"{prefix}state_{ts}.json")
    img_path = os.path.join(base_dir, f"{prefix}frame_{ts}.png")

    write_state_json(state, json_path)
    cv2.imwrite(img_path, frame)

    print(f"[LOG] Saved game state -> {json_path}")
    print(f"[LOG] Saved frame      -> {img_path}")


def main():    
    cfg = load_config()
    cap = ScreenCapture(cfg)
    tracker = TableTracker(cfg)

    # Find hero seat (hero_bottom)
    hero_seat_id = None
    for i, seat_cfg in enumerate(cfg["seats"]):
        if seat_cfg["name"] == "hero_bottom":
            hero_seat_id = i
            break

    if hero_seat_id is None:
        print("[WARN] hero_bottom seat not found in config; auto logging disabled.")

    TARGET_FPS = 10
    FRAME_DELAY = 1.0 / TARGET_FPS

    print("Press 'q' to quit, 'r' to force full resync, 's' to log snapshot.")

    frame_idx = 0
    prev_hero_bar = False          # did hero have the timebar last frame?
    last_logged_street: Optional[str] = None  # to avoid double-logging same street

    while True:
        start = time.time()
        frame = cap.grab_table_frame()

        # Default: light OCR, you can tune this
        max_tasks = 10

        tracker.update(frame, max_tasks_per_frame=max_tasks)
        state = tracker.state

        # Is hero's timebar ON now? (we require actual timebar, not heuristic hero_to_act)
        hero_bar_now = (
            hero_seat_id is not None
            and tracker.acting_seat_id == hero_seat_id
        )

        # ---------- Automatic snapshot: hero timebar rising edge ----------
        if hero_bar_now and not prev_hero_bar and hero_seat_id is not None:
            # One snapshot per street (flop/turn/river)
            if state.street != last_logged_street:
                # Make sure state is fully updated for THIS frame
                tracker.drain_all_tasks(frame)
                state_fresh = tracker.snapshot_copy()
                frame_fresh = frame.copy()

                log_game_state(state_fresh, frame_fresh, tag=state.street)
                print(f"[INFO] hero_bottom timebar ON on {state.street} -> logged snapshot for GoT input.")

                last_logged_street = state.street

        # ---------- Debug overlay ----------
        debug = frame.copy()
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
            f"pot: {state.pot_size:.2f} total: {state.total_pot:.2f}",
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

        # Optionally draw hero timebar ROI for sanity
        if hero_seat_id is not None:
            tb_roi = cfg["seats"][hero_seat_id].get("timebar_roi")
            if tb_roi:
                tx, ty, tw, th = tb_roi
                color = (0, 255, 0) if hero_bar_now else (0, 0, 255)
                cv2.rectangle(debug, (tx, ty), (tx + tw, ty + th), color, 2)

        cv2.imshow("Table + OCR debug", debug)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            tracker.prev_rois.clear()
            tracker.tasks.clear()
            print("[INFO] Forced full resync.")
        elif key == ord("s"):
            # Manual snapshot at any time
            tracker.drain_all_tasks(frame)
            log_game_state(tracker.snapshot_copy(), frame.copy(), tag="manual")
            print("[INFO] Manual snapshot logged.")

        # Track hero timebar state for rising-edge detection
        prev_hero_bar = hero_bar_now

        frame_idx += 1
        elapsed = time.time() - start
        if elapsed < FRAME_DELAY:
            time.sleep(FRAME_DELAY - elapsed)

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
