# scripts/test_dealer_and_next.py

import time
import cv2
import numpy as np

from poker.capture import load_config, ScreenCapture
from poker.ocr import crop_roi


def dealer_button_score(img: np.ndarray) -> float:
    if img is None or img.size == 0:
        return 0.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    white_mask = (v >= 200) & (s <= 40)
    white_ratio = white_mask.mean()

    red_mask = (
        ((h <= 10) | (h >= 170)) &
        (s >= 90) &
        (v >= 80)
    )
    red_ratio = red_mask.mean()

    # Combined confidence
    return float(white_ratio * red_ratio)


def detect_time_bar(img: np.ndarray) -> bool:
    if img is None or img.size == 0:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask = (
        (h >= 20) & (h <= 90) &
        (s >= 80) &
        (v >= 80)
    )
    return mask.mean() > 0.10


def main():
    cfg = load_config()
    cap = ScreenCapture(cfg)
    seats_cfg = cfg["seats"]
    seat_names = [s["name"] for s in seats_cfg]

    # Find bottom_right seat
    bottom_right_id = None
    for i, s in enumerate(seats_cfg):
        if s["name"] == "bottom_right":
            bottom_right_id = i
            break

    if bottom_right_id is None:
        print("[ERROR] bottom_right seat not in config")
        return

    timebar_roi = seats_cfg[bottom_right_id].get("timebar_roi")
    if timebar_roi is None:
        print("[ERROR] bottom_right timebar_roi missing")
        return

    TARGET_FPS = 10
    FRAME_DELAY = 1 / TARGET_FPS

    prev_dealer = None
    prev_status = None

    print("Running dealer & next-turn detector…")
    print("Press 'q' to exit.")

    while True:
        start = time.time()
        frame = cap.grab_table_frame()

        # -------- Detect dealer button --------
        best_score = 0
        dealer_id = None

        for i, seat_cfg in enumerate(seats_cfg):
            roi = seat_cfg.get("button_roi")
            if not roi:
                continue

            img = crop_roi(frame, roi)
            score = dealer_button_score(img)

            if score > best_score:
                best_score = score
                dealer_id = i

        # Require score above threshold to avoid random noise
        if best_score < 0.01:
            dealer_id = None

        dealer_name = seat_names[dealer_id] if dealer_id is not None else None

        # -------- Detect bottom_right "you are next" --------
        tb_img = crop_roi(frame, timebar_roi)
        you_next = detect_time_bar(tb_img)
        status_msg = "Prepare, you are next" if you_next else "Not your turn"

        # -------- Print change only --------
        if dealer_name != prev_dealer or status_msg != prev_status:
            print("------------------------------------------------")
            print(f"Dealer: {dealer_name}")
            print(f"Status: {status_msg}")
            print("------------------------------------------------")
            prev_dealer = dealer_name
            prev_status = status_msg

        # Visualization
        debug = frame.copy()

        # Draw dealer
        if dealer_id is not None:
            x, y, w, h = seats_cfg[dealer_id]["button_roi"]
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Draw bottom_right timebar
        tx, ty, tw, th = timebar_roi
        cv2.rectangle(debug, (tx, ty), (tx + tw, ty + th),
                      (0, 255, 0) if you_next else (0, 0, 255), 2)

        cv2.putText(debug, status_msg, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Dealer / Next Debug", debug)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        elapsed = time.time() - start
        if elapsed < FRAME_DELAY:
            time.sleep(FRAME_DELAY - elapsed)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
