# scripts/test_turn_and_dealer.py

import time
import cv2
import numpy as np

from poker.capture import load_config, ScreenCapture
from poker.ocr import crop_roi, detect_button  # using your existing helpers


def detect_time_bar(img: np.ndarray) -> bool:
    """
    Detect the yellow/green action timer bar in the given ROI.

    Assumes:
      - When NOT your turn: ROI is mostly table felt (dark/unsaturated).
      - When your turn: colored bar (yellow/green) appears in this ROI.
    """
    if img is None or img.size == 0:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Yellow/green band, reasonably saturated and bright
    mask = (
        (h >= 20) & (h <= 90) &
        (s >= 80) &
        (v >= 80)
    )

    ratio = mask.mean()
    return ratio > 0.10  # tweak if needed


def main():
    cfg = load_config()
    cap = ScreenCapture(cfg)

    seats_cfg = cfg["seats"]
    seat_names = [s["name"] for s in seats_cfg]

    # Find bottom_right seat and its timebar_roi
    bottom_right_id = None
    bottom_right_timebar_roi = None

    for i, seat_cfg in enumerate(seats_cfg):
        if seat_cfg["name"] == "bottom_right":
            bottom_right_id = i
            bottom_right_timebar_roi = seat_cfg.get("timebar_roi")
            break

    if bottom_right_id is None or bottom_right_timebar_roi is None:
        print("[ERROR] bottom_right seat or its timebar_roi not found in config.")
        return

    TARGET_FPS = 10
    FRAME_DELAY = 1.0 / TARGET_FPS

    prev_dealer = None
    prev_status = None

    print("Running dealer / 'you are next' detector…")
    print("Press 'q' to exit.\n")

    while True:
        start = time.time()

        frame = cap.grab_table_frame()

        # -------- Dealer detection (via button_roi) --------
        dealer_id = None
        for i, seat_cfg in enumerate(seats_cfg):
            btn_roi = seat_cfg.get("button_roi")
            if not btn_roi:
                continue
            btn_img = crop_roi(frame, btn_roi)
            if detect_button(btn_img):
                dealer_id = i
                break

        dealer_name = seat_names[dealer_id] if dealer_id is not None else None

        # -------- bottom_right timebar detection --------
        tb_img = crop_roi(frame, bottom_right_timebar_roi)
        you_are_next = detect_time_bar(tb_img)

        status_msg = "Prepare, you are next" if you_are_next else "Not your turn"

        # Print only when something changes
        if dealer_name != prev_dealer or status_msg != prev_status:
            print("------------------------------------------------------")
            print(f"Dealer: {dealer_name}")
            print(f"Status: {status_msg}")
            print("------------------------------------------------------")
            prev_dealer = dealer_name
            prev_status = status_msg

        # -------- Debug overlay --------
        debug = frame.copy()

        # Draw dealer button ROI
        if dealer_id is not None:
            bx, by, bw, bh = seats_cfg[dealer_id]["button_roi"]
            cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)
            cv2.putText(debug, "DEALER", (bx, by - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw bottom_right timebar ROI
        tx, ty, tw, th = bottom_right_timebar_roi
        cv2.rectangle(debug, (tx, ty), (tx + tw, ty + th),
                      (0, 255, 0) if you_are_next else (0, 0, 255), 2)

        # Text on screen
        cv2.putText(debug, status_msg, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Dealer / Next Turn Debug", debug)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # Keep FPS stable
        elapsed = time.time() - start
        if elapsed < FRAME_DELAY:
            time.sleep(FRAME_DELAY - elapsed)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
