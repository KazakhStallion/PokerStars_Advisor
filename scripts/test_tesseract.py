# scripts/test_tesseract.py

import time
import re

import cv2
import pytesseract

from poker.capture import load_config, ScreenCapture
from poker.ocr import crop_roi, get_seat_status, ocr_text_fast, ocr_amount_fast


def main():
    cfg = load_config()
    cap = ScreenCapture(cfg)

    frame = cap.grab_table_frame()
    cv2.imshow("full frame", frame)

    start = time.perf_counter()

    print("=== POT ROIS ===")
    # Pot
    pot_img = crop_roi(frame, cfg["pot_text"])
    pot_raw = ocr_text_fast(pot_img, "0123456789.,$", psm=7)
    pot_val = ocr_amount_fast(pot_img)
    print(f"pot_text: raw='{pot_raw}'  -> value={pot_val}")
    # cv2.imshow("pot_text", pot_img)

    # Total pot
    total_img = crop_roi(frame, cfg["total_pot_text"])
    total_raw = ocr_text_fast(total_img, "0123456789.,$", psm=7)
    total_val = ocr_amount_fast(total_img)
    print(f"total_pot_text: raw='{total_raw}'  -> value={total_val}")
    # cv2.imshow("total_pot_text", total_img)

    print("\n=== SEAT ROIS ===")
    for i, seat_cfg in enumerate(cfg["seats"]):
        name = seat_cfg["name"]

        # Stack
        stack_img = crop_roi(frame, seat_cfg["stack"])
        stack_raw = ocr_text_fast(stack_img, "0123456789.,$", psm=7)
        stack_val = ocr_amount_fast(stack_img)

        # Bet
        bet_img = crop_roi(frame, seat_cfg["bet"])
        bet_raw = ocr_text_fast(bet_img, "0123456789.,$", psm=7)
        bet_val = ocr_amount_fast(bet_img)

        # Status (Fold, Call, Check, Bet, Raise, All In, Sitting Out)
        status_img = crop_roi(frame, seat_cfg["status_roi"])
        status_norm, status_raw, stack_status_raw = get_seat_status(status_img, stack_img)

        print(
            f"[seat {i} - {name}] "
            f"stack_raw='{stack_raw}' -> {stack_val} | "
            f"bet_raw='{bet_raw}' -> {bet_val} | "
            f"status_raw='{status_raw}' "
            f"(stack_status_raw='{stack_status_raw}') -> {status_norm}"
        )

        # cv2.imshow(f"{name}_stack", stack_img)
        # cv2.imshow(f"{name}_bet", bet_img)
        # cv2.imshow(f"{name}_status", status_img)

    elapsed = time.perf_counter() - start
    print(f"\nTotal OCR time for this frame: {elapsed*1000:.1f} ms")

    print("\nPress any key in an image window to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
