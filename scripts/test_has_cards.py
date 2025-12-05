import time
import cv2
import numpy as np

from poker.capture import load_config, ScreenCapture
from poker.ocr import crop_roi, roi_has_card, normalize_status, ocr_text_fast, STATUS_WHITELIST, STACK_STATUS_WHITELIST


def detect_status(status_img, stack_img):
    """Small helper to mimic your actual status inference."""
    status_raw = ocr_text_fast(status_img, STATUS_WHITELIST, psm=7)
    status_norm = normalize_status(status_raw)

    stack_raw = ocr_text_fast(stack_img, STACK_STATUS_WHITELIST, psm=7)
    stack_norm = normalize_status(stack_raw)

    # Sitting Out / All In only appear in stack area
    if stack_norm in ("sit_out", "all_in"):
        return stack_norm
    return status_norm


def main():
    cfg = load_config()
    cap = ScreenCapture(cfg)

    print("Capturing one frame...")
    frame = cap.grab_table_frame()  # single snapshot

    print("\n=== SEAT CARD / ACTIVE TEST ===\n")

    for seat_id, seat_cfg in enumerate(cfg["seats"]):
        name = seat_cfg["name"]

        # ROIs
        card_img = crop_roi(frame, seat_cfg["card_region"])
        status_img = crop_roi(frame, seat_cfg["status_roi"])
        stack_img  = crop_roi(frame, seat_cfg["stack"])

        # Variance for has_cards
        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        var = float(gray.var())

        has_cards = roi_has_card(card_img)

        status = detect_status(status_img, stack_img)
        is_sitting_out = status == "sit_out"

        # is_active = has_cards AND not folded/sit out
        is_active = has_cards and status not in ("fold", "sit_out")

        print(
            f"Seat {seat_id}: {name}\n"
            f"  variance={var:.1f}\n"
            f"  has_cards={has_cards}\n"
            f"  status={status}\n"
            f"  is_sitting_out={is_sitting_out}\n"
            f"  is_active={is_active}\n"
        )

    print("=== END TEST ===\n")


if __name__ == "__main__":
    main()
