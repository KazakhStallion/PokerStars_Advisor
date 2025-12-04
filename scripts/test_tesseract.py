# scripts/test_tesseract_rois.py

import cv2

from poker.capture import load_config, ScreenCapture
from poker.ocr import (
    crop_roi,
    ocr_text,
    ocr_amount,
    normalize_status,
)


def show_roi(name, img):
    cv2.imshow(name, img)


def main():
    cfg = load_config()
    cap = ScreenCapture(cfg)

    frame = cap.grab_table_frame()

    print("=== POT ROIS ===")
    # Pot
    pot_img = crop_roi(frame, cfg["pot_text"])
    pot_raw = ocr_text(pot_img, "0123456789.,$", psm=7)
    pot_val = ocr_amount(pot_img)
    print(f"pot_text: raw='{pot_raw}'  -> value={pot_val}")
    show_roi("pot_text", pot_img)

    # Total pot
    total_img = crop_roi(frame, cfg["total_pot_text"])
    total_raw = ocr_text(total_img, "0123456789.,$", psm=7)
    total_val = ocr_amount(total_img)
    print(f"total_pot_text: raw='{total_raw}'  -> value={total_val}")
    show_roi("total_pot_text", total_img)

    print("\n=== SEAT ROIS ===")
    for i, seat_cfg in enumerate(cfg["seats"]):
        name = seat_cfg["name"]

        # Stack
        stack_img = crop_roi(frame, seat_cfg["stack"])
        stack_raw = ocr_text(stack_img, "0123456789.,$", psm=7)
        stack_val = ocr_amount(stack_img)

        # Bet
        bet_img = crop_roi(frame, seat_cfg["bet"])
        bet_raw = ocr_text(bet_img, "0123456789.,$", psm=7)
        bet_val = ocr_amount(bet_img)

        # Status
        status_img = crop_roi(frame, seat_cfg["status_roi"])
        status_raw = ocr_text(
            status_img,
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz 0123456789,:.",
            psm=7,
        )
        status_norm = normalize_status(status_raw)

        print(
            f"[seat {i} - {name}] "
            f"stack_raw='{stack_raw}' -> {stack_val} | "
            f"bet_raw='{bet_raw}' -> {bet_val} | "
            f"status_raw='{status_raw}' -> {status_norm}"
        )

        # Optional visual debug
        show_roi(f"{name}_stack", stack_img)
        show_roi(f"{name}_bet", bet_img)
        show_roi(f"{name}_status", status_img)

    print("\nPress any key in an image window to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
