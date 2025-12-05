# scripts/test_card_templates.py

import cv2
import numpy as np

from poker.capture import load_config, ScreenCapture
from poker.card_templates import load_all_templates
from poker.ocr import crop_roi, card_present, prep_gray, scan_templates, has_back_cards, board_card_present


def main():
    # Load templates
    rank_templates, suit_templates = load_all_templates()

    # Grab one frame
    cfg = load_config()
    cap = ScreenCapture(cfg)
    frame = cap.grab_table_frame()

    results = []

    # Hero cards
    for i, roi in enumerate(cfg["hero_cards"]):
        card_img = crop_roi(frame, roi)
        label = f"hero_{i}"

        if not card_present(card_img):
            results.append((label, "no_card", None, None))
            continue

        gray = prep_gray(card_img)

        r_label, r_score = scan_templates(gray, rank_templates)
        s_label, s_score = scan_templates(gray, suit_templates)

        card_str = f"{r_label}{s_label}"
        results.append((label, card_str, (r_label, r_score), (s_label, s_score)))

        cv2.imshow(f"{label}_full", card_img)

    # Board cards
    for i, roi in enumerate(cfg["board_cards"]):
        card_img = crop_roi(frame, roi)
        label = f"board_{i}"

        if not card_present(card_img):
            results.append((label, "no_card", None, None))
            continue

        gray = prep_gray(card_img)

        r_label, r_score = scan_templates(gray, rank_templates)
        s_label, s_score = scan_templates(gray, suit_templates)

        card_str = f"{r_label}{s_label}"
        results.append((label, card_str, (r_label, r_score), (s_label, s_score)))

        cv2.imshow(f"{label}_full", card_img)

    # Print results
    for label, card_str, rank_info, suit_info in results:
        if rank_info is None:
            print(f"{label}: {card_str}")
        else:
            (r_label, r_score) = rank_info
            (s_label, s_score) = suit_info
            print(
                f"{label}: {card_str}  "
                f"(rank={r_label} {r_score:.3f}, suit={s_label} {s_score:.3f})"
            )

    print("Press any key in an image window to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
