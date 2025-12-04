# scripts/test_card_templates.py

import cv2
import numpy as np

from poker.capture import load_config, ScreenCapture
from poker.card_templates import load_all_templates


def crop_roi(frame, roi):
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]


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


def main():
    # 1) Load templates
    rank_templates, suit_templates = load_all_templates()

    # 2) Grab one frame
    cfg = load_config()
    cap = ScreenCapture(cfg)
    frame = cap.grab_table_frame()

    results = []

    # ---------- HERO CARDS ----------
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

    # ---------- BOARD CARDS ----------
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

    # 4) Print results
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
