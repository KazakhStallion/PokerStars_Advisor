import cv2
from poker.capture import load_config, ScreenCapture

def draw_rois(frame, cfg):
    # Hero cards
    for i, (x, y, w, h) in enumerate(cfg["hero_cards"]):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"H{i}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Board cards
    for i, (x, y, w, h) in enumerate(cfg["board_cards"]):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"B{i}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Pot text
    x, y, w, h = cfg["pot_text"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(frame, "pot", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Total pot text
    x, y, w, h = cfg["total_pot_text"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 200), 2)
    cv2.putText(frame, "total", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

    # Seats
    for seat_idx, seat in enumerate(cfg["seats"]):
        name = seat["name"]

        def draw_box(key, color):
            x, y, w, h = seat[key]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name}:{key}", (x, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        draw_box("stack",       (255, 255, 255))
        draw_box("bet",         (0, 0, 255))
        draw_box("card_region", (0, 255, 0))
        draw_box("button_roi",  (255, 255, 0))
        draw_box("status_roi",  (255, 0, 255))


def main():
    cfg = load_config()
    cap = ScreenCapture(cfg)

    while True:
        frame = cap.grab_table_frame()        # cropped to the table window
        vis = frame.copy()
        draw_rois(vis, cfg)

        cv2.imshow("PokerStars table + ROIs", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("debug_frame_with_rois.png", vis)
            print("Saved debug_frame_with_rois.png")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
