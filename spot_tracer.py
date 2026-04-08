import json
import cv2
import os

CFG_PATH = "config.json"
WIN = "Spot Tracer (4-click ROI)"

# Window target size
MAX_W = 1600
MAX_H = 900

# Label controls (tweak these)
SHOW_LAST_N_LABELS = 10      # show labels only for last N spots
LABEL_FONT_SCALE = 0.45      # smaller label text
LABEL_THICKNESS = 1          # thinner text outline
SHOW_LABELS = True           # press L to toggle


def load_cfg():
    if not os.path.exists(CFG_PATH):
        return {}
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cfg(cfg):
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def put_text_shadow(img, text, org, font_scale=1.0, thickness=2):
    # outline
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # text
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def compute_scale(w, h, max_w, max_h):
    scale = min(max_w / w, max_h / h, 1.0)
    dw, dh = int(w * scale), int(h * scale)
    return scale, dw, dh


def scale_poly(poly, s):
    return [[int(x * s), int(y * s)] for x, y in poly]


def main():
    global SHOW_LABELS

    cfg = load_cfg()

    img_path = cfg.get("source_image", "images/lot.jpg")
    img_orig = cv2.imread(img_path)
    if img_orig is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    H, W = img_orig.shape[:2]
    cfg["trace_image_size"] = {"w": W, "h": H}

    # Spot ID formatting
    lot_prefix = cfg.get("lot_prefix", "GLL")
    spot_row = cfg.get("spot_row", "B")
    start_number = int(cfg.get("spot_start_number", 1))

    spots = cfg.get("spots", [])   # ORIGINAL coordinates
    current = []                   # ORIGINAL coordinates

    scale, disp_w, disp_h = compute_scale(W, H, MAX_W, MAX_H)

    def next_id():
        return f"{lot_prefix}-{spot_row}{start_number + len(spots):02d}"

    def draw_overlay():
        disp = cv2.resize(img_orig, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        # draw saved spots
        for idx, spt in enumerate(spots):
            poly_disp = scale_poly(spt["poly"], scale)
            sid = spt["spot_id"]

            # polygon lines
            for i in range(4):
                x1, y1 = poly_disp[i]
                x2, y2 = poly_disp[(i + 1) % 4]
                cv2.line(disp, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

            # labels: only last N to reduce clutter
            if SHOW_LABELS and (idx >= max(0, len(spots) - SHOW_LAST_N_LABELS)):
                x0, y0 = poly_disp[0]
                put_text_shadow(
                    disp,
                    sid,  # keep full label like GLL-B01
                    (x0, max(30, y0 - 8)),
                    font_scale=LABEL_FONT_SCALE,
                    thickness=LABEL_THICKNESS
                )

        # draw current clicks
        cur_disp = scale_poly(current, scale)
        for p in cur_disp:
            cv2.circle(disp, tuple(p), 6, (255, 255, 255), -1, cv2.LINE_AA)

        if len(cur_disp) >= 2:
            for i in range(len(cur_disp) - 1):
                cv2.line(disp, tuple(cur_disp[i]), tuple(cur_disp[i + 1]), (255, 255, 255), 2, cv2.LINE_AA)

        # UI text
        FS_TITLE = 0.70
        FS_INFO = 0.75
        TH = 2

        put_text_shadow(disp, "Trace spots: click 4 corners per spot (auto-saves on 4th click)",
                        (20, 45), font_scale=FS_TITLE, thickness=TH)
        put_text_shadow(disp, "Keys: [S]=save  [U]=undo  [C]=clear  [L]=toggle labels  [Q]=quit",
                        (20, 85), font_scale=FS_INFO, thickness=TH)
        put_text_shadow(disp, f"Current clicks: {len(current)}/4   Spots saved: {len(spots)}   Next: {next_id()}",
                        (20, 125), font_scale=FS_INFO, thickness=TH)

        return disp

    def on_mouse(event, x, y, flags, param):
        nonlocal current, spots
        if event == cv2.EVENT_LBUTTONDOWN:
            ox = int(x / scale)
            oy = int(y / scale)
            ox = max(0, min(W - 1, ox))
            oy = max(0, min(H - 1, oy))

            current.append([ox, oy])

            if len(current) == 4:
                spots.append({"spot_id": next_id(), "poly": current[:]})
                current = []

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)
    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        cv2.imshow(WIN, draw_overlay())
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("u"):
            if current:
                current.pop()
            elif spots:
                spots.pop()
        elif key == ord("c"):
            spots = []
            current = []
        elif key == ord("l"):
            SHOW_LABELS = not SHOW_LABELS
        elif key == ord("s"):
            cfg["spots"] = spots
            save_cfg(cfg)
            print(f"[SAVED] {len(spots)} spots -> {CFG_PATH}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()