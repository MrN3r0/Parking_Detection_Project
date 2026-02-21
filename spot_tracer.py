import json
import cv2
import os

CFG_PATH = "config.json"
WIN = "Spot Tracer (4-click ROI)"

# Window target size (change if you want bigger)
MAX_W = 1600
MAX_H = 900


def load_cfg():
    if not os.path.exists(CFG_PATH):
        return {}
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cfg(cfg):
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def put_text_shadow(img, text, org, font_scale=1.0, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def compute_scale(w, h, max_w, max_h):
    scale = min(max_w / w, max_h / h, 1.0)
    dw, dh = int(w * scale), int(h * scale)
    return scale, dw, dh


def scale_poly(poly, s):
    return [[int(x * s), int(y * s)] for x, y in poly]


def main():
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

    spots = cfg.get("spots", [])   # stored in ORIGINAL coordinates
    current = []                   # current clicks in ORIGINAL coordinates

    # Display scaling so it fits screen but text stays readable
    scale, disp_w, disp_h = compute_scale(W, H, MAX_W, MAX_H)

    def next_id():
        return f"{lot_prefix}-{spot_row}{start_number + len(spots):02d}"

    def draw_overlay():
        # resize original to display size
        disp = cv2.resize(img_orig, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        # draw saved spots (scaled to display)
        for spt in spots:
            poly_disp = scale_poly(spt["poly"], scale)
            sid = spt["spot_id"]

            for i in range(4):
                x1, y1 = poly_disp[i]
                x2, y2 = poly_disp[(i + 1) % 4]
                cv2.line(disp, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

            x0, y0 = poly_disp[0]
            put_text_shadow(disp, sid, (x0, max(30, y0 - 10)), font_scale=1.0, thickness=2)

        # draw current clicks (scaled to display)
        cur_disp = scale_poly(current, scale)
        for p in cur_disp:
            cv2.circle(disp, tuple(p), 6, (255, 255, 255), -1, cv2.LINE_AA)

        if len(cur_disp) >= 2:
            for i in range(len(cur_disp) - 1):
                cv2.line(disp, tuple(cur_disp[i]), tuple(cur_disp[i + 1]), (255, 255, 255), 2, cv2.LINE_AA)

        # BIG ON-SCREEN TEXT (because we draw it on the display image)
        FS_TITLE = 0.50
        FS_INFO = 0.75
        TH = 2

        put_text_shadow(disp, "Trace spots: click 4 corners per spot (auto-saves on 4th click)",
                        (20, 45), font_scale=FS_TITLE, thickness=TH)
        put_text_shadow(disp, "Keys: [S]=save  [U]=undo  [C]=clear  [Q]=quit",
                        (20, 85), font_scale=FS_INFO, thickness=TH)
        put_text_shadow(disp, f"Current clicks: {len(current)}/4   Spots saved: {len(spots)}   Next: {next_id()}",
                        (20, 125), font_scale=FS_INFO, thickness=TH)

        return disp

    def on_mouse(event, x, y, flags, param):
        nonlocal current, spots
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert display coords -> original coords
            ox = int(x / scale)
            oy = int(y / scale)
            ox = max(0, min(W - 1, ox))
            oy = max(0, min(H - 1, oy))

            current.append([ox, oy])

            # auto-save after 4 clicks
            if len(current) == 4:
                spots.append({"spot_id": next_id(), "poly": current[:]})
                current = []

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)  # make window match display image size
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
        elif key == ord("s"):
            cfg["spots"] = spots
            save_cfg(cfg)
            print(f"[SAVED] {len(spots)} spots -> {CFG_PATH}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
