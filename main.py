import json
import math
import cv2
import numpy as np
from ultralytics import YOLO

CFG_PATH = "config.json"


def load_cfg():
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_display_scale(w, h, max_w, max_h):
    s = min(max_w / w, max_h / h, 1.0)
    return s, int(w * s), int(h * s)


def scale_poly(poly, s):
    return [[int(x * s), int(y * s)] for x, y in poly]


def scale_box(box, s):
    x1, y1, x2, y2 = box
    return (int(x1 * s), int(y1 * s), int(x2 * s), int(y2 * s))


def scale_spots_to_frame(spots, from_w, from_h, to_w, to_h):
    sx = to_w / from_w
    sy = to_h / from_h
    out = []
    for sp in spots:
        new_poly = [[int(x * sx), int(y * sy)] for x, y in sp["poly"]]
        out.append({**sp, "poly": new_poly})
    return out


# ------------------------------
# Normalize polygon order
# ------------------------------
def normalize_poly(poly):
    pts = np.array(poly, dtype=np.float32)
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    angles = [math.atan2(p[1] - cy, p[0] - cx) for p in pts]
    pts = pts[np.argsort(angles)]
    # make clockwise
    if cv2.contourArea(pts) < 0:
        pts = pts[::-1]
    return pts.astype(int).tolist()


def normalize_spots(spots):
    return [{**sp, "poly": normalize_poly(sp["poly"])} for sp in spots]


# ------------------------------
# Shrink polygon inward (fixes YOLO bleed into neighbor spots)
# ------------------------------
def shrink_polygon(poly, shrink_px=8):
    if shrink_px <= 0:
        return poly

    pts = np.array(poly, dtype=np.int32)

    x, y, w, h = cv2.boundingRect(pts)
    pad = shrink_px + 6
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = x + w + pad
    y1 = y + h + pad

    # local coords
    local = pts.copy()
    local[:, 0] -= x0
    local[:, 1] -= y0

    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(mask, [local], 255)

    k = max(1, int(shrink_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    eroded = cv2.erode(mask, kernel, iterations=1)

    cnts, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return poly  # shrink too strong, fallback

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 20:
        return poly

    cnt = cnt.reshape(-1, 2)
    cnt[:, 0] += x0
    cnt[:, 1] += y0
    return cnt.astype(int).tolist()


# ------------------------------
# YOLO occupancy: intersection pixels
# ------------------------------
def box_poly(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def expand_box(x1, y1, x2, y2, px=12):
    return (x1 - px, y1 - px, x2 + px, y2 + px)


def spot_occupied_by_intersection_px(spot_poly, vehicle_boxes, min_intersection_px=400):
    spot = np.array(spot_poly, dtype=np.float32)
    best = 0.0
    best_idx = -1

    for i, (x1, y1, x2, y2) in enumerate(vehicle_boxes):
        vb = box_poly(x1, y1, x2, y2)
        inter_area, _ = cv2.intersectConvexConvex(spot, vb)
        inter_area = float(inter_area)

        if inter_area > best:
            best = inter_area
            best_idx = i

        if inter_area >= float(min_intersection_px):
            return True, best, best_idx

    return False, best, best_idx


# ------------------------------
# Pixel-diff fallback (baseline vs current inside polygon)
# ------------------------------
def polygon_mask(shape_hw, poly):
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(poly, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def spot_occupied_by_pixel_diff(
    baseline_bgr,
    current_bgr,
    spot_poly,
    diff_ratio_thresh=0.05,
    diff_pixel_thresh=25,
    blur_ksize=5
):
    h, w = current_bgr.shape[:2]
    mask = polygon_mask((h, w), spot_poly)

    base_gray = cv2.cvtColor(baseline_bgr, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)

    if blur_ksize and blur_ksize > 1:
        base_gray = cv2.GaussianBlur(base_gray, (blur_ksize, blur_ksize), 0)
        cur_gray = cv2.GaussianBlur(cur_gray, (blur_ksize, blur_ksize), 0)

    diff = cv2.absdiff(cur_gray, base_gray)
    diff_masked = cv2.bitwise_and(diff, diff, mask=mask)

    changed = (diff_masked > diff_pixel_thresh).astype(np.uint8)
    inside = (mask > 0).astype(np.uint8)

    changed_count = int((changed & inside).sum())
    inside_count = int(inside.sum()) + 1
    ratio = changed_count / inside_count
    return (ratio >= float(diff_ratio_thresh)), float(ratio)


# ------------------------------
# Drawing helpers
# ------------------------------
def put_label(img, text, x, y, color, font_scale=1.0, thickness=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def main():
    cfg = load_cfg()

    img_path = cfg.get("source_image", "Parking_Lot_Photo2.jpg")
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    H, W = img.shape[:2]

    baseline_path = cfg.get("baseline_image", None)
    baseline = None
    if baseline_path:
        baseline = cv2.imread(baseline_path)
        if baseline is None:
            raise RuntimeError(f"Could not read baseline_image: {baseline_path}")
        if baseline.shape[:2] != (H, W):
            raise RuntimeError(
                f"Baseline size {baseline.shape[1]}x{baseline.shape[0]} "
                f"!= current size {W}x{H} (must match)."
            )

    spots = cfg.get("spots", [])
    if not spots:
        raise RuntimeError("No spots found in config.json. Trace spots first.")

    ts = cfg.get("trace_image_size")
    if ts:
        tw, th = int(ts["w"]), int(ts["h"])
        if tw != W or th != H:
            spots = scale_spots_to_frame(spots, tw, th, W, H)
            print(f"[SCALE] spots scaled {tw}x{th} -> {W}x{H}")

    spots = normalize_spots(spots)

    # NEW: shrink polygons to prevent neighbor overlap false OCC
    spot_shrink_px = int(cfg.get("spot_shrink_px", 10))
    spots_shrunk = []
    for sp in spots:
        shr = shrink_polygon(sp["poly"], shrink_px=spot_shrink_px)
        spots_shrunk.append({**sp, "poly_shrunk": shr})

    max_w = int(cfg.get("display_max_w", 1600))
    max_h = int(cfg.get("display_max_h", 900))
    disp_scale, disp_w, disp_h = compute_display_scale(W, H, max_w, max_h)
    disp = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    line_thickness = int(cfg.get("line_thickness", 8))
    label_scale = float(cfg.get("label_font_scale", 1.1))
    label_th = int(cfg.get("label_thickness", 3))

    model = YOLO(cfg.get("yolo_model", "yolov8n.pt"))

    # IMPORTANT: if you changed lots, don’t keep conf super low
    conf = float(cfg.get("conf_thresh", 0.20))

    allow_list = cfg.get("vehicle_classes", ["car", "truck", "bus", "motorcycle"])
    allow = set(allow_list) if allow_list else None

    # reduce box expand so it doesn't bleed into neighbor spots
    expand_px = int(cfg.get("box_expand_px", 8))

    # raise intersection requirement to stop tiny overlaps
    min_intersection_px = int(cfg.get("min_intersection_px", 650))

    show_spot_debug = bool(cfg.get("show_spot_debug", False))
    show_debug_boxes = bool(cfg.get("show_debug_boxes", False))
    draw_only_overlapping = bool(cfg.get("draw_only_overlapping_boxes", True))

    use_pixel_fallback = bool(cfg.get("use_pixel_fallback", True))
    diff_ratio_thresh = float(cfg.get("diff_ratio_thresh", 0.05))
    diff_pixel_thresh = int(cfg.get("diff_pixel_thresh", 25))
    blur_ksize = int(cfg.get("diff_blur_ksize", 5))

    results = model.predict(img, conf=conf, verbose=False)[0]
    names = results.names

    vehicle_boxes = []
    debug_dets = []

    for b in results.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        confv = float(b.conf[0].tolist())
        cls_idx = int(b.cls[0].tolist())
        cls_name = names.get(cls_idx, str(cls_idx))

        if allow is not None and cls_name not in allow:
            continue

        x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, px=expand_px)
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))

        vehicle_boxes.append((x1, y1, x2, y2))
        debug_dets.append((x1, y1, x2, y2, confv, cls_name))

    # Decide occupancy
    occ_count = 0

    # track which YOLO box overlapped any spot (so we can optionally draw only those)
    overlapping_box_indices = set()

    for sp in spots_shrunk:
        poly = sp["poly"]           # original
        poly_shr = sp["poly_shrunk"]  # shrunk for checking
        sid = sp["spot_id"]

        occ_yolo, best_area, best_idx = spot_occupied_by_intersection_px(
            poly_shr, vehicle_boxes, min_intersection_px=min_intersection_px
        )
        if best_idx is not None and best_idx >= 0 and best_area > 1:
            overlapping_box_indices.add(best_idx)

        occ_px = False
        px_ratio = 0.0
        if (not occ_yolo) and use_pixel_fallback and (baseline is not None):
            occ_px, px_ratio = spot_occupied_by_pixel_diff(
                baseline, img, poly_shr,
                diff_ratio_thresh=diff_ratio_thresh,
                diff_pixel_thresh=diff_pixel_thresh,
                blur_ksize=blur_ksize
            )

        occ = occ_yolo or occ_px
        if occ:
            occ_count += 1

        color = (0, 0, 255) if occ else (0, 255, 0)

        # draw ORIGINAL polygon (looks nicer), but CHECKING uses shrunk polygon
        poly_disp = scale_poly(poly, disp_scale)
        for i in range(len(poly_disp)):
            xA, yA = poly_disp[i]
            xB, yB = poly_disp[(i + 1) % len(poly_disp)]
            cv2.line(disp, (xA, yA), (xB, yB), color, line_thickness, cv2.LINE_AA)

        x0, y0 = poly_disp[0]
        put_label(disp, f"{sid} ({'OCC' if occ else 'VAC'})", x0, max(60, y0 - 12),
                  color, label_scale, label_th)

        if show_spot_debug:
            method = "YOLO" if occ_yolo else ("PIX" if occ_px else "-")
            put_label(
                disp,
                f"{method} area={int(best_area)} diff={px_ratio:.3f}",
                x0, max(90, y0 + 20),
                (255, 255, 255),
                0.8, 2
            )

    # Draw debug boxes ONLY if enabled (and optionally only overlapping ones)
    if show_debug_boxes:
        for i, (x1, y1, x2, y2, confv, cls_name) in enumerate(debug_dets):
            if draw_only_overlapping and i not in overlapping_box_indices:
                continue
            dx1, dy1, dx2, dy2 = scale_box((x1, y1, x2, y2), disp_scale)
            cv2.rectangle(disp, (dx1, dy1), (dx2, dy2), (255, 255, 255), 3, cv2.LINE_AA)
            put_label(disp, f"{cls_name} {confv:.2f}", dx1, max(30, dy1 - 8),
                      (255, 255, 255), 0.9, 2)

    # Top status text (fits screen)
    put_label(
        disp,
        f"Occupied: {occ_count}/{len(spots_shrunk)}   YOLO: {len(vehicle_boxes)}   conf={conf:.2f}",
        20, 45, (255, 255, 255), 1.0, 3
    )
    put_label(
        disp,
        f"shrink_px={spot_shrink_px}  min_px={min_intersection_px}  expand_px={expand_px}",
        20, 85, (255, 255, 255), 1.0, 3
    )

    cv2.namedWindow("ROI Spot Occupancy", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI Spot Occupancy", disp_w, disp_h)
    cv2.imshow("ROI Spot Occupancy", disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()