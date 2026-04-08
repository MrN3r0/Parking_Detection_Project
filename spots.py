import cv2
import numpy as np


def clean_poly(poly):
    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
    hull = cv2.convexHull(pts).reshape((-1, 2))
    return hull.tolist()


def normalize_spots(spots):
    out = []
    for s in spots:
        if "poly" in s and s["poly"]:
            out.append({"spot_id": s.get("spot_id", "SPOT"), "poly": clean_poly(s["poly"])})
    return out


def scale_spots_to_frame(spots, from_w, from_h, to_w, to_h):
    sx = to_w / from_w
    sy = to_h / from_h
    scaled = []
    for s in spots:
        new_poly = [[int(x * sx), int(y * sy)] for x, y in s["poly"]]
        scaled.append({**s, "poly": new_poly})
    return scaled


def polygon_mask(shape_hw, poly):
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(poly, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def extract_spot_crop(frame_bgr, poly, padding_ratio=0.10, crop_size=224):
    """
    Extract a masked parking-space crop for classification.
    Pixels outside the polygon are zeroed so the classifier focuses on the spot.
    """
    polygon = np.array(clean_poly(poly), dtype=np.float32)
    h, w = frame_bgr.shape[:2]

    min_xy = polygon.min(axis=0)
    max_xy = polygon.max(axis=0)
    bbox_w = max_xy[0] - min_xy[0]
    bbox_h = max_xy[1] - min_xy[1]

    pad_x = max(4.0, bbox_w * padding_ratio)
    pad_y = max(4.0, bbox_h * padding_ratio)

    x0 = max(0, int(np.floor(min_xy[0] - pad_x)))
    y0 = max(0, int(np.floor(min_xy[1] - pad_y)))
    x1 = min(w, int(np.ceil(max_xy[0] + pad_x)))
    y1 = min(h, int(np.ceil(max_xy[1] + pad_y)))

    if x1 <= x0 or y1 <= y0:
        return None

    crop = frame_bgr[y0:y1, x0:x1].copy()
    local_poly = polygon - np.array([x0, y0], dtype=np.float32)

    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [local_poly.astype(np.int32)], 255)
    crop[mask == 0] = 0

    return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
