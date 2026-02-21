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

def point_in_poly(x, y, poly):
    pts = np.array(poly, dtype=np.int32)
    return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0

def poly_area(poly):
    pts = np.array(poly, dtype=np.float32)
    return float(cv2.contourArea(pts))

def box_poly(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

def intersect_area_convex(poly_a, poly_b):
    a = np.array(poly_a, dtype=np.float32)
    inter_area, _ = cv2.intersectConvexConvex(a, poly_b)
    return float(inter_area)

def spot_occupancy_center_or_overlap(spots, vehicle_boxes, overlap_thresh=0.08):
    out = {}
    for s in spots:
        sid = s["spot_id"]
        poly = clean_poly(s["poly"])
        spot_area = max(poly_area(poly), 1.0)

        occupied = False
        best_ratio = 0.0

        for (x1, y1, x2, y2) in vehicle_boxes:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if point_in_poly(cx, cy, poly):
                occupied = True
                best_ratio = 1.0
                break

            vb_poly = box_poly(x1, y1, x2, y2)
            inter = intersect_area_convex(poly, vb_poly)
            ratio = inter / spot_area
            best_ratio = max(best_ratio, ratio)

            if ratio >= overlap_thresh:
                occupied = True
                break

        out[sid] = {"occupied": occupied, "score": float(best_ratio)}
    return out
