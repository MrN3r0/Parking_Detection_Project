import json

import cv2
from ultralytics import YOLO

from spots import extract_spot_crop, normalize_spots, scale_spots_to_frame

CFG_PATH = "config.json"


def load_cfg():
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_display_scale(width, height, max_width, max_height):
    scale = min(max_width / width, max_height / height, 1.0)
    return scale, int(width * scale), int(height * scale)


def scale_poly(poly, scale):
    return [[int(x * scale), int(y * scale)] for x, y in poly]


def put_label(img, text, x, y, color, font_scale=0.7, thickness=2):
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def classify_spot(model, crop_bgr):
    result = model.predict(crop_bgr, verbose=False)[0]
    probs = result.probs
    top1_idx = int(probs.top1)
    confidence = float(probs.top1conf.item())
    label = str(model.names[top1_idx]).lower()
    return {
        "label": label,
        "occupied": label == "occupied",
        "confidence": confidence,
    }


def main():
    cfg = load_cfg()

    image_path = cfg.get("source_image", "Parking_Photo.jpg")
    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    frame_h, frame_w = frame.shape[:2]
    spots = cfg.get("spots", [])
    if not spots:
        raise RuntimeError("No spots found in config.json. Trace spots first.")

    trace_size = cfg.get("trace_image_size")
    if trace_size:
        trace_w = int(trace_size["w"])
        trace_h = int(trace_size["h"])
        if (trace_w, trace_h) != (frame_w, frame_h):
            spots = scale_spots_to_frame(spots, trace_w, trace_h, frame_w, frame_h)
            print(f"[SCALE] spots scaled {trace_w}x{trace_h} -> {frame_w}x{frame_h}")

    spots = normalize_spots(spots)

    classifier_model = cfg.get("classifier_model", "runs/classify/train/weights/best.pt")
    crop_size = int(cfg.get("classifier_crop_size", 224))
    crop_padding = float(cfg.get("classifier_crop_padding", 0.10))
    show_spot_debug = bool(cfg.get("show_spot_debug", False))

    model = YOLO(classifier_model)

    max_w = int(cfg.get("display_max_w", 1600))
    max_h = int(cfg.get("display_max_h", 900))
    disp_scale, disp_w, disp_h = compute_display_scale(frame_w, frame_h, max_w, max_h)
    display = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    line_thickness = max(2, int(cfg.get("line_thickness", 4)))
    label_scale = float(cfg.get("label_font_scale", 0.55))
    label_thickness = max(1, int(cfg.get("label_thickness", 2)))

    occupied_count = 0
    spot_results = {}

    for spot in spots:
        crop = extract_spot_crop(
            frame,
            spot["poly"],
            padding_ratio=crop_padding,
            crop_size=crop_size,
        )
        if crop is None:
            info = {"label": "vacant", "occupied": False, "confidence": 0.0}
        else:
            info = classify_spot(model, crop)

        spot_results[spot["spot_id"]] = info

        if info["occupied"]:
            occupied_count += 1

        color = (0, 0, 255) if info["occupied"] else (0, 255, 0)
        poly_disp = scale_poly(spot["poly"], disp_scale)

        for idx in range(len(poly_disp)):
            x1, y1 = poly_disp[idx]
            x2, y2 = poly_disp[(idx + 1) % len(poly_disp)]
            cv2.line(display, (x1, y1), (x2, y2), color, line_thickness, cv2.LINE_AA)

        label_x, label_y = poly_disp[0]
        state = "OCC" if info["occupied"] else "VAC"
        put_label(
            display,
            f"{spot['spot_id']} ({state})",
            label_x,
            max(25, label_y - 8),
            color,
            label_scale,
            label_thickness,
        )

        if show_spot_debug:
            put_label(
                display,
                f"{info['label']} {info['confidence']:.2f}",
                label_x,
                max(45, label_y + 16),
                (255, 255, 255),
                0.5,
                1,
            )

    put_label(
        display,
        f"Occupied: {occupied_count}/{len(spots)}",
        20,
        35,
        (255, 255, 255),
        0.9,
        2,
    )
    put_label(
        display,
        f"Classifier: {classifier_model}",
        20,
        65,
        (255, 255, 255),
        0.55,
        1,
    )

    print(json.dumps(spot_results, indent=2))

    cv2.namedWindow("ROI Spot Occupancy Classifier", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI Spot Occupancy Classifier", disp_w, disp_h)
    cv2.imshow("ROI Spot Occupancy Classifier", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
