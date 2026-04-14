import argparse
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

CLASS_MAP = {
    "0": "vacant",
    "1": "occupied",
}

CLASS_NAMES = {
    0: "empty",
    1: "occupied",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert the Roboflow parking YOLOv8 polygon dataset into "
            "vacant/occupied classifier crops."
        )
    )
    parser.add_argument(
        "--archive",
        default=r"C:\Users\nnobl\OneDrive\Desktop\Parking.v1i.yolov8 (1).zip",
        help="Path to the Roboflow YOLOv8 ZIP with empty/occupied labels.",
    )
    parser.add_argument(
        "--output",
        default=str(Path("training") / "fine_tune_parking"),
        help="Output classifier dataset directory.",
    )
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--padding", type=float, default=0.10)
    return parser.parse_args()


def ensure_clean_dirs(output_dir):
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ("train", "val", "test"):
        for label in ("vacant", "occupied"):
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)


def clamp_bbox(x0, y0, x1, y1, width, height):
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return x0, y0, x1, y1


def polygon_from_yolo(parts, width, height):
    values = [float(v) for v in parts]

    # YOLO detection boxes have 4 values after the class.
    if len(values) == 4:
        cx, cy, bw, bh = values
        x0 = (cx - bw / 2.0) * width
        y0 = (cy - bh / 2.0) * height
        x1 = (cx + bw / 2.0) * width
        y1 = (cy + bh / 2.0) * height
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)

    if len(values) < 6 or len(values) % 2 != 0:
        return None

    pts = []
    for idx in range(0, len(values), 2):
        pts.append([values[idx] * width, values[idx + 1] * height])

    return np.array(pts, dtype=np.float32)


def extract_polygon_crop(image_bgr, polygon, crop_size, padding):
    height, width = image_bgr.shape[:2]

    min_xy = polygon.min(axis=0)
    max_xy = polygon.max(axis=0)
    bbox_w = max_xy[0] - min_xy[0]
    bbox_h = max_xy[1] - min_xy[1]

    pad_x = max(4.0, bbox_w * padding)
    pad_y = max(4.0, bbox_h * padding)

    x0 = int(np.floor(min_xy[0] - pad_x))
    y0 = int(np.floor(min_xy[1] - pad_y))
    x1 = int(np.ceil(max_xy[0] + pad_x))
    y1 = int(np.ceil(max_xy[1] + pad_y))
    x0, y0, x1, y1 = clamp_bbox(x0, y0, x1, y1, width, height)

    crop = image_bgr[y0:y1, x0:x1].copy()
    local_poly = polygon - np.array([x0, y0], dtype=np.float32)

    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [local_poly.astype(np.int32)], 255)
    crop[mask == 0] = 0

    return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)


def matching_label_path(image_name):
    split, _, image_file = image_name.partition("/images/")
    return f"{split}/labels/{Path(image_file).stem}.txt"


def main():
    args = parse_args()
    archive_path = Path(args.archive)
    output_dir = Path(args.output).resolve()

    ensure_clean_dirs(output_dir)

    counts = {split: Counter() for split in ("train", "val", "test")}
    manifest = {
        "archive": str(archive_path),
        "class_map": CLASS_MAP,
        "source_class_names": CLASS_NAMES,
        "crop_size": args.crop_size,
        "padding": args.padding,
        "samples": [],
    }

    with zipfile.ZipFile(archive_path) as zf:
        image_names = [
            name
            for name in zf.namelist()
            if "/images/" in name and name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        for image_name in image_names:
            source_split = image_name.split("/", 1)[0]
            split = SPLIT_MAP.get(source_split)
            if split is None:
                continue

            label_name = matching_label_path(image_name)
            if label_name not in zf.namelist():
                continue

            image_bytes = zf.read(image_name)
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"[SKIP] could not decode image: {image_name}")
                continue

            height, width = image_bgr.shape[:2]
            label_text = zf.read(label_name).decode("utf-8").strip()
            if not label_text:
                continue

            for idx, line in enumerate(label_text.splitlines()):
                parts = line.split()
                if not parts:
                    continue

                class_id = parts[0]
                mapped_label = CLASS_MAP.get(class_id)
                if mapped_label is None:
                    continue

                polygon = polygon_from_yolo(parts[1:], width, height)
                if polygon is None:
                    continue

                crop = extract_polygon_crop(
                    image_bgr=image_bgr,
                    polygon=polygon,
                    crop_size=args.crop_size,
                    padding=args.padding,
                )

                out_name = f"{Path(image_name).stem}_spot_{idx:03d}.jpg"
                out_path = output_dir / split / mapped_label / out_name
                cv2.imwrite(str(out_path), crop)

                counts[split][mapped_label] += 1
                manifest["samples"].append(
                    {
                        "split": split,
                        "label": mapped_label,
                        "source_class_id": class_id,
                        "source_image": image_name,
                        "source_label": label_name,
                        "output_image": str(out_path.relative_to(output_dir)),
                    }
                )

    manifest["counts"] = {split: dict(counter) for split, counter in counts.items()}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Created fine-tuning classifier dataset at:", output_dir)
    for split in ("train", "val", "test"):
        split_total = sum(counts[split].values())
        print(
            f"{split}: total={split_total} "
            f"vacant={counts[split]['vacant']} occupied={counts[split]['occupied']}"
        )


if __name__ == "__main__":
    main()
