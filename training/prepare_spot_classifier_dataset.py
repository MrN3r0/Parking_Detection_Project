import argparse
import json
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import cv2
import numpy as np


LABEL_MAP = {
    "free_parking_space": "vacant",
    "not_free_parking_space": "occupied",
    "partially_free_parking_space": "occupied",
}

SPLITS = ("train", "val", "test")
CLASSES = ("vacant", "occupied")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the parking-space polygon dataset in archive.zip into "
            "per-spot classifier crops."
        )
    )
    parser.add_argument(
        "--archive",
        default=str(Path.cwd().parent / "archive.zip"),
        help="Path to archive.zip containing annotations.xml and images/.",
    )
    parser.add_argument(
        "--output",
        default=str(Path("training") / "spot_classifier"),
        help="Output directory for the classifier dataset.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=224,
        help="Final square crop size written for each sample.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.12,
        help="Extra padding around each polygon crop, as a fraction of bbox size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when splitting by source image.",
    )
    return parser.parse_args()


def ensure_clean_dirs(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in SPLITS:
        for class_name in CLASSES:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)


def parse_points(points_text: str) -> np.ndarray:
    pts = []
    for item in points_text.split(";"):
        x_str, y_str = item.split(",")
        pts.append([float(x_str), float(y_str)])
    return np.array(pts, dtype=np.float32)


def clamp_bbox(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return x0, y0, x1, y1


def extract_polygon_crop(
    image_bgr: np.ndarray,
    polygon: np.ndarray,
    crop_size: int,
    padding: float,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]

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
    x0, y0, x1, y1 = clamp_bbox(x0, y0, x1, y1, w, h)

    crop = image_bgr[y0:y1, x0:x1].copy()
    local_poly = polygon - np.array([x0, y0], dtype=np.float32)

    # Mask out everything outside the traced parking-space polygon so the
    # classifier focuses on the spot content instead of neighboring stalls.
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [local_poly.astype(np.int32)], 255)
    crop[mask == 0] = 0

    return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)


def build_split_map(image_names: Iterable[str], seed: int) -> dict[str, str]:
    names = list(image_names)
    rng = random.Random(seed)
    rng.shuffle(names)

    total = len(names)
    if total == 0:
        return {}

    train_cut = max(1, int(round(total * 0.70)))
    val_cut = max(train_cut + 1, int(round(total * 0.85))) if total >= 3 else min(total, train_cut + 1)

    split_map: dict[str, str] = {}
    for idx, name in enumerate(names):
        if idx < train_cut:
            split_map[name] = "train"
        elif idx < val_cut:
            split_map[name] = "val"
        else:
            split_map[name] = "test"

    # Ensure all splits exist when there are enough source images.
    if total >= 3 and len(set(split_map.values())) < 3:
        ordered = list(names)
        split_map[ordered[0]] = "train"
        split_map[ordered[1]] = "val"
        split_map[ordered[2]] = "test"

    return split_map


def main() -> None:
    args = parse_args()

    archive_path = Path(args.archive).resolve()
    output_dir = Path(args.output).resolve()
    ensure_clean_dirs(output_dir)

    split_counts: dict[str, Counter] = {split: Counter() for split in SPLITS}
    image_to_records: dict[str, list[dict[str, object]]] = defaultdict(list)

    with zipfile.ZipFile(archive_path) as zf:
        with zf.open("annotations.xml") as f:
            root = ET.parse(f).getroot()

        image_elements = root.findall("image")
        split_map = build_split_map((img.get("name") for img in image_elements), args.seed)

        for image_el in image_elements:
            image_name = image_el.get("name")
            if not image_name:
                continue

            image_bytes = zf.read(image_name)
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Could not decode image from archive: {image_name}")

            split = split_map[image_name]
            image_stem = Path(image_name).stem
            example_index = 0

            for polygon_el in image_el.findall("polygon"):
                raw_label = polygon_el.get("label", "")
                mapped_label = LABEL_MAP.get(raw_label)
                if mapped_label is None:
                    continue

                points_text = polygon_el.get("points", "")
                polygon = parse_points(points_text)
                crop = extract_polygon_crop(
                    image_bgr=image_bgr,
                    polygon=polygon,
                    crop_size=args.crop_size,
                    padding=args.padding,
                )

                out_name = f"{image_stem}_spot_{example_index:03d}.png"
                out_path = output_dir / split / mapped_label / out_name
                cv2.imwrite(str(out_path), crop)

                record = {
                    "split": split,
                    "label": mapped_label,
                    "source_image": image_name,
                    "output_image": str(out_path.relative_to(output_dir)),
                    "original_label": raw_label,
                    "points": polygon.astype(float).tolist(),
                }
                image_to_records[image_name].append(record)
                split_counts[split][mapped_label] += 1
                example_index += 1

    manifest = {
        "archive": str(archive_path),
        "crop_size": args.crop_size,
        "padding": args.padding,
        "seed": args.seed,
        "label_map": LABEL_MAP,
        "counts": {split: dict(counter) for split, counter in split_counts.items()},
        "images": image_to_records,
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Created classifier dataset at:", output_dir)
    for split in SPLITS:
        counts = split_counts[split]
        total = sum(counts.values())
        print(f"{split}: total={total} vacant={counts['vacant']} occupied={counts['occupied']}")


if __name__ == "__main__":
    main()
