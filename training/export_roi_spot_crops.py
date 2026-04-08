import json
from pathlib import Path

import cv2

from spots import extract_spot_crop, normalize_spots, scale_spots_to_frame


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = PROJECT_ROOT / "config.json"
OUTPUT_ROOT = PROJECT_ROOT / "training" / "local_finetune_review"


def load_cfg():
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def export_image_spots(image_path, spots, crop_size, crop_padding):
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[SKIP] could not read: {image_path}")
        return 0

    frame_h, frame_w = frame.shape[:2]
    count = 0

    for spot in spots:
        crop = extract_spot_crop(
            frame,
            spot["poly"],
            padding_ratio=crop_padding,
            crop_size=crop_size,
        )
        if crop is None:
            continue

        out_name = f"{image_path.stem}__{spot['spot_id']}.jpg"
        out_path = OUTPUT_ROOT / "unsorted" / out_name
        cv2.imwrite(str(out_path), crop)
        count += 1

    print(f"[OK] {image_path.name}: exported {count} crops from {frame_w}x{frame_h}")
    return count


def main():
    cfg = load_cfg()

    crop_size = int(cfg.get("classifier_crop_size", 224))
    crop_padding = float(cfg.get("classifier_crop_padding", 0.10))
    source_image = cfg.get("source_image", "test_images/Parking_Photo.jpg")
    same_view_images = cfg.get("same_view_images", [source_image])

    spots = cfg.get("spots", [])
    if not spots:
        raise RuntimeError("No spots found in config.json.")

    spots = normalize_spots(spots)

    trace_size = cfg.get("trace_image_size")
    if not trace_size:
        raise RuntimeError("trace_image_size is required in config.json.")

    trace_w = int(trace_size["w"])
    trace_h = int(trace_size["h"])

    (OUTPUT_ROOT / "unsorted").mkdir(parents=True, exist_ok=True)

    manifest = {
        "crop_size": crop_size,
        "crop_padding": crop_padding,
        "trace_image_size": {"w": trace_w, "h": trace_h},
        "images": [],
    }

    total = 0
    for rel_path in same_view_images:
        image_path = (PROJECT_ROOT / rel_path).resolve()
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[SKIP] missing or unreadable: {rel_path}")
            continue

        frame_h, frame_w = frame.shape[:2]
        image_spots = spots
        if (frame_w, frame_h) != (trace_w, trace_h):
            image_spots = scale_spots_to_frame(spots, trace_w, trace_h, frame_w, frame_h)
            print(f"[SCALE] {rel_path}: {trace_w}x{trace_h} -> {frame_w}x{frame_h}")

        count = export_image_spots(image_path, image_spots, crop_size, crop_padding)
        total += count
        manifest["images"].append(
            {
                "image": rel_path,
                "width": frame_w,
                "height": frame_h,
                "exported_crops": count,
            }
        )

    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] exported {total} crops")
    print(f"[DONE] review folder: {OUTPUT_ROOT / 'unsorted'}")
    print(f"[DONE] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
