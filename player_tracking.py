#!/usr/bin/env python3
"""Step 2: offline person detection and ID tracking for badminton footage.

Uses the permissively licensed RF-DETR Nano/Small detector and ByteTrack.
Does not identify named athletes, recognize strokes, or measure shuttle speed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from badminton_proto import (
    VERSION, _font, _run, build_contact_sheet,
    probe_video, project_points, sha256_file,
)


TRACKING_SCHEMA_VERSION = "0.2.3"
COLORS = ("#b9ff52", "#42dfdd", "#ffa55b", "#eb88f4", "#fc7575", "#72b7ff")


def parse_roi(value: str | None) -> tuple[float, float, float, float] | None:
    """Optional normalized rectangle x_min,y_min,x_max,y_max (0..1)."""
    if value is None:
        return None
    try:
        box = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("--roi must contain four numbers separated by commas") from exc
    if len(box) != 4 or not all(np.isfinite(box)):
        raise ValueError("--roi must contain exactly four finite numbers")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("--roi values must form a rectangle inside 0..1")
    return box


def read_calibration(path: Path | None, camera_id: str, image_size: tuple[int, int]) -> np.ndarray | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("camera_id") != camera_id:
        raise ValueError("Calibration camera_id does not match --camera")
    size = data.get("image_size", {})
    if (size.get("width_px"), size.get("height_px")) != image_size:
        raise ValueError("Calibration image dimensions do not match the video; use a full-resolution frame")
    matrix = np.asarray(data.get("pixel_to_court_homography"), dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Calibration does not contain a valid 3x3 homography")
    if abs(np.linalg.det(matrix)) < 1e-14:
        raise ValueError("Calibration homography is singular")
    return matrix


def foot_points(boxes: np.ndarray) -> np.ndarray:
    """Approximate floor contact at the bottom-centre of each person box."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]))


def person_detection_mask(detections: Any, fallback_class_id: int = 1) -> np.ndarray:
    """Select 'person' by model-supplied name, falling back to COCO category ID 1.

    Pretrained RF-DETR COCO checkpoints can emit sparse COCO IDs (1 = person),
    while some fine-tuned checkpoints use zero-based IDs. Prefer class names when
    available so we do not silently confuse the two conventions.
    """
    if detections.class_id is None:
        raise ValueError("Detector returned no class IDs; person labels are required")
    labels = getattr(detections, "data", None)
    names = labels.get("class_name") if isinstance(labels, dict) else None
    if names is not None:
        if len(names) != len(detections):
            raise ValueError("Detector class_name count does not match detection count")
        return np.array([str(name).strip().casefold() == "person" for name in names], dtype=bool)
    return np.asarray(detections.class_id, dtype=int) == fallback_class_id


def select_person_detections(
    detections: Any,
    image_size: tuple[int, int],
    roi: tuple[float, float, float, float] | None,
    homography: np.ndarray | None,
    person_class_id: int = 1,
) -> Any:
    """Keep model-labeled people, optionally restricted to the court footprint."""
    persons = detections[person_detection_mask(detections, person_class_id)]
    if len(persons) == 0:
        return persons
    feet = foot_points(persons.xyxy)
    keep = np.ones(len(persons), dtype=bool)
    if roi is not None:
        width, height = image_size
        x1, y1, x2, y2 = roi
        keep &= (feet[:, 0] >= x1 * width) & (feet[:, 0] <= x2 * width)
        keep &= (feet[:, 1] >= y1 * height) & (feet[:, 1] <= y2 * height)
    if homography is not None:
        court = project_points(feet, homography)
        # Some bounding boxes slightly overshoot the visible floor boundary.
        keep &= (court[:, 0] >= -0.30) & (court[:, 0] <= 6.40)
        keep &= (court[:, 1] >= -0.30) & (court[:, 1] <= 13.70)
    return persons[keep]


def predict_with_crop(
    detector: Any,
    image: Image.Image,
    confidence: float,
    detect_roi: tuple[float, float, float, float] | None,
) -> Any:
    """Optionally enlarge a court crop for inference; return boxes in original pixels."""
    if detect_roi is None:
        return detector.predict(image, threshold=confidence)
    width, height = image.size
    x1, y1, x2, y2 = detect_roi
    bounds = (
        int(x1 * width), int(y1 * height),
        max(1, int(np.ceil(x2 * width))), max(1, int(np.ceil(y2 * height))),
    )
    crop = image.crop(bounds)
    detections = detector.predict(crop, threshold=confidence)
    # RF-DETR's crop-relative bounding boxes must be mapped to the full frame
    # before court filtering, video annotation and metric calculations.
    detections.xyxy = np.asarray(detections.xyxy, dtype=float) + np.array(
        [bounds[0], bounds[1], bounds[0], bounds[1]], dtype=float,
    )
    return detections


def make_detector_and_tracker(
    model_name: str,
    sample_fps: float,
    track_high_confidence: float,
    track_activation: float,
) -> tuple[Any, Any]:
    try:
        from rfdetr import RFDETRNano, RFDETRSmall
        from trackers import ByteTrackTracker
    except ImportError as exc:
        raise RuntimeError(
            "Vision packages are missing. Run: python -m pip install -r requirements-vision.txt"
        ) from exc
    model_type = {"nano": RFDETRNano, "small": RFDETRSmall}[model_name]
    # RF-DETR downloads its pretrained COCO checkpoint on the first run.
    return model_type(), ByteTrackTracker(
        frame_rate=sample_fps,
        high_conf_det_threshold=track_high_confidence,
        track_activation_threshold=track_activation,
    )


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _detections_to_records(detections: Any, homography: np.ndarray | None) -> list[dict[str, Any]]:
    if len(detections) == 0:
        return []
    ids = detections.tracker_id
    confidence = detections.confidence
    feet = foot_points(detections.xyxy)
    court = project_points(feet, homography) if homography is not None else None
    result = []
    for i, bbox in enumerate(detections.xyxy):
        track_id = int(ids[i]) if ids is not None and int(ids[i]) >= 0 else None
        item = {
            "track_id": track_id,
            "confidence": round(float(confidence[i]), 4) if confidence is not None else None,
            "bbox_xyxy_px": [round(float(n), 1) for n in bbox],
            "foot_px": [round(float(n), 1) for n in feet[i]],
        }
        if court is not None:
            item["foot_court_xy_m"] = [round(float(n), 3) for n in court[i]]
        result.append(item)
    return result


def annotate_frame(image: Image.Image, records: list[dict[str, Any]], trails: dict[int, deque]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for person in records:
        track_id = person["track_id"]
        color = COLORS[track_id % len(COLORS)] if track_id is not None else "#aeb7bf"
        xyxy = person["bbox_xyxy_px"]
        foot = tuple(person["foot_px"])
        if track_id is not None:
            trails[track_id].append(foot)
            if len(trails[track_id]) >= 2:
                draw.line(list(trails[track_id]), fill=color, width=3)
        draw.rectangle(xyxy, outline=color, width=3)
        draw.ellipse((foot[0] - 5, foot[1] - 5, foot[0] + 5, foot[1] + 5), fill=color)
        label = f"ID {track_id}" if track_id is not None else "Unconfirmed"
        if person["confidence"] is not None:
            label += f"  {person['confidence']:.2f}"
        x = int(max(0, min(image.width - 1, xyxy[0])))
        y = int(max(0, min(image.height - 24, xyxy[1] - 25)))
        text_width = max(1, min(image.width - x, int(len(label) * 9 + 10)))
        draw.rectangle((x, y, x + text_width, y + 24), fill="#15232a")
        draw.text((x + 4, y + 4), label, fill=color, font=_font(14))
    draw.rectangle((0, image.height - 25, min(image.width, 315), image.height), fill="#15232a")
    draw.text((5, image.height - 22), "PERSON TRACKING - UNVERIFIED", fill="white", font=_font(12))
    return output


def annotate_detector_frame(
    image: Image.Image,
    raw_persons: Any,
    filtered_persons: Any,
    roi: tuple[float, float, float, float] | None,
    detect_roi: tuple[float, float, float, float] | None,
) -> Image.Image:
    """Diagnostic view before ByteTrack. Grey = raw person, orange = court kept."""
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for group, color, width in ((raw_persons, "#a0b1bb", 1), (filtered_persons, "#ffa33a", 3)):
        for box in group.xyxy:
            draw.rectangle(tuple(float(n) for n in box), outline=color, width=width)
    for region, color in ((roi, "#ffe062"), (detect_roi, "#42dfdd")):
        if region is None:
            continue
        left, top, right, bottom = region
        draw.rectangle(
            (left * image.width, top * image.height, right * image.width, bottom * image.height),
            outline=color, width=2,
        )
    caption = f"DETECTOR: {len(raw_persons)} PERSON / {len(filtered_persons)} IN COURT"
    draw.rectangle((0, image.height - 26, image.width, image.height), fill="#15232a")
    draw.text((5, image.height - 22), caption, fill="#ffa33a", font=_font(12))
    return output


def _extract_sampled_video(video: Path, frame_dir: Path, fps: float, max_seconds: float | None, start_seconds: float = 0.0) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start_seconds:
        command.extend(["-ss", str(start_seconds)])
    command.extend(["-i", str(video)])
    if max_seconds is not None:
        command.extend(["-t", str(max_seconds)])
    command.extend(["-vf", f"fps={fps:g}", "-q:v", "3", str(frame_dir / "frame_%06d.jpg")])
    _run(command)
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("FFmpeg did not extract any frames")
    return frames


def _encode_preview(frame_dir: Path, output_file: Path, fps: float) -> None:
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", f"{fps:g}", "-start_number", "1",
        "-i", str(frame_dir / "frame_%06d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24",
        "-movflags", "+faststart", str(output_file),
    ])
    if not output_file.is_file():
        raise RuntimeError("Annotated preview was not created")


def track_players(
    video: Path,
    output: Path,
    camera_id: str = "court-01",
    sample_fps: float = 5.0,
    max_seconds: float | None = None,
    model_name: str = "nano",
    confidence: float = 0.25,
    roi: tuple[float, float, float, float] | None = None,
    detect_roi: tuple[float, float, float, float] | None = None,
    track_high_confidence: float = 0.6,
    track_activation: float = 0.7,
    person_class_id: int = 1,
    calibration_path: Path | None = None,
    detector: Any | None = None,
    tracker: Any | None = None,
    start_seconds: float = 0.0,
) -> Path:
    if model_name not in ("nano", "small"):
        raise ValueError("Model must be nano or small")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1")
    if not (0 < track_high_confidence <= track_activation < 1):
        raise ValueError("Tracker thresholds must satisfy 0 < high-confidence <= activation < 1")
    if not 0 <= person_class_id <= 1000:
        raise ValueError("--person-class-id must be between 0 and 1000")
    meta = probe_video(video)
    if not 0 < sample_fps <= meta["frame_rate_fps"]:
        raise ValueError("--sample-fps must be positive and no higher than the source video FPS")
    if not np.isfinite(start_seconds) or not 0 <= start_seconds < meta["duration_seconds"]:
        raise ValueError("--start-seconds must be finite, non-negative and before the video ends")
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("--max-seconds must be finite and positive")
    if (detector is None) != (tracker is None):
        raise ValueError("Tests must supply both detector and tracker, or neither")
    calibration = read_calibration(calibration_path, camera_id, (meta["width_px"], meta["height_px"]))
    output.mkdir(parents=True, exist_ok=True)
    if detector is None:
        detector, tracker = make_detector_and_tracker(
            model_name, sample_fps, track_high_confidence, track_activation,
        )
    assert tracker is not None
    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=20))
    ids_seen: set[int] = set()
    total_detections = 0
    stage_totals = {
        "all_class_detections": 0,
        "raw_person_detections": 0,
        "court_filtered_person_detections": 0,
        "tracker_output_boxes": 0,
        "boxes_with_track_id": 0,
    }
    class_id_histogram: Counter[int] = Counter()
    review_frames: list[dict[str, Any]] = []
    detector_review_frames: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="badminton-tracking-") as directory:
        temp = Path(directory)
        frames = _extract_sampled_video(video, temp / "decoded", sample_fps, max_seconds, start_seconds)
        annotated = temp / "annotated"
        annotated.mkdir()
        sample_indices = set(np.linspace(0, len(frames) - 1, min(9, len(frames)), dtype=int).tolist())
        review_dir = output / "review_frames"
        review_dir.mkdir(exist_ok=True)
        detector_review_dir = output / "detector_review"
        detector_review_dir.mkdir(exist_ok=True)
        jsonl = output / "player_tracks.jsonl"
        with jsonl.open("w", encoding="utf-8") as stream:
            for i, frame_path in enumerate(frames):
                clip_timestamp = round(i / sample_fps, 3)
                timestamp = round(start_seconds + i / sample_fps, 3)
                with Image.open(frame_path) as image_file:
                    image = image_file.convert("RGB")
                detections = predict_with_crop(detector, image, confidence, detect_roi)
                if detections.class_id is None:
                    raise ValueError("Detector returned no class IDs; person labels are required")
                class_id_histogram.update(int(class_id) for class_id in detections.class_id)
                raw_persons = detections[person_detection_mask(detections, person_class_id)]
                filtered = select_person_detections(detections, image.size, roi, calibration, person_class_id)
                tracked = tracker.update(filtered, timestamp=clip_timestamp)
                records = _detections_to_records(tracked, calibration)
                stage = {
                    "all_class_detections": len(detections),
                    "raw_person_detections": len(raw_persons),
                    "court_filtered_person_detections": len(filtered),
                    "tracker_output_boxes": len(tracked),
                    "boxes_with_track_id": sum(p["track_id"] is not None for p in records),
                }
                for key, count in stage.items():
                    stage_totals[key] += count
                ids_seen.update(p["track_id"] for p in records if p["track_id"] is not None)
                total_detections += len(records)
                row = {"frame_index": i, "timestamp_seconds": timestamp, "persons": records, "stage_counts": stage}
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                preview = annotate_frame(image, records, trails)
                preview.save(annotated / f"frame_{i+1:06d}.jpg", quality=90)
                if i in sample_indices:
                    target = review_dir / f"frame_{len(review_frames)+1:02d}.jpg"
                    shutil.copyfile(annotated / f"frame_{i+1:06d}.jpg", target)
                    review_frames.append({"timestamp_seconds": timestamp, "image": str(target.relative_to(output))})
                    detector_frame = annotate_detector_frame(image, raw_persons, filtered, roi, detect_roi)
                    detector_file = detector_review_dir / f"frame_{len(detector_review_frames)+1:02d}.jpg"
                    detector_frame.save(detector_file, quality=92)
                    detector_review_frames.append({
                        "timestamp_seconds": timestamp,
                        "image": detector_file.name,
                    })
        preview_path = output / "tracked_preview.mp4"
        _encode_preview(annotated, preview_path, sample_fps)
    contact = build_contact_sheet(output, review_frames)
    detector_contact = build_contact_sheet(detector_review_dir, detector_review_frames)
    warnings = [
        "Track IDs may change after occlusions or camera cuts; they are not verified athlete identities.",
        "Bounding-box foot positions are approximate and must be validated by a coach.",
        "This step does not detect shuttlecocks, hits, rallies, or shot types.",
    ]
    if calibration is None:
        warnings.append("No calibration: coordinates are image pixels, not metres on court.")
    if roi is None and calibration is None:
        warnings.append("No court region: spectators and officials can be detected as people.")
    if stage_totals["raw_person_detections"] == 0:
        warnings.append("No person boxes after label selection: inspect diagnostics.class_id_histogram and detector_review.")
    elif stage_totals["court_filtered_person_detections"] == 0:
        warnings.append("All person boxes were rejected by the court region: review --roi or calibration.")
    elif stage_totals["tracker_output_boxes"] == 0:
        warnings.append("Tracker discarded all court person boxes: try lower --track-high-confidence and --track-activation.")
    elif not ids_seen:
        warnings.append("People were detected but no stable IDs assigned: tune tracking thresholds and sampling FPS.")
    summary = {
        "schema_version": TRACKING_SCHEMA_VERSION,
        "step_1_tool_version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": f"{camera_id}-{sha256_file(video)[:12]}",
        "camera_id": camera_id,
        "video": meta,
        "model": {
            "detector": "RF-DETR " + model_name,
            "detector_package_version": _package_version("rfdetr"),
            "tracker": "ByteTrack",
            "tracker_package_version": _package_version("trackers"),
            "confidence_threshold": confidence,
            "track_high_confidence_threshold": track_high_confidence,
            "track_activation_threshold": track_activation,
            "person_class_id_fallback": person_class_id,
            "person_label_selection": "class_name == person if supplied, otherwise COCO class_id fallback",
        },
        "sampling": {
            "fps": sample_fps,
            "start_seconds": start_seconds,
            "max_seconds": max_seconds,
            "frames_processed": len(frames),
            "timestamps": "source video seconds: start_seconds + frame_index / fps; approximate original video time",
        },
        "region": {
            "normalized_roi": list(roi) if roi else None,
            "normalized_detector_crop": list(detect_roi) if detect_roi else None,
            "calibration_provided": calibration is not None,
        },
        "diagnostics": {
            "unique_track_ids_observed": len(ids_seen),
            "person_boxes_reported": total_detections,
            "stage_totals": stage_totals,
            "class_id_histogram": {str(class_id): count for class_id, count in sorted(class_id_histogram.items())},
            "note": "These are diagnostic counts, not validated athlete counts.",
        },
        "artifacts": {
            "tracks_jsonl": "player_tracks.jsonl",
            "annotated_video": preview_path.name,
            "contact_sheet": contact,
            "review_frames": review_frames,
            "detector_contact_sheet": str(Path("detector_review") / detector_contact),
            "detector_review_frames": [
                {"timestamp_seconds": item["timestamp_seconds"], "image": str(Path("detector_review") / item["image"])}
                for item in detector_review_frames
            ],
        },
        "warnings": warnings,
        "coaching_metrics": {"status": "not_available"},
    }
    target = output / "tracking_summary.json"
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 2 badminton player detection and tracking")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--camera", default="court-01")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Seek into the source video before sampling; exported timestamps remain source-relative")
    parser.add_argument("--model", choices=("nano", "small"), default="nano")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--roi", default=None, help="Optional normalized court rectangle: left,top,right,bottom")
    parser.add_argument("--detect-roi", default=None, help="Optional normalized crop for model inference; boxes are restored to full-frame pixels")
    parser.add_argument("--track-high-confidence", type=float, default=0.6)
    parser.add_argument("--track-activation", type=float, default=0.7)
    parser.add_argument("--person-class-id", type=int, default=1, help="Fallback if model omits class_name; pretrained COCO person ID is 1")
    parser.add_argument("--calibration", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        summary = track_players(
            video=args.video, output=args.out, camera_id=args.camera,
            sample_fps=args.sample_fps, max_seconds=args.max_seconds,
            start_seconds=args.start_seconds,
            model_name=args.model, confidence=args.confidence,
            roi=parse_roi(args.roi), detect_roi=parse_roi(args.detect_roi),
            track_high_confidence=args.track_high_confidence,
            track_activation=args.track_activation,
            person_class_id=args.person_class_id,
            calibration_path=args.calibration,
        )
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Created {summary.resolve()}")
    print(f"Preview: {(summary.parent / 'tracked_preview.mp4').resolve()}")
    print(f"Review sheet: {(summary.parent / 'review_contact_sheet.jpg').resolve()}")
    data = json.loads(summary.read_text(encoding="utf-8"))
    stages = data["diagnostics"]["stage_totals"]
    print("Detections: " + ", ".join(f"{key}={value}" for key, value in stages.items()))
    print(f"Detector-only sheet: {(summary.parent / 'detector_review' / 'review_contact_sheet.jpg').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
