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


TRACKING_SCHEMA_VERSION = "0.4.1"
COLORS = ("#b9ff52", "#42dfdd", "#ffa55b", "#eb88f4", "#fc7575", "#72b7ff")


def scene_thumbnail(image: Image.Image) -> np.ndarray:
    """Small full-frame RGB image for inexpensive edit-boundary checks."""
    return np.asarray(image.convert("RGB").resize((64, 36), Image.Resampling.BILINEAR), dtype=float) / 255


def scene_change_score(previous: np.ndarray, current: np.ndarray) -> float:
    """Mean absolute RGB change, normalized to 0..1; not a view classifier."""
    return float(np.abs(current - previous).mean())


def suppress_duplicate_people(detections: Any, iou_threshold: float) -> Any:
    """Score-ordered person NMS; keep metadata and avoid merging boxes."""
    if not iou_threshold or len(detections) < 2:
        return detections
    boxes = np.asarray(detections.xyxy, dtype=float)
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    scores = detections.confidence if detections.confidence is not None else np.ones(len(boxes))
    edge_clipped = np.asarray(getattr(detections, "data", {}).get("crop_edge_clipped", np.zeros(len(boxes))), dtype=bool)
    order = np.argsort(-np.asarray(scores), kind="stable")
    kept = []
    while len(order):
        index = order[0]
        remaining = order[1:]
        intersection_size = np.maximum(
            0, np.minimum(boxes[index, 2:], boxes[remaining, 2:])
            - np.maximum(boxes[index, :2], boxes[remaining, :2]),
        )
        intersection = intersection_size[:, 0] * intersection_size[:, 1]
        # Crop truncation can score slightly higher than the complete person.
        # Prefer a covering non-edge box with comparable confidence in that case.
        if edge_clipped[index] and areas[index] > 0:
            fuller = (~edge_clipped[remaining] & (areas[remaining] > areas[index])
                      & (intersection / areas[index] >= .65)
                      & (np.asarray(scores)[remaining] >= scores[index] - .10))
            if np.any(fuller):
                order = remaining
                continue
        kept.append(index)
        union = areas[index] + areas[remaining] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        coverage = np.divide(intersection, areas[remaining], out=np.zeros_like(intersection), where=areas[remaining] > 0)
        edge_fragment = edge_clipped[remaining] & (areas[index] > areas[remaining]) & (coverage >= .65)
        order = remaining[(iou <= iou_threshold) & ~edge_fragment]
    return detections[np.asarray(sorted(kept), dtype=int)]


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
        detections = detector.predict(image, threshold=confidence)
        detections.data["crop_edge_clipped"] = np.zeros(len(detections), dtype=bool)
        return detections
    width, height = image.size
    x1, y1, x2, y2 = detect_roi
    bounds = (
        int(x1 * width), int(y1 * height),
        max(1, int(np.ceil(x2 * width))), max(1, int(np.ceil(y2 * height))),
    )
    crop = image.crop(bounds)
    detections = detector.predict(crop, threshold=confidence)
    crop_boxes = np.asarray(detections.xyxy, dtype=float)
    # A box touching an inference-crop edge may be just a body fragment.
    # Full-frame edges are excluded because a second crop cannot restore them.
    detections.data["crop_edge_clipped"] = (
        ((crop_boxes[:, 0] <= 2) & (bounds[0] > 0))
        | ((crop_boxes[:, 1] <= 2) & (bounds[1] > 0))
        | ((crop_boxes[:, 2] >= crop.width - 2) & (bounds[2] < width))
        | ((crop_boxes[:, 3] >= crop.height - 2) & (bounds[3] < height))
    )
    # RF-DETR's crop-relative bounding boxes must be mapped to the full frame
    # before court filtering, video annotation and metric calculations.
    detections.xyxy = np.asarray(detections.xyxy, dtype=float) + np.array(
        [bounds[0], bounds[1], bounds[0], bounds[1]], dtype=float,
    )
    # RF-DETR 1.11 carries crop-specific image metadata too. Once coordinates
    # are restored, those fields must describe the original frame for merging.
    if "source_image" in getattr(detections, "metadata", {}):
        detections.metadata["source_image"] = np.asarray(image)
    if "source_shape" in getattr(detections, "data", {}):
        detections.data["source_shape"] = np.tile(np.array([height, width]), (len(detections), 1))
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


def predict_with_secondary_crop(detector: Any, image: Image.Image, confidence: float,
                                detect_roi: Any, secondary_roi: Any,
                                secondary_confidence: float) -> tuple[Any, int]:
    primary = predict_with_crop(detector, image, confidence, detect_roi)
    if secondary_roi is None:
        return primary, 0
    secondary = predict_with_crop(detector, image, secondary_confidence, secondary_roi)
    # supervision preserves class names and other per-box metadata. Keep both
    # passes in raw diagnostics, then suppress person duplicates before tracking.
    return type(primary).merge([primary, secondary]), len(secondary)


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
    """Before ByteTrack: grey = raw people, orange = final tracker input."""
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
    caption = f"DETECTOR: {len(raw_persons)} PERSON / {len(filtered_persons)} TRACK INPUT"
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
    scene_cut_threshold: float = 0.15,
    court_reference_path: Path | None = None,
    court_view_threshold: float = 0.08,
    person_nms_iou: float = 0.5,
    secondary_detect_roi: tuple[float, float, float, float] | None = None,
    secondary_confidence: float = 0.35,
) -> Path:
    if model_name not in ("nano", "small"):
        raise ValueError("Model must be nano or small")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1")
    if not (0 < track_high_confidence <= track_activation < 1):
        raise ValueError("Tracker thresholds must satisfy 0 < high-confidence <= activation < 1")
    if not 0 <= person_class_id <= 1000:
        raise ValueError("--person-class-id must be between 0 and 1000")
    if not np.isfinite(scene_cut_threshold) or not 0 <= scene_cut_threshold <= 1:
        raise ValueError("--scene-cut-threshold must be finite and between 0 and 1 (0 disables)")
    if not np.isfinite(court_view_threshold) or not 0 < court_view_threshold <= 1:
        raise ValueError("--court-view-threshold must be finite and in (0, 1]")
    if not np.isfinite(person_nms_iou) or not 0 <= person_nms_iou <= 1:
        raise ValueError("--person-nms-iou must be finite and in [0, 1] (0 disables)")
    if not 0 < secondary_confidence < 1:
        raise ValueError("--secondary-confidence must be between 0 and 1")
    if secondary_detect_roi is not None and not person_nms_iou:
        raise ValueError("Secondary-crop inference requires person NMS to remove duplicate boxes")
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
    court_reference = None
    if court_reference_path is not None:
        with Image.open(court_reference_path) as reference:
            if reference.size != (meta["width_px"], meta["height_px"]):
                raise ValueError("Court reference dimensions must match the source video")
            court_reference = scene_thumbnail(reference)
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
        "secondary_all_class_detections": 0,
        "raw_person_detections": 0,
        "court_filtered_person_detections": 0,
        "deduplicated_person_detections": 0,
        "view_accepted_person_detections": 0,
        "tracker_output_boxes": 0,
        "boxes_with_track_id": 0,
    }
    class_id_histogram: Counter[int] = Counter()
    review_frames: list[dict[str, Any]] = []
    detector_review_frames: list[dict[str, Any]] = []
    scene_id = 0
    scene_events: list[dict[str, Any]] = []
    scene_track_ids: dict[int, int] = {}
    active_calibration = calibration
    previous_view_match: bool | None = None
    rejected_view_frames = 0
    with TemporaryDirectory(prefix="badminton-tracking-") as directory:
        temp = Path(directory)
        frames = _extract_sampled_video(video, temp / "decoded", sample_fps, max_seconds, start_seconds)
        annotated = temp / "annotated"
        annotated.mkdir()
        sample_indices = set(np.linspace(0, len(frames) - 1, min(9, len(frames)), dtype=int).tolist())
        # Check full frames, not detector crops. Include both sides of every
        # candidate boundary in the review sheets so resets can be audited.
        cut_scores: dict[int, float] = {}
        if scene_cut_threshold or court_reference is not None:
            previous = None
            previous_match = None
            for index, path in enumerate(frames):
                with Image.open(path) as source:
                    current = scene_thumbnail(source)
                if previous is not None and scene_cut_threshold:
                    score = scene_change_score(previous, current)
                    if score >= scene_cut_threshold:
                        cut_scores[index] = score
                        sample_indices.update((index - 1, index))
                if court_reference is not None:
                    match = scene_change_score(court_reference, current) <= court_view_threshold
                    if previous_match is not None and match != previous_match:
                        sample_indices.update((index - 1, index))
                    previous_match = match
                previous = current
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
                view_difference = scene_change_score(court_reference, scene_thumbnail(image)) if court_reference is not None else None
                view_match = view_difference is None or view_difference <= court_view_threshold
                view_changed = previous_view_match is not None and view_match != previous_view_match
                boundary = i in cut_scores or view_changed
                if boundary:
                    tracker.reset()
                    trails.clear()
                    scene_track_ids.clear()
                    scene_id += 1
                    # A different view cannot reuse the original floor mapping,
                    # even if a later shot appears to return to the same court.
                    active_calibration = None
                    scene_events.append({"frame_index": i, "timestamp_seconds": timestamp,
                                         "scene_id": scene_id, "change_score": round(cut_scores[i], 4) if i in cut_scores else None,
                                         "reason": "candidate_cut" if i in cut_scores else "reference_view_transition"})
                previous_view_match = view_match
                rejected_view_frames += int(not view_match)
                detections, secondary_count = predict_with_secondary_crop(
                    detector, image, confidence, detect_roi,
                    secondary_detect_roi if view_match else None, secondary_confidence,
                )
                if detections.class_id is None:
                    raise ValueError("Detector returned no class IDs; person labels are required")
                class_id_histogram.update(int(class_id) for class_id in detections.class_id)
                raw_persons = detections[person_detection_mask(detections, person_class_id)]
                filtered = select_person_detections(detections, image.size, roi, active_calibration, person_class_id)
                deduplicated = suppress_duplicate_people(filtered, person_nms_iou)
                accepted = deduplicated if view_match else deduplicated[np.zeros(len(deduplicated), dtype=bool)]
                tracked = tracker.update(accepted, timestamp=clip_timestamp)
                records = _detections_to_records(tracked, active_calibration)
                for person in records:
                    local_id = person["track_id"]
                    if scene_id and local_id is not None:
                        if local_id not in scene_track_ids:
                            scene_track_ids[local_id] = max(ids_seen | set(scene_track_ids.values()), default=-1) + 1
                        person["track_id"] = scene_track_ids[local_id]
                stage = {
                    "all_class_detections": len(detections),
                    "secondary_all_class_detections": secondary_count,
                    "raw_person_detections": len(raw_persons),
                    "court_filtered_person_detections": len(filtered),
                    "deduplicated_person_detections": len(deduplicated),
                    "view_accepted_person_detections": len(accepted),
                    "tracker_output_boxes": len(tracked),
                    "boxes_with_track_id": sum(p["track_id"] is not None for p in records),
                }
                for key, count in stage.items():
                    stage_totals[key] += count
                ids_seen.update(p["track_id"] for p in records if p["track_id"] is not None)
                total_detections += len(records)
                view_status = "requires_review" if scene_id == 0 else "invalidated_after_cut"
                if court_reference is not None:
                    view_status = "reference_match" if view_match else "reference_mismatch"
                geometry_status = "not_calibrated"
                if active_calibration is not None:
                    geometry_status = "available_unverified"
                elif calibration is not None and scene_id:
                    geometry_status = "invalidated_after_boundary"
                row = {
                    "frame_index": i, "timestamp_seconds": timestamp,
                    "persons": records, "stage_counts": stage,
                    "scene_id": scene_id, "scene_cut": boundary,
                    "court_view_difference": round(view_difference, 4) if view_difference is not None else None,
                    "court_geometry_status": geometry_status, "court_view_status": view_status,
                }
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                preview = annotate_frame(image, records, trails)
                badge = ImageDraw.Draw(preview)
                badge.rectangle((max(0, image.width - 290), 0, image.width, 25), fill="#15232a")
                badge.text((max(0, image.width - 284), 5), f"Scene {scene_id} | {row['court_view_status']}", fill="yellow", font=_font(14))
                preview.save(annotated / f"frame_{i+1:06d}.jpg", quality=90)
                if i in sample_indices:
                    target = review_dir / f"frame_{len(review_frames)+1:02d}.jpg"
                    shutil.copyfile(annotated / f"frame_{i+1:06d}.jpg", target)
                    review_frames.append({"timestamp_seconds": timestamp, "image": str(target.relative_to(output))})
                    detector_frame = annotate_detector_frame(image, raw_persons, accepted, roi, detect_roi)
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
    if scene_events:
        warnings.append("Candidate camera cuts detected: tracker and trails reset; court calibration disabled after the first cut. Review every scene before using positions.")
    warnings.append("Scene checks do not classify court views, detect every edit, or validate the initial view. Detector crop and ROI remain unverified after camera changes.")
    if court_reference is not None:
        warnings.append("Reference-view matching is an experimental fixed-view similarity check, not court recognition or calibration validation. Rejected frames retain detector diagnostics but export no player tracks.")
    if roi is None and calibration is None:
        warnings.append("No court region: spectators and officials can be detected as people.")
    if stage_totals["raw_person_detections"] == 0:
        warnings.append("No person boxes after label selection: inspect diagnostics.class_id_histogram and detector_review.")
    elif stage_totals["court_filtered_person_detections"] == 0:
        warnings.append("All person boxes were rejected by the court region: review --roi or calibration.")
    elif stage_totals["view_accepted_person_detections"] == 0:
        warnings.append("All court candidates were rejected by reference-view matching; inspect the court reference and detector diagnostics.")
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
            "person_nms_iou": person_nms_iou,
            "secondary_confidence_threshold": secondary_confidence,
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
            "normalized_secondary_detector_crop": list(secondary_detect_roi) if secondary_detect_roi else None,
            "calibration_provided": calibration is not None,
        },
        "scene_review": {
            "method": "mean absolute RGB change on 64x36 full-frame thumbnails",
            "cut_threshold": scene_cut_threshold,
            "enabled": bool(scene_cut_threshold),
            "candidate_cuts": scene_events,
            "scenes_observed": scene_id + 1,
            "status": "requires_manual_review",
            "track_ids": "unique within this run across scene resets; not athlete identities",
        },
        "court_view_review": {
            "enabled": court_reference is not None,
            "reference_filename": court_reference_path.name if court_reference_path else None,
            "reference_sha256": sha256_file(court_reference_path) if court_reference_path else None,
            "maximum_mean_rgb_difference": court_view_threshold,
            "rejected_frames": rejected_view_frames,
            "status": "requires_manual_review",
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
    parser.add_argument("--scene-cut-threshold", type=float, default=0.15, help="Full-frame change threshold 0..1; 0 disables candidate-cut resets")
    parser.add_argument("--court-reference", type=Path, default=None, help="Reviewed full-resolution wide-court reference; reject tracking on mismatched views")
    parser.add_argument("--court-view-threshold", type=float, default=0.08, help="Maximum reference RGB difference, in (0, 1]")
    parser.add_argument("--person-nms-iou", type=float, default=0.5, help="Suppress overlapping person duplicates above this IoU; 0 disables")
    parser.add_argument("--secondary-detect-roi", default=None, help="Optional second crop for jump coverage; requires person NMS")
    parser.add_argument("--secondary-confidence", type=float, default=0.35)
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
            scene_cut_threshold=args.scene_cut_threshold,
            court_reference_path=args.court_reference, court_view_threshold=args.court_view_threshold,
            person_nms_iou=args.person_nms_iou,
            secondary_detect_roi=parse_roi(args.secondary_detect_roi), secondary_confidence=args.secondary_confidence,
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
