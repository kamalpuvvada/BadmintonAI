"""Export explicitly reviewed image positions, without movement or coaching scores."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from tracking_review import verified_snapshot


EXPORT_FILE = "reviewed_tracks.json"


def build_export(run: Path) -> dict:
    verification, summary, rows, review = verified_snapshot(run)
    dimensions = summary["video"]
    width, height = dimensions["width_px"], dimensions["height_px"]
    if any(type(value) is not int or value <= 0 for value in (width, height)):
        raise ValueError("Video dimensions must be positive integers")
    # Bind to the validated review snapshot as well as the exact source bytes.
    # Revoking/replacing a decision invalidates an existing export.
    review_hash = sha256(json.dumps(review, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode("utf-8")).hexdigest()
    segments = []
    for interval_index, interval in enumerate(verification["approved_intervals"]):
        for track_id in interval["track_ids"]:
            samples = []
            for frame_index in range(interval["start_frame"], interval["end_frame"] + 1):
                row = rows[frame_index]
                person = next(p for p in row["persons"] if p["track_id"] == track_id)
                box, foot, confidence = person["bbox_xyxy_px"], person["foot_px"], person["confidence"]
                if (any(type(v) not in (int, float) for v in box + foot + [confidence])
                        or not math.isfinite(confidence) or not 0 <= confidence <= 1
                        or not 0 <= box[0] < box[2] <= width
                        or not 0 <= box[1] < box[3] <= height
                        or not 0 <= foot[0] <= width or not 0 <= foot[1] <= height):
                    raise ValueError("Reviewed position or confidence is outside the image/data range")
                samples.append({"frame_index": frame_index, "source_seconds": row["timestamp_seconds"],
                                "bbox_xyxy_px": list(box), "box_bottom_px": list(foot),
                                "detector_confidence": confidence})
            segments.append({"interval_index": interval_index, "scene_id": interval["scene_id"],
                             "track_id": track_id, "samples": samples})
    return {"schema_version": "0.1.0", "session_id": summary["session_id"],
            "source_binding": verification["source_binding"], "review_snapshot_sha256": review_hash,
            "approval_scope": verification["approval_scope"],
            "status": "available" if segments else "no_approved_intervals",
            "image_size_px": {"width": width, "height": height},
            "sampling_fps": summary["sampling"]["fps"],
            "coordinate_system": "source_image_pixels_top_left_origin",
            "position_meaning": "bounding_box_bottom_candidate_not_verified_floor_contact",
            "identity_scope": "source_binding_and_scene_id_and_track_id",
            "review_decisions": verification["decisions"],
            "approved_intervals": verification["approved_intervals"], "segments": segments,
            "movement_metrics": {"status": "not_available"},
            "coaching_metrics": {"status": "not_available"}}


def export(run: Path) -> Path:
    payload = build_export(run)
    path = run / EXPORT_FILE
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def verify_export(run: Path) -> dict:
    """Recheck source and review now; a saved export is only a snapshot."""
    current = build_export(run)
    saved = json.loads((run / EXPORT_FILE).read_text(encoding="utf-8"))
    # Canonical JSON comparison also distinguishes booleans from numeric IDs.
    if json.dumps(saved, sort_keys=True, allow_nan=False) != json.dumps(current, sort_keys=True, allow_nan=False):
        raise ValueError("Export is stale or modified; regenerate it from the current review")
    return {"status": "valid", "data_status": current["status"],
            "segments": len(current["segments"]), "source_binding": current["source_binding"],
            "coaching_metrics": current["coaching_metrics"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "verify"))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "export":
            print(export(args.run))
        else:
            print(json.dumps(verify_export(args.run), indent=2))
    except (ValueError, OSError, KeyError, TypeError, AttributeError, IndexError, StopIteration) as exc:
        parser.exit(1, f"Export error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
