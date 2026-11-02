"""Review player identity assignments without changing detections or granting tracking approval."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from badminton_proto import sha256_file
from tracking_review import verified_snapshot

SCHEMA = "1.0.0"
SLOTS = ("P1", "P2", "P3", "P4")


def _local(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact escapes the job folder")
    return path


def source_snapshot(job_path: Path):
    job_path = job_path.resolve()
    job = json.loads((job_path / "job.json").read_text(encoding="utf-8"))
    if job.get("schema_version") != SCHEMA or job.get("status") not in ("completed", "needs_review"):
        raise ValueError("Identity review requires a current completed or needs_review job")
    result_path = job_path / "result.json"
    if sha256_file(result_path) != job["stages"]["export"]["artifacts"]["result.json"]:
        raise ValueError("Public result differs from the job export")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    players = _local(job_path, job["stages"]["players"]["directory"])
    verification, summary, rows, _ = verified_snapshot(players)
    artifacts = job["stages"]["players"]["artifacts"]
    for name in ("tracking_summary.json", "player_tracks.jsonl"):
        if sha256_file(players / name) != artifacts[name]:
            raise ValueError("Player observations differ from the completed job")
    source = job["binding"]["source"]
    if sha256_file(Path(source["path"])) != source["sha256"] or source != result["source"]:
        raise ValueError("Source video changed since processing")
    if summary["session_id"] != job["session_id"]:
        raise ValueError("Player source belongs to another session")
    binding = {**verification["source_binding"], "source_video_sha256": source["sha256"], "session_id": job["session_id"]}
    return job, result, summary, rows, binding


def tracklets(rows: list[dict]) -> list[dict]:
    """Split each ID at absent frames or scene boundaries, never bridge gaps."""
    groups, active = [], {}
    for index, row in enumerate(rows):
        present = set()
        for person in row["persons"]:
            track = person["track_id"]
            if track is None:
                continue
            if track in present:
                raise ValueError("Duplicate tracker ID in a source frame")
            present.add(track)
            previous = active.get(track)
            if previous is None or previous["end_frame"] != index-1 or previous["scene_id"] != row["scene_id"]:
                previous = {"scene_id": row["scene_id"], "track_id": track, "start_frame": index,
                            "end_frame": index, "start_seconds": row["timestamp_seconds"],
                            "end_seconds": row["timestamp_seconds"], "samples": []}
                groups.append(previous)
                active[track] = previous
            previous["end_frame"] = index
            previous["end_seconds"] = row["timestamp_seconds"]
            previous["samples"].append({"frame_index": index, "source_seconds": row["timestamp_seconds"],
                                         "bbox_xyxy_px": person["bbox_xyxy_px"]})
        active = {track: value for track, value in active.items() if track in present}
    for group in groups:
        group["id"] = f"S{group['scene_id']}:T{group['track_id']}:F{group['start_frame']}-{group['end_frame']}"
    return groups


def prepare(job_path: Path, output: Path):
    if output.exists():
        raise ValueError("Choose a new identity editor folder")
    job, result, summary, rows, binding = source_snapshot(job_path)
    preview = _local(job_path, result["artifacts"]["preview"])
    render = job["stages"]["render"]
    render_root = _local(job_path, render["directory"])
    preview_hash = sha256_file(preview)
    if preview_hash != render["artifacts"][str(preview.relative_to(render_root))]:
        raise ValueError("Preview does not match the finished render")
    data = {"schema_version": SCHEMA, "source_binding": binding,
            "expected_players": job["binding"]["configuration"]["expected_players"],
            "sampling": summary["sampling"], "image_size_px": [summary["video"]["width_px"], summary["video"]["height_px"]],
            "preview_sha256": preview_hash, "preview": result["preview"], "tracklets": tracklets(rows)}
    # Recheck observations and source immediately before publishing the editor.
    if source_snapshot(job_path)[4] != binding:
        raise ValueError("Source changed while preparing identity review")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as work:
        copied = Path(work) / "preview.mp4"
        shutil.copyfile(preview, copied)
        if sha256_file(copied) != preview_hash or source_snapshot(job_path)[4] != binding:
            raise ValueError("Preview or observations changed while copying identity review")
        output.mkdir()
        copied.replace(output / "preview.mp4")
    (output / "identity_source.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    payload = json.dumps(data, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    page = Path(__file__).with_suffix(".html").read_text(encoding="utf-8").replace("/*IDENTITY_DATA*/", payload)
    (output / "index.html").write_text(page, encoding="utf-8")
    return output / "index.html"


def apply(job_path: Path, patch_path: Path, output: Path):
    if output.exists():
        raise ValueError("Choose a new identity result folder")
    job, result, summary, rows, binding = source_snapshot(job_path)
    patch_bytes = patch_path.read_bytes()
    patch = json.loads(patch_bytes.decode("utf-8"))
    if (type(patch) is not dict or set(patch) != {"schema_version", "source_binding", "reviewer", "reviewer_kind", "assignments"}
            or patch["schema_version"] != SCHEMA or patch["source_binding"] != binding):
        raise ValueError("Identity patch schema or source binding is invalid/stale")
    if (type(patch["reviewer"]) is not str or not patch["reviewer"].strip()
            or patch["reviewer_kind"] not in ("human", "automated")):
        raise ValueError("Explicit reviewer and reviewer kind are required")
    if type(patch["assignments"]) is not list or not patch["assignments"]:
        raise ValueError("Identity patch must contain assignments")
    groups = {g["id"]: g for g in tracklets(rows)}
    used_tracks, used_slots, assignments, positions = set(), set(), [], []
    expected = job["binding"]["configuration"]["expected_players"]
    for assignment in patch["assignments"]:
        if type(assignment) is not dict or set(assignment) != {"tracklet_id", "start_frame", "end_frame", "player_slot"}:
            raise ValueError("Invalid identity assignment fields")
        if type(assignment["tracklet_id"]) is not str:
            raise ValueError("Tracklet ID must be text")
        group = groups.get(assignment["tracklet_id"])
        start, end, slot = assignment["start_frame"], assignment["end_frame"], assignment["player_slot"]
        if (group is None or type(start) is not int or type(end) is not int
                or not group["start_frame"] <= start <= end <= group["end_frame"] or slot not in SLOTS[:expected]):
            raise ValueError("Identity assignment range or slot is invalid")
        for index in range(start, end+1):
            track_key = (index, group["track_id"])
            slot_key = (index, slot)
            if track_key in used_tracks or slot_key in used_slots:
                raise ValueError("Overlapping tracker or player assignments in the same frame")
            used_tracks.add(track_key); used_slots.add(slot_key)
            sample = next(p for p in rows[index]["persons"] if p["track_id"] == group["track_id"])
            positions.append({"frame_index": index, "source_seconds": rows[index]["timestamp_seconds"],
                              "scene_id": group["scene_id"], "track_id": group["track_id"], "player_slot": slot,
                              "bbox_xyxy_px": sample["bbox_xyxy_px"], "foot_px": sample["foot_px"],
                              "detection_review_status": "unchanged; consult current tracking review"})
        assignments.append(dict(assignment))
    if source_snapshot(job_path)[4] != binding:
        raise ValueError("Source changed during identity correction")
    from hashlib import sha256
    payload = {"schema_version": SCHEMA, "status": "reviewer_assigned_identity_snapshot",
               "created_utc": datetime.now(timezone.utc).isoformat(), "source_binding": binding,
               "patch_sha256": sha256(patch_bytes).hexdigest(), "reviewer": patch["reviewer"].strip(),
               "reviewer_kind": patch["reviewer_kind"], "reviewer_identity_authenticated": False,
               "scope": "Identity labels only; does not approve boxes, gaps, calibration, events or coaching",
               "assignments": assignments, "positions": sorted(positions, key=lambda r: (r["frame_index"], r["player_slot"])),
               "coverage": {"assigned_observations": len(positions), "total_confirmed_observations": sum(
                   p["track_id"] is not None for r in rows for p in r["persons"]),
                   "unconfirmed_observations": sum(p["track_id"] is None for r in rows for p in r["persons"])},
               "coaching_metrics": {"status": "not_available"}}
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True)
    (output / "player_identities.json").write_text(encoded, encoding="utf-8")
    (output / "corrections.json").write_bytes(patch_bytes)
    (output / "identity_manifest.json").write_text(json.dumps({"schema_version": SCHEMA, "source_binding": binding,
        "artifacts": {name: sha256_file(output / name) for name in ("player_identities.json", "corrections.json")}}, indent=2)+"\n", encoding="utf-8")
    return output / "player_identities.json"


def verify(job_path: Path, output: Path):
    binding = source_snapshot(job_path)[4]
    manifest = json.loads((output / "identity_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA or manifest.get("source_binding") != binding:
        raise ValueError("Identity export belongs to stale observations")
    for name in ("player_identities.json", "corrections.json"):
        if sha256_file(output / name) != manifest["artifacts"][name]:
            raise ValueError("Identity export or correction file was modified")
    return {"status": "valid_identity_snapshot", "source_binding": binding,
            "scope": "Identity assignments only; no tracking/coaching approval"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "apply", "verify"))
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--patch", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "apply" and args.patch is None:
            raise ValueError("--patch is required for apply")
        if args.command == "verify":
            print(json.dumps(verify(args.job, args.out), indent=2))
        else:
            print(prepare(args.job, args.out) if args.command == "prepare" else apply(args.job, args.patch, args.out))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Identity review error: {exc}\n")


if __name__ == "__main__":
    main()
