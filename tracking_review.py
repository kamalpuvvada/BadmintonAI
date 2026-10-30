"""Local scene review and hash-bound approval intervals; no coaching metrics."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
from urllib.parse import quote
import os


CHECKS = ("boxes", "ids", "view", "boundaries")
REVIEW_FILE = "tracking_review.json"


def _read_run(run: Path) -> tuple[dict, list[dict], dict]:
    summary_path, tracks_path = run / "tracking_summary.json", run / "player_tracks.jsonl"
    summary_bytes, tracks_bytes = summary_path.read_bytes(), tracks_path.read_bytes()
    summary = json.loads(summary_bytes.decode("utf-8"))
    rows = [json.loads(line) for line in tracks_bytes.decode("utf-8").splitlines() if line.strip()]
    sampling = summary["sampling"]
    fps, start = sampling["fps"], sampling.get("start_seconds", 0)
    if not rows or len(rows) != sampling["frames_processed"] or not math.isfinite(fps) or fps <= 0 or not math.isfinite(start):
        raise ValueError("Tracking frame count or sampling rate is invalid")
    previous_scene = 0
    for index, row in enumerate(rows):
        scene = row.get("scene_id")
        if type(scene) is not int or scene < 0:
            raise ValueError("Scene boundaries are required; rerun the current tracking tool")
        changed = index > 0 and scene != previous_scene
        if (row["frame_index"] != index or (index == 0 and scene != 0)
                or (changed and scene != previous_scene + 1)
                or type(row.get("scene_cut")) is not bool or row["scene_cut"] != changed):
            raise ValueError("Frame indices or scene boundaries are inconsistent")
        timestamp = row["timestamp_seconds"]
        if not math.isfinite(timestamp) or abs(timestamp - (start + index / fps)) > .0011:
            raise ValueError("Tracking timestamps do not match source sampling")
        if not isinstance(row["persons"], list):
            raise ValueError("Person records must be a list")
        for person in row["persons"]:
            track_id = person["track_id"]
            if track_id is not None and (type(track_id) is not int or track_id < 0):
                raise ValueError("Track IDs must be non-negative integers or null")
            box, foot = person["bbox_xyxy_px"], person["foot_px"]
            if (len(box) != 4 or len(foot) != 2 or not all(math.isfinite(v) for v in box + foot)
                    or box[0] >= box[2] or box[1] >= box[3]):
                raise ValueError("Invalid person box or foot position")
        previous_scene = scene
    if previous_scene + 1 != summary["scene_review"]["scenes_observed"]:
        raise ValueError("Scene count does not match tracking summary")
    binding = {"summary_sha256": sha256(summary_bytes).hexdigest(), "tracks_sha256": sha256(tracks_bytes).hexdigest()}
    return summary, rows, binding


def _scene_facts(rows: list[dict], expected: int | None) -> list[dict]:
    facts = []
    owners: dict[int, set[int]] = {}
    for row in rows:
        for person in row["persons"]:
            if person["track_id"] is not None:
                owners.setdefault(person["track_id"], set()).add(row["scene_id"])
    for scene in sorted({row["scene_id"] for row in rows}):
        frames = [row for row in rows if row["scene_id"] == scene]
        ids = sorted({p["track_id"] for row in frames for p in row["persons"] if p["track_id"] is not None})
        blocked = []
        flags = []
        if any(row.get("court_view_status") == "reference_mismatch" for row in frames):
            blocked.append("Excluded camera view")
        if not ids:
            blocked.append("No confirmed tracks")
        if any(len(owners[track_id]) > 1 for track_id in ids):
            blocked.append("Track IDs are reused across scenes")
        if any(len([p["track_id"] for p in row["persons"] if p["track_id"] is not None])
               != len({p["track_id"] for p in row["persons"] if p["track_id"] is not None}) for row in frames):
            blocked.append("Duplicate track ID within a frame")
        if expected is not None:
            if len(ids) != expected:
                blocked.append("Confirmed ID count differs from expected players; inspect continuity")
            if any(len(row["persons"]) != expected for row in frames):
                blocked.append("Person count differs from expected players in some frames")
        unconfirmed = sum(any(p["track_id"] is None for p in row["persons"]) for row in frames)
        if unconfirmed:
            flags.append(f"{unconfirmed} frame(s) contain unconfirmed IDs; these frames cannot enter approved intervals")
        flags.append("Box bottoms are approximate image positions, including during jumps")
        facts.append({"scene_id": scene, "start_frame": frames[0]["frame_index"],
                      "end_frame": frames[-1]["frame_index"],
                      "start_seconds": frames[0]["timestamp_seconds"],
                      "end_seconds": frames[-1]["timestamp_seconds"],
                      "frames": len(frames), "track_ids": ids,
                      "person_count_histogram": dict(Counter(str(len(row["persons"])) for row in frames)),
                      "blockers": blocked, "flags": flags,
                      "review": {"decision": "pending", "reviewer": "", "notes": "",
                                 "checks": {key: False for key in CHECKS}, "reviewed_utc": None},
                      "review_history": []})
    return facts


def _load(run: Path, expected: int | None = None, create: bool = False) -> tuple[dict, dict, list[dict]]:
    summary, rows, binding = _read_run(run)
    path = run / REVIEW_FILE
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if old is None and not create:
        raise ValueError("Prepare a review packet first")
    if old is not None:
        if old.get("schema_version") != "0.1.0":
            raise ValueError("Unsupported review schema version")
        if old.get("source_binding") != binding:
            raise ValueError("Review is stale: tracking files changed; preserve it and prepare a new tracking run")
        if expected is not None and expected != old["expected_players"]:
            raise ValueError("Existing review uses a different expected player count")
        expected = old["expected_players"]
    if expected is not None and (type(expected) is not int or not 1 <= expected <= 4):
        raise ValueError("Expected players must be between 1 and 4, or omitted")
    review = {"schema_version": "0.1.0", "session_id": summary["session_id"],
              "approval_scope": "tracking_in_image_pixels",
              "source_binding": binding, "expected_players": expected,
              "scenes": _scene_facts(rows, expected), "coaching_metrics": {"status": "not_available"}}
    if old is not None:
        decisions = {scene["scene_id"]: scene["review"] for scene in old["scenes"]}
        if set(decisions) != {scene["scene_id"] for scene in review["scenes"]} or len(decisions) != len(old["scenes"]):
            raise ValueError("Review scene list does not match tracking data")
        for scene in review["scenes"]:
            scene["review"] = decisions[scene["scene_id"]]
            scene["review_history"] = next(item.get("review_history", []) for item in old["scenes"] if item["scene_id"] == scene["scene_id"])
    for scene in review["scenes"]:
        decision = scene["review"]
        if (not isinstance(decision, dict) or not isinstance(decision.get("reviewer"), str)
                or not isinstance(decision.get("notes"), str) or not isinstance(decision.get("checks"), dict)
                or not isinstance(scene["review_history"], list)):
            raise ValueError("Malformed review decision or history")
        if decision["decision"] not in ("pending", "accepted", "rejected"):
            raise ValueError("Unknown scene review decision")
        if decision["decision"] != "pending" and not decision["reviewer"].strip():
            raise ValueError("Reviewed scenes require a reviewer")
        if decision["decision"] == "accepted":
            if scene["blockers"] or not all(decision["checks"].get(key) is True for key in CHECKS):
                raise ValueError("Scene cannot be accepted: resolve blockers and verify boxes, IDs, view and boundaries")
        if decision["decision"] != "pending":
            recorded = decision.get("reviewed_utc")
            if not isinstance(recorded, str) or datetime.fromisoformat(recorded).utcoffset() is None:
                raise ValueError("Reviewed scenes require a timezone-aware review timestamp")
    return review, summary, rows


def _approved_intervals(review: dict, rows: list[dict]) -> list[dict]:
    """Only confirmed, consecutive frames in explicitly accepted scenes."""
    approved = {scene["scene_id"] for scene in review["scenes"] if scene["review"]["decision"] == "accepted"}
    intervals: list[dict] = []
    active = None
    for row in rows:
        ids = sorted(p["track_id"] for p in row["persons"] if p["track_id"] is not None)
        eligible = (row["scene_id"] in approved and bool(ids)
                    and len(ids) == len(row["persons"]) and len(set(ids)) == len(ids)
                    and (review["expected_players"] is None or len(ids) == review["expected_players"]))
        if not eligible:
            active = None
            continue
        if (active is None or active["scene_id"] != row["scene_id"] or active["track_ids"] != ids
                or active["end_frame"] + 1 != row["frame_index"]):
            active = {"scene_id": row["scene_id"], "start_frame": row["frame_index"],
                      "end_frame": row["frame_index"], "start_seconds": row["timestamp_seconds"],
                      "end_seconds": row["timestamp_seconds"], "track_ids": ids}
            intervals.append(active)
        else:
            active["end_frame"] = row["frame_index"]
            active["end_seconds"] = row["timestamp_seconds"]
    return intervals


def _artifact(run: Path, value: str) -> str:
    path = (run / value).resolve()
    if not path.is_relative_to(run.resolve()):
        raise ValueError("Review artifact must stay inside the tracking directory")
    if not path.is_file():
        raise ValueError(f"Review artifact is missing: {value}")
    return quote(os.path.relpath(path, run).replace(os.sep, "/"))


def _save(run: Path, review: dict, summary: dict, rows: list[dict]) -> Path:
    review["approved_intervals"] = _approved_intervals(review, rows)
    # Validate paths before writing either artifact. All source prose is escaped.
    video = _artifact(run, summary["artifacts"]["annotated_video"])
    images = [(item["timestamp_seconds"], _artifact(run, item["image"]))
              for item in summary["artifacts"]["review_frames"]]
    cards = []
    for scene in review["scenes"]:
        decision = scene["review"]
        labels = {"boxes": "Box placement", "ids": "ID continuity", "view": "Camera view", "boundaries": "Scene boundaries"}
        checks_text = " · ".join(f"{labels[key]}: {'verified' if decision['checks'].get(key) is True else 'not verified'}" for key in CHECKS)
        id_text = ", ".join(map(str, scene["track_ids"])) or "None confirmed"
        clip_start = scene["start_frame"] / summary["sampling"]["fps"]
        clip_end = (scene["end_frame"] + 1) / summary["sampling"]["fps"]
        pictures = "".join(f'<figure><img src="{url}" alt="Tracked scene frame"><figcaption>Source {time:.3f} s</figcaption></figure>'
                           for time, url in images if scene["start_seconds"] <= time <= scene["end_seconds"])
        notes = "".join(f"<li>{escape(message)}</li>" for message in scene["blockers"] + scene["flags"])
        cards.append(f'''<section><h2>Scene {scene['scene_id']} · {scene['start_seconds']:.3f}–{scene['end_seconds']:.3f} s</h2>
<p class="status">{escape(decision['decision'].capitalize())}</p><p>{scene['frames']} sampled frames · Temporary IDs: {id_text}</p>
<p><a href="{video}#t={clip_start:.3f},{clip_end:.3f}">Watch this scene</a></p>
<p>Reviewer: {escape(decision['reviewer'] or 'Not recorded')}</p><p>{escape(decision['notes'])}</p>
<p>{checks_text}</p>
<ul>{notes}</ul><div class="frames">{pictures or '<p>No saved review frame in this scene. Check the video.</p>'}</div></section>''')
    status = "Review required" if any(scene["review"]["decision"] == "pending" for scene in review["scenes"]) else "Review completed"
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BadmintonAI tracking review</title><style>
body{{font:16px/1.5 system-ui,sans-serif;background:#10202b;color:#eff5fa;margin:0}}main{{max-width:1100px;margin:auto;padding:28px}}
h1,h2{{line-height:1.2}}section{{background:#192f3d;border:1px solid #49606d;border-radius:12px;padding:20px;margin:22px 0}}
.status{{color:#ffdf8c;font-weight:700}}video{{width:100%;max-height:500px;background:#000}}.frames{{display:flex;flex-wrap:wrap;gap:14px}}
figure{{margin:0;width:320px;max-width:100%}}img{{width:100%;height:auto}}figcaption{{color:#b9cbd8}}li{{margin:6px 0}}a{{color:#74e5d2}}
</style><main><h1>Tracking review</h1><p class="status">{status} · Coaching scores unavailable</p>
<p>Session: {escape(review['session_id'])}</p><p>{len(review['scenes'])} scenes · Expected players: {review['expected_players'] if review['expected_players'] is not None else 'Not specified'}</p>
<p>Review each scene's boxes, ID continuity, camera view and start/end boundaries.
Record an explicit acceptance or rejection with a reviewer. No scene is approved automatically.</p>
<p>Times below refer to the source video. This preview starts at source {summary['sampling'].get('start_seconds', 0):.3f} s.
Unconfirmed frames cannot enter approved intervals. IDs do not identify named athletes.</p>
<video controls preload="metadata" src="{video}">Open the tracked preview in a video player.</video>{''.join(cards)}
<p>Decisions are bound to this tracking output. Recheck the source files before using any approved interval.
Movement measurements and coaching scores are not produced by this review packet.</p></main></html>'''
    path = run / REVIEW_FILE
    path.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    (run / "tracking_review.html").write_text(html, encoding="utf-8")
    return path


def prepare(run: Path, expected_players: int | None = None) -> Path:
    review, summary, rows = _load(run, expected_players, create=True)
    return _save(run, review, summary, rows)


def decide(run: Path, scene_id: int, decision: str, reviewer: str, notes: str,
           checks: dict[str, bool]) -> Path:
    review, summary, rows = _load(run)
    matches = [scene for scene in review["scenes"] if scene["scene_id"] == scene_id]
    if not matches or decision not in ("accepted", "rejected", "pending"):
        raise ValueError("Unknown scene or decision")
    scene = matches[0]
    if decision != "pending" and not reviewer.strip():
        raise ValueError("A reviewer is required")
    if decision == "accepted" and (scene["blockers"] or not all(checks.get(key) is True for key in CHECKS)):
        raise ValueError("Acceptance requires no blockers and all four verification checks")
    scene["review"] = {"decision": decision, "reviewer": reviewer.strip(), "notes": notes,
                       "checks": {key: checks.get(key) is True for key in CHECKS},
                       "reviewed_utc": datetime.now(timezone.utc).isoformat()}
    scene["review_history"].append(dict(scene["review"]))
    return _save(run, review, summary, rows)


def verified_snapshot(run: Path) -> tuple[dict, dict, list[dict], dict]:
    """Read and validate one source/review snapshot for downstream consumers."""
    review, summary, rows = _load(run)
    decisions = Counter(scene["review"]["decision"] for scene in review["scenes"])
    verification = {"source_binding": review["source_binding"], "approval_scope": "tracking_in_image_pixels",
            "decisions": dict(decisions),
            "all_scenes_decided": not decisions["pending"],
            "approved_intervals": _approved_intervals(review, rows),
            "coaching_metrics": {"status": "not_available"}}
    return verification, summary, rows, review


def verify(run: Path) -> dict:
    return verified_snapshot(run)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "decide", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--expected-players", type=int)
        elif name == "decide":
            command.add_argument("--scene", type=int, required=True)
            command.add_argument("--decision", choices=("accepted", "rejected", "pending"), required=True)
            command.add_argument("--reviewer", default="")
            command.add_argument("--notes", default="")
            for check in CHECKS:
                command.add_argument(f"--{check}-verified", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            print(prepare(args.run, args.expected_players))
        elif args.command == "decide":
            print(decide(args.run, args.scene, args.decision, args.reviewer, args.notes,
                         {key: getattr(args, f"{key}_verified") for key in CHECKS}))
        else:
            print(json.dumps(verify(args.run), indent=2))
    except (ValueError, OSError, KeyError, TypeError, AttributeError, IndexError) as exc:
        parser.exit(1, f"Review error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
