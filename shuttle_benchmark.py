"""Hash-bound development labels and diagnostic evaluation of shuttle proposals."""
from __future__ import annotations

import argparse
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from badminton_proto import probe_video, sha256_file
from player_tracking import _extract_sampled_video


def _point(value, width, height):
    if (not isinstance(value, list) or len(value) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
            or not 0 <= value[0] < width or not 0 <= value[1] < height):
        raise ValueError("Invalid benchmark point")


def _validate(data: dict):
    if data.get("schema_version") != "0.1.0" or data.get("split") != "development_seed":
        raise ValueError("Unsupported benchmark schema or split")
    width, height = data["image_size_px"]
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError("Invalid image dimensions")
    if not data["cases"]:
        raise ValueError("Benchmark must contain cases")
    identifiers, frames = set(), set()
    for case in data["cases"]:
        key = (case["source_run"], case["frame_index"])
        if case["id"] in identifiers or key in frames:
            raise ValueError("Duplicate case or source frame")
        identifiers.add(case["id"])
        frames.add(key)
        sampling = data["sources"][case["source_run"]]
        if (type(sampling["frames"]) is not int or sampling["frames"] <= 0
                or not math.isfinite(sampling["fps"]) or sampling["fps"] <= 0
                or not math.isfinite(sampling["start_seconds"]) or sampling["start_seconds"] < 0
                or type(case["frame_index"]) is not int
                or not 0 <= case["frame_index"] < sampling["frames"]):
            raise ValueError("Invalid benchmark sampling or frame")
        if (not case["reviewer"].strip() or case["reviewer_kind"] not in ("human", "automated")
                or case["visibility"] not in ("visible", "uncertain")):
            raise ValueError("Labels require explicit provenance and visibility")
        if case.get("view", "unspecified") not in ("wide_court", "closeup", "unknown", "unspecified"):
            raise ValueError("Invalid annotation view")
        if case["visibility"] == "visible":
            _point(case["point_px"], width, height)
        elif case["point_px"] is not None:
            raise ValueError("Uncertain frames cannot assert a shuttle location")
        for region in case["not_shuttle_regions"]:
            _point(region["center_px"], width, height)
            radius = region["radius_px"]
            if type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0:
                raise ValueError("Invalid negative-region radius")
            if not region["identity"].strip():
                raise ValueError("Negative regions require an observed identity")
            if case["point_px"] is not None and math.dist(region["center_px"], case["point_px"]) <= radius:
                raise ValueError("Visible point contradicts a local negative label")
    if "sequences" in data or data.get("role") == "continuous_development":
        sequences = data.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise ValueError("Continuous annotation requires sequence intervals")
        covered, names = set(), set()
        for sequence in sequences:
            name = sequence["id"]
            source = sequence["source_run"]
            first, last = sequence["start_frame"], sequence["end_frame"]
            if (not isinstance(name, str) or not name.strip() or name in names
                    or source not in data["sources"] or type(first) is not int or type(last) is not int
                    or not 0 <= first <= last < data["sources"][source]["frames"]):
                raise ValueError("Invalid or duplicate continuous sequence")
            names.add(name)
            interval = {(source, i) for i in range(first, last + 1)}
            if covered & interval:
                raise ValueError("Continuous sequences must not overlap")
            covered |= interval
        if covered != frames:
            raise ValueError("Continuous sequences require exactly one annotation for every frame, including uncertain frames")


def _continuous_metrics(data: dict, loaded: dict, tolerance: float) -> list[dict]:
    """Measure observed errors and fragmentation; uncertainty always breaks runs."""
    cases = {(c["source_run"], c["frame_index"]): c for c in data["cases"]}
    # Segment IDs must refer to actual consecutive observations within one scene.
    for rows in loaded.values():
        previous = {}
        for index, row in enumerate(rows):
            proposal = row["proposal"]
            if proposal is None:
                continue
            sid, scene = proposal.get("proposal_segment_id"), row.get("scene_id")
            if type(sid) is not int or sid < 0 or type(scene) is not int or scene < 0:
                raise ValueError("Continuous evaluation requires valid scene and proposal segment IDs")
            if sid in previous and previous[sid] != (index - 1, scene):
                raise ValueError("Proposal segment crosses a gap or scene boundary")
            previous[sid] = (index, scene)
    results = []
    for sequence in data["sequences"]:
        source = sequence["source_run"]
        timeline = []
        longest_match = longest_missing = match_run = missing_run = 0
        same_segment_pairs = segment_changes = matched_view_changes = 0
        previous_match = None
        counts = {"visible_frames": 0, "uncertain_frames": 0, "candidate_point_hits": 0,
                  "proposal_point_hits": 0, "visible_frames_without_candidates": 0,
                  "visible_frames_without_proposal": 0, "visible_frames_with_displaced_proposal": 0,
                  "uncertain_frames_with_proposals_unscored": 0}
        errors = []
        by_view, status_counts, scene_changes, view_changes = {}, {}, [], []
        previous_scene = None
        previous_view = None
        previous_motion = None
        step_errors = []
        for index in range(sequence["start_frame"], sequence["end_frame"] + 1):
            case, row = cases[source, index], loaded[source][index]
            proposal = row["proposal"]
            view = case.get("view", "unspecified")
            view_counts = by_view.setdefault(view, {"annotated_frames": 0, "visible_frames": 0,
                                                   "uncertain_frames": 0, "proposal_point_hits": 0,
                                                   "visible_frames_without_proposal": 0,
                                                   "visible_frames_with_displaced_proposal": 0})
            view_counts["annotated_frames"] += 1
            status = row.get("status", "unspecified")
            status_counts[status] = status_counts.get(status, 0) + 1
            if index > sequence["start_frame"] and row.get("scene_id") != previous_scene:
                scene_changes.append(index)
                missing_run = 0
            previous_scene = row.get("scene_id")
            if index > sequence["start_frame"] and view != previous_view:
                view_changes.append(index)
                missing_run = 0
            previous_view = view
            item = {"frame_index": index, "visibility": case["visibility"], "proposal_error_px": None,
                    "annotation_view": view, "prediction_status": status,
                    "scene_id": row.get("scene_id"),
                    "proposal_segment_id": proposal["proposal_segment_id"] if proposal else None}
            if case["visibility"] == "uncertain":
                counts["uncertain_frames"] += 1
                view_counts["uncertain_frames"] += 1
                counts["uncertain_frames_with_proposals_unscored"] += proposal is not None
                item["result"] = "uncertain_unscored"
                match_run = missing_run = 0
                previous_match = None
                previous_motion = None
            else:
                counts["visible_frames"] += 1
                view_counts["visible_frames"] += 1
                candidate_hit = any(math.dist(c["center_px"], case["point_px"]) <= tolerance for c in row["candidates"])
                counts["candidate_point_hits"] += candidate_hit
                counts["visible_frames_without_candidates"] += not row["candidates"]
                error = math.dist(proposal["position_px"], case["point_px"]) if proposal else None
                hit = error is not None and error <= tolerance
                counts["proposal_point_hits"] += hit
                view_counts["proposal_point_hits"] += hit
                view_counts["visible_frames_without_proposal"] += proposal is None
                view_counts["visible_frames_with_displaced_proposal"] += proposal is not None and not hit
                counts["visible_frames_without_proposal"] += proposal is None
                counts["visible_frames_with_displaced_proposal"] += proposal is not None and not hit
                if error is not None:
                    errors.append(error)
                    item["proposal_error_px"] = round(error, 4)
                motion_key = (row.get("scene_id"), proposal["proposal_segment_id"], view) if proposal else None
                if proposal and previous_motion is not None and previous_motion[0] == motion_key:
                    predicted_step = [a-b for a,b in zip(proposal["position_px"], previous_motion[1])]
                    labeled_step = [a-b for a,b in zip(case["point_px"], previous_motion[2])]
                    step_errors.append(math.dist(predicted_step, labeled_step))
                previous_motion = (motion_key, proposal["position_px"], case["point_px"]) if proposal else None
                item["result"] = "point_match" if hit else ("missing_proposal" if proposal is None else "displaced_proposal")
                missing_run = missing_run + 1 if proposal is None else 0
                longest_missing = max(longest_missing, missing_run)
                if hit:
                    key = (row["scene_id"], proposal["proposal_segment_id"], view)
                    if previous_match is not None:
                        same_segment_pairs += previous_match[:2] == key[:2]
                        segment_changes += previous_match[:2] != key[:2]
                        matched_view_changes += previous_match[2] != view
                    match_run = match_run + 1 if previous_match == key else 1
                    previous_match = key
                    longest_match = max(longest_match, match_run)
                else:
                    match_run = 0
                    previous_match = None
            timeline.append(item)
        results.append({**sequence, "annotated_frames": len(timeline), **counts,
                        "longest_correct_same_segment_run_frames": longest_match,
                        "longest_consecutive_visible_missing_proposal_frames": longest_missing,
                        "adjacent_point_matches_in_same_segment": same_segment_pairs,
                        "adjacent_point_matches_with_segment_or_scene_change": segment_changes,
                        "adjacent_point_matches_with_annotation_view_change": matched_view_changes,
                        "mean_error_on_visible_proposed_frames_px": round(sum(errors)/len(errors), 4) if errors else None,
                        "max_error_on_visible_proposed_frames_px": round(max(errors), 4) if errors else None,
                        "annotation_view_counts": by_view, "prediction_status_counts": status_counts,
                        "scene_change_frames": scene_changes,
                        "annotation_view_change_frames": view_changes,
                        "adjacent_visible_proposed_pairs_same_segment_and_view": len(step_errors),
                        "mean_step_vector_error_px": round(sum(step_errors)/len(step_errors), 4) if step_errors else None,
                        "max_step_vector_error_px": round(max(step_errors), 4) if step_errors else None,
                        "timeline": timeline})
    return results


def prepare(seed: Path, video: Path, output: Path) -> Path:
    data = json.loads(seed.read_text(encoding="utf-8"))
    _validate(data)
    if output.exists():
        raise ValueError("Choose a new benchmark directory")
    if sha256_file(video) != data["source_video_sha256"]:
        raise ValueError("Benchmark source video hash does not match")
    meta = probe_video(video)
    if [meta["width_px"], meta["height_px"]] != data["image_size_px"]:
        raise ValueError("Benchmark source dimensions do not match")
    if any(abs(s["fps"] - meta["frame_rate_fps"]) > 1e-6 for s in data["sources"].values()):
        raise ValueError("Benchmark must retain the native source sampling rate")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    cards = []
    with TemporaryDirectory() as temporary:
        for number, (name, sampling) in enumerate(data["sources"].items()):
            paths = _extract_sampled_video(video, Path(temporary) / str(number), sampling["fps"],
                                           sampling["frames"] / sampling["fps"], sampling["start_seconds"])
            if len(paths) != sampling["frames"]:
                raise ValueError("Decoded benchmark frame count differs from seed")
            for case in (c for c in data["cases"] if c["source_run"] == name):
                # Case IDs never become filesystem paths.
                index = data["cases"].index(case)
                relative = f"images/case_{index:04}.png"
                image = Image.open(paths[case["frame_index"]]).convert("RGB")
                image.save(output / relative)
                case["image"] = relative
                case["image_sha256"] = sha256_file(output / relative)
                annotated = image.copy()
                draw = ImageDraw.Draw(annotated)
                if case["point_px"] is not None:
                    x, y = case["point_px"]
                    draw.ellipse((x-6, y-6, x+6, y+6), outline="#38ff93", width=2)
                for region in case["not_shuttle_regions"]:
                    x, y = region["center_px"]; r = region["radius_px"]
                    draw.rectangle((x-r, y-r, x+r, y+r), outline="#ff8855", width=2)
                annotated.save(output / f"images/labeled_{index:04}.png")
                time = sampling["start_seconds"] + case["frame_index"] / sampling["fps"]
                cards.append(f'<section><h2>{escape(case["id"])} · {time:.3f} s</h2><p>{escape(case["visibility"])} · {escape(case["reviewer"])} / {escape(case["reviewer_kind"])}</p><p>{escape(case["notes"])}</p><img src="images/labeled_{index:04}.png" alt="Labeled source frame"><p><a href="{relative}">Unmarked source image</a></p></section>')
    data["seed_sha256"] = sha256_file(seed)
    target = output / "benchmark.json"
    target.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "index.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Shuttle development benchmark</title><style>body{font:17px/1.6 system-ui;background:#10202b;color:#edf5fa;max-width:1000px;margin:auto;padding:24px}img{width:100%}section{border:1px solid #587480;border-radius:10px;padding:20px;margin:20px 0}a{color:#80e5d0}</style><h1>Shuttle development benchmark</h1><p>Green: observed shuttle point. Orange: a local region known to contain another object. An uncertain frame is not labeled as shuttle-absent. These selected development cases are not an independent accuracy test. Labels are explicitly attributed; source images are hash-bound.</p>'''+''.join(cards), encoding="utf-8")
    return target


def evaluate(benchmark: Path, predictions: dict[str, Path], tolerance_px: float = 4) -> dict:
    if not math.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError("Point tolerance must be finite and positive")
    data = json.loads(benchmark.read_text(encoding="utf-8"))
    _validate(data)
    root = benchmark.resolve().parent
    for case in data["cases"]:
        path = (root / case["image"]).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != case["image_sha256"]:
            raise ValueError("Benchmark image is changed or outside the packet")
        with Image.open(path) as image:
            if list(image.size) != data["image_size_px"]:
                raise ValueError("Benchmark image dimensions changed")
    needed = {c["source_run"] for c in data["cases"]}
    if set(predictions) != needed:
        raise ValueError("Provide exactly one prediction run per benchmark source")
    loaded, bindings = {}, {}
    for name, path in predictions.items():
        summary_path, rows_path = path / "shuttle_summary.json", path / "shuttle_candidates.jsonl"
        summary_bytes, rows_bytes = summary_path.read_bytes(), rows_path.read_bytes()
        summary = json.loads(summary_bytes)
        rows = [json.loads(line) for line in rows_bytes.decode("utf-8").splitlines() if line.strip()]
        sampling = data["sources"][name]
        if (summary["source_video_sha256"] != data["source_video_sha256"] or summary["sampling"] != sampling
                or [summary["video"]["width_px"], summary["video"]["height_px"]] != data["image_size_px"]
                or len(rows) != sampling["frames"]):
            raise ValueError("Prediction source or sampling does not match benchmark")
        for i, row in enumerate(rows):
            expected = sampling["start_seconds"] + i / sampling["fps"]
            if type(row["frame_index"]) is not int or row["frame_index"] != i or not math.isfinite(row["timestamp_seconds"]) or abs(row["timestamp_seconds"] - expected) > .00001:
                raise ValueError("Invalid prediction frame or timestamp")
            for candidate in row["candidates"]:
                _point(candidate["center_px"], *data["image_size_px"])
            proposal = row["proposal"]
            if proposal is not None:
                index = proposal["candidate_index"]
                if (type(index) is not int or not 0 <= index < len(row["candidates"])
                        or proposal["position_px"] != row["candidates"][index]["center_px"]):
                    raise ValueError("Proposal does not reference a candidate")
        loaded[name] = rows
        bindings[name] = {"summary_sha256": sha256(summary_bytes).hexdigest(), "rows_sha256": sha256(rows_bytes).hexdigest()}
    visible, candidate_hits, proposal_hits, negatives, negative_candidate_hits, negative_proposal_hits = (0,) * 6
    details = []
    distance = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    for case in data["cases"]:
        row = loaded[case["source_run"]][case["frame_index"]]
        points = [c["center_px"] for c in row["candidates"]]
        proposal = row["proposal"]["position_px"] if row["proposal"] is not None else None
        detail = {"id": case["id"], "visibility": case["visibility"]}
        if case["visibility"] == "visible":
            visible += 1
            hit = any(distance(p, case["point_px"]) <= tolerance_px for p in points)
            linked_hit = proposal is not None and distance(proposal, case["point_px"]) <= tolerance_px
            candidate_hits += hit; proposal_hits += linked_hit
            detail.update(candidate_hit=hit, proposal_hit=linked_hit)
        detail["negative_region_hits"] = []
        for region in case["not_shuttle_regions"]:
            negatives += 1
            hit = any(distance(p, region["center_px"]) <= region["radius_px"] for p in points)
            linked_hit = proposal is not None and distance(proposal, region["center_px"]) <= region["radius_px"]
            negative_candidate_hits += hit; negative_proposal_hits += linked_hit
            detail["negative_region_hits"].append({"identity": region["identity"], "candidate_hit": hit, "proposal_hit": linked_hit})
        details.append(detail)
    result = {"schema_version": "0.1.0", "evaluation_scope": "selected_development_seed_only",
            "benchmark_sha256": sha256_file(benchmark), "prediction_bindings": bindings, "point_tolerance_px": tolerance_px,
            "visible_points": visible, "candidate_point_hits": candidate_hits, "proposal_point_hits": proposal_hits,
            "uncertain_frames_excluded_from_point_scoring": len(data["cases"]) - visible,
            "known_negative_regions": negatives, "negative_regions_with_candidates": negative_candidate_hits,
            "negative_regions_with_proposals": negative_proposal_hits,
            "cases": details, "verified_shuttle_positions": 0,
            "limitations": ["Selected, correlated development examples; not held-out data or full-video precision/recall.",
                            "Negative regions label only local distractors, not absence of a shuttle elsewhere in the frame.",
                            "Evaluation never promotes proposals to verified trajectories or enables events/scoring."]}
    if "sequences" in data:
        result["evaluation_scope"] = "continuously_annotated_development_intervals_only"
        result["continuous_sequences"] = _continuous_metrics(data, loaded, tolerance_px)
        result["limitations"].append("Uncertain annotations break continuity statistics and never imply shuttle absence; point matches do not certify object identity or physical trajectories.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("prepare")
    make.add_argument("--seed", type=Path, required=True)
    make.add_argument("--video", type=Path, required=True)
    make.add_argument("--out", type=Path, required=True)
    score = commands.add_parser("evaluate")
    score.add_argument("--benchmark", type=Path, required=True)
    score.add_argument("--prediction", action="append", required=True, help="Benchmark source name=prediction directory")
    score.add_argument("--tolerance-px", type=float, default=4)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            print(prepare(args.seed, args.video, args.out))
        else:
            mappings = {}
            for item in args.prediction:
                name, path = item.split("=", 1)
                if name in mappings:
                    raise ValueError("Duplicate prediction mapping")
                mappings[name] = Path(path)
            print(json.dumps(evaluate(args.benchmark, mappings, args.tolerance_px), indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, AttributeError, IndexError) as exc:
        parser.exit(1, f"Benchmark error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
