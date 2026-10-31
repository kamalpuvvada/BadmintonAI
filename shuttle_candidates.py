"""Experimental moving-bright-blob proposals; never verified shuttle detections."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageDraw

from badminton_proto import _font, probe_video, sha256_file
from player_tracking import _encode_preview, _extract_sampled_video, parse_roi
from tracking_review import verified_snapshot


def moving_candidates(previous: np.ndarray, current: np.ndarray, following: np.ndarray,
                      roi: tuple[float, float, float, float], player_boxes: list[list[float]],
                      background: np.ndarray | None = None, allow_tinted_streaks: bool = False) -> list[dict]:
    """Tiny bright components changing on BOTH sides of the current frame.

    Player masks deliberately trade away near-body shuttle recall. Colour,
    shape and motion are heuristics, not a learned shuttle classifier.
    """
    if previous.shape != current.shape or following.shape != current.shape:
        raise ValueError("Temporal images must have equal dimensions")
    gray = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in (previous, current, following)]
    motion = np.minimum(cv2.absdiff(gray[1], gray[0]), cv2.absdiff(gray[1], gray[2]))
    low, high = current.min(axis=2), current.max(axis=2)
    mask = ((motion >= 12) & (low >= 120) & (high >= 180)
            & (high.astype(int) - low.astype(int) <= 70)).astype(np.uint8)
    if background is not None:
        if background.shape != current.shape:
            raise ValueError("Background dimensions must match temporal images")
        background_gray = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
        # Persistent white court markings and graphics can flicker under video
        # compression. Require positive contrast against a scene-local median.
        contrast = gray[1].astype(int) - background_gray.astype(int)
        mask &= (contrast >= 20).astype(np.uint8)
        if allow_tinted_streaks:
            # Blur over a coloured backdrop can tint the shuttle. Keep the
            # motion/background tests; colour alone cannot identify it.
            tinted = ((motion >= 12) & (low >= 80) & (high >= 160)
                      & (high.astype(int) - low.astype(int) <= 160) & (contrast >= 25))
            mask |= tinted.astype(np.uint8)
    elif allow_tinted_streaks:
        raise ValueError("Tinted-streak proposals require a scene background")
    height, width = mask.shape
    allowed = np.zeros_like(mask)
    left, top, right, bottom = roi
    allowed[int(top * height):math.ceil(bottom * height), int(left * width):math.ceil(right * width)] = 1
    mask &= allowed
    for x1, y1, x2, y2 in player_boxes:
        mask[max(0, math.floor(y1) - 4):min(height, math.ceil(y2) + 4),
             max(0, math.floor(x1) - 4):min(width, math.ceil(x2) + 4)] = 0
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = []
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if 1 <= area <= 50 and w <= 20 and h <= 20:
            result.append({"center_px": [round(float(v), 2) for v in centers[label]],
                           "bbox_xyxy_px": [x, y, x + w, y + h], "area_px": area})
    return result


class ProposalLinker:
    """Consecutive-frame proposals only; no interpolation, no gap bridging."""
    def __init__(self, max_step_px: float = 60, ambiguity_margin_px: float = 8):
        if not all(math.isfinite(v) and v > 0 for v in (max_step_px, ambiguity_margin_px)):
            raise ValueError("Link distances must be positive and finite")
        self.max_step_px = max_step_px
        self.margin = ambiguity_margin_px
        self.next_id = 0
        self.reset()

    def reset(self):
        self.scene = None
        self.last_frame = None
        self.points = []
        self.segment_id = None

    def update(self, frame: int, scene: int, candidates: list[dict]) -> dict | None:
        if self.scene != scene or self.last_frame != frame - 1:
            self.reset()
        if not candidates:
            self.reset()
            return None
        if not self.points:
            if len(candidates) != 1:
                self.reset()
                return None
            index = 0
            self.segment_id = self.next_id
            self.next_id += 1
        else:
            last = np.array(self.points[-1])
            predicted = last if len(self.points) == 1 else 2 * last - np.array(self.points[-2])
            choices = sorted((float(np.linalg.norm(np.array(c["center_px"]) - predicted)), i)
                             for i, c in enumerate(candidates)
                             if np.linalg.norm(np.array(c["center_px"]) - last) <= self.max_step_px)
            if not choices or (len(choices) > 1 and choices[1][0] - choices[0][0] < self.margin):
                self.reset()
                return None
            index = choices[0][1]
        point = candidates[index]["center_px"]
        self.points = (self.points + [point])[-2:]
        self.scene, self.last_frame = scene, frame
        return {"candidate_index": index, "proposal_segment_id": self.segment_id,
                "position_px": list(point), "verification_status": "unverified"}


def frame_context(timestamp: float, summary: dict, rows: list[dict], intervals: list[dict]) -> dict:
    """Conservatively restrict proposals to reviewed player intervals.

    This is inherited player-view eligibility, NOT shuttle review or proof
    that higher-rate frames were themselves reviewed.
    """
    fps, start = summary["sampling"]["fps"], summary["sampling"].get("start_seconds", 0)
    offset = (timestamp - start) * fps
    index = math.floor(offset + 1e-8)
    if index < 0 or index >= len(rows):
        return {"scene_id": None, "eligible": False, "player_boxes": []}
    row = rows[index]
    eligible = any(i["scene_id"] == row["scene_id"]
                   and start + i["start_frame"] / fps - 1e-8 <= timestamp
                   <= start + i["end_frame"] / fps + 1e-8 for i in intervals)
    # Include adjacent sampled boxes only within this scene. Fast-moving
    # bodies otherwise leak small white fragments between player samples.
    nearby = rows[max(0, index - 1):min(len(rows), index + 2)]
    boxes = [p["bbox_xyxy_px"] for r in nearby if r["scene_id"] == row["scene_id"] for p in r["persons"]]
    return {"scene_id": row["scene_id"], "eligible": eligible, "player_boxes": boxes}


def run(video: Path, player_run: Path, output: Path,
        roi: tuple[float, float, float, float] = (.10, .08, .90, 1.0),
        allow_tinted_streaks: bool = False) -> Path:
    if output.exists():
        raise ValueError("Choose a new output directory; existing evidence is never overwritten")
    verification, summary, player_rows, review = verified_snapshot(player_run)
    meta = probe_video(video)
    source_hash = sha256_file(video)
    if (not summary["session_id"].endswith(source_hash[:12])
            or (meta["width_px"], meta["height_px"]) != (summary["video"]["width_px"], summary["video"]["height_px"])):
        raise ValueError("Source video does not match the reviewed player run")
    fps = meta["frame_rate_fps"]
    if not math.isfinite(fps) or not 0 < fps <= 120:
        raise ValueError("Source FPS must be within 0..120")
    start = summary["sampling"].get("start_seconds", 0)
    duration = len(player_rows) / summary["sampling"]["fps"]
    review_hash = sha256(json.dumps(review, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    linker = ProposalLinker()
    output.mkdir(parents=True)
    counts = Counter()
    records = []
    with TemporaryDirectory() as temporary:
        work = Path(temporary)
        paths = _extract_sampled_video(video, work / "source", fps, duration, start)
        contexts = [frame_context(start + i / fps, summary, player_rows, verification["approved_intervals"])
                    for i in range(len(paths))]
        backgrounds = {}
        for scene in sorted({c["scene_id"] for c in contexts if c["eligible"]}):
            indices = [i for i, c in enumerate(contexts) if c["eligible"] and c["scene_id"] == scene]
            selected = [indices[j] for j in np.linspace(0, len(indices) - 1, min(31, len(indices)), dtype=int)]
            backgrounds[scene] = np.median(np.stack([
                np.array(Image.open(paths[i]).convert("RGB")) for i in selected]), axis=0).astype(np.uint8)
        annotated = work / "annotated"
        annotated.mkdir()
        # Keep a three-frame window in memory, not the full video.
        previous, current = None, np.array(Image.open(paths[0]).convert("RGB"))
        for index, path in enumerate(paths):
            following = np.array(Image.open(paths[index + 1]).convert("RGB")) if index + 1 < len(paths) else None
            context = contexts[index]
            scene = context["scene_id"]
            same_context = (index > 0 and following is not None and all(
                c["eligible"] and c["scene_id"] == scene for c in contexts[index - 1:index + 2]))
            candidates, proposal = [], None
            if not context["eligible"]:
                status = "excluded_player_review"
            elif not same_context:
                status = "no_temporal_context"
            else:
                candidates = moving_candidates(previous, current, following, roi, context["player_boxes"], backgrounds[scene], allow_tinted_streaks)
                if len(candidates) > 12:
                    status = "excessive_clutter"
                else:
                    proposal = linker.update(index, scene, candidates)
                    status = "proposal" if proposal else ("no_candidate" if not candidates else "ambiguous_candidates")
            if proposal is None:
                linker.reset()
            timestamp = start + index / fps
            record = {"frame_index": index, "timestamp_seconds": round(timestamp, 6), "scene_id": scene,
                      "status": status, "candidates": candidates, "proposal": proposal,
                      "verified_shuttle_position_px": None}
            records.append(record)
            counts[status] += 1
            image = Image.fromarray(current)
            draw = ImageDraw.Draw(image)
            for n, candidate in enumerate(candidates):
                x, y = candidate["center_px"]
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline="#ffb14e", width=2)
                draw.text((x + 7, y), str(n), fill="#ffb14e", font=_font(12))
            if proposal:
                x, y = proposal["position_px"]
                draw.rectangle((x - 9, y - 9, x + 9, y + 9), outline="#74e6e5", width=2)
            draw.rectangle((0, 0, image.width, 24), fill="#10202b")
            draw.text((5, 4), f"{timestamp:.3f}s S{scene} {status} | UNVERIFIED CANDIDATES", fill="white", font=_font(13))
            image.save(annotated / f"frame_{index + 1:06d}.jpg", quality=93)
            previous, current = current, following
        _encode_preview(annotated, output / "candidate_preview.mp4", fps)
        # Nine diagnostic frames retain source resolution in individual files.
        selected = sorted(set(np.linspace(0, len(paths) - 1, min(9, len(paths)), dtype=int).tolist()))
        sheet = Image.new("RGB", (960, math.ceil(len(selected) / 3) * 204), "#10202b")
        for j, index in enumerate(selected):
            im = Image.open(annotated / f"frame_{index + 1:06d}.jpg").convert("RGB")
            im.save(output / f"review_{index:06d}.jpg", quality=95)
            im.thumbnail((320, 180))
            sheet.paste(im, ((j % 3) * 320, (j // 3) * 204 + 24))
            ImageDraw.Draw(sheet).text(((j % 3) * 320 + 4, (j // 3) * 204 + 3), f"{start + index / fps:.3f}s", fill="white", font=_font(14))
        sheet.save(output / "candidate_contact_sheet.jpg", quality=95)
    (output / "shuttle_candidates.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records), encoding="utf-8")
    result = {"schema_version": "0.1.0", "status": "experimental_unverified_candidates",
              "method": "three_frame_motion_bright_components_scene_median_and_player_masks",
              "source_video_sha256": source_hash, "player_source_binding": verification["source_binding"],
              "player_review_snapshot_sha256": review_hash, "player_review_provenance": verification["review_provenance"],
              "video": meta, "opencv_version": cv2.__version__,
              "configuration": {"motion_difference_min": 12, "rgb_channel_min": 120, "rgb_peak_min": 180,
                                "allow_tinted_streaks": allow_tinted_streaks,
                                "tinted_rgb_channel_min": 80, "tinted_rgb_peak_min": 160,
                                "tinted_rgb_chroma_max": 160, "tinted_background_contrast_min": 25,
                                "rgb_chroma_max": 70, "background_contrast_min": 20,
                                "component_area_range_px": [1, 50], "component_dimension_max_px": 20,
                                "player_mask_margin_px": 4, "scene_background_samples_max": 31,
                                "clutter_candidate_limit": 12, "link_step_max_px": linker.max_step_px,
                                "link_ambiguity_margin_px": linker.margin},
              "sampling": {"fps": fps, "start_seconds": start, "frames": len(records)},
              "roi": list(roi), "frame_status_counts": dict(counts),
              "proposal_segments": linker.next_id, "verified_shuttle_positions": 0,
              "limitations": ["Bright moving rackets, shoes, graphics and compression noise can become proposals.",
                              "Player masks suppress near-body shuttles; blur, low contrast and static shuttles can be missed.",
                              "Scene medians use offline look-ahead within one scene and can suppress slow or stationary objects.",
                              "Player-view review is not shuttle validation; source timestamps and boundaries are sampling approximations.",
                              "No interpolation or linking across gaps, rejected views or scene boundaries.",
                              "Pixel thresholds are experimental for this 640x360 source, not resolution-independent settings."],
              "shot_events": {"status": "not_available"}, "coaching_metrics": {"status": "not_available"}}
    (output / "shuttle_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "index.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Shuttle candidate review</title><style>body{font:17px/1.6 system-ui;background:#10202b;color:#edf5fa;max-width:1000px;margin:auto;padding:24px}video,img{width:100%}a{color:#7ce6ce}</style><h1>Experimental shuttle candidates</h1><p>Orange circles are moving bright components. Cyan boxes mark linked proposals. Neither is a verified shuttle detection. No shot events or coaching scores are available.</p><video controls src="candidate_preview.mp4"></video><img src="candidate_contact_sheet.jpg" alt="Candidate review frames"><h2>Diagnostics</h2><pre>'''
        + escape(json.dumps(result, indent=2)) + '</pre></html>', encoding="utf-8")
    return output / "shuttle_summary.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--player-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roi", default="0.10,0.08,0.90,1.00")
    parser.add_argument("--allow-tinted-streaks", action="store_true", help="Experimental colour-tinted blur proposals; may admit more distractors")
    args = parser.parse_args()
    try:
        print(run(args.video, args.player_run, args.out, parse_roi(args.roi), args.allow_tinted_streaks))
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f"Shuttle candidate error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
