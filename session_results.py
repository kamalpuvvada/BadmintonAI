"""Native-frame experimental preview and a self-contained offline results page."""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryFile

import cv2
import numpy as np

from badminton_proto import probe_video


def observation_at(rows: list[dict], times: list[float], timestamp: float, fps: float):
    """Hold the preceding observation for at most one sampling interval; never interpolate."""
    index = bisect_right(times, timestamp + 1e-6) - 1
    if index < 0 or timestamp - times[index] >= 1 / fps - 1e-6:
        return None
    return rows[index]


def render_preview(video: Path, output: Path, summary: dict, rows: list[dict], shuttle: Path | None,
                   cancel=lambda: None):
    meta = probe_video(video)
    width, height, fps = meta["width_px"], meta["height_px"], meta["frame_rate_fps"]
    # H.264 yuv420p requires even dimensions; source is never stretched or cropped.
    preview_width, preview_height = width + width % 2, height + height % 2 + 64
    sample_fps = summary["sampling"]["fps"]
    maximum = summary["sampling"].get("max_seconds")
    start = summary["sampling"].get("start_seconds", 0)
    if start != 0:
        raise ValueError("Session preview currently requires start_seconds=0")
    # Tracker JSON rounds seconds to milliseconds. Restore its known sampling grid.
    times = [start + r["frame_index"] / sample_fps for r in rows]
    shuttle_rows = [json.loads(line) for line in (shuttle / "shuttle_candidates.jsonl").read_text(encoding="utf-8").splitlines()] if shuttle else []
    shuttle_times = [r["timestamp_seconds"] for r in shuttle_rows]
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError("Cannot decode source video")
    target = output / "preview.mp4"
    arguments = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{preview_width}x{preview_height}", "-r", str(fps), "-i", "pipe:0", "-an",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(target)]
    count, sampled_frames = 0, 0
    source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    estimated = min(source_count, math.ceil(maximum * fps)) if maximum is not None else source_count
    selected = set(np.linspace(0, max(0, estimated-1), min(9, estimated), dtype=int).tolist())
    samples = []
    try:
        with TemporaryFile() as errors:
            process = subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
            try:
                while maximum is None or count / fps < maximum - 1e-6:
                    cancel()
                    ok, frame = capture.read()
                    if not ok:
                        break
                    timestamp = count / fps
                    observation = observation_at(rows, times, timestamp, sample_fps)
                    row = observation_at(shuttle_rows, shuttle_times, timestamp, fps) if shuttle_rows else None
                    canvas = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)
                    canvas[64:64+height, :width] = frame
                    text = f"{timestamp:.3f}s | UNVERIFIED tracking proposals"
                    if observation is not None:
                        sampled_frames += 1
                        age = max(0, timestamp - observation["timestamp_seconds"])
                        text += f" | sample age {age:.3f}s"
                        for person in observation["persons"]:
                            x1, y1, x2, y2 = map(lambda n: int(round(n)), person["bbox_xyxy_px"])
                            color = (230, 220, 50) if person["track_id"] is not None else (30, 160, 255)
                            cv2.rectangle(canvas, (x1, y1+64), (x2, y2+64), color, 2)
                            label = f"T{person['track_id']}" if person["track_id"] is not None else "ID pending"
                            cv2.putText(canvas, label, (max(0, x1), max(80, y1+60)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
                    if row and row.get("proposal"):
                        x, y = map(lambda n: int(round(n)), row["proposal"]["position_px"])
                        cv2.circle(canvas, (x, y+64), 10, (255, 170, 40), 2)
                    cv2.putText(canvas, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .4, (245, 245, 245), 1)
                    cv2.putText(canvas, "T = temporary tracker ID; original video graphics are not our measurements", (8, 48),
                                cv2.FONT_HERSHEY_SIMPLEX, .36, (180, 210, 245), 1)
                    process.stdin.write(canvas.tobytes())
                    if count in selected:
                        filename = f"review-{count:06d}.jpg"
                        cv2.imwrite(str(output / filename), canvas)
                        samples.append({"frame_index": count, "source_seconds": timestamp, "image": filename})
                    count += 1
                process.stdin.close()
                if process.wait(timeout=60):
                    errors.seek(0)
                    raise RuntimeError("Preview encoding failed: " + errors.read().decode(errors="replace")[-2000:])
            finally:
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
    finally:
        capture.release()
    if not count:
        raise ValueError("Source contains no decodable frames")
    # Decode the complete output; a successful encoder exit alone is insufficient.
    check = cv2.VideoCapture(str(target))
    decoded = 0
    try:
        while True:
            cancel()
            ok, frame = check.read()
            if not ok:
                break
            if frame.shape[:2] != (preview_height, preview_width):
                raise ValueError("Preview dimensions differ from the declared geometry")
            decoded += 1
    finally:
        check.release()
    if decoded != count:
        raise ValueError("Preview frame count does not match decoded source")
    result = {"schema_version": "1.0.0", "frames": count, "decoded_preview_frames": decoded,
              "fps": fps, "source_size_px": [width, height], "preview_size_px": [preview_width, preview_height],
              "source_rectangle_xywh_px": [0, 64, width, height], "source_frame_index_offset": 0,
              "frames_with_sampled_player_observation": sampled_frames, "player_observation_display": "preceding sample held for less than one sampling interval; no interpolation",
              "timebase": "frame_index / probed average FPS; constant-frame-rate input required for exact alignment",
              "audio": "omitted_from_preview; source unchanged", "review_samples": samples}
    (output / "preview.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def write_results(output: Path, source: dict, config: dict, players: Path, inspected: Path,
                  rendered: Path, shuttle: Path | None, verification: dict):
    summary = json.loads((players / "tracking_summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (players / "player_tracks.jsonl").read_text(encoding="utf-8").splitlines()]
    preview = json.loads((rendered / "preview.json").read_text(encoding="utf-8"))
    def relative(path):
        return path.relative_to(output).as_posix()
    status = "needs_review" if not verification["all_scenes_decided"] else "completed"
    result = {"schema_version": "1.0.0", "session_id": summary["session_id"], "status": status,
              "source": source, "configuration": config,
              "players": {"status": "experimental", "track_ids_are_athlete_identities": False,
                          "sampled_frames": len(rows), "person_count_histogram": dict(Counter(str(len(r["persons"])) for r in rows)),
                          "diagnostics": summary["diagnostics"], "review": verification},
              "shuttle": {"status": "experimental_unverified_candidates" if shuttle else "not_available",
                          "reason": None if shuttle else "No currently accepted player intervals", "verified_positions": 0},
              "metrics": {name: {"status": "not_available", "value": None, "reason": reason} for name, reason in (
                  ("distance_metres", "Ground-contact positions and court calibration have not been validated"),
                  ("shuttle_speed", "No validated physical flight reconstruction"),
                  ("contacts", "Contact detection has not been implemented and validated"),
                  ("shot_types", "Shot classification has not been implemented and validated"),
                  ("coaching", "No validated supporting measurements"))},
              "preview": preview,
              "limitations": ["All new tracking overlays are proposals; temporary T IDs are not P1-P4 athlete assignments.",
                  "Sample-and-hold display is not an observation on every native video frame.",
                  "Existing source graphics are not BadmintonAI measurements or independent labels."] if config["processed_overlay"] else [
                  "All new tracking overlays are proposals; temporary T IDs are not athlete assignments.",
                  "Sample-and-hold display is not an observation on every native video frame."],
              "artifacts": {"preview": relative(rendered / "preview.mp4"), "preview_metadata": relative(rendered / "preview.json"),
                            "player_review": relative(players / "tracking_review.html"),
                            "player_summary": relative(players / "tracking_summary.json"), "observations": relative(players / "player_tracks.jsonl"),
                            "inspection": relative(inspected / "session.json"),
                            "detector_sheet": relative(players / "detector_review" / "review_contact_sheet.jpg"),
                            "shuttle_summary": relative(shuttle / "shuttle_summary.json") if shuttle else None}}
    text = json.dumps(result, indent=2, allow_nan=False)
    (output / "result.json.tmp").write_text(text + "\n", encoding="utf-8")
    (output / "result.json.tmp").replace(output / "result.json")
    template = (Path(__file__).parent / "session_results.html").read_text(encoding="utf-8")
    # Safe inside a script element even if metadata includes </script> or HTML.
    payload = json.dumps(result, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    page = template.replace("/*SESSION_DATA*/", payload)
    (output / "index.html.tmp").write_text(page, encoding="utf-8")
    (output / "index.html.tmp").replace(output / "index.html")
    return result
