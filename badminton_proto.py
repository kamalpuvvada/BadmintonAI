#!/usr/bin/env python3
"""Step 1 of a fixed-camera badminton analysis pipeline.

This module intentionally does not claim to recognize players, hits, or shot
speeds. It establishes repeatable video ingestion and court calibration before
we introduce learned models in subsequent steps.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


VERSION = "0.1.0"
DOUBLES_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
CORNER_NAMES = ("far_left", "far_right", "near_right", "near_left")


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "Unknown error").strip()
        raise RuntimeError(f"{command[0]} failed: {detail}") from exc


def probe_video(video: Path) -> dict[str, Any]:
    if not video.is_file():
        raise ValueError(f"Video file not found: {video}")
    result = _run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=index,codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate",
        "-of", "json", str(video),
    ])
    metadata = json.loads(result.stdout)
    stream = next((s for s in metadata.get("streams", []) if s.get("codec_type") == "video"), None)
    if stream is None:
        raise ValueError("Input has no video stream")
    duration = float(metadata.get("format", {}).get("duration", 0))
    fps_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    fps = float(Fraction(fps_text))
    if duration <= 0 or fps <= 0:
        raise ValueError("Video requires positive duration and frame rate")
    return {
        "filename": video.name,
        "duration_seconds": round(duration, 3),
        "width_px": int(stream["width"]),
        "height_px": int(stream["height"]),
        "frame_rate_fps": round(fps, 3),
        "video_codec": stream.get("codec_name", "unknown"),
        "audio_present": any(s.get("codec_type") == "audio" for s in metadata.get("streams", [])),
        "file_size_bytes": int(metadata.get("format", {}).get("size", video.stat().st_size)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def extract_review_frames(video: Path, output: Path, duration: float, count: int = 9) -> list[dict[str, Any]]:
    frame_dir = output / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(count):
        # Avoid the very beginning/end of clips, where cuts/fades are common.
        timestamp = duration * (i + 0.5) / count
        path = frame_dir / f"frame_{i + 1:02}.jpg"
        _run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{timestamp:.3f}", "-i", str(video),
            "-frames:v", "1", "-vf", "scale=360:-2",
            "-q:v", "3", str(path),
        ])
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Could not extract review frame at {timestamp:.2f}s")
        frames.append({"timestamp_seconds": round(timestamp, 3), "image": str(path.relative_to(output))})
    return frames


def build_contact_sheet(output: Path, frames: list[dict[str, Any]]) -> str:
    width, height = 360, 680
    sheet = Image.new("RGB", (width * 3, height * 3), "#131b24")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(frames):
        image_path = output / item["image"]
        with Image.open(image_path) as source:
            thumbnail = source.convert("RGB")
            thumbnail.thumbnail((width, height - 40))
            x = (index % 3) * width + (width - thumbnail.width) // 2
            y = (index // 3) * height + 32
            sheet.paste(thumbnail, (x, y))
        draw.text(((index % 3) * width + 12, (index // 3) * height + 9),
                  f"{item['timestamp_seconds']:.1f} sec", fill="white", font=_font(16))
    filename = "review_contact_sheet.jpg"
    sheet.save(output / filename, quality=90)
    return filename


def capture_warnings(meta: dict[str, Any], overlay_present: bool) -> list[str]:
    warnings: list[str] = []
    if meta["frame_rate_fps"] < 60:
        warnings.append("Under 60 fps: fast shuttle motion and hit moments may be missed.")
    if meta["width_px"] < meta["height_px"]:
        warnings.append("Portrait footage: verify that the entire court and every player remain visible.")
    if overlay_present:
        warnings.append("Burned-in graphics may confuse detection; obtain original camera footage.")
    warnings.append("One camera cannot directly provide ground-truth 3D shuttle speed or height.")
    return warnings


def inspect(video: Path, output: Path, camera_id: str, overlay_present: bool) -> Path:
    metadata = probe_video(video)
    digest = sha256_file(video)
    output.mkdir(parents=True, exist_ok=True)
    frames = extract_review_frames(video, output, metadata["duration_seconds"])
    contact_sheet = build_contact_sheet(output, frames)
    result = {
        "schema_version": "0.1.0",
        "tool_version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": f"{camera_id}-{digest[:12]}",
        "camera_id": camera_id,
        "capture_mode": "fixed_camera_offline",
        "video_sha256": digest,
        "video": metadata,
        "capture_warnings": capture_warnings(metadata, overlay_present),
        "artifacts": {"contact_sheet": contact_sheet, "review_frames": frames},
        "calibration": {"status": "not_provided", "required_before_court_metrics": True},
        "analytics": {
            "status": "not_run",
            "players": [],
            "rallies": [],
            "shots": [],
            "note": "No AI detections or measurements have been produced in Step 1.",
        },
    }
    report = output / "session.json"
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return report


def project_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Map 2D pixel points to metric court coordinates using a homography."""
    points = np.asarray(points, dtype=np.float64)
    homog = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if np.any(np.abs(homog[:, 2]) < 1e-12):
        raise ValueError("Homography projects a point to infinity")
    return homog[:, :2] / homog[:, 2:3]


def solve_homography(pixel: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Find a planar projective transform with four point correspondences."""
    if np.asarray(pixel).shape != (4, 2) or np.asarray(world).shape != (4, 2):
        raise ValueError("Expected exactly four 2D court corners")
    pixel = np.asarray(pixel, dtype=np.float64)
    world = np.asarray(world, dtype=np.float64)
    a, b = [], []
    for (x, y), (u, v) in zip(pixel, world):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.extend((u, v))
    try:
        values = np.linalg.solve(np.asarray(a), np.asarray(b))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Court points are degenerate or incorrectly ordered") from exc
    matrix = np.append(values, 1).reshape(3, 3)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Invalid calibration transform")
    return matrix


def _polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _convex_quad(points: np.ndarray) -> bool:
    edges = np.roll(points, -1, axis=0) - points
    next_edges = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    return bool(np.all(turns > 0) or np.all(turns < 0))


def calibrate(frame: Path, corners_file: Path, output: Path, camera_id: str) -> Path:
    if not frame.is_file():
        raise ValueError(f"Frame not found: {frame}")
    corners = json.loads(corners_file.read_text(encoding="utf-8"))
    points = np.asarray([corners[name] for name in CORNER_NAMES], dtype=np.float64)
    if not np.all(np.isfinite(points)) or not _convex_quad(points):
        raise ValueError("Corners must form a convex quadrilateral in the specified order")
    with Image.open(frame) as image:
        image_size = image.size
        if np.any(points < 0) or np.any(points[:, 0] >= image_size[0]) or np.any(points[:, 1] >= image_size[1]):
            raise ValueError("Every corner must fall inside the image")
        if _polygon_area(points) < image_size[0] * image_size[1] * 0.02:
            raise ValueError("Court footprint is too small to calibrate reliably")
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        ordered = [tuple(map(float, row)) for row in points]
        draw.line(ordered + [ordered[0]], fill="#a9ff46", width=4)
        for name, point in zip(CORNER_NAMES, ordered):
            x, y = point
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#a9ff46")
            draw.text((x + 8, y - 22), name, fill="#ffffff", stroke_width=2, stroke_fill="#0b1622", font=_font(15))
        output.mkdir(parents=True, exist_ok=True)
        preview = output / "court_calibration_preview.jpg"
        image.save(preview, quality=92)
    world = np.asarray([
        [0, 0], [DOUBLES_WIDTH_M, 0],
        [DOUBLES_WIDTH_M, COURT_LENGTH_M], [0, COURT_LENGTH_M],
    ], dtype=np.float64)
    matrix = solve_homography(points, world)
    residual = np.linalg.norm(project_points(points, matrix) - world, axis=1)
    result = {
        "schema_version": "0.1.0",
        "camera_id": camera_id,
        "frame_filename": frame.name,
        "coordinate_system": "metres; x from left to right, y from far baseline to near baseline",
        "court": {"type": "doubles", "width_m": DOUBLES_WIDTH_M, "length_m": COURT_LENGTH_M},
        "image_size": {"width_px": image_size[0], "height_px": image_size[1]},
        "corner_order": list(CORNER_NAMES),
        "pixel_corners": points.tolist(),
        "world_corners_m": world.tolist(),
        "pixel_to_court_homography": matrix.tolist(),
        "corner_fit_max_error_m": float(np.max(residual)),
        "preview": preview.name,
        "limitations": [
            "A 2D homography only maps points on the floor plane.",
            "Four-point fit error does not measure real-world calibration accuracy.",
            "Recalibrate if the camera is moved, zoomed, or re-aimed.",
        ],
    }
    target = output / "calibration.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Badminton academy AI — Step 1 video QA and court calibration")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect", help="Make a video QA/session report")
    inspect_parser.add_argument("--video", type=Path, required=True)
    inspect_parser.add_argument("--out", type=Path, required=True)
    inspect_parser.add_argument("--camera", default="court-01")
    inspect_parser.add_argument("--processed-overlay", action="store_true",
                                help="Mark that the video includes burned-in analytics or labels")
    calibrate_parser = commands.add_parser("calibrate", help="Calibrate a fixed camera from four visible corners")
    calibrate_parser.add_argument("--frame", type=Path, required=True)
    calibrate_parser.add_argument("--corners", type=Path, required=True)
    calibrate_parser.add_argument("--out", type=Path, required=True)
    calibrate_parser.add_argument("--camera", default="court-01")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "inspect":
            report = inspect(arguments.video, arguments.out, arguments.camera, arguments.processed_overlay)
        else:
            report = calibrate(arguments.frame, arguments.corners, arguments.out, arguments.camera)
    except (ValueError, KeyError, TypeError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Created {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
