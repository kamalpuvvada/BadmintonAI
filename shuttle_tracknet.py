"""Offline TrackNetV3 heatmap proposals, without inpainting or verified events."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from badminton_proto import _font, probe_video, sha256_file
from download_tracknet import CHECKPOINT_SHA256
from player_tracking import _extract_sampled_video, _encode_preview
from shuttle_candidates import frame_context, ProposalLinker
from tracking_review import verified_snapshot
from third_party.tracknetv3.tracknet import TrackNet


MODEL_SIZE = (512, 288)


def _validate_heatmap_threshold(threshold: float):
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Heatmap threshold must be finite and strictly between zero and one")


def sequence_windows(indices: list[int], sequence_length: int, stride: int | None = None) -> list[list[int]]:
    if type(sequence_length) is not int or not 1 <= sequence_length <= 16:
        raise ValueError("Sequence length must be an integer in 1..16")
    stride = sequence_length if stride is None else stride
    if type(stride) is not int or not 1 <= stride <= sequence_length:
        raise ValueError("Window stride must be an integer in 1..sequence length")
    if any(type(i) is not int or i < 0 for i in indices) or any(b != a + 1 for a, b in zip(indices, indices[1:])):
        raise ValueError("A sequence group must contain consecutive frame indices")
    if len(indices) < sequence_length:
        return []
    starts = list(range(0, len(indices) - sequence_length + 1, stride))
    if starts[-1] != len(indices) - sequence_length:
        starts.append(len(indices) - sequence_length)
    return [indices[start:start + sequence_length] for start in starts]


def prepare_input(images: list[Image.Image], background: np.ndarray, size=MODEL_SIZE) -> np.ndarray:
    """RGB background first, then sequence RGB; upstream concat convention."""
    if not images or background.shape != (images[0].height, images[0].width, 3):
        raise ValueError("Background and source image sizes must match")
    if any(image.mode != "RGB" or image.size != images[0].size for image in images):
        raise ValueError("Sequence images must be equal-sized RGB")
    channels = []
    for image in [Image.fromarray(background.astype(np.uint8)), *images]:
        array = np.asarray(image.resize(size, Image.Resampling.BICUBIC), dtype=np.float32)
        channels.append(np.moveaxis(array, -1, 0))
    return np.concatenate(channels, axis=0)[None] / np.float32(255)


def decode_heatmap(heatmap: np.ndarray, image_size: tuple[int, int], threshold: float = .5) -> list[dict]:
    _validate_heatmap_threshold(threshold)
    if (heatmap.ndim != 2 or not all(heatmap.shape) or not np.all(np.isfinite(heatmap))
            or np.any(heatmap < 0) or np.any(heatmap > 1)):
        raise ValueError("Invalid heatmap or threshold")
    if any(type(v) is not int or v <= 0 for v in image_size):
        raise ValueError("Invalid source image size")
    mask = (heatmap > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sx, sy = image_size[0] / heatmap.shape[1], image_size[1] / heatmap.shape[0]
    boxes = sorted((cv2.boundingRect(contour) for contour in contours), key=lambda b: (-b[2]*b[3], b[1], b[0]))
    return [{"center_px": [round((x + w/2)*sx, 3), round((y + h/2)*sy, 3)],
             "bbox_xyxy_px": [round(x*sx, 3), round(y*sy, 3), round((x+w)*sx, 3), round((y+h)*sy, 3)],
             "heatmap_peak": round(float(heatmap[y:y+h, x:x+w].max()), 6)} for x, y, w, h in boxes]


class TrackNetPredictor:
    def __init__(self, weights: Path, expected_sha256: str):
        actual = sha256_file(weights)
        if actual != expected_sha256:
            raise ValueError("Checkpoint SHA-256 does not match the requested artifact")
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        parameters = checkpoint["param_dict"]
        self.sequence_length = parameters["seq_len"]
        if (type(self.sequence_length) is not int or not 1 <= self.sequence_length <= 16
                or parameters["bg_mode"] != "concat" or parameters["model_name"] != "TrackNet"):
            raise ValueError("Only TrackNet concat-background checkpoints with 1..16 frames are supported")
        self.model = TrackNet((self.sequence_length + 1)*3, self.sequence_length)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        self.metadata = {"name": "TrackNetV3 tracking module", "checkpoint_sha256": actual,
                         "sequence_length": self.sequence_length, "background_mode": "concat",
                         "upstream_commit": "6eda442ada1740573f200f836d93edc9a541ee86",
                         "torch_version": torch.__version__, "device": "cpu", "inpainting": False}

    def __call__(self, input_array: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return self.model(torch.from_numpy(input_array)).cpu().numpy()


def _read_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def infer_groups(paths: list[Path], contexts: list[dict], predictor, stride: int | None = None,
                 heatmap_threshold: float = .5, frame_peaks: dict | None = None) -> dict[int, list[dict]]:
    """Stride-controlled blocks plus a covering tail, averaging observations.

    No padding, synthetic positions, cross-boundary context or repair model.
    A too-short accepted group simply supplies no heatmaps.
    """
    if len(paths) != len(contexts):
        raise ValueError("Frame and context counts differ")
    _validate_heatmap_threshold(heatmap_threshold)
    sequence_windows([], predictor.sequence_length, stride)  # Validate even with no eligible groups.
    groups = []
    for index, context in enumerate(contexts):
        if not context["eligible"]:
            continue
        if not groups or groups[-1][-1] != index - 1 or contexts[groups[-1][-1]]["scene_id"] != context["scene_id"]:
            groups.append([])
        groups[-1].append(index)
    detections = {}
    for indices in groups:
        windows = sequence_windows(indices, predictor.sequence_length, stride)
        if not windows:
            continue
        samples = [indices[j] for j in np.linspace(0, len(indices)-1, min(31, len(indices)), dtype=int)]
        background = np.median(np.stack([np.array(_read_image(paths[i])) for i in samples]), axis=0).astype(np.uint8)
        means = np.zeros((len(indices), MODEL_SIZE[1], MODEL_SIZE[0]), dtype=np.float32)
        counts = np.zeros(len(indices), dtype=np.int32)
        image_size = _read_image(paths[indices[0]]).size
        for window in windows:
            batch = prepare_input([_read_image(paths[i]) for i in window], background)
            heatmaps = predictor(batch)
            if heatmaps.shape != (1, predictor.sequence_length, MODEL_SIZE[1], MODEL_SIZE[0]) or not np.all(np.isfinite(heatmaps)):
                raise ValueError("Model returned invalid heatmap shape or values")
            for offset, index in enumerate(window):
                means[index - indices[0]] += heatmaps[0, offset]
                counts[index - indices[0]] += 1
        for offset, index in enumerate(indices):
            averaged = means[offset] / counts[offset]
            detections[index] = decode_heatmap(averaged, image_size, heatmap_threshold)
            if frame_peaks is not None:
                frame_peaks[index] = round(float(averaged.max()), 6)
    return detections


def run(video: Path, player_run: Path, output: Path, weights: Path,
        expected_sha256: str = CHECKPOINT_SHA256, threads: int = 4, window_stride: int | None = None,
        heatmap_threshold: float = .5) -> Path:
    if output.exists():
        raise ValueError("Choose a new output directory")
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError("CPU threads must be within 1..32")
    _validate_heatmap_threshold(heatmap_threshold)
    verification, summary, player_rows, review = verified_snapshot(player_run)
    meta, video_hash = probe_video(video), sha256_file(video)
    if (not summary["session_id"].endswith(video_hash[:12])
            or (meta["width_px"], meta["height_px"]) != (summary["video"]["width_px"], summary["video"]["height_px"])):
        raise ValueError("Video does not match the reviewed player run")
    torch.set_num_threads(threads)
    predictor = TrackNetPredictor(weights, expected_sha256)
    sequence_windows([], predictor.sequence_length, window_stride)
    fps, start = meta["frame_rate_fps"], summary["sampling"].get("start_seconds", 0)
    if not math.isfinite(fps) or not 0 < fps <= 120:
        raise ValueError("Source FPS must be within 0..120")
    duration = len(player_rows) / summary["sampling"]["fps"]
    output.mkdir(parents=True)
    timer = time.perf_counter()
    records, counts = [], Counter()
    linker = ProposalLinker()
    with TemporaryDirectory() as temporary:
        work = Path(temporary)
        paths = _extract_sampled_video(video, work / "source", fps, duration, start)
        contexts = [frame_context(start+i/fps, summary, player_rows, verification["approved_intervals"]) for i in range(len(paths))]
        frame_peaks = {}
        detections = infer_groups(paths, contexts, predictor, window_stride, heatmap_threshold, frame_peaks)
        annotated = work / "annotated"
        annotated.mkdir()
        selected = set(np.linspace(0, len(paths)-1, min(9, len(paths)), dtype=int).tolist())
        sheet = Image.new("RGB", (960, math.ceil(len(selected)/3)*204), "#10202b")
        for index, path in enumerate(paths):
            context = contexts[index]
            candidates = detections.get(index, [])
            proposal = None
            if not context["eligible"]:
                status = "excluded_player_review"
            elif index not in detections:
                status = "insufficient_sequence_context"
            else:
                proposal = linker.update(index, context["scene_id"], candidates)
                status = "proposal" if proposal else ("no_heatmap_detection" if not candidates else "ambiguous_candidates")
            if proposal is None:
                linker.reset()
            timestamp = start+index/fps
            records.append({"frame_index": index, "timestamp_seconds": round(timestamp, 6), "scene_id": context["scene_id"],
                            "status": status, "candidates": candidates, "proposal": proposal,
                            "heatmap_peak": frame_peaks.get(index),
                            "verified_shuttle_position_px": None})
            counts[status] += 1
            image = _read_image(path)
            draw = ImageDraw.Draw(image)
            for candidate in candidates:
                x,y = candidate["center_px"]
                draw.ellipse((x-6,y-6,x+6,y+6), outline="#ffb14e", width=2)
            if proposal:
                x,y = proposal["position_px"]
                draw.rectangle((x-9,y-9,x+9,y+9), outline="#74e6e5", width=2)
            draw.rectangle((0,0,image.width,24), fill="#10202b")
            draw.text((5,4),f"{timestamp:.3f}s S{context['scene_id']} {status} | TRACKNET UNVERIFIED", fill="white", font=_font(13))
            image.save(annotated/f"frame_{index+1:06d}.jpg",quality=93)
            if index in selected:
                image.save(output/f"review_{index:06d}.jpg",quality=95)
                j=sorted(selected).index(index)
                image.thumbnail((320,180));sheet.paste(image,((j%3)*320,(j//3)*204+24))
                ImageDraw.Draw(sheet).text(((j%3)*320+4,(j//3)*204+3),f"{timestamp:.3f}s",fill="white",font=_font(14))
        _encode_preview(annotated,output/"candidate_preview.mp4",fps)
        sheet.save(output/"candidate_contact_sheet.jpg",quality=95)
    result={"schema_version":"0.1.0","status":"experimental_unverified_candidates","method":"tracknetv3_concat_scene_bounded_heatmaps",
            "source_video_sha256":video_hash,"player_source_binding":verification["source_binding"],
            "player_review_snapshot_sha256":sha256(json.dumps(review,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),
            "player_review_provenance":verification["review_provenance"],"video":meta,"model":predictor.metadata,
            "sampling":{"fps":fps,"start_seconds":start,"frames":len(records)},"frame_status_counts":dict(counts),
            "configuration":{"heatmap_threshold":heatmap_threshold,"input_size_px":list(MODEL_SIZE),"background_samples_max":31,
                             "window_mode":"nonoverlap_plus_overlapping_tail_mean" if window_stride in (None, predictor.sequence_length) else "strided_windows_plus_tail_mean",
                             "window_stride":predictor.sequence_length if window_stride is None else window_stride,
                             "padding":False,"cpu_threads":threads},
            "elapsed_seconds":round(time.perf_counter()-timer,3),"proposal_segments":linker.next_id,"verified_shuttle_positions":0,
            "limitations":["Heatmap activation is uncalibrated; wrong-object and missed detections remain possible.",
                           "Scene-local median and sequence inputs use offline look-ahead, never cross gaps or cuts.",
                           "Player review is inherited view eligibility, not shuttle validation. No model inpainting or gap interpolation.",
                           "Modified inference adapter; results are not the authors' published evaluation protocol."],
            "shot_events":{"status":"not_available"},"coaching_metrics":{"status":"not_available"}}
    (output/"shuttle_candidates.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in records),encoding="utf-8")
    target=output/"shuttle_summary.json"
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (output/"index.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TrackNet shuttle review</title><style>body{font:17px/1.6 system-ui;background:#10202b;color:#edf5fa;max-width:1000px;margin:auto;padding:24px}video,img{width:100%}</style><h1>TrackNetV3 shuttle proposals</h1><p>Orange: heatmap observations. Cyan: linked proposals. All remain unverified. No inpainting, gap interpolation, shot events or coaching scores.</p><video controls src="candidate_preview.mp4"></video><img src="candidate_contact_sheet.jpg" alt="TrackNet review frames"><pre>'''+escape(json.dumps(result,indent=2))+'</pre></html>',encoding="utf-8")
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video",type=Path,required=True)
    parser.add_argument("--player-run",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--weights",type=Path,default=Path("out/models/TrackNetV3_TrackNet_best.pt"))
    parser.add_argument("--weights-sha256",default=CHECKPOINT_SHA256)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--window-stride",type=int,help="Experimental window step, 1..checkpoint sequence length; default is non-overlap")
    parser.add_argument("--heatmap-threshold",type=float,default=.5,help="Experimental activation cutoff strictly between 0 and 1; default 0.5")
    args=parser.parse_args()
    try:
        print(run(args.video,args.player_run,args.out,args.weights,args.weights_sha256,args.threads,args.window_stride,args.heatmap_threshold))
    except (ValueError,OSError,RuntimeError,KeyError,TypeError,pickle.UnpicklingError) as exc:
        parser.exit(1,f"TrackNet error: {exc}\n")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
