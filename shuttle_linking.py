"""Experimental confidence/motion linking of observed TrackNet candidates only."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path

from shuttle_candidates import ProposalLinker, frame_context
from tracking_review import verified_snapshot


LINKER_MODES = ("motion", "confidence_motion")


class ConfidenceMotionLinker(ProposalLinker):
    """Resolve a dominant observed activation, still enforcing motion/gap gates.

    Activations are uncalibrated. Equal or weak peaks retain the original motion
    ambiguity rule. No synthetic positions, future frames or gap bridging.
    """
    peak_min = .65
    peak_margin = .20

    def update(self, frame: int, scene: int, candidates: list[dict]) -> dict | None:
        for candidate in candidates:
            peak = candidate.get("heatmap_peak")
            if type(peak) not in (int, float) or not math.isfinite(peak) or not 0 <= peak <= 1:
                raise ValueError("Confidence linking requires valid observed heatmap peaks")
        indices = list(range(len(candidates)))
        dominant = False
        if len(candidates) > 1:
            ranked = sorted(indices, key=lambda i: candidates[i]["heatmap_peak"], reverse=True)
            best, second = (candidates[i]["heatmap_peak"] for i in ranked[:2])
            if best >= self.peak_min and best - second >= self.peak_margin - 1e-12:
                indices = ranked[:1]
                dominant = True
        proposal = super().update(frame, scene, [candidates[i] for i in indices])
        if proposal is not None:
            proposal["candidate_index"] = indices[proposal["candidate_index"]]
            proposal["selection_method"] = "dominant_activation_with_motion_gate" if dominant else "motion"
        return proposal


def make_linker(mode: str):
    if mode not in LINKER_MODES:
        raise ValueError("Unknown shuttle linker mode")
    return ConfidenceMotionLinker() if mode == "confidence_motion" else ProposalLinker()


def linker_configuration(mode: str) -> dict:
    linker = make_linker(mode)
    return {"linker_mode": mode, "link_step_max_px": linker.max_step_px,
            "link_ambiguity_margin_px": linker.margin,
            "dominant_peak_min": ConfidenceMotionLinker.peak_min if mode == "confidence_motion" else None,
            "dominant_peak_margin": ConfidenceMotionLinker.peak_margin if mode == "confidence_motion" else None}


def relink(source_run: Path, player_run: Path, output: Path, mode: str) -> Path:
    """Replay existing observations against the current exact player review."""
    if output.exists():
        raise ValueError("Choose a new relinking output directory")
    linker = make_linker(mode)
    summary_path, rows_path = source_run/"shuttle_summary.json", source_run/"shuttle_candidates.jsonl"
    raw_summary, raw_rows = summary_path.read_bytes(), rows_path.read_bytes()
    summary = json.loads(raw_summary)
    rows = [json.loads(line) for line in raw_rows.decode("utf-8").splitlines() if line.strip()]
    video_hash = summary.get("source_video_sha256")
    if (summary.get("method") != "tracknetv3_concat_scene_bounded_heatmaps"
            or summary.get("status") != "experimental_unverified_candidates"
            or not isinstance(video_hash,str) or len(video_hash)!=64
            or any(c not in "0123456789abcdef" for c in video_hash)
            or summary.get("verified_shuttle_positions") != 0):
        raise ValueError("Only unverified TrackNet observation runs can be relinked")
    verification, players, player_rows, review = verified_snapshot(player_run)
    review_hash = sha256(json.dumps(review,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if (summary["player_source_binding"] != verification["source_binding"]
            or summary["player_review_snapshot_sha256"] != review_hash
            or not players["session_id"].endswith(summary["source_video_sha256"][:12])
            or summary["player_review_provenance"] != verification["review_provenance"]
            or any(summary["video"][k] != players["video"][k] for k in ("width_px","height_px"))):
        raise ValueError("Relinking source does not match the current player review")
    sampling = summary["sampling"]
    fps, start = sampling["fps"], sampling["start_seconds"]
    player_start = players["sampling"].get("start_seconds",0)
    duration = len(player_rows)/players["sampling"]["fps"]
    if (not math.isfinite(fps) or not 0 < fps <= 120 or not math.isfinite(start)
            or fps != summary["video"]["frame_rate_fps"]
            or start != player_start or type(sampling["frames"]) is not int
            or len(rows) != sampling["frames"] or not rows
            or abs(len(rows)/fps-duration) > 1/fps):
        raise ValueError("Invalid relinking sampling")
    output_rows, counts = [], Counter()
    for index, original in enumerate(rows):
        timestamp = start+index/fps
        if (type(original["frame_index"]) is not int or original["frame_index"] != index
                or not math.isfinite(original["timestamp_seconds"])
                or abs(original["timestamp_seconds"]-timestamp) > .00001
                or original.get("verified_shuttle_position_px") is not None):
            raise ValueError("Invalid relinking frame or verification state")
        context = frame_context(timestamp,players,player_rows,verification["approved_intervals"])
        if original["scene_id"] != context["scene_id"]:
            raise ValueError("Relinking scene does not match current review")
        candidates = original["candidates"]
        if not isinstance(candidates,list):
            raise ValueError("Invalid relinking candidate list")
        for candidate in candidates:
            point = candidate["center_px"]
            if (not isinstance(point,list) or len(point)!=2
                    or any(type(v) not in (int,float) or not math.isfinite(v) for v in point)
                    or not 0 <= point[0] < summary["video"]["width_px"]
                    or not 0 <= point[1] < summary["video"]["height_px"]):
                raise ValueError("Invalid relinking candidate location")
            peak = candidate.get("heatmap_peak")
            if type(peak) not in (int,float) or not math.isfinite(peak) or not 0 <= peak <= 1:
                raise ValueError("Invalid relinking heatmap peak")
        original_proposal = original["proposal"]
        if original_proposal is not None:
            selected = original_proposal["candidate_index"]
            if (original["status"] != "proposal" or type(selected) is not int
                    or not 0 <= selected < len(candidates)
                    or original_proposal["position_px"] != candidates[selected]["center_px"]
                    or original_proposal.get("verification_status") != "unverified"):
                raise ValueError("Invalid original proposal reference")
        elif original["status"] == "proposal":
            raise ValueError("Original proposal status requires an observation reference")
        if ((original["status"] == "no_heatmap_detection" and candidates)
                or (original["status"] == "ambiguous_candidates" and not candidates)):
            raise ValueError("Original status does not match observations")
        proposal = None
        if not context["eligible"]:
            if original["status"] != "excluded_player_review" or candidates or original["proposal"] is not None:
                raise ValueError("Excluded frames must not contain observations or proposals")
            status = "excluded_player_review"
        elif original["status"] == "insufficient_sequence_context":
            if candidates or original["proposal"] is not None:
                raise ValueError("Insufficient context must not contain observations")
            status = "insufficient_sequence_context"
        else:
            if original["status"] not in ("proposal","no_heatmap_detection","ambiguous_candidates"):
                raise ValueError("Invalid eligible observation status")
            proposal = linker.update(index,context["scene_id"],candidates)
            status = "proposal" if proposal else ("no_heatmap_detection" if not candidates else "ambiguous_candidates")
        if proposal is None:
            linker.reset()
        row = deepcopy(original)
        row.update(status=status,proposal=proposal,verified_shuttle_position_px=None)
        output_rows.append(row);counts[status]+=1
    # Recheck bindings before writing: revocation or concurrent changes invalidate replay.
    current, _, _, current_review = verified_snapshot(player_run)
    if (current["source_binding"] != verification["source_binding"] or current_review != review
            or summary_path.read_bytes() != raw_summary or rows_path.read_bytes() != raw_rows):
        raise ValueError("Relinking inputs changed while processing")
    result = deepcopy(summary)
    result["configuration"].update(linker_configuration(mode))
    result.update(frame_status_counts=dict(counts),proposal_segments=linker.next_id,
                  verified_shuttle_positions=0,shot_events={"status":"not_available"},
                  coaching_metrics={"status":"not_available"},
                  relink_source_binding={"summary_sha256":sha256(raw_summary).hexdigest(),
                                         "rows_sha256":sha256(raw_rows).hexdigest()},
                  relink_operation="observations_reused_no_new_inference")
    result.pop("elapsed_seconds",None)  # Source inference time is not replay time.
    result["limitations"].append("Experimental relinking reuses observed candidates; activations are not calibrated probabilities. No new inference or trajectory approval.")
    rows_text = "".join(json.dumps(r,allow_nan=False)+"\n" for r in output_rows)
    summary_text = json.dumps(result,indent=2,allow_nan=False)+"\n"
    output.mkdir(parents=True)
    (output/"shuttle_candidates.jsonl").write_text(rows_text,encoding="utf-8")
    target=output/"shuttle_summary.json"
    target.write_text(summary_text,encoding="utf-8")
    (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Shuttle relinking diagnostics</title><h1>Experimental relinking</h1><p>Existing observed candidates only. No preview was copied from the old linking run. No verified trajectories or coaching scores.</p><pre>'+escape(json.dumps(result,indent=2))+'</pre>',encoding="utf-8")
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run",type=Path,required=True)
    parser.add_argument("--player-run",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--mode",choices=LINKER_MODES,required=True)
    args=parser.parse_args()
    try:
        print(relink(args.source_run,args.player_run,args.out,args.mode))
    except (ValueError,OSError,KeyError,TypeError,AttributeError) as exc:
        parser.exit(1,f"Relinking error: {exc}\n")


if __name__=="__main__":
    main()
