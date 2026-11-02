"""Resumable offline session jobs. Processing never grants review approval."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version, PackageNotFoundError
from html import escape
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from badminton_proto import probe_video, sha256_file
from player_tracking import parse_roi
from tracking_review import prepare, verified_snapshot

ROOT = Path(__file__).resolve().parent
SCHEMA = "1.0.0"
STAGES = ("inspect", "players", "review", "shuttle", "render", "export")


class Cancelled(RuntimeError):
    pass


def atomic_json(path: Path, data: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(data) -> str:
    return sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


@contextmanager
def job_lock(output: Path):
    """OS releases the lock even after a killed worker; never guess PID liveness."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".job.lock").open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ValueError("This job is already running") from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ValueError("This job is already running") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def check_cancel(output: Path):
    if (output / "CANCEL").exists():
        raise Cancelled("Cancellation requested")


def invalidate_report(output: Path, reason: str):
    if (output / "result.json").exists():
        previous = json.loads((output / "result.json").read_text(encoding="utf-8"))
        previous.update(status="stale", invalidation_reason=reason)
        atomic_json(output / "result.json", previous)
    if (output / "index.html").exists():
        (output / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<title>Session refresh required</title><h1>Session results need refreshing</h1><p>'
            + escape(reason) + '</p><p><a href="job.json">Check current job status</a></p></html>', encoding="utf-8")


def package_versions():
    result = {}
    for name in ("rfdetr", "trackers", "torch", "numpy", "Pillow", "opencv-python"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def code_binding():
    # Conservative invalidation includes all first-party pipeline/UI sources.
    paths = sorted(ROOT.glob("*.py")) + sorted(ROOT.glob("session_*.html"))
    paths += sorted((ROOT / "third_party" / "tracknetv3").glob("*.py"))
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths if not p.name.startswith("test_")}


def stage_binding(binding: dict, name: str) -> dict:
    """UI-only changes must not force another expensive detection pass."""
    common = {"session_runner.py", "badminton_proto.py"}
    sources = {
        "inspect": common,
        "players": common | {"player_tracking.py", "player_regions.py"},
        "shuttle": common | {"player_tracking.py", "player_regions.py", "tracking_review.py", "shuttle_tracknet.py",
                             "shuttle_candidates.py", "shuttle_linking.py", "download_tracknet.py"},
        "render": common | {"session_results.py", "session_results.html"},
    }[name]
    code = {key: value for key, value in binding["code"].items()
            if key.replace('\\', '/') in sources or (name == "shuttle" and key.replace('\\', '/').startswith('third_party/'))}
    return {**binding, "code": code}


def artifact_binding(directory: Path) -> dict:
    return {str(p.relative_to(directory)): sha256_file(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name not in ("tracking_review.json", "tracking_review.html")}


def cached_stage(output: Path, stage: dict, fingerprint: str) -> Path | None:
    if stage.get("status") != "completed" or stage.get("fingerprint") != fingerprint:
        return None
    directory = (output / stage["directory"]).resolve()
    if not directory.is_relative_to(output.resolve()):
        raise ValueError("Stage directory escapes the job")
    if not directory.is_dir() or not stage.get("artifacts"):
        return None
    return directory if artifact_binding(directory) == stage["artifacts"] else None


def execute(command: list[str], directory: Path, output: Path, environment: dict):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "stage.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                check_cancel(output)
                time.sleep(.2)
            check_cancel(output)
            if process.returncode:
                raise RuntimeError(f"Stage exited with code {process.returncode}; see {directory / 'stage.log'}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def validate_configuration(config: dict):
    if (type(config["sample_fps"]) not in (int, float) or not math.isfinite(config["sample_fps"])
            or config["sample_fps"] <= 0):
        raise ValueError("Sample FPS must be positive and finite")
    if type(config["expected_players"]) is not int or not 1 <= config["expected_players"] <= 4:
        raise ValueError("Expected players must be 1..4")
    if type(config["threads"]) is not int or not 1 <= config["threads"] <= 32:
        raise ValueError("CPU threads must be 1..32")
    for key in ("roi", "detect_roi"):
        parse_roi(config[key])
    from player_regions import parse_polygon
    parse_polygon(config.get("court_polygon"))
    parse_roi(config.get("secondary_detect_roi"))
    maximum = config["max_seconds"]
    if maximum is not None and (type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum <= 0):
        raise ValueError("Maximum seconds must be positive and finite")


def run_job(video: Path, output: Path, config: dict, person_weights: Path,
            shuttle_weights: Path, resume: bool = False) -> dict:
    validate_configuration(config)
    video, output, person_weights, shuttle_weights = [p.resolve() for p in (video, output, person_weights, shuttle_weights)]
    if not video.is_file():
        raise ValueError("Source video is missing")
    # Existing model files only: this runner does not silently download weights.
    if not person_weights.is_file() or person_weights.name != "rf-detr-nano.pth":
        raise ValueError("Provide an existing rf-detr-nano.pth checkpoint")
    if video.is_relative_to(output):
        raise ValueError("Keep the source video outside the job output")
    if output.exists() and not resume:
        raise ValueError("Choose a new job output or pass --resume")
    with job_lock(output):
        path = output / "job.json"
        if resume and not path.exists():
            raise ValueError("No job manifest to resume")
        source = {"path": str(video), "sha256": sha256_file(video), "video": probe_video(video)}
        if config["sample_fps"] > source["video"]["frame_rate_fps"]:
            raise ValueError("Sample FPS exceeds source FPS")
        models = {"person": {"path": str(person_weights), "sha256": sha256_file(person_weights)},
                  "shuttle": {"path": str(shuttle_weights), "sha256": sha256_file(shuttle_weights) if shuttle_weights.is_file() else None}}
        binding = {"source": source, "configuration": config, "models": models,
                   "code": code_binding(), "packages": package_versions()}
        job = json.loads(path.read_text(encoding="utf-8")) if resume else {
            "schema_version": SCHEMA, "session_id": f"{config['camera']}-{source['sha256'][:12]}",
            "created_utc": datetime.now(timezone.utc).isoformat(), "stages": {name: {"status": "pending"} for name in STAGES}}
        if job["schema_version"] != SCHEMA:
            raise ValueError("Unsupported job schema")
        if resume and job["binding"]["source"]["sha256"] != source["sha256"]:
            raise ValueError("Cannot resume a job with a different source video")
        (output / "CANCEL").unlink(missing_ok=True)
        job.update(binding=binding, status="running", error=None, active_stage=None)

        def save():
            job["updated_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(path, job)

        def stage(name, dependencies, action):
            check_cancel(output)
            fingerprint = digest({"binding": stage_binding(binding, name), "dependencies": dependencies, "stage": name})
            current = job["stages"][name]
            cached = cached_stage(output, current, fingerprint)
            if cached is not None:
                print(f"Reusing verified {name} artifacts", flush=True)
                return cached
            attempt = 1
            while (output / f"{name}-{attempt:03d}").exists():
                attempt += 1
            directory = output / f"{name}-{attempt:03d}"
            directory.mkdir()
            job["active_stage"] = name
            job["stages"][name] = {"status": "running", "directory": directory.name, "fingerprint": fingerprint}
            save()
            print(f"Running {name}: {directory}", flush=True)
            started = time.perf_counter()
            action(directory)
            check_cancel(output)
            if sha256_file(video) != source["sha256"] or sha256_file(person_weights) != models["person"]["sha256"]:
                raise ValueError("Video or person checkpoint changed during processing")
            if models["shuttle"]["sha256"] and sha256_file(shuttle_weights) != models["shuttle"]["sha256"]:
                raise ValueError("Shuttle checkpoint changed during processing")
            if stage_binding({**binding, "code": code_binding()}, name)["code"] != stage_binding(binding, name)["code"]:
                raise ValueError("Pipeline source changed during processing; resume with current code")
            job["stages"][name].update(status="completed", artifacts=artifact_binding(directory),
                                       elapsed_seconds=round(time.perf_counter()-started, 3))
            save()
            return directory

        environment = dict(os.environ, RF_HOME=str(person_weights.parent), HF_HUB_OFFLINE="1",
                           TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS=str(config["threads"]), MKL_NUM_THREADS=str(config["threads"]))
        def command(script, args, directory):
            execute([sys.executable, str(ROOT / script), *map(str, args)], directory, output, environment)

        save()
        invalidate_report(output, "Analysis is being refreshed; previous results are not current.")
        try:
            inspected = stage("inspect", {}, lambda d: command("badminton_proto.py", ["inspect", "--video", video,
                "--out", d, "--camera", config["camera"], *(["--processed-overlay"] if config["processed_overlay"] else [])], d))
            arguments = ["--video", video, "--camera", config["camera"], "--sample-fps", config["sample_fps"],
                         "--confidence", .3, "--track-high-confidence", .3, "--track-activation", .4]
            for key, flag in (("roi", "--roi"), ("detect_roi", "--detect-roi"), ("max_seconds", "--max-seconds")):
                if config[key] is not None:
                    arguments.extend([flag, config[key]])
            for key, flag in (("court_polygon", "--court-polygon"), ("secondary_detect_roi", "--secondary-detect-roi")):
                if config.get(key) is not None:
                    arguments.extend([flag, config[key]])
            players = stage("players", {}, lambda d: command("player_tracking.py", [*arguments, "--out", d], d))
            # Preparing an existing packet preserves decisions and checks source bindings.
            prepare(players, config["expected_players"])
            verification, summary, rows, review = verified_snapshot(players)
            review_binding = {"verification": verification, "review_sha256": sha256_file(players / "tracking_review.json")}
            job["stages"]["review"] = {"status": "completed" if verification["all_scenes_decided"] else "needs_review",
                                        "directory": players.name, **review_binding}
            shuttle = None
            if verification["approved_intervals"]:
                if not shuttle_weights.is_file():
                    raise ValueError("Reviewed intervals are ready but the shuttle checkpoint is missing")
                shuttle = stage("shuttle", review_binding, lambda d: command("shuttle_tracknet.py", ["--video", video,
                    "--player-run", players, "--out", d / "data", "--weights", shuttle_weights,
                    "--weights-sha256", models["shuttle"]["sha256"], "--threads", config["threads"]], d)) / "data"
            else:
                # Invalidate old downstream eligibility immediately on review revocation.
                job["stages"]["shuttle"] = {"status": "needs_review" if not verification["all_scenes_decided"] else "not_available",
                    "reason": "No currently accepted player intervals; no review is granted automatically"}
            save()
            from session_results import render_preview, write_results
            dependencies = {"players": job["stages"]["players"]["artifacts"], "review": review_binding,
                            "shuttle": job["stages"]["shuttle"]}
            rendered = stage("render", dependencies, lambda d: render_preview(video, d, summary, rows, shuttle,
                lambda: check_cancel(output)))
            # Review can change while rendering: never publish a stale capability claim.
            latest_verification = verified_snapshot(players)[0]
            if sha256_file(players / "tracking_review.json") != review_binding["review_sha256"]:
                raise ValueError("Review changed during rendering; resume to refresh downstream artifacts")
            result = write_results(output, source, config, players, inspected, rendered, shuttle, latest_verification)
            if sha256_file(players / "tracking_review.json") != review_binding["review_sha256"]:
                raise ValueError("Review changed while publishing; resume to refresh downstream artifacts")
            job["stages"]["export"] = {"status": "completed", "artifacts": {
                name: sha256_file(output / name) for name in ("result.json", "index.html")}}
            job.update(status="needs_review" if not verification["all_scenes_decided"] else "completed",
                       active_stage=None, result="result.json", preview=result["artifacts"]["preview"])
            save()
            return job
        except (Exception, KeyboardInterrupt) as exc:
            state = "cancelled" if isinstance(exc, (Cancelled, KeyboardInterrupt)) else "failed"
            active = job.get("active_stage")
            if active:
                job["stages"][active]["status"] = state
            job.update(status=state, error=str(exc) or type(exc).__name__, active_stage=None)
            # A previous successful report must not look current after failed refresh.
            invalidate_report(output, job["error"])
            save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "cancel", "status"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--camera", default="academy-court-01")
    parser.add_argument("--sample-fps", type=float, default=15)
    parser.add_argument("--expected-players", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--roi")
    parser.add_argument("--detect-roi")
    parser.add_argument("--court-polygon")
    parser.add_argument("--secondary-detect-roi")
    parser.add_argument("--processed-overlay", action="store_true")
    cache = os.environ.get("RF_HOME") or os.environ.get("ROBOFLOW_HOME") or str(Path.home() / ".roboflow" / "models")
    parser.add_argument("--person-weights", type=Path, default=Path(cache) / "rf-detr-nano.pth")
    parser.add_argument("--shuttle-weights", type=Path, default=ROOT / "out/models/TrackNetV3_TrackNet_best.pt")
    args = parser.parse_args()
    try:
        if args.command == "status":
            print((args.out / "job.json").read_text(encoding="utf-8"))
        elif args.command == "cancel":
            if not (args.out / "job.json").is_file():
                raise ValueError("No job manifest to cancel")
            (args.out / "CANCEL").touch()
            print("Cancellation requested")
        else:
            if args.video is None:
                raise ValueError("--video is required for run")
            keys = ("camera", "sample_fps", "expected_players", "threads", "max_seconds", "roi", "detect_roi", "processed_overlay", "court_polygon", "secondary_detect_roi")
            job = run_job(args.video, args.out, {key: getattr(args, key) for key in keys},
                          args.person_weights, args.shuttle_weights, args.resume)
            print(f"Job {job['status']}: {(args.out / 'index.html').resolve()}")
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"Job error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
