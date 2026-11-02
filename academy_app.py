"""Small Windows desktop front end for the offline worker; no cloud services."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import re
from pathlib import Path
import subprocess
import sys
import threading
import uuid
import webbrowser

from badminton_proto import probe_video, sha256_file

ROOT = Path(__file__).resolve().parent
PORTRAIT_DEMO_SHA256 = "2c08a15287f336a5a9711f7199b15bb03c8347a0bb5e1dfa3c1d79b61b793232"


def byte_range(header: str, size: int):
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not match or size <= 0 or not any(match.groups()):
        raise ValueError("Unsupported byte range")
    first, last = match.groups()
    if not first:
        suffix = int(last)
        if suffix <= 0:
            raise ValueError("Empty suffix range")
        return max(0, size-suffix), size-1
    start, end = int(first), min(size-1, int(last)) if last else size-1
    if start >= size or end < start:
        raise ValueError("Unsatisfiable byte range")
    return start, end


def worker_arguments(video: Path, output: Path, expected: int, processed: bool, portrait: bool):
    meta = probe_video(video)
    if expected not in (2, 4):
        raise ValueError("Choose singles (2) or doubles (4)")
    if portrait and sha256_file(video) != PORTRAIT_DEMO_SHA256:
        raise ValueError("The portrait demo settings match only the supplied reference clip. Choose Standard for another video.")
    command = [sys.executable, str(ROOT / "session_runner.py"), "run", "--video", str(video.resolve()),
               "--out", str(output.resolve()), "--expected-players", str(expected),
               "--sample-fps", str(min(15, meta["frame_rate_fps"]))]
    if processed or portrait:
        command.append("--processed-overlay")
    if portrait:
        command.extend(["--court-polygon", "0,0.64;0.55,0.535;1,0.535;1,1;0,1",
                        "--detect-roi", "0.00,0.38,1.00,1.00", "--secondary-detect-roi", "0.38,0.45,1.00,0.64"])
    return command


def resume_arguments(output: Path):
    job = json.loads((output / "job.json").read_text(encoding="utf-8"))
    if job.get("schema_version") != "1.0.0":
        raise ValueError("Unsupported saved job")
    binding, config = job["binding"], job["binding"]["configuration"]
    command = [sys.executable, str(ROOT / "session_runner.py"), "run", "--resume", "--out", str(output.resolve()),
               "--video", binding["source"]["path"], "--person-weights", binding["models"]["person"]["path"],
               "--shuttle-weights", binding["models"]["shuttle"]["path"]]
    for key, flag in (("camera","--camera"),("sample_fps","--sample-fps"),("expected_players","--expected-players"),
                      ("threads","--threads"),("max_seconds","--max-seconds"),("roi","--roi"),
                      ("detect_roi","--detect-roi"),("court_polygon","--court-polygon"),
                      ("secondary_detect_roi","--secondary-detect-roi")):
        if config.get(key) is not None:
            command.extend([flag,str(config[key])])
    if config["processed_overlay"]:
        command.append("--processed-overlay")
    return command


class LocalResults:
    """Read-only loopback server for a single output folder; no directory browsing."""
    def __init__(self):
        self.server = None
        self.root = None

    def url(self, directory: Path):
        directory = directory.resolve()
        if self.root == directory and self.server:
            return f"http://127.0.0.1:{self.server.server_port}/"
        self.close()
        root = directory
        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(root), **kwargs)
            def log_message(self, *args):
                pass
            def list_directory(self, path):
                self.send_error(404)
                return None
            def send_head(self):
                self._range = None
                resolved = Path(self.translate_path(self.path)).resolve()
                if not resolved.is_relative_to(root) or any(part.startswith('.') for part in resolved.relative_to(root).parts):
                    self.send_error(404)
                    return None
                requested = getattr(self, 'headers', {}).get('Range')
                if requested and resolved.is_file():
                    size = resolved.stat().st_size
                    try:
                        start, end = byte_range(requested, size)
                    except ValueError:
                        self.send_response(416)
                        self.send_header('Content-Range', f'bytes */{size}')
                        self.send_header('Content-Length', '0')
                        self.end_headers()
                        return None
                    stream = resolved.open('rb')
                    stream.seek(start)
                    self._range = (start, end)
                    self.send_response(206)
                    self.send_header('Content-Type', self.guess_type(str(resolved)))
                    self.send_header('Content-Length', str(end-start+1))
                    self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                    self.send_header('Accept-Ranges', 'bytes')
                    self.end_headers()
                    return stream
                return super().send_head()
            def copyfile(self, source, target):
                if self._range is None:
                    return super().copyfile(source, target)
                remaining = self._range[1]-self._range[0]+1
                while remaining:
                    block = source.read(min(65536, remaining))
                    if not block:
                        break
                    target.write(block)
                    remaining -= len(block)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.root = root
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.server.server_port}/"

    def close(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        self.server, self.root = None, None


class Desktop:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.window, self.tk, self.ttk = root, tk, ttk
        self.worker = None
        self.log = None
        self.job = None
        self.log_path = None
        self.pending = None
        self.cancel_requested = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.results = LocalResults()
        root.title("BadmintonAI — Offline academy analysis")
        root.geometry("820x620")
        root.minsize(650, 560)
        style = ttk.Style(root)
        style.configure("TButton", padding=8)
        panel = ttk.Frame(root, padding=24)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="Analyze an academy recording", font=("Segoe UI", 21)).pack(anchor="w")
        ttk.Label(panel, text="Local processing · experimental tracking · review before measurements", wraplength=740).pack(anchor="w", pady=(4, 20))
        self.video = tk.StringVar()
        row = ttk.Frame(panel); row.pack(fill="x")
        ttk.Entry(row, textvariable=self.video).pack(side="left", fill="x", expand=True)
        self.browse = ttk.Button(row, text="Choose video", command=self.choose)
        self.browse.pack(side="left", padx=(8, 0))
        ttk.Label(panel, text="Players").pack(anchor="w", pady=(16, 2))
        self.players = tk.StringVar(value="Doubles — 4 players")
        self.player_select = ttk.Combobox(panel, textvariable=self.players, values=("Singles — 2 players", "Doubles — 4 players"), state="readonly")
        self.player_select.pack(anchor="w")
        ttk.Label(panel, text="Capture settings").pack(anchor="w", pady=(12, 2))
        self.profile = tk.StringVar(value="Standard — choose region later")
        self.profile_select = ttk.Combobox(panel, textvariable=self.profile, values=("Standard — choose region later", "Supplied portrait demo — experimental"), state="readonly", width=50)
        self.profile_select.pack(anchor="w")
        self.processed = tk.BooleanVar(value=False)
        ttk.Checkbutton(panel, text="Video already contains analysis graphics", variable=self.processed).pack(anchor="w", pady=12)
        row = ttk.Frame(panel); row.pack(fill="x", pady=8)
        self.start_button = ttk.Button(row, text="Start analysis", command=self.start)
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(row, text="Cancel analysis", state="disabled", command=self.cancel)
        self.cancel_button.pack(side="left", padx=8)
        self.open_button = ttk.Button(row, text="Open results", state="disabled", command=self.open_results)
        self.open_button.pack(side="left")
        row = ttk.Frame(panel); row.pack(fill="x", pady=4)
        self.resume_button = ttk.Button(row, text="Resume a saved job", command=self.resume)
        self.resume_button.pack(side="left")
        self.identity_button = ttk.Button(row, text="Review player identities", state="disabled", command=self.open_identities)
        self.identity_button.pack(side="left", padx=8)
        row = ttk.Frame(panel); row.pack(fill="x", pady=4)
        self.apply_button = ttk.Button(row, text="Apply identity correction file", state="disabled", command=self.apply_identities)
        self.apply_button.pack(side="left")
        self.status = tk.StringVar(value="Choose a video to begin. Models must already be installed locally.")
        ttk.Label(panel, textvariable=self.status, wraplength=740, font=("Segoe UI", 11)).pack(anchor="w", pady=14)
        self.progress = ttk.Progressbar(panel, mode="indeterminate")
        self.progress.pack(fill="x")
        self.details = tk.Text(panel, height=7, wrap="word", state="disabled", font=("Segoe UI", 10))
        self.details.pack(fill="both", expand=True, pady=(12, 0))
        self.show_details("Results will include a preview and review files. Temporary tracker IDs are not athlete identities. Distance, shot types, physical shuttle speed and coaching remain unavailable until their prerequisites are verified.")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(500, self.poll)

    def choose(self):
        from tkinter import filedialog
        selected = filedialog.askopenfilename(title="Choose an academy recording", filetypes=[("Video recordings", "*.mp4 *.mov *.mkv *.avi *.webm"), ("All files", "*.*")])
        if selected:
            self.video.set(selected)

    def show_details(self, text):
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")

    def busy(self, active):
        self.start_button.configure(state="disabled" if active else "normal")
        self.browse.configure(state="disabled" if active else "normal")
        self.player_select.configure(state="disabled" if active else "readonly")
        self.profile_select.configure(state="disabled" if active else "readonly")
        self.resume_button.configure(state="disabled" if active else "normal")
        self.cancel_button.configure(state="normal" if active else "disabled")
        if active: self.progress.start(12)
        else: self.progress.stop()

    def start(self):
        video = Path(self.video.get().strip())
        if not video.is_file():
            self.status.set("Choose an existing video recording.")
            return
        suffix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.job = ROOT / "out" / "sessions" / suffix
        expected = 2 if self.players.get().startswith("Singles") else 4
        self.busy(True)
        self.open_button.configure(state="disabled")
        self.identity_button.configure(state="disabled")
        self.apply_button.configure(state="disabled")
        self.cancel_requested = False
        self.status.set("Checking the recording and selected settings…")
        self.pending = self.executor.submit(worker_arguments, video, self.job, expected,
                                            self.processed.get(), self.profile.get().startswith("Supplied"))

    def resume(self):
        from tkinter import filedialog
        selected = filedialog.askopenfilename(title="Choose a saved job.json", initialdir=str(ROOT / 'out'), filetypes=[("Saved job manifest", "job.json")])
        if selected:
            self.job = Path(selected).parent
            self.cancel_requested = False
            self.open_button.configure(state="disabled")
            self.identity_button.configure(state="disabled")
            self.apply_button.configure(state="disabled")
            self.busy(True)
            self.status.set("Checking saved job…")
            self.pending = self.executor.submit(resume_arguments, self.job)

    def cancel(self):
        self.cancel_requested = True
        if self.job and (self.job / "job.json").exists():
            (self.job / "CANCEL").touch()
        self.status.set("Cancellation requested. Waiting for the current stage to stop safely…")

    def poll(self):
        if self.pending and self.pending.done():
            pending, self.pending = self.pending, None
            try:
                command = pending.result()
                if self.cancel_requested:
                    self.busy(False)
                    self.status.set("Cancelled before analysis started.")
                else:
                    logs = ROOT / "out" / "launcher-logs"
                    logs.mkdir(parents=True, exist_ok=True)
                    self.log_path = logs / (self.job.name + ".log")
                    self.log = self.log_path.open("w", encoding="utf-8")
                    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
                    self.worker = subprocess.Popen(command, cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT, **options)
                    self.status.set("Starting local analysis…")
            except (ValueError, OSError, RuntimeError, KeyError) as exc:
                if self.log:
                    self.log.close(); self.log = None
                self.busy(False)
                self.status.set(str(exc))
        if self.worker:
            manifest = None
            if (self.job / "job.json").is_file():
                try:
                    manifest = json.loads((self.job / "job.json").read_text(encoding="utf-8"))
                    if self.cancel_requested:
                        (self.job / "CANCEL").touch()
                    else:
                        names = {"inspect":"Inspecting recording", "players":"Detecting and tracking players", "shuttle":"Analyzing eligible shuttle sequences", "render":"Rendering and checking the preview"}
                        self.status.set(names.get(manifest.get("active_stage"), "Publishing session results") + "…")
                except (OSError, ValueError):
                    pass
            code = self.worker.poll()
            if code is not None:
                self.worker = None
                self.log.close(); self.log = None
                self.busy(False)
                if code == 0 and manifest and manifest["status"] in ("needs_review", "completed"):
                    self.status.set("Analysis ready — tracking review is needed." if manifest["status"] == "needs_review" else "Processing finished — check measurement availability in results.")
                    self.open_button.configure(state="normal")
                    self.identity_button.configure(state="normal")
                    self.apply_button.configure(state="normal")
                    self.show_details(f"Results: {self.job}\n\nOpen results to play the preview and inspect scene decisions. No review has been granted automatically. Close this window after finishing with the results browser.")
                elif self.cancel_requested or (manifest and manifest.get("status") == "cancelled"):
                    self.status.set("Analysis cancelled. Completed stage files were preserved.")
                else:
                    self.status.set("Analysis could not finish. Details are shown below.")
                    self.show_details(self.log_path.read_text(encoding="utf-8", errors="replace")[-4000:])
        self.window.after(500, self.poll)

    def open_results(self):
        try:
            webbrowser.open(self.results.url(self.job))
        except OSError as exc:
            self.status.set(f"Could not open local results: {exc}. Results remain at {self.job}")

    def open_identities(self):
        from player_identity import prepare, source_snapshot
        try:
            binding = source_snapshot(self.job)[4]
            number = 1
            while True:
                folder = self.job / f"identity-editor-{number:03d}"
                if not folder.exists():
                    prepare(self.job, folder)
                    break
                try:
                    data = json.loads((folder / "identity_source.json").read_text(encoding="utf-8"))
                    if data["source_binding"] == binding and sha256_file(folder / "preview.mp4") == data["preview_sha256"]:
                        break
                except (ValueError, OSError, KeyError):
                    pass
                number += 1
            webbrowser.open(self.results.url(self.job) + folder.name + "/")
        except (ValueError, OSError, KeyError) as exc:
            self.status.set(f"Could not prepare identity review: {exc}")

    def apply_identities(self):
        from tkinter import filedialog
        from player_identity import apply, verify
        selected = filedialog.askopenfilename(title="Choose downloaded identity corrections", filetypes=[("Correction JSON", "*.json")])
        if not selected:
            return
        try:
            number = 1
            while (self.job / f"identity-result-{number:03d}").exists():
                number += 1
            folder = self.job / f"identity-result-{number:03d}"
            target = apply(self.job, Path(selected), folder)
            verify(self.job, folder)
            result = json.loads(target.read_text(encoding="utf-8"))
            self.status.set("Identity corrections validated. Scene review and coaching availability are unchanged.")
            self.show_details(f"Identity results: {target}\n\nAssigned {result['coverage']['assigned_observations']} observed positions. Reviewer type: {result['reviewer_kind']}. These are separate identity labels, not verified movement measurements.")
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.status.set(f"Identity corrections were not applied: {exc}")

    def close(self):
        if self.worker or self.pending:
            # Closing the launcher never leaves an orphaned analysis or kills it
            # halfway through publishing. Cancel and wait; no approval prompt.
            self.cancel()
            self.status.set("Stopping analysis safely. Close this window again after it stops.")
            return
        self.results.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.window.destroy()


def main():
    import tkinter as tk
    root = tk.Tk()
    Desktop(root)
    root.mainloop()


if __name__ == "__main__":
    main()
