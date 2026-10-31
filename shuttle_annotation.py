"""Offline, hash-bound shuttle label corrections; never approves trajectories."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from PIL import Image

from shuttle_benchmark import _validate


LABEL_FIELDS = ("visibility", "point_px", "view", "notes")
CHANGE_FIELDS = {"case_id", *LABEL_FIELDS}


def _read_packet(benchmark: Path):
    raw = benchmark.read_bytes()
    data = json.loads(raw)
    _validate(data)
    root = benchmark.resolve().parent
    images = []
    for case in data["cases"]:
        if not isinstance(case["id"], str) or not case["id"].strip():
            raise ValueError("Editor case IDs must be nonempty strings")
        path = (root / case["image"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Image is outside the benchmark packet")
        image_bytes = path.read_bytes()
        if sha256(image_bytes).hexdigest() != case["image_sha256"]:
            raise ValueError("Benchmark image has changed")
        with Image.open(BytesIO(image_bytes)) as image:
            if list(image.size) != data["image_size_px"] or image.format not in ("PNG", "JPEG"):
                raise ValueError("Invalid benchmark image dimensions or format")
            image.load()
            suffix = ".png" if image.format == "PNG" else ".jpg"
        images.append((image_bytes, suffix))
    return data, sha256(raw).hexdigest(), images


def _write_images(data: dict, images: list, output: Path):
    (output / "images").mkdir()
    for index, (case, (raw, suffix)) in enumerate(zip(data["cases"], images)):
        # IDs and source image names never become output paths.
        relative = f"images/case_{index:04d}{suffix}"
        (output / relative).write_bytes(raw)
        case["image"] = relative
        case["image_sha256"] = sha256(raw).hexdigest()


def prepare_editor(benchmark: Path, output: Path) -> Path:
    if output.exists():
        raise ValueError("Choose a new editor directory")
    data, binding, images = _read_packet(benchmark)
    output.mkdir(parents=True)
    _write_images(data, images, output)
    editor_data = {"schema_version": "0.1.0", "benchmark_sha256": binding,
                   "image_size_px": data["image_size_px"], "sources": data["sources"],
                   "cases": data["cases"]}
    # JSON is inert script data. Escape HTML delimiters even inside notes/IDs.
    payload = json.dumps(editor_data, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = Path(__file__).with_name("shuttle_annotation.html").read_text(encoding="utf-8")
    target = output / "index.html"
    target.write_text(template.replace("__ANNOTATION_DATA__", payload), encoding="utf-8")
    (output / "editor.js").write_bytes(Path(__file__).with_name("shuttle_annotation.js").read_bytes())
    (output / "editor_source.json").write_text(json.dumps({
        "schema_version": "0.1.0", "benchmark_sha256": binding,
        "images": [{"image": c["image"], "sha256": c["image_sha256"]} for c in data["cases"]],
        "status": "annotation_workspace_only", "verified_shuttle_positions": 0}, indent=2) + "\n", encoding="utf-8")
    return target


def apply_corrections(benchmark: Path, correction: Path, output: Path) -> Path:
    if output.exists():
        raise ValueError("Choose a new corrected packet directory")
    data, binding, images = _read_packet(benchmark)
    patch_bytes = correction.read_bytes()
    patch = json.loads(patch_bytes)
    if (set(patch) != {"schema_version", "benchmark_sha256", "reviewer", "reviewer_kind", "changes"}
            or patch["schema_version"] != "0.1.0" or patch["benchmark_sha256"] != binding):
        raise ValueError("Correction does not match this exact benchmark")
    reviewer, kind = patch["reviewer"], patch["reviewer_kind"]
    if not isinstance(reviewer, str) or not reviewer.strip() or kind not in ("human", "automated"):
        raise ValueError("Corrections require an explicit reviewer and reviewer kind")
    changes = patch["changes"]
    if not isinstance(changes, list) or not changes or len(changes) > len(data["cases"]):
        raise ValueError("Corrections require a nonempty list of frame changes")
    cases = {case["id"]: case for case in data["cases"]}
    seen = set()
    now = datetime.now(timezone.utc).isoformat()
    patch_hash = sha256(patch_bytes).hexdigest()
    for change in changes:
        if not isinstance(change, dict) or set(change) != CHANGE_FIELDS:
            raise ValueError("Only label fields can be corrected")
        identifier = change["case_id"]
        if not isinstance(identifier, str) or identifier not in cases or identifier in seen:
            raise ValueError("Unknown or duplicate corrected frame")
        if not isinstance(change["notes"], str):
            raise ValueError("Correction notes must be text")
        seen.add(identifier)
        case = cases[identifier]
        history = case.setdefault("label_history", [])
        if not isinstance(history, list):
            raise ValueError("Invalid existing label history")
        history.append({"previous_label": {k: deepcopy(case.get(k)) for k in (*LABEL_FIELDS, "reviewer", "reviewer_kind", "reviewed_at_utc")},
                        "parent_benchmark_sha256": binding, "correction_sha256": patch_hash,
                        "changed_at_utc": now})
        case.update({k: deepcopy(change[k]) for k in LABEL_FIELDS})
        case.update(reviewer=reviewer.strip(), reviewer_kind=kind, reviewed_at_utc=now)
    _validate(data)  # Includes location bounds, unknown labels and local negatives.
    revisions = data.setdefault("annotation_revisions", [])
    if not isinstance(revisions, list):
        raise ValueError("Invalid existing annotation revisions")
    revisions.append({"parent_benchmark_sha256": binding, "correction_sha256": patch_hash,
                      "reviewer": reviewer.strip(), "reviewer_kind": kind,
                      "case_ids": sorted(seen), "changed_at_utc": now})
    data["annotation_status"] = "development_labels_not_verified_trajectories"
    output.mkdir(parents=True)
    _write_images(data, images, output)
    target = output / "benchmark.json"
    target.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # A new editor is also the corrected packet's local review page.
    prepare_editor(target, output / "editor")
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Corrected shuttle labels</title><p>Development annotations; trajectories remain unverified.</p><a href="editor/index.html">Review corrected labels</a>', encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    editor = commands.add_parser("prepare")
    editor.add_argument("--benchmark", type=Path, required=True)
    editor.add_argument("--out", type=Path, required=True)
    apply = commands.add_parser("apply")
    apply.add_argument("--benchmark", type=Path, required=True)
    apply.add_argument("--corrections", type=Path, required=True)
    apply.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare_editor(args.benchmark, args.out) if args.command == "prepare" else apply_corrections(args.benchmark, args.corrections, args.out)
        print(result)
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
        parser.exit(1, f"Annotation error: {exc}\n")


if __name__ == "__main__":
    main()
