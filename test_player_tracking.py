import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from PIL import Image

from player_tracking import (
    _detections_to_records, parse_roi, person_detection_mask, predict_with_crop, read_calibration,
    select_person_detections, track_players,
)


class FakeDetections:
    """Minimal stand-in for supervision.Detections, for offline unit tests."""

    def __init__(self, xyxy, class_id, confidence, tracker_id=None, class_names=None):
        self.xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
        self.class_id = np.asarray(class_id, dtype=int)
        self.confidence = np.asarray(confidence, dtype=float)
        self.tracker_id = None if tracker_id is None else np.asarray(tracker_id, dtype=int)
        self.data = {} if class_names is None else {"class_name": np.asarray(class_names)}

    def __len__(self):
        return len(self.xyxy)

    def __getitem__(self, mask):
        return FakeDetections(
            self.xyxy[mask], self.class_id[mask], self.confidence[mask],
            None if self.tracker_id is None else self.tracker_id[mask],
            None if "class_name" not in self.data else self.data["class_name"][mask],
        )


class FakeDetector:
    def predict(self, _image, threshold):
        assert threshold == 0.25
        return FakeDetections(
            xyxy=[[100, 270, 180, 530], [30, 20, 70, 90], [210, 250, 260, 430]],
            class_id=[1, 1, 2],
            confidence=[0.91, 0.72, 0.88],
        )


class FakeTracker:
    def update(self, detections, timestamp):
        assert timestamp >= 0
        return FakeDetections(
            detections.xyxy, detections.class_id, detections.confidence,
            tracker_id=[7] * len(detections),
        )


class FakeEmptyTracker:
    def update(self, detections, timestamp):
        return FakeDetections([], [], [], tracker_id=[])


class PlayerTrackingTests(unittest.TestCase):
    def test_roi_validation(self):
        self.assertEqual(parse_roi("0.1,0.2,0.9,1.0"), (0.1, 0.2, 0.9, 1.0))
        for bad in ("0,0,1", "0,0,2,1", "0.9,0,0.1,1", "cat,0,1,1"):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                parse_roi(bad)

    def test_filter_people_in_court_region(self):
        detections = FakeDetector().predict(None, 0.25)
        kept = select_person_detections(detections, (360, 640), (0.05, 0.40, 0.95, 0.95), None)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept.class_id.tolist(), [1])
        self.assertEqual(kept.xyxy.tolist(), [[100, 270, 180, 530]])

    def test_class_name_takes_priority_over_numeric_class_id(self):
        detections = FakeDetections(
            [[10, 10, 20, 30], [50, 50, 80, 100]],
            class_id=[0, 1], confidence=[0.9, 0.8],
            class_names=["Person", "bicycle"],
        )
        self.assertEqual(person_detection_mask(detections, fallback_class_id=1).tolist(), [True, False])
        selected = select_person_detections(detections, (360, 640), None, None)
        self.assertEqual(selected.class_id.tolist(), [0])

    def test_records_include_id_and_footpoint(self):
        people = FakeDetections([[100, 250, 180, 530]], [0], [0.77], [5])
        result = _detections_to_records(people, None)
        self.assertEqual(result[0]["track_id"], 5)
        self.assertEqual(result[0]["foot_px"], [140.0, 530.0])
        self.assertNotIn("foot_court_xy_m", result[0])

    def test_detector_crop_restores_full_frame_coordinates(self):
        class CropDetector:
            def predict(self, image, threshold):
                self_seen_size.append(image.size)
                return FakeDetections([[20, 30, 60, 180]], [0], [0.8])

        self_seen_size = []
        detections = predict_with_crop(
            CropDetector(), Image.new("RGB", (360, 640)), 0.2,
            (0.1, 0.4, 0.9, 0.9),
        )
        self.assertEqual(self_seen_size, [(288, 320)])
        self.assertEqual(detections.xyxy.tolist(), [[56.0, 286.0, 96.0, 436.0]])

    def test_calibration_rejects_wrong_video_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "calibration.json"
            target.write_text(json.dumps({
                "camera_id": "court-01", "image_size": {"width_px": 720, "height_px": 1280},
                "pixel_to_court_homography": np.eye(3).tolist(),
            }))
            with self.assertRaisesRegex(ValueError, "dimensions"):
                read_calibration(target, "court-01", (360, 640))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools unavailable")
    def test_end_to_end_with_mock_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "synthetic.mp4"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=green:s=360x640:r=5:d=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
            ], check=True)
            out = root / "out"
            report = track_players(
                video, out, sample_fps=2, max_seconds=0.8,
                roi=(0.05, 0.4, 0.95, 0.95),
                detector=FakeDetector(), tracker=FakeTracker(),
            )
            data = json.loads(report.read_text())
            self.assertGreaterEqual(data["sampling"]["frames_processed"], 1)
            self.assertEqual(data["diagnostics"]["unique_track_ids_observed"], 1)
            totals = data["diagnostics"]["stage_totals"]
            self.assertEqual(totals["raw_person_detections"], 2 * data["sampling"]["frames_processed"])
            self.assertEqual(totals["court_filtered_person_detections"], data["sampling"]["frames_processed"])
            self.assertEqual(totals["boxes_with_track_id"], data["sampling"]["frames_processed"])
            self.assertEqual(data["diagnostics"]["class_id_histogram"]["1"], 2 * data["sampling"]["frames_processed"])
            self.assertEqual(data["model"]["person_class_id_fallback"], 1)
            self.assertTrue((out / "tracked_preview.mp4").is_file())
            self.assertTrue((out / "review_contact_sheet.jpg").is_file())
            self.assertTrue((out / "detector_review" / "review_contact_sheet.jpg").is_file())
            rows = (out / "player_tracks.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), data["sampling"]["frames_processed"])
            self.assertEqual(json.loads(rows[0])["persons"][0]["track_id"], 7)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools unavailable")
    def test_report_explains_when_tracker_hides_valid_detections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "synthetic.mp4"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=green:s=360x640:r=5:d=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
            ], check=True)
            out = root / "out"
            result = track_players(
                video, out, sample_fps=2, max_seconds=0.8,
                roi=(0.05, 0.4, 0.95, 0.95),
                detector=FakeDetector(), tracker=FakeEmptyTracker(),
            )
            summary = json.loads(result.read_text())
            totals = summary["diagnostics"]["stage_totals"]
            self.assertGreater(totals["court_filtered_person_detections"], 0)
            self.assertEqual(totals["tracker_output_boxes"], 0)
            self.assertTrue(any("Tracker discarded" in warning for warning in summary["warnings"]))
            self.assertTrue((out / "detector_review" / "review_contact_sheet.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
