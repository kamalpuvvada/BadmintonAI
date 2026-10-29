import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from player_tracking import (
    _detections_to_records, _extract_sampled_video, parse_roi, person_detection_mask, predict_with_crop, read_calibration,
    predict_with_secondary_crop, select_person_detections, suppress_duplicate_people, track_players,
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

    @classmethod
    def merge(cls, groups):
        result = cls(np.concatenate([g.xyxy for g in groups]),
                   np.concatenate([g.class_id for g in groups]),
                   np.concatenate([g.confidence for g in groups]))
        result.data = {key: np.concatenate([g.data[key] for g in groups]) for key in groups[0].data}
        return result

    def __getitem__(self, mask):
        result = FakeDetections(
            self.xyxy[mask], self.class_id[mask], self.confidence[mask],
            None if self.tracker_id is None else self.tracker_id[mask],
            None if "class_name" not in self.data else self.data["class_name"][mask],
        )
        result.data = {key: value[mask] for key, value in self.data.items()}
        return result


class FakeDetector:
    def predict(self, _image, threshold):
        assert threshold == 0.25
        return FakeDetections(
            xyxy=[[100, 270, 180, 530], [30, 20, 70, 90], [210, 250, 260, 430]],
            class_id=[1, 1, 2],
            confidence=[0.91, 0.72, 0.88],
        )


class FakeTracker:
    def reset(self):
        pass

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
    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg unavailable")
    def test_matching_view_edit_resets_tracks_without_rejecting_court_frames(self):
        from PIL import ImageDraw
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "source.mp4"
            video.write_bytes(b"test source")
            frames = []
            for index, x in enumerate((30, 30, 200)):
                image = Image.new("RGB", (360, 640), "green")
                ImageDraw.Draw(image).rectangle((x, 100, x + 50, 250), fill="white")
                path = root / f"source-{index}.jpg"
                image.save(path)
                frames.append(path)
            meta = {"width_px": 360, "height_px": 640, "frame_rate_fps": 1, "duration_seconds": 3}
            for threshold in (0, .01):
                tracker = FakeTracker()
                tracker.reset = Mock()
                with patch("player_tracking.probe_video", return_value=meta), patch("player_tracking._extract_sampled_video", return_value=frames):
                    report = track_players(video, root / str(threshold), sample_fps=1,
                        detector=FakeDetector(), tracker=tracker, court_reference_path=frames[0],
                        court_cut_threshold=threshold, roi=(.05, .4, .95, .95))
                rows = [json.loads(line) for line in (report.parent / "player_tracks.jsonl").read_text().splitlines()]
                summary = json.loads(report.read_text())
                self.assertTrue(all(row["court_view_status"] == "reference_match" for row in rows))
                self.assertTrue(all(len(row["persons"]) == 1 for row in rows))
                self.assertEqual(tracker.reset.call_count, int(bool(threshold)))
                if threshold:
                    self.assertTrue(rows[2]["scene_cut"])
                    self.assertNotEqual(rows[1]["persons"][0]["track_id"], rows[2]["persons"][0]["track_id"])
                    self.assertEqual(summary["scene_review"]["candidate_cuts"][0]["reason"], "matching_view_jump_cut")
                else:
                    self.assertFalse(summary["scene_review"]["candidate_cuts"])

    def test_matching_view_cut_threshold_validation(self):
        for value in (-1, 2, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "court-cut-threshold"):
                track_players(Path("unused"), Path("unused"), court_cut_threshold=value)
        with self.assertRaisesRegex(ValueError, "requires --court-reference"):
            track_players(Path("unused"), Path("unused"), court_cut_threshold=.01)

    def test_crop_edge_fragment_suppression_requires_coverage_and_higher_score(self):
        detections = FakeDetections([[10, 10, 40, 100], [12, 72, 42, 104], [60, 72, 90, 104]],
            [1, 1, 1], [.8, .4, .6])
        detections.data["crop_edge_clipped"] = np.array([False, True, True])
        kept = suppress_duplicate_people(detections, .4)
        self.assertEqual(kept.xyxy.tolist(), [[10, 10, 40, 100], [60, 72, 90, 104]])
        detections.confidence[1] = .85
        kept = suppress_duplicate_people(detections, .4)
        self.assertEqual(kept.xyxy.tolist(), [[10, 10, 40, 100], [60, 72, 90, 104]])
        # Without crop-edge evidence, retain the lower-overlap box for review.
        detections.data["crop_edge_clipped"][:] = False
        self.assertEqual(len(suppress_duplicate_people(detections, .4)), 3)

    def test_secondary_crop_with_real_supervision_source_metadata(self):
        try:
            import supervision as sv
        except ImportError:
            self.skipTest("Optional supervision package is unavailable")
        class Detector:
            def predict(self, image, threshold):
                result = sv.Detections(xyxy=np.array([[10., 10., 30., 50.]]),
                    class_id=np.array([1]), confidence=np.array([.9]))
                result.metadata["source_image"] = np.asarray(image)
                result.data["source_shape"] = np.array([[image.height, image.width]])
                result.data["class_name"] = np.array(["person"])
                return result
        image = Image.new("RGB", (100, 80), "green")
        merged, count = predict_with_secondary_crop(Detector(), image, .25,
            (0, .25, 1, 1), (0, 0, 1, 1), .35)
        self.assertEqual(len(merged), 2)
        self.assertEqual(count, 1)
        self.assertEqual(merged.metadata["source_image"].shape, (80, 100, 3))
        self.assertEqual(merged.data["source_shape"].tolist(), [[80, 100], [80, 100]])
        self.assertEqual(merged.data["class_name"].tolist(), ["person", "person"])

    def test_secondary_crop_restores_coordinates_and_merges_labels_before_nms(self):
        calls = []
        class Detector:
            def predict(self, image, threshold):
                calls.append((image.size, threshold))
                box = [10, 10, 30, 50] if threshold == .25 else [10, 30, 30, 70]
                return FakeDetections([box], [1], [.9 if threshold == .25 else .8], class_names=["person"])
        merged, count = predict_with_secondary_crop(Detector(), Image.new("RGB", (100, 100)),
            .25, (0, .2, 1, 1), (0, 0, 1, 1), .35)
        self.assertEqual(calls, [((100, 80), .25), ((100, 100), .35)])
        self.assertEqual(count, 1)
        self.assertEqual(merged.xyxy.tolist(), [[10, 30, 30, 70], [10, 30, 30, 70]])
        self.assertEqual(merged.data["class_name"].tolist(), ["person", "person"])
        self.assertEqual(len(suppress_duplicate_people(merged, .4)), 1)

    def test_person_nms_removes_duplicates_preserving_other_people_and_labels(self):
        detections = FakeDetections(
            [[10, 10, 50, 100], [11, 11, 51, 101], [40, 10, 80, 100]],
            [1, 1, 1], [.9, .7, .8], class_names=["person"] * 3)
        kept = suppress_duplicate_people(detections, .5)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept.confidence.tolist(), [.9, .8])
        self.assertEqual(kept.data["class_name"].tolist(), ["person", "person"])
        self.assertEqual(len(suppress_duplicate_people(detections, 0)), 3)
        self.assertEqual(len(suppress_duplicate_people(FakeDetections([], [], []), .5)), 0)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg unavailable")
    def test_reference_gate_rejects_closeup_even_when_cut_detection_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "source.mp4"
            video.write_bytes(b"test source")
            frames = []
            for index, color in enumerate(("red", "blue", "red")):
                path = root / f"source-{index}.jpg"
                Image.new("RGB", (360, 640), color).save(path)
                frames.append(path)
            tracker = FakeTracker()
            tracker.reset = Mock()
            meta = {"width_px": 360, "height_px": 640, "frame_rate_fps": 1, "duration_seconds": 3}
            with patch("player_tracking.probe_video", return_value=meta), patch("player_tracking._extract_sampled_video", return_value=frames):
                report = track_players(video, root / "output", sample_fps=1, detector=FakeDetector(),
                    tracker=tracker, scene_cut_threshold=0, court_reference_path=frames[0],
                    roi=(.05, .4, .95, .95))
            rows = [json.loads(line) for line in (report.parent / "player_tracks.jsonl").read_text().splitlines()]
            self.assertEqual(rows[1]["persons"], [])
            self.assertEqual(rows[1]["court_view_status"], "reference_mismatch")
            self.assertEqual(rows[1]["stage_counts"]["raw_person_detections"], 2)
            self.assertEqual(rows[1]["stage_counts"]["view_accepted_person_detections"], 0)
            self.assertEqual(rows[2]["court_view_status"], "reference_match")
            self.assertNotEqual(rows[0]["persons"][0]["track_id"], rows[2]["persons"][0]["track_id"])
            self.assertEqual(tracker.reset.call_count, 2)
            summary = json.loads(report.read_text())
            self.assertEqual(summary["court_view_review"]["rejected_frames"], 1)
            with patch("player_tracking.probe_video", return_value=meta), patch("player_tracking._extract_sampled_video", return_value=[frames[1], frames[0], frames[1]]):
                report = track_players(video, root / "starts-in-closeup", sample_fps=1,
                    detector=FakeDetector(), tracker=FakeTracker(), scene_cut_threshold=0,
                    court_reference_path=frames[0], roi=(.05, .4, .95, .95))
            rows = [json.loads(line) for line in (report.parent / "player_tracks.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["persons"], [])
            self.assertEqual(rows[0]["court_view_status"], "reference_mismatch")
            self.assertTrue(rows[1]["persons"])
            self.assertEqual(rows[2]["persons"], [])

    def test_reference_dimensions_and_filter_threshold_validation(self):
        for name in ("person_nms_iou", "court_view_threshold"):
            for value in (-1, 2, float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    track_players(Path("unused"), Path("unused"), **{name: value})
        for value in (0, 1, float("nan"), float("inf")):
            with self.subTest(secondary_confidence=value), self.assertRaisesRegex(ValueError, "secondary-confidence"):
                track_players(Path("unused"), Path("unused"), secondary_confidence=value)
        with self.assertRaisesRegex(ValueError, "requires person NMS"):
            track_players(Path("unused"), Path("unused"), person_nms_iou=0, secondary_detect_roi=(0, 0, 1, 1))
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.jpg"
            Image.new("RGB", (64, 64)).save(reference)
            meta = {"width_px": 360, "height_px": 640, "frame_rate_fps": 30, "duration_seconds": 10}
            with patch("player_tracking.probe_video", return_value=meta), self.assertRaisesRegex(ValueError, "reference dimensions"):
                track_players(Path("unused"), Path(directory) / "out", court_reference_path=reference)

    def test_scene_threshold_validation(self):
        for threshold in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, "scene-cut-threshold"):
                track_players(Path("unused.mp4"), Path("unused-output"), scene_cut_threshold=threshold)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools unavailable")
    def test_cut_resets_identity_trails_and_invalidates_calibration(self):
        from player_tracking import annotate_frame
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "cut.mp4"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=red:s=360x640:r=5:d=1",
                "-f", "lavfi", "-i", "color=c=blue:s=360x640:r=5:d=1",
                "-f", "lavfi", "-i", "color=c=red:s=360x640:r=5:d=1",
                "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
            ], check=True)
            calibration = root / "calibration.json"
            calibration.write_text(json.dumps({"camera_id": "court-01",
                "image_size": {"width_px": 360, "height_px": 640},
                "pixel_to_court_homography": np.diag([.01, .01, 1]).tolist()}))
            for threshold in (.15, 0):
                tracker = FakeTracker()
                tracker.reset = Mock()
                trail_sizes = []
                def observe(image, records, trails):
                    trail_sizes.append(sum(len(t) for t in trails.values()))
                    return annotate_frame(image, records, trails)
                out = root / str(threshold)
                with patch("player_tracking.annotate_frame", side_effect=observe):
                    report = track_players(video, out, sample_fps=5,
                        roi=(.05, .4, .95, .95), calibration_path=calibration,
                        detector=FakeDetector(), tracker=tracker, scene_cut_threshold=threshold)
                summary = json.loads(report.read_text())
                rows = [json.loads(line) for line in (out / "player_tracks.jsonl").read_text().splitlines()]
                before, after = rows[4], rows[5]
                if threshold:
                    self.assertEqual(tracker.reset.call_count, 2)
                    self.assertEqual(after["scene_id"], 1)
                    self.assertTrue(after["scene_cut"])
                    self.assertNotEqual(before["persons"][0]["track_id"], after["persons"][0]["track_id"])
                    self.assertIn("foot_court_xy_m", before["persons"][0])
                    self.assertNotIn("foot_court_xy_m", after["persons"][0])
                    self.assertEqual(after["court_view_status"], "invalidated_after_cut")
                    self.assertEqual(after["court_geometry_status"], "invalidated_after_boundary")
                    self.assertEqual(trail_sizes[5], 0)
                    self.assertEqual(summary["scene_review"]["candidate_cuts"][0]["timestamp_seconds"], 1)
                    self.assertEqual(summary["scene_review"]["scenes_observed"], 3)
                    self.assertEqual(len({rows[i]["persons"][0]["track_id"] for i in (0, 5, 10)}), 3)
                    self.assertNotIn("foot_court_xy_m", rows[10]["persons"][0])
                else:
                    tracker.reset.assert_not_called()
                    self.assertEqual(after["scene_id"], 0)
                    self.assertIn("foot_court_xy_m", after["persons"][0])
                    self.assertGreater(trail_sizes[5], 0)
                    self.assertFalse(summary["scene_review"]["candidate_cuts"])

    def test_segment_rejects_invalid_time_bounds_before_inference(self):
        with patch("player_tracking.probe_video", return_value={"frame_rate_fps": 30, "duration_seconds": 10}):
            for start in (-1, 10, 11, float("nan"), float("inf")):
                with self.subTest(start=start), self.assertRaisesRegex(ValueError, "start-seconds"):
                    track_players(Path("unused.mp4"), Path("unused-output"), start_seconds=start)
            for duration in (0, -1, float("nan"), float("inf")):
                with self.subTest(duration=duration), self.assertRaisesRegex(ValueError, "max-seconds"):
                    track_players(Path("unused.mp4"), Path("unused-output"), max_seconds=duration)

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
    def test_segment_seek_extracts_requested_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "two-colors.mp4"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=red:s=64x64:r=10:d=1",
                "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=10:d=1",
                "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
            ], check=True)
            frames = _extract_sampled_video(video, root / "frames", 5, 0.4, start_seconds=1.2)
            self.assertEqual(len(frames), 2)
            for path in frames:
                with Image.open(path) as image:
                    red, green, blue = image.getpixel((32, 32))
                    self.assertGreater(blue, red + green)

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
                video, out, sample_fps=2, max_seconds=0.6, start_seconds=0.4,
                roi=(0.05, 0.4, 0.95, 0.95),
                detector=FakeDetector(), tracker=FakeTracker(),
            )
            data = json.loads(report.read_text())
            self.assertEqual(data["sampling"]["start_seconds"], 0.4)
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
            for i, row in enumerate(rows):
                self.assertAlmostEqual(json.loads(row)["timestamp_seconds"], 0.4 + i / 2)
            self.assertEqual(data["artifacts"]["review_frames"][0]["timestamp_seconds"], 0.4)

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
