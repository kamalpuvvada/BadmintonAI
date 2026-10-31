from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from badminton_proto import sha256_file
from player_tracking import _encode_preview
from shuttle_benchmark import evaluate, prepare


class ShuttleBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.packet = self.root / "packet"
        self.packet.mkdir()
        image = self.packet / "frame.png"
        Image.new("RGB", (10, 10), "green").save(image)
        self.sampling = {"fps": 30, "start_seconds": 0, "frames": 3}
        template = {"source_run": "source", "visibility": "uncertain", "point_px": None,
                    "not_shuttle_regions": [], "reviewer": "Synthetic reviewer", "reviewer_kind": "automated",
                    "notes": "Synthetic only", "image": "frame.png", "image_sha256": sha256_file(image)}
        cases = [{**deepcopy(template), "id": f"case-{i}", "frame_index": i} for i in range(3)]
        cases[0].update(visibility="visible", point_px=[2, 2])
        cases[2]["not_shuttle_regions"] = [{"center_px": [8, 8], "radius_px": 1, "identity": "racket_head"}]
        self.data = {"schema_version": "0.1.0", "split": "development_seed", "image_size_px": [10, 10],
                     "source_video_sha256": "a" * 64, "sources": {"source": self.sampling}, "cases": cases}
        self.predictions = self.root / "predictions"
        self.predictions.mkdir()
        self.summary = {"source_video_sha256": "a" * 64, "sampling": self.sampling,
                        "video": {"width_px": 10, "height_px": 10}}
        self.rows = [{"frame_index": i, "timestamp_seconds": round(i / 30, 6), "candidates": [], "proposal": None} for i in range(3)]
        self.rows[0]["candidates"] = [{"center_px": [2, 2]}, {"center_px": [7, 7]}]
        self.rows[0]["proposal"] = {"candidate_index": 1, "position_px": [7, 7]}
        self.rows[2]["candidates"] = [{"center_px": [8, 8]}]
        self.rows[2]["proposal"] = {"candidate_index": 0, "position_px": [8, 8]}
        self.save()

    def save(self):
        self.benchmark = self.packet / "benchmark.json"
        self.benchmark.write_text(json.dumps(self.data), encoding="utf-8")
        (self.predictions / "shuttle_summary.json").write_text(json.dumps(self.summary), encoding="utf-8")
        (self.predictions / "shuttle_candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rows), encoding="utf-8")

    def score(self, **kwargs):
        return evaluate(self.benchmark, {"source": self.predictions}, **kwargs)

    def test_point_hits_and_local_negatives_are_separate_and_never_verified(self):
        result = self.score()
        self.assertEqual((result["visible_points"], result["candidate_point_hits"], result["proposal_point_hits"]), (1, 1, 0))
        self.assertEqual((result["known_negative_regions"], result["negative_regions_with_candidates"], result["negative_regions_with_proposals"]), (1, 1, 1))
        self.assertEqual(result["verified_shuttle_positions"], 0)
        self.assertNotIn("precision", result)
        self.assertEqual(result["uncertain_frames_excluded_from_point_scoring"], 2)

    def test_unknown_frames_do_not_become_absence_labels(self):
        before = self.score()
        self.rows[1]["candidates"] = [{"center_px": [1, 1]}]
        self.rows[1]["proposal"] = {"candidate_index": 0, "position_px": [1, 1]}
        self.save()
        after = self.score()
        for key in ("candidate_point_hits", "proposal_point_hits", "negative_regions_with_proposals"):
            self.assertEqual(before[key], after[key])

    def test_tolerance_boundary_and_invalid_points(self):
        self.rows[0]["candidates"] = [{"center_px": [6, 2]}]
        self.rows[0]["proposal"] = {"candidate_index": 0, "position_px": [6, 2]}
        self.save()
        self.assertEqual(self.score(tolerance_px=4)["proposal_point_hits"], 1)
        self.assertEqual(self.score(tolerance_px=3.99)["proposal_point_hits"], 0)
        self.rows[0]["candidates"][0]["center_px"] = [float("nan"), 2]
        self.save()
        with self.assertRaisesRegex(ValueError, "point"):
            self.score()

    def test_changed_or_escaping_images_are_rejected(self):
        Image.new("RGB", (10, 10), "red").save(self.packet / "frame.png")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.score()
        outside = self.root / "outside.png"
        Image.new("RGB", (10, 10), "red").save(outside)
        for case in self.data["cases"]:
            case.update(image="../outside.png", image_sha256=sha256_file(outside))
        self.save()
        with self.assertRaisesRegex(ValueError, "outside"):
            self.score()

    def test_mismatched_source_sampling_and_malformed_rows_are_rejected(self):
        original = deepcopy(self.summary)
        for change in ({"source_video_sha256": "b" * 64}, {"sampling": {**self.sampling, "fps": 15}}):
            self.summary = {**original, **change}
            self.save()
            with self.assertRaisesRegex(ValueError, "does not match"):
                self.score()
        self.summary = original
        self.rows[1]["timestamp_seconds"] = 100
        self.save()
        with self.assertRaisesRegex(ValueError, "timestamp"):
            self.score()

    def test_duplicate_labels_and_inconsistent_visibility_are_rejected(self):
        self.data["cases"].append(deepcopy(self.data["cases"][0]))
        self.save()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.score()
        self.data["cases"].pop()
        self.data["cases"][1]["point_px"] = [2, 2]
        self.save()
        with self.assertRaisesRegex(ValueError, "Uncertain"):
            self.score()

    def test_contradictory_positive_and_local_negative_are_rejected(self):
        self.data["cases"][0]["not_shuttle_regions"] = [{"center_px": [2, 2], "radius_px": 2, "identity": "racket"}]
        self.save()
        with self.assertRaisesRegex(ValueError, "contradicts"):
            self.score()

    def test_prepare_decodes_hash_bound_native_images_and_preserves_existing_packet(self):
        frames = self.root / "frames"
        frames.mkdir()
        for i in range(3):
            Image.new("RGB", (10, 10), (i * 80, 30, 10)).save(frames / f"frame_{i + 1:06d}.jpg")
        video = self.root / "source.mp4"
        _encode_preview(frames, video, 30)
        self.data["source_video_sha256"] = sha256_file(video)
        seed = self.root / "seed.json"
        seed.write_text(json.dumps(self.data))
        target = prepare(seed, video, self.root / "new-packet")
        data = json.loads(target.read_text())
        self.assertEqual(len(data["cases"]), 3)
        for case in data["cases"]:
            self.assertEqual(sha256_file(target.parent / case["image"]), case["image_sha256"])
        before = target.read_bytes()
        with self.assertRaisesRegex(ValueError, "new benchmark"):
            prepare(seed, video, target.parent)
        self.assertEqual(target.read_bytes(), before)

    def continuous_fixture(self):
        template = self.data["cases"][0]
        self.sampling["frames"] = 10
        self.data["role"] = "continuous_development"
        self.data["sequences"] = [{"id": "interval", "source_run": "source", "start_frame": 0, "end_frame": 9}]
        self.data["cases"] = [{**deepcopy(template), "id": f"case-{i}", "frame_index": i} for i in range(10)]
        self.data["cases"][2].update(visibility="uncertain", point_px=None)
        self.rows = []
        for i in range(10):
            point = [8, 8] if i == 7 else [2, 2]
            missing = i in (5, 6, 8)
            self.rows.append({"frame_index": i, "timestamp_seconds": round(i/30, 6), "scene_id": int(i == 9),
                              "candidates": [] if missing else [{"center_px": point}],
                              "proposal": None if missing else {"candidate_index": 0, "position_px": point,
                                  "proposal_segment_id": 0 if i < 4 else (1 if i == 4 else i)}})
        self.save()

    def test_continuous_metrics_separate_missing_displaced_and_unknown_and_split_runs(self):
        self.continuous_fixture()
        result = self.score()
        sequence = result["continuous_sequences"][0]
        self.assertEqual(sequence["annotated_frames"], 10)
        self.assertEqual(sequence["visible_frames"], 9)
        self.assertEqual(sequence["proposal_point_hits"], 5)
        self.assertEqual(sequence["visible_frames_without_proposal"], 3)
        self.assertEqual(sequence["visible_frames_with_displaced_proposal"], 1)
        self.assertEqual(sequence["uncertain_frames_with_proposals_unscored"], 1)
        self.assertEqual(sequence["longest_correct_same_segment_run_frames"], 2)
        self.assertEqual(sequence["longest_consecutive_visible_missing_proposal_frames"], 2)
        self.assertEqual(sequence["adjacent_point_matches_in_same_segment"], 1)
        self.assertEqual(sequence["adjacent_point_matches_with_segment_or_scene_change"], 1)
        self.assertAlmostEqual(sequence["mean_error_on_visible_proposed_frames_px"], 1.4142, places=4)
        self.assertEqual(sequence["timeline"][2]["result"], "uncertain_unscored")
        self.assertEqual(result["verified_shuttle_positions"], 0)

    def test_independent_scene_boundary_splits_motion_and_runs_with_same_wide_view(self):
        self.continuous_fixture()
        for case in self.data["cases"]:
            case.update(view="wide_court",annotation_scene_id=int(case["frame_index"]>=1))
        self.save()
        sequence=self.score()["continuous_sequences"][0]
        self.assertEqual(sequence["annotation_scene_change_frames"],[1])
        self.assertEqual(sequence["annotation_view_change_frames"],[])
        self.assertEqual(sequence["scene_change_frames"],[9])
        self.assertEqual(sequence["adjacent_visible_proposed_pairs_same_segment_and_view"],0)
        self.assertEqual(sequence["longest_correct_same_segment_run_frames"],1)
        self.assertEqual(sequence["adjacent_point_matches_with_annotation_scene_change"],1)
        self.assertIsNone(sequence["mean_step_vector_error_px"])
        self.assertEqual(sequence["timeline"][1]["annotation_scene_id"],1)
        for case in self.data["cases"]:case["annotation_scene_id"]=int(case["frame_index"]>=6)
        self.save()
        self.assertEqual(self.score()["continuous_sequences"][0]["longest_consecutive_visible_missing_proposal_frames"],1)

    def test_annotation_scenes_require_complete_nonnegative_integer_monotonic_ids(self):
        self.continuous_fixture()
        for case in self.data["cases"]:case["annotation_scene_id"]=0
        for invalid in (None,-1,True,1.5):
            self.data["cases"][0]["annotation_scene_id"]=invalid;self.save()
            with self.assertRaisesRegex(ValueError,"annotation scene ID"):self.score()
        self.data["cases"][0]["annotation_scene_id"]=1;self.save()
        with self.assertRaisesRegex(ValueError,"cannot decrease"):self.score()
        self.data["cases"][0].pop("annotation_scene_id");self.save()
        with self.assertRaisesRegex(ValueError,"Every case"):self.score()

    def test_continuous_coverage_requires_all_frames_and_no_overlapping_intervals(self):
        self.continuous_fixture()
        original = deepcopy(self.data)
        self.data["cases"].pop(5)
        self.save()
        with self.assertRaisesRegex(ValueError, "every frame"):
            self.score()
        self.data = deepcopy(original)
        self.data["sequences"].append({"id":"overlap", "source_run":"source", "start_frame":8, "end_frame":9})
        self.save()
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.score()
        self.data = deepcopy(original)
        del self.data["sequences"]
        self.save()
        with self.assertRaisesRegex(ValueError, "requires sequence"):
            self.score()

    def test_continuous_evaluation_rejects_reused_segment_across_gap_or_scene(self):
        self.continuous_fixture()
        self.rows[7]["proposal"]["proposal_segment_id"] = 0
        self.save()
        with self.assertRaisesRegex(ValueError, "gap or scene"):
            self.score()
        self.continuous_fixture()
        self.rows[1]["scene_id"] = 1
        self.save()
        with self.assertRaisesRegex(ValueError, "gap or scene"):
            self.score()
        self.continuous_fixture()
        del self.rows[0]["proposal"]["proposal_segment_id"]
        self.save()
        with self.assertRaisesRegex(ValueError, "valid scene"):
            self.score()

    def test_all_uncertain_sequence_produces_no_location_error_or_continuity_claim(self):
        self.continuous_fixture()
        for case in self.data["cases"]:
            case.update(visibility="uncertain", point_px=None)
        self.save()
        sequence = self.score()["continuous_sequences"][0]
        self.assertEqual(sequence["visible_frames"], 0)
        self.assertEqual(sequence["longest_correct_same_segment_run_frames"], 0)
        self.assertEqual(sequence["longest_consecutive_visible_missing_proposal_frames"], 0)
        self.assertIsNone(sequence["mean_error_on_visible_proposed_frames_px"])

    def test_motion_error_uses_observed_adjacent_points_and_breaks_at_unknown_gap_cut_and_view(self):
        self.continuous_fixture()
        # One observed label step is zero but the proposal moves one pixel.
        self.rows[1]["candidates"][0]["center_px"] = [3, 2]
        self.rows[1]["proposal"]["position_px"] = [3, 2]
        self.save()
        sequence = self.score()["continuous_sequences"][0]
        self.assertEqual(sequence["adjacent_visible_proposed_pairs_same_segment_and_view"], 1)
        self.assertEqual(sequence["mean_step_vector_error_px"], 1)
        self.assertEqual(sequence["max_step_vector_error_px"], 1)
        self.assertEqual(sequence["scene_change_frames"], [9])
        self.data["cases"][0]["view"] = "closeup"
        self.data["cases"][1]["view"] = "wide_court"
        self.save()
        sequence = self.score()["continuous_sequences"][0]
        self.assertEqual(sequence["adjacent_visible_proposed_pairs_same_segment_and_view"], 0)
        self.assertEqual(sequence["adjacent_point_matches_with_segment_or_scene_change"], 1)
        self.assertEqual(sequence["adjacent_point_matches_with_annotation_view_change"], 1)
        self.assertEqual(sequence["annotation_view_change_frames"], [1, 2])
        self.assertIsNone(sequence["mean_step_vector_error_px"])
        self.assertEqual(sequence["annotation_view_counts"]["closeup"]["visible_frames"], 1)
        self.data["cases"][0]["view"] = "invalid"
        self.save()
        with self.assertRaisesRegex(ValueError, "annotation view"):
            self.score()

    def test_view_breakdown_does_not_hide_visible_closeup_or_gate_failures(self):
        self.continuous_fixture()
        self.data["cases"][5]["view"] = "closeup"
        self.rows[5]["status"] = "excluded_player_review"
        self.save()
        result = self.score()
        sequence = result["continuous_sequences"][0]
        self.assertEqual(result["visible_points"], 9)
        self.assertEqual(sequence["annotation_view_counts"]["closeup"]["visible_frames_without_proposal"], 1)
        self.assertEqual(sequence["prediction_status_counts"]["excluded_player_review"], 1)
        self.assertEqual(sequence["timeline"][5]["result"], "missing_proposal")
        self.assertEqual(sequence["timeline"][5]["annotation_view"], "closeup")


if __name__ == "__main__":
    unittest.main()
