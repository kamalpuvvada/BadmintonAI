import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch

from badminton_proto import sha256_file
from download_tracknet import download
from player_tracking import _encode_preview
from shuttle_tracknet import MODEL_SIZE, TrackNetPredictor, decode_heatmap, infer_groups, prepare_input, run, sequence_windows
import test_tracking_review as review_tests
from tracking_review import prepare


class UnsupportedCheckpoint:
    """A plain unsupported pickle type, without any executable payload."""
    pass


class FakePredictor:
    sequence_length = 3
    metadata = {"name": "Synthetic test predictor", "sequence_length": 3, "inpainting": False}

    def __init__(self):
        self.calls = []

    def __call__(self, batch):
        self.calls.append(np.round(batch[0, 3::3, 0, 0]*255).astype(int).tolist())
        output = np.zeros((1, 3, MODEL_SIZE[1], MODEL_SIZE[0]), dtype=np.float32)
        output[:, :, 100:103, 200:203] = .9
        return output


class ShuttleTrackNetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_preprocessing_retains_rgb_order_background_first_and_normalization(self):
        background = np.full((10, 12, 3), [10, 20, 30], dtype=np.uint8)
        images = [Image.new("RGB", (12, 10), (40, 50, 60)), Image.new("RGB", (12, 10), (70, 80, 90))]
        batch = prepare_input(images, background, size=(8, 8))
        self.assertEqual(batch.shape, (1, 9, 8, 8))
        self.assertEqual(batch.dtype, np.float32)
        np.testing.assert_allclose(batch[0, :, 0, 0], np.arange(10, 100, 10)/255, atol=1e-7)

    def test_heatmap_coordinates_map_to_source_and_missing_is_not_origin(self):
        heatmap = np.zeros((4, 8), dtype=np.float32)
        self.assertFalse(decode_heatmap(heatmap, (16, 12)))
        heatmap[1:3, 2:4] = .9
        heatmap[0, 0] = .5  # Strict upstream threshold, not a detected origin.
        found = decode_heatmap(heatmap, (16, 12))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["center_px"], [6, 6])
        self.assertEqual(found[0]["bbox_xyxy_px"], [4, 3, 8, 9])
        for invalid in (np.empty((0, 8)), np.full((4, 8), float("nan")), np.full((4, 8), 1.1)):
            with self.assertRaisesRegex(ValueError, "Invalid heatmap"):
                decode_heatmap(invalid, (16, 12))

    def test_windows_cover_tail_without_padding_or_crossing_index_gaps(self):
        self.assertEqual(sequence_windows(list(range(10, 21)), 8), [list(range(10, 18)), list(range(13, 21))])
        self.assertEqual(sequence_windows(list(range(7)), 8), [])
        with self.assertRaisesRegex(ValueError, "consecutive"):
            sequence_windows([0, 2, 3], 3)
        with self.assertRaisesRegex(ValueError, "Sequence length"):
            sequence_windows([0], True)

    def test_inference_never_shares_sequence_frames_between_scenes_or_gaps(self):
        paths = []
        contexts = []
        for i in range(12):
            path = self.root/f"{i}.png"
            Image.new("RGB", (16, 16), (10*i, 0, 0)).save(path)
            paths.append(path)
            contexts.append({"eligible": i not in (0, 4), "scene_id": 0 if i < 5 else (1 if i < 7 else 2)})
        predictor = FakePredictor()
        result = infer_groups(paths, contexts, predictor)
        self.assertEqual(predictor.calls, [[10,20,30], [70,80,90], [90,100,110]])
        self.assertEqual(sorted(result), [1,2,3,7,8,9,10,11])
        self.assertTrue(all(result[i] for i in result))
        # The too-short scene at 5,6 is left empty, never padded.
        self.assertNotIn(5, result)

    def test_overlapping_windows_cover_once_per_start_and_reject_invalid_stride(self):
        self.assertEqual(sequence_windows(list(range(11)), 8, 4), [list(range(8)), list(range(3, 11))])
        self.assertEqual(sequence_windows(list(range(17)), 8, 4),
                         [list(range(8)), list(range(4, 12)), list(range(8, 16)), list(range(9, 17))])
        for stride in (0, -1, 9, True, 1.5):
            with self.assertRaisesRegex(ValueError, "Window stride"):
                sequence_windows([], 8, stride)

    def test_overlap_averages_heatmaps_before_threshold_without_crossing_gap_or_cut(self):
        paths = []
        contexts = []
        for i in range(15):
            path = self.root/f"overlap-{i}.png"
            Image.new("RGB", (16, 16), (i*10, 0, 0)).save(path)
            paths.append(path)
            contexts.append({"eligible": i != 5, "scene_id": 0 if i < 10 else 1})

        class OffsetPredictor(FakePredictor):
            def __call__(self, batch):
                result = super().__call__(batch)
                # First sequence position is weak; later positions are strong.
                result[:, 0, 100:103, 200:203] = .3
                return result

        predictor = OffsetPredictor()
        result = infer_groups(paths, contexts, predictor, stride=2)
        self.assertEqual(predictor.calls, [[0,10,20], [20,30,40], [60,70,80],
                                         [70,80,90], [100,110,120], [120,130,140]])
        self.assertNotIn(5, result)
        self.assertFalse(result[0])
        self.assertTrue(result[2])  # Mean (.9 + .3)/2 = .6, above threshold.
        self.assertAlmostEqual(result[2][0]["heatmap_peak"], .6, places=6)
        self.assertFalse(result[6])
        self.assertFalse(result[10])

    def test_checkpoint_hash_and_restricted_loading_fail_without_unsafe_fallback(self):
        checkpoint = self.root/"weights.pt"
        torch.save({"param_dict": {"seq_len": 8, "bg_mode": "subtract", "model_name": "TrackNet"}}, checkpoint)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            TrackNetPredictor(checkpoint, "0"*64)
        with self.assertRaisesRegex(ValueError, "Only TrackNet concat"):
            TrackNetPredictor(checkpoint, sha256_file(checkpoint))
        torch.save(UnsupportedCheckpoint(), checkpoint)
        with self.assertRaises(pickle.UnpicklingError):
            TrackNetPredictor(checkpoint, sha256_file(checkpoint))

    def test_experimental_threshold_rejects_invalid_values_before_reading_inputs(self):
        for threshold in (0,1,-.1,float("nan"),float("inf"),True,"0.35"):
            with self.assertRaisesRegex(ValueError,"Heatmap threshold"):
                infer_groups([],[],FakePredictor(),heatmap_threshold=threshold)
            with self.assertRaisesRegex(ValueError,"Heatmap threshold"):
                run(self.root/"missing.mp4",self.root/"missing-run",self.root/"invalid-output",
                    self.root/"missing.pt",heatmap_threshold=threshold)
            self.assertFalse((self.root/"invalid-output").exists())

    def test_threshold_uses_averaged_heatmaps_and_peaks_do_not_fill_gaps(self):
        paths=[]
        for i in range(6):
            path=self.root/f"threshold-{i}.png"
            Image.new("RGB",(16,16),(i*10,0,0)).save(path); paths.append(path)
        class WeakFirst(FakePredictor):
            def __call__(self,batch):
                result=super().__call__(batch)
                result[:,0,100:103,200:203]=.3
                return result
        contexts=[{"eligible":i!=5,"scene_id":0} for i in range(6)]
        peaks={}
        result=infer_groups(paths,contexts,WeakFirst(),stride=2,heatmap_threshold=.65,frame_peaks=peaks)
        self.assertEqual(peaks,{0:.3,1:.9,2:.6,3:.9,4:.9})
        self.assertFalse(result[0]);self.assertFalse(result[2]);self.assertTrue(result[1])
        self.assertNotIn(5,result);self.assertNotIn(5,peaks)
        lower=infer_groups(paths,contexts,WeakFirst(),stride=2,heatmap_threshold=.2)
        self.assertTrue(lower[0]);self.assertTrue(lower[2])
        # Explicit default reproduces the unchanged decoder behavior.
        default=infer_groups(paths,contexts,WeakFirst(),stride=2)
        self.assertEqual(default,infer_groups(paths,contexts,WeakFirst(),stride=2,heatmap_threshold=.5))

    def test_native_pipeline_keeps_view_gates_and_scene_local_proposals(self):
        fixture = review_tests.TrackingReviewTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.summary["sampling"]["start_seconds"] = 0
        fixture.summary["video"] = {"width_px": 100, "height_px": 100}
        for i, row in enumerate(fixture.rows):
            row["timestamp_seconds"] = round(i/15, 3)
        frames = self.root/"frames"
        frames.mkdir()
        for i in range(12):
            Image.new("RGB", (100, 100), (10*i, 50, 0)).save(frames/f"frame_{i+1:06d}.jpg")
        video = self.root/"source.mp4"
        _encode_preview(frames, video, 30)
        fixture.summary["session_id"] = "test-"+sha256_file(video)[:12]
        fixture.write_source()
        prepare(fixture.run, 2)
        fixture.accept(0); fixture.accept(1)
        output = self.root/"result"
        with patch("shuttle_tracknet.TrackNetPredictor", return_value=FakePredictor()):
            target = run(video, fixture.run, output, self.root/"synthetic-only.pt")
        summary = json.loads(target.read_text())
        rows = [json.loads(line) for line in (output/"shuttle_candidates.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 12)
        self.assertEqual(summary["frame_status_counts"], {"excluded_player_review": 6, "proposal": 6})
        self.assertTrue(all(r["verified_shuttle_position_px"] is None for r in rows))
        self.assertEqual({r["proposal"]["proposal_segment_id"] for r in rows[:6] if r["proposal"]}, {0})
        self.assertEqual({r["proposal"]["proposal_segment_id"] for r in rows[6:] if r["proposal"]}, {1})
        self.assertEqual(summary["coaching_metrics"]["status"], "not_available")
        self.assertEqual(summary["configuration"]["heatmap_threshold"],.5)
        self.assertTrue(all(r["heatmap_peak"] is None for r in rows if r["status"]=="excluded_player_review"))
        self.assertTrue(all(r["heatmap_peak"]==.9 for r in rows if r["proposal"]))
        self.assertFalse(summary["model"]["inpainting"])
        self.assertTrue((output/"candidate_preview.mp4").exists())
        with self.assertRaisesRegex(ValueError, "new output"):
            run(video, fixture.run, output, self.root/"synthetic-only.pt")

    def test_download_reads_only_named_member_and_preserves_existing_weights(self):
        directory = self.root/"download"
        directory.mkdir()
        archive = directory/"TrackNetV3_ckpts.zip"
        with zipfile.ZipFile(archive,"w") as bundle:
            bundle.writestr("ckpts/TrackNet_best.pt", b"synthetic weights")
            bundle.writestr("../escape.txt", b"must not extract")
        checkpoint = self.root/"expected.pt"
        checkpoint.write_bytes(b"synthetic weights")
        with patch("download_tracknet.ARCHIVE_SHA256", sha256_file(archive)), patch("download_tracknet.CHECKPOINT_SHA256", sha256_file(checkpoint)):
            target = download(directory)
            self.assertEqual(target.read_bytes(), b"synthetic weights")
            self.assertEqual(download(directory), target)
            self.assertFalse((self.root/"escape.txt").exists())
            target.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "different hash"):
                download(directory)
            self.assertEqual(target.read_bytes(), b"changed")


if __name__ == "__main__":
    unittest.main()
