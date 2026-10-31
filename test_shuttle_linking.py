from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shuttle_linking import ConfidenceMotionLinker, make_linker, relink


def candidate(x,peak=.9):
    return {"center_px":[x,20],"heatmap_peak":peak}


class ConfidenceLinkingTests(unittest.TestCase):
    def test_dominant_activation_references_original_candidate_without_mutation(self):
        candidates=[candidate(50,.4),candidate(10,.85)]
        original=deepcopy(candidates)
        found=ConfidenceMotionLinker().update(0,0,candidates)
        self.assertEqual(found["candidate_index"],1)
        self.assertEqual(found["position_px"],[10,20])
        self.assertEqual(found["selection_method"],"dominant_activation_with_motion_gate")
        self.assertEqual(candidates,original)
        self.assertEqual(found["verification_status"],"unverified")
        self.assertIsNotNone(ConfidenceMotionLinker().update(0,0,[candidate(10,.85),candidate(50,.65)]))

    def test_weak_or_close_peaks_remain_ambiguous_but_observed_motion_can_select(self):
        for peaks in ((.6,.3),(.8,.7),(.9,.9)):
            linker=ConfidenceMotionLinker()
            self.assertIsNone(linker.update(0,0,[candidate(10,peaks[0]),candidate(30,peaks[1])]))
        linker=ConfidenceMotionLinker()
        linker.update(0,0,[candidate(10)])
        linker.update(1,0,[candidate(15)])
        found=linker.update(2,0,[candidate(20,.8),candidate(50,.8)])
        self.assertEqual(found["position_px"],[20,20])
        self.assertEqual(found["selection_method"],"motion")

    def test_confidence_cannot_override_step_gate_or_bridge_gaps_and_cuts(self):
        linker=ConfidenceMotionLinker()
        first=linker.update(0,0,[candidate(10)])
        self.assertIsNone(linker.update(1,0,[candidate(90,.95),candidate(15,.4)]))
        second=linker.update(2,0,[candidate(15)])
        self.assertIsNone(linker.update(3,0,[]))
        third=linker.update(4,0,[candidate(20,.9),candidate(50,.4)])
        fourth=linker.update(5,1,[candidate(20)])
        fifth=linker.update(7,1,[candidate(25)])
        self.assertEqual([p["proposal_segment_id"] for p in [first,second,third,fourth,fifth]],list(range(5)))

    def test_invalid_peaks_and_modes_are_rejected(self):
        for peak in (None,float("nan"),float("inf"),-.1,1.1,True,".9"):
            with self.assertRaisesRegex(ValueError,"heatmap peaks"):
                ConfidenceMotionLinker().update(0,0,[candidate(10,peak)])
        with self.assertRaisesRegex(ValueError,"linker mode"): make_linker("bad")


class RelinkingTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.source=self.root/"source";self.source.mkdir()
        self.review={"synthetic":"review"}
        self.players={"session_id":"synthetic-"+"a"*12,"video":{"width_px":100,"height_px":100},
                      "sampling":{"fps":30,"start_seconds":0}}
        self.player_rows=[{"scene_id":0,"persons":[]} for _ in range(5)]
        self.verification={"source_binding":{"summary_sha256":"b"*64,"tracks_sha256":"c"*64},
                           "approved_intervals":[{"scene_id":0,"start_frame":1,"end_frame":4}],
                           "review_provenance":{"accepted_by_reviewer_kind":{"automated":1}}}
        self.summary={"method":"tracknetv3_concat_scene_bounded_heatmaps","status":"experimental_unverified_candidates",
                      "source_video_sha256":"a"*64,"verified_shuttle_positions":0,
                      "player_source_binding":self.verification["source_binding"],
                      "player_review_snapshot_sha256":sha256(json.dumps(self.review,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
                      "player_review_provenance":self.verification["review_provenance"],
                      "sampling":{"fps":30,"start_seconds":0,"frames":5},
                      "video":{"width_px":100,"height_px":100,"frame_rate_fps":30},"configuration":{},"limitations":[],"elapsed_seconds":99}
        self.rows=[{"frame_index":i,"timestamp_seconds":round(i/30,6),"scene_id":0,
                    "status":"ambiguous_candidates" if i else "excluded_player_review",
                    "candidates":[candidate(10+i,.85),candidate(50,.4)] if i else [],
                    "proposal":None,"heatmap_peak":.85 if i else None,"verified_shuttle_position_px":None} for i in range(5)]
        self.rows[3].update(status="insufficient_sequence_context",candidates=[],heatmap_peak=None)
        self.snapshot=patch("shuttle_linking.verified_snapshot",side_effect=lambda _:self.state())
        self.snapshot.start();self.addCleanup(self.snapshot.stop)
        self.save()

    def state(self):return deepcopy((self.verification,self.players,self.player_rows,self.review))

    def save(self):
        (self.source/"shuttle_summary.json").write_text(json.dumps(self.summary))
        (self.source/"shuttle_candidates.jsonl").write_text("".join(json.dumps(r)+"\n" for r in self.rows))

    def test_replay_preserves_candidates_gates_and_history_bindings(self):
        before=(self.source/"shuttle_candidates.jsonl").read_bytes()
        target=relink(self.source,self.root/"players",self.root/"result","confidence_motion")
        result=json.loads(target.read_text())
        rows=[json.loads(s) for s in (target.parent/"shuttle_candidates.jsonl").read_text().splitlines()]
        self.assertEqual([r["candidates"] for r in rows],[r["candidates"] for r in self.rows])
        self.assertEqual([r["status"] for r in rows],["excluded_player_review","proposal","proposal","insufficient_sequence_context","proposal"])
        self.assertEqual([rows[i]["proposal"]["proposal_segment_id"] for i in [1,2,4]],[0,0,1])
        self.assertEqual(result["relink_source_binding"]["rows_sha256"],sha256(before).hexdigest())
        self.assertEqual((self.source/"shuttle_candidates.jsonl").read_bytes(),before)
        self.assertNotIn("elapsed_seconds",result)
        self.assertEqual(result["coaching_metrics"]["status"],"not_available")
        self.assertTrue(all(r["verified_shuttle_position_px"] is None for r in rows))
        self.assertFalse((target.parent/"candidate_preview.mp4").exists())
        with self.assertRaisesRegex(ValueError,"new relinking"):relink(self.source,self.root,target.parent,"confidence_motion")

    def test_stale_review_invalid_sampling_and_forged_excluded_observations_fail(self):
        original=deepcopy(self.rows)
        for mutate in [lambda:self.rows[0].update(candidates=[candidate(10)]),
                       lambda:self.rows[1].update(scene_id=1),
                       lambda:self.rows[1].update(timestamp_seconds=9),
                       lambda:self.rows[1].update(verified_shuttle_position_px=[1,2]),
                       lambda:self.rows[1].update(candidates=[candidate(100)]),
                       lambda:self.rows[1].update(candidates=[candidate(10,float("nan"))]),
                       lambda:self.rows[1].update(status="proposal"),
                       lambda:self.rows[1].update(status="no_heatmap_detection"),
                       lambda:self.rows[1].update(proposal={"candidate_index":99,"position_px":[1,2]}),
                       lambda:self.rows[3].update(candidates=[candidate(10)])]:
            self.rows=deepcopy(original);mutate();self.save()
            with self.assertRaises(ValueError):relink(self.source,self.root,self.root/"invalid","confidence_motion")
            self.assertFalse((self.root/"invalid").exists())
        self.rows=original;self.save()
        self.review["changed"]=True
        with self.assertRaisesRegex(ValueError,"current player review"):relink(self.source,self.root,self.root/"stale","motion")

    def test_review_revocation_during_replay_prevents_output(self):
        first=self.state();second=self.state();second[3]["revoked"]=True
        with patch("shuttle_linking.verified_snapshot",side_effect=[first,second]):
            with self.assertRaisesRegex(ValueError,"changed while processing"):
                relink(self.source,self.root,self.root/"revoked","confidence_motion")
        self.assertFalse((self.root/"revoked").exists())

    def test_invalid_source_sampling_is_rejected_before_output(self):
        original=deepcopy(self.summary)
        for values in ({"fps":float("nan")},{"fps":15},{"start_seconds":1},{"frames":True},{"frames":4}):
            self.summary=deepcopy(original);self.summary["sampling"].update(values);self.save()
            with self.assertRaises(ValueError):
                relink(self.source,self.root,self.root/"invalid-sampling","confidence_motion")
            self.assertFalse((self.root/"invalid-sampling").exists())

    def test_nonfinite_metadata_is_rejected_without_partial_output(self):
        self.summary["configuration"]["bad_extension"]=float("nan");self.save()
        with self.assertRaises(ValueError):
            relink(self.source,self.root,self.root/"nonfinite","confidence_motion")
        self.assertFalse((self.root/"nonfinite").exists())


if __name__=="__main__":unittest.main()
