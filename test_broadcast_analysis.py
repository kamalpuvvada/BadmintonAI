"""Geometry, exclusion and uncertainty regressions for the broadcast prototype."""
import unittest
import json
from pathlib import Path
import tempfile
import numpy as np

from broadcast_analysis import active_difference, body_core, contiguous, fit_court, project, reuse_shuttle, shot_hint
from badminton_proto import sha256_file


class BroadcastTests(unittest.TestCase):
    def test_camera_transform_preserves_court_coordinates(self):
        pixel=np.array([[100,100],[600,100],[700,900],[0,900],[350,500]],float)
        expected=np.array([[0,0],[1,0],[1,1],[0,1],[.5,.583333333]],float)
        matrix=fit_court(pixel[:4],expected[:4])
        camera=np.array([[1.1,0,-80],[0,1.1,20],[0,0,1]])
        moved=project(pixel,camera)
        np.testing.assert_allclose(project(moved,matrix@np.linalg.inv(camera)),project(pixel,matrix),atol=1e-8)

    def test_degenerate_setup_rejected(self):
        with self.assertRaises(ValueError):fit_court([[0,0],[1,1],[2,2],[3,3]],[[0,0],[1,0],[1,1],[0,1]])
        with self.assertRaises(ValueError):fit_court([[0,0],[1,1],[2,2]],[[0,0],[1,0],[1,1]])

    def test_static_portrait_padding_does_not_dilute_cut(self):
        first=np.zeros((1000,600,3),np.uint8);second=first.copy()
        second[400:860]=255
        self.assertAlmostEqual(active_difference(first,second,(0,.4,1,.86)),1)
        # Change outside the picture cannot trigger the active-picture guard.
        third=first.copy();third[:300]=255
        self.assertEqual(active_difference(first,third,(0,.4,1,.86)),0)

    def test_movement_never_joins_missing_or_other_scene(self):
        a={'scene_id':0,'native_frame_index':0}
        self.assertTrue(contiguous(a,{'scene_id':0,'native_frame_index':4},4))
        self.assertFalse(contiguous(a,{'scene_id':1,'native_frame_index':4},4))
        self.assertFalse(contiguous(a,{'scene_id':0,'native_frame_index':8},4))
        self.assertFalse(contiguous(None,a,4))

    def test_missing_outgoing_trajectory_cannot_produce_named_shot(self):
        name,_=shot_hint([50,40],[0,0,100,200],[[60,30]])
        self.assertEqual(name,'Unknown shot')

    def test_named_hints_explicitly_remain_ambiguous(self):
        name,reason=shot_hint([50,150],[0,0,100,200],[[50,100],[50,80],[50,50],[50,30]])
        self.assertEqual(name,'Lift / clear?')
        self.assertIn('ambiguous',reason)

    def test_invalid_active_roi_rejected(self):
        im=np.zeros((20,20,3),np.uint8)
        with self.assertRaises(ValueError):active_difference(im,im,(0,.9,1,.2))

    def test_body_texture_mask_preserves_racket_space_and_marks_only_central_body(self):
        person={'bbox_xyxy_px':[100,100,200,300]}
        self.assertTrue(body_core([150,200],[person]))
        self.assertFalse(body_core([90,150],[person]))
        self.assertFalse(body_core([150,80],[person]))
        self.assertFalse(body_core([150,120],[person]))

    def test_reuse_requires_unchanged_source_config_and_observation_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';target=root/'target'
            source.mkdir();target.mkdir()
            inputs={'source':'exact-source','configuration':'exact-config'}
            (target/'analysis_binding.json').write_text(json.dumps(inputs))
            names=['shuttle_frames.jsonl','shuttle_summary.json']
            for name in names:(source/name).write_text('{}\n')
            metadata={'inputs':inputs,'artifacts_sha256':{name:sha256_file(source/name) for name in names}}
            (source/'shuttle_binding.json').write_text(json.dumps(metadata))
            (source/names[0]).write_text('modified')
            with self.assertRaisesRegex(ValueError,'artifact changed'):reuse_shuttle(source,{},target)
            (source/names[0]).write_text('{}\n')
            (target/'analysis_binding.json').write_text(json.dumps({'source':'different'}))
            with self.assertRaisesRegex(ValueError,'source/configuration/players differ'):reuse_shuttle(source,{},target)
            (target/'analysis_binding.json').write_text(json.dumps(inputs))
            reuse_shuttle(source,{},target)
            for name in names:self.assertEqual(sha256_file(source/name),sha256_file(target/name))
            self.assertFalse(json.loads((target/'shuttle_reuse.json').read_text())['model_inference_repeated'])


if __name__=='__main__':unittest.main()
