import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_session_runner as fixtures
from player_identity import prepare, apply, source_snapshot, tracklets, verify


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.SessionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.run_fake()
        self.job=self.fixture.output
        self.output=self.fixture.root/'identities'
        self.patch_path=self.fixture.root/'patch.json'
        self.binding=source_snapshot(self.job)[4]
        self.patch={'schema_version':'1.0.0','source_binding':self.binding,
                    'reviewer':'Synthetic automated fixture','reviewer_kind':'automated','assignments':[
                        {'tracklet_id':'S0:T0:F0-4','start_frame':0,'end_frame':4,'player_slot':'P1'},
                        {'tracklet_id':'S0:T1:F0-4','start_frame':0,'end_frame':4,'player_slot':'P2'}]}

    def write(self):
        self.patch_path.write_text(json.dumps(self.patch),encoding='utf-8')

    def test_prepare_has_no_identity_approval_or_automatic_assignment(self):
        editor=self.fixture.root/'editor'
        prepare(self.job,editor)
        data=json.loads((editor/'identity_source.json').read_text())
        self.assertEqual(data['source_binding'],self.binding)
        self.assertEqual(len(data['tracklets']),2)
        self.assertNotIn('assignments',data)
        self.assertEqual((editor/'preview.mp4').read_bytes(),(self.job/'render-001/preview.mp4').read_bytes())
        self.assertTrue((editor/'index.html').exists())

    def test_apply_preserves_observations_pending_review_and_automated_provenance(self):
        before=(self.job/'players-001/player_tracks.jsonl').read_bytes()
        review=(self.job/'players-001/tracking_review.json').read_bytes()
        self.write();path=apply(self.job,self.patch_path,self.output)
        result=json.loads(path.read_text())
        self.assertEqual(result['reviewer_kind'],'automated')
        self.assertFalse(result['reviewer_identity_authenticated'])
        self.assertEqual(len(result['positions']),10)
        self.assertEqual(result['coverage']['assigned_observations'],10)
        self.assertEqual(result['coaching_metrics']['status'],'not_available')
        self.assertEqual((self.job/'players-001/player_tracks.jsonl').read_bytes(),before)
        self.assertEqual((self.job/'players-001/tracking_review.json').read_bytes(),review)
        self.assertEqual(verify(self.job,self.output)['status'],'valid_identity_snapshot')
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'modified'):
            verify(self.job,self.output)

    def test_conflicting_assignments_invalid_ranges_and_stale_binding_leave_no_output(self):
        variants=[
            [{'tracklet_id':'S0:T0:F0-4','start_frame':0,'end_frame':4,'player_slot':'P3'}],
            [{'tracklet_id':'S0:T0:F0-4','start_frame':False,'end_frame':4,'player_slot':'P1'}],
            [{'tracklet_id':'S0:T0:F0-4','start_frame':0,'end_frame':5,'player_slot':'P1'}],
            [self.patch['assignments'][0],{**self.patch['assignments'][1],'player_slot':'P1'}],
            [self.patch['assignments'][0],{**self.patch['assignments'][0],'player_slot':'P2'}]]
        for assignments in variants:
            self.patch['assignments']=assignments;self.write()
            with self.assertRaises(ValueError):
                apply(self.job,self.patch_path,self.output)
            self.assertFalse(self.output.exists())
        self.patch['source_binding']={**self.binding,'tracks_sha256':'0'*64};self.write()
        with self.assertRaisesRegex(ValueError,'stale'):
            apply(self.job,self.patch_path,self.output)

    def test_split_identity_range_is_explicit_and_gaps_are_not_interpolated(self):
        self.patch['assignments']=[{'tracklet_id':'S0:T0:F0-4','start_frame':0,'end_frame':1,'player_slot':'P1'},
                                   {'tracklet_id':'S0:T0:F0-4','start_frame':3,'end_frame':4,'player_slot':'P2'}]
        self.write();result=json.loads(apply(self.job,self.patch_path,self.output).read_text())
        self.assertEqual([r['frame_index'] for r in result['positions']],[0,1,3,4])
        self.assertEqual([r['player_slot'] for r in result['positions']],['P1','P1','P2','P2'])
        rows=[{'scene_id':s,'timestamp_seconds':i,'persons':([] if i==1 else [{'track_id':0,'bbox_xyxy_px':[0,0,1,1]}])}
              for i,s in enumerate([0,0,0,1])]
        self.assertEqual([g['id'] for g in tracklets(rows)],['S0:T0:F0-0','S0:T0:F2-2','S1:T0:F3-3'])

    def test_duplicate_ids_and_copy_changes_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            tracklets([{'scene_id':0,'timestamp_seconds':0,'persons':[{'track_id':0,'bbox_xyxy_px':[0,0,1,1]}]*2}])
        editor=self.fixture.root/'editor'
        def corrupted(source,target):
            Path(target).write_bytes(b'changed preview')
        with patch('player_identity.shutil.copyfile',side_effect=corrupted):
            with self.assertRaisesRegex(ValueError,'changed while copying'):
                prepare(self.job,editor)
        self.assertFalse(editor.exists())

    def test_missing_reviewer_unknown_fields_and_changed_source_are_rejected(self):
        self.patch['reviewer']='';self.write()
        with self.assertRaisesRegex(ValueError,'reviewer'):
            apply(self.job,self.patch_path,self.output)
        self.patch['reviewer']='Fixture';self.patch['approval']=True;self.write()
        with self.assertRaisesRegex(ValueError,'schema'):
            apply(self.job,self.patch_path,self.output)
        self.fixture.video.write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError,'Source video changed'):
            source_snapshot(self.job)
