from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import unittest

import test_session_runner as fixtures
from contact_proposals import find_contacts,prepare
from match_analysis import verify,validate_document,new_outcome
from tracking_review import decide,CHECKS


def observations():
    return [{'frame_index':i,'timestamp_seconds':i/10,'scene_id':0,'status':'proposal',
             'proposal':{'position_px':[10+6*(i if i<=5 else 10-i),25],'proposal_segment_id':0}}
            for i in range(10)]


class ContactTests(unittest.TestCase):
    def setUp(self):
        self.players=[{'persons':[{'track_id':1,'bbox_xyxy_px':[30,10,60,60]}]} for _ in range(5)]

    def test_direction_change_near_player_is_only_a_proposal(self):
        result=find_contacts(observations(),self.players,10,5,(100,80),set(range(10)))
        self.assertEqual([p['frame_index'] for p in result],[5])
        self.assertEqual(result[0]['nearby_track_ids'],[1])
        self.assertIn('unknown',result[0]['hitter_status'])

    def test_gaps_cuts_link_segments_and_ineligible_context_are_not_bridged(self):
        for kind in ('gap','scene','segment','eligibility'):
            rows=observations();eligible=set(range(10))
            if kind=='gap':rows[4]['proposal']=None
            if kind=='scene':
                for row in rows[5:]:row['scene_id']=1
            if kind=='segment':
                for row in rows[5:]:row['proposal']['proposal_segment_id']=1
            if kind=='eligibility':eligible.remove(4)
            self.assertEqual(find_contacts(rows,self.players,10,5,(100,80),eligible),[],kind)

    def test_straight_motion_or_no_player_near_contact_does_not_create_contact(self):
        rows=observations()
        for i,row in enumerate(rows):row['proposal']['position_px']=[10+i*6,25]
        self.assertEqual(find_contacts(rows,self.players,10,5,(100,80),set(range(10))),[])
        players=[{'persons':[{'track_id':2,'bbox_xyxy_px':[0,65,5,79]}]} for _ in range(5)]
        self.assertEqual(find_contacts(observations(),players,10,5,(100,80),set(range(10))),[])

    def test_malformed_or_nonfinite_observations_are_rejected(self):
        for mutation in (lambda r:r[5]['proposal'].update(position_px=[float('nan'),25]),
                         lambda r:r[5].update(timestamp_seconds=100),lambda r:r[5]['proposal'].update(proposal_segment_id=True)):
            rows=observations();mutation(rows)
            with self.assertRaises(ValueError):find_contacts(rows,self.players,10,5,(100,80),set(range(10)))


class ContactPacketTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.SessionTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups);self.fixture.run_fake()

    def test_pending_player_review_yields_no_guessed_contacts(self):
        out=self.fixture.root/'suggested';prepare(self.fixture.output,out)
        packet=verify(self.fixture.output,out)
        self.assertEqual(packet['automatic_proposals']['status'],'not_available')
        self.assertEqual(packet['document']['events'],[])

    def test_eligible_bound_observations_generate_unknown_shots_and_revocation_blocks_reuse(self):
        players=self.fixture.output/'players-001'
        decide(players,0,'accepted','Automated fixture','Synthetic only',dict.fromkeys(CHECKS,True),reviewer_kind='automated')
        original=self.fixture.fake_execute
        def execute(command,directory,output,environment):
            if Path(command[1]).name!='shuttle_tracknet.py':return original(command,directory,output,environment)
            data=directory/'data';data.mkdir()
            job=json.loads((output/'job.json').read_text())
            review=json.loads((players/'tracking_review.json').read_text())
            summary={'source_video_sha256':job['binding']['source']['sha256'],
                     'player_review_snapshot_sha256':sha256(json.dumps(review,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
                     'sampling':{'start_seconds':0,'fps':10}}
            (data/'shuttle_summary.json').write_text(json.dumps(summary))
            (data/'shuttle_candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in observations()))
        self.fixture.fake_execute=execute;self.fixture.run_fake(True)
        out=self.fixture.root/'suggested';prepare(self.fixture.output,out);packet=verify(self.fixture.output,out)
        self.assertEqual(packet['automatic_proposals']['status'],'experimental_proposals')
        self.assertEqual(len(packet['document']['events']),1)
        self.assertEqual(packet['document']['events'][0]['status'],'proposed')
        self.assertIsNone(packet['document']['events'][0]['player'])
        self.assertEqual(packet['document']['events'][0]['shot_family'],'unknown')
        self.assertEqual(packet['document']['rallies'][0]['coverage'],'fragment')
        decide(players,0,'rejected','Automated fixture','Revoked',{},reviewer_kind='automated')
        with self.assertRaisesRegex(ValueError,'stale'):prepare(self.fixture.output,self.fixture.root/'stale')
        self.assertFalse((self.fixture.root/'stale').exists())
