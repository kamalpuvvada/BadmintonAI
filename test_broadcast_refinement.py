import unittest
from copy import deepcopy
from pathlib import Path
import tempfile
import cv2
import numpy as np
from broadcast_refinement import clean_baseline, identify_hitter, recognize_shot, recover_observed_gaps


def observation(i, point=None, scene=0, candidates=None):
    return {'frame_index':i,'native_frame_index':i*2,'timestamp_seconds':i/30,'scene_id':scene,
            'status':'proposal' if point else 'unobserved','candidates':candidates or [],
            'proposal':{'position_px':point,'proposal_segment_id':0} if point else None}


class RefinementTests(unittest.TestCase):
    crop=[0,0,512,288]
    players=[{'persons':[]}]*100

    def test_recovers_actual_candidate_without_inventing_a_position(self):
        base=[observation(0,[100,100]),observation(1),observation(2,[140,100])]
        trial=deepcopy(base);trial[1]['candidates']=[{'center_px':[121,101],'heatmap_peak':.6}]
        result,gaps=recover_observed_gaps(base,trial,self.players,self.crop)
        self.assertEqual(result[1]['proposal']['position_px'],[121,101])
        self.assertEqual(gaps[0]['samples'],1)
        self.assertIsNone(base[1]['proposal'])

    def test_no_candidate_long_gap_or_camera_cut_cannot_be_filled(self):
        base=[observation(0,[100,100]),observation(1),observation(2,[140,100])]
        trial=deepcopy(base)
        self.assertFalse(recover_observed_gaps(base,trial,self.players,self.crop)[1])
        trial[1]['candidates']=[{'center_px':[120,100],'heatmap_peak':.6}]
        trial[1]['scene_id']=1
        self.assertFalse(recover_observed_gaps(base,trial,self.players,self.crop)[1])
        trial[1]['scene_id']=0
        self.assertFalse(recover_observed_gaps(base,trial,self.players,self.crop,max_gap=0)[1])

    def test_stationary_advertising_and_scoreboard_are_rejected(self):
        base=[observation(0,[100,100]),observation(1),observation(2,[100,100])]
        trial=deepcopy(base);trial[1]['candidates']=[{'center_px':[100,100],'heatmap_peak':.9}]
        self.assertFalse(recover_observed_gaps(base,trial,self.players,self.crop)[1])
        base=[observation(0,[100,400]),observation(1),observation(2,[140,400])]
        trial=deepcopy(base);trial[1]['candidates']=[{'center_px':[120,400],'heatmap_peak':.9}]
        self.assertFalse(recover_observed_gaps(base,trial,self.players,self.crop)[1])

    def test_hitter_margin_preserves_ambiguity(self):
        people=[{'track_id':0,'bbox_xyxy_px':[100,100,200,300]}, {'track_id':1,'bbox_xyxy_px':[150,100,250,300]}]
        self.assertIsNone(identify_hitter([175,150],people))
        self.assertEqual(identify_hitter([90,150],people)['track_id'],0)

    def test_low_rising_contact_is_lift_hypothesis_not_clear(self):
        hint,_=recognize_shot([50,160],[0,0,100,200],[[50,130],[50,90],[50,60],[50,30],[50,10]])
        self.assertEqual(hint,'Lift?')
        self.assertEqual(recognize_shot([50,160],[0,0,100,200],[])[0],'Unknown shot')

    def test_real_slow_apex_is_preserved_but_persistent_background_texture_is_withheld(self):
        court=[{'scene_id':0,'pixel_to_court':np.eye(3).tolist()} for _ in range(30)]
        rows=[observation(i,[100,100] if 20<=i<=28 else None) for i in range(60)]
        with tempfile.TemporaryDirectory() as temp:
            for persistent in [False,True]:
                video=Path(temp)/f'{persistent}.mp4'
                writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),60,(640,480))
                self.assertTrue(writer.isOpened())
                for n in range(120):
                    frame=np.zeros((480,640,3),np.uint8)
                    if persistent or 40<=n<=56:cv2.circle(frame,(100,100),4,(255,255,255),-1)
                    writer.write(frame)
                writer.release()
                result,removed=clean_baseline(rows,court,self.crop,video)
                self.assertEqual(len(removed),9 if persistent else 0)
                self.assertEqual(result[24]['proposal'] is None,persistent)


if __name__=='__main__':unittest.main()
