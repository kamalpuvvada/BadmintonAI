"""Regression checks for independent pose/flight predictions and abstention."""
import unittest

from shot_recognition import body_frame, classify_motion, swing_windows


def motion(**changes):
    record = {'motion_coherence': .9, 'speed_torso_per_second': 10,
              'highest_hand': .2, 'vertical_sweep': .1, 'horizontal_sweep': 1,
              'contact_height': 0, 'court_position': [.5,.8]}
    record.update(changes)
    return record


class ShotRecognitionTests(unittest.TestCase):
    def test_drive_requires_lateral_below_overhead_motion(self):
        flight = {'rise_torso':0, 'travel_torso':2}
        self.assertEqual(classify_motion(motion(),flight)[0],'Drive')
        self.assertIsNone(classify_motion(motion(highest_hand=1.1),flight))

    def test_smash_requires_high_contact_and_downward_stroke(self):
        flight = {'rise_torso':0, 'travel_torso':2}
        swing = motion(highest_hand=1.2,contact_height=.7,vertical_sweep=1)
        self.assertEqual(classify_motion(swing,flight)[0],'Smash')
        self.assertIsNone(classify_motion(motion(highest_hand=1.2,vertical_sweep=1),flight))

    def test_rising_flight_supports_lift_and_clear_at_different_contact_heights(self):
        flight = {'rise_torso':1.1, 'travel_torso':2}
        self.assertEqual(classify_motion(motion(vertical_sweep=-.7),flight)[0],'Lift')
        self.assertEqual(classify_motion(motion(highest_hand=1.3,contact_height=1.1,vertical_sweep=-.7),flight)[0],'Clear')

    def test_jitter_and_slow_motion_abstain(self):
        flight = {'rise_torso':0, 'travel_torso':2}
        self.assertIsNone(classify_motion(motion(motion_coherence=.1),flight))
        self.assertIsNone(classify_motion(motion(speed_torso_per_second=1),flight))

    def test_incomplete_cut_context_cannot_supply_swing_features(self):
        rows = [{'native_frame_index':4*i,'timestamp_seconds':i/15,'status':'experimental_court',
                 'scene_id':0 if i<2 else 1,'players':[]} for i in range(5)]
        self.assertEqual(swing_windows(rows),[])

    def test_wrong_player_torso_in_expanded_crop_is_rejected(self):
        landmarks = {name:{'point_px':point,'visibility':1} for name,point in {
            'left_shoulder':[100,100], 'right_shoulder':[140,100],
            'left_hip':[100,180], 'right_hip':[140,180], 'left_wrist':[90,120]}.items()}
        player = {'landmarks':landmarks,'bbox_xyxy_px':[0,0,50,200]}
        self.assertIsNone(body_frame(player,'left'))
        player['bbox_xyxy_px']=[80,80,160,200]
        self.assertIsNotNone(body_frame(player,'left'))
        player['landmarks']['left_wrist']['visibility']=.2
        self.assertIsNone(body_frame(player,'left'))


if __name__ == '__main__':
    unittest.main()
