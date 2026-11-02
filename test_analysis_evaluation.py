from copy import deepcopy
import unittest

from analysis_evaluation import match_contacts,compare,agreement
from match_analysis import empty_document,new_rally,new_event,new_outcome
from test_match_analysis import example,SOURCE


def packet(document):
    return {'document':document,'source':SOURCE,'source_binding':{'source_video_sha256':'a'*64},'reviewer_kind':'automated'}


class EvaluationTests(unittest.TestCase):
    def test_matching_maximizes_one_to_one_count_instead_of_greedy_nearest(self):
        def event(name,frame):return {'id':name,'start_frame':frame,'end_frame':frame}
        pairs=match_contacts([event('p1',8),event('p2',10)],[event('t1',10),event('t2',12)],2)
        self.assertEqual([(p['id'],t['id']) for p,t in pairs],[('p1','t1'),('p2','t2')])

    def test_duplicate_contacts_lower_precision_and_unknown_shots_are_failures_not_omissions(self):
        truth=packet(example());pred=deepcopy(truth)
        duplicate=deepcopy(pred['document']['events'][2]);duplicate.update(id='duplicate',start_frame=51,end_frame=51,status='proposed',shot_status='proposed')
        pred['document']['events'].append(duplicate)
        pred['document']['outcomes'][0].update(finish_status='proposed',preceding_pressure_event_id=None)
        result=compare(pred,truth,allow_development=True)
        self.assertEqual(result['contacts']['matched'],4);self.assertEqual(result['contacts']['precision'],.8)
        self.assertEqual(result['contacts']['recall'],1)
        pred=deepcopy(truth);pred['document']['events'][2].update(shot_family='unknown',shot_status='unknown')
        result=compare(pred,truth,allow_development=True)
        self.assertEqual(result['shot_classes']['per_class']['smash']['recall'],0)
        self.assertFalse(result['release_approved'])

    def test_unknown_point_predictions_reduce_accuracy_and_coverage(self):
        truth=packet(example());pred=deepcopy(truth);pred['document']['outcomes']=[]
        result=compare(pred,truth,allow_development=True)
        self.assertEqual(result['point_winner']['accuracy'],0)
        self.assertEqual(result['point_winner']['prediction_coverage'],0)
        self.assertEqual(result['finish_labels']['agreement'],0)

    def test_uncertain_reference_events_are_not_false_positive_ground_truth(self):
        doc=empty_document();doc['match']['teams']={'A':['P1','P2'],'B':['P3','P4']}
        doc['rallies']=[new_rally('r1',0,99)]
        event=new_event('uncertain','r1',10);event.update(start_frame=0,end_frame=20,status='accepted',player='P1',side='A',
              evidence=[{'kind':'video','start_frame':0,'end_frame':20,'note':'Synthetic uncertain interval'}])
        doc['events']=[event];truth=packet(doc);pred=deepcopy(truth)
        pred['document']['events'][0].update(id='prediction',start_frame=10,end_frame=10,status='proposed')
        result=compare(pred,truth,allow_development=True)
        self.assertEqual(result['contacts']['reference_uncertain_excluded'],1)
        self.assertEqual(result['contacts']['predictions_unscored_near_uncertainty'],1)
        self.assertIsNone(result['contacts']['precision'])

    def test_source_side_and_rubric_mismatch_or_unreviewed_reference_fail(self):
        truth=packet(example())
        for key in ('source','teams','rubric','reviewer'):
            pred=deepcopy(truth);target=deepcopy(truth)
            if key=='source':pred['source_binding']['source_video_sha256']='b'*64
            if key=='teams':pred['document']['match']['teams']={'A':['P1'],'B':['P3']}
            if key=='rubric':pred['document']['analysis_rubric']['skill_context']='Another skill level'
            if key=='reviewer':target['reviewer_kind']=None
            with self.subTest(key=key),self.assertRaises(ValueError):compare(pred,target,allow_development=True)

    def test_kappa_discloses_no_variability_and_does_not_assert_rater_independence(self):
        self.assertIsNone(agreement([('forced_error','forced_error')])['cohens_kappa'])
        metric=agreement([('forced_error','forced_error'),('unforced_error','unforced_error')])
        self.assertEqual(metric['cohens_kappa'],1)
        self.assertFalse(metric['independent_raters_verified'])


if __name__=='__main__':unittest.main()
