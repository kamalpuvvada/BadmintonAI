from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from shot_model import FEATURE_VERSION,FEATURES,features_at,fit,predict,train_file,load_model,validate_records,propose,export_examples,combine_files
from test_contact_proposals import observations
import test_session_runner as fixtures
from match_analysis import prepare,verify,apply
from test_match_analysis import example
from badminton_proto import sha256_file


def synthetic_dataset():
    records=[]
    for group,split,n in [('a','train',10),('b','train',10),('c','test',5)]:
        for kind,label in enumerate(['drop','smash']):
            for i in range(n):
                features=[(-1 if kind==0 else 1)+.001*i]*len(FEATURES)
                records.append({'example_id':f'{group}-{label}-{i}','session_id':group,'source_video_sha256':group*64,
                    'rally_id':'r1','event_id':f'e{kind}-{i}','frame_index':kind*100+i,'reviewer_kind':'automated',
                    'label':label,'split':split,'features':features})
    return {'schema_version':'1.0.0','feature_version':FEATURE_VERSION,'description':'Synthetic numerical fixture, not badminton performance evidence','records':records}


class FeatureTests(unittest.TestCase):
    def test_features_are_resolution_invariant_and_never_interpolate_missing_context(self):
        rows=observations();a=features_at(rows,5,10,(100,80),set(range(10)))
        scaled=deepcopy(rows)
        for row in scaled:row['proposal']['position_px']=[x*2 for x in row['proposal']['position_px']]
        self.assertTrue(np.allclose(a,features_at(scaled,5,10,(200,160),set(range(10)))))
        rows[4]['proposal']=None;self.assertIsNone(features_at(rows,5,10,(100,80),set(range(10))))
        self.assertIsNone(features_at(observations(),0,10,(100,80),set(range(10))))

    def test_model_cannot_use_a_window_across_cuts_or_link_segments(self):
        rows=observations();rows[5]['scene_id']=1
        self.assertIsNone(features_at(rows,5,10,(100,80),set(range(10))))
        rows=observations();rows[5]['proposal']['proposal_segment_id']=1
        self.assertIsNone(features_at(rows,5,10,(100,80),set(range(10))))


class ModelTests(unittest.TestCase):
    def test_synthetic_training_is_deterministic_development_only_and_abstains_on_ambiguity(self):
        data=synthetic_dataset();model,metrics=fit(data,allow_development=True)
        other,_=fit(data,allow_development=True);self.assertEqual(model,other)
        self.assertEqual(metrics['test']['examples'],10)
        self.assertGreater(metrics['test']['accuracy_all_examples'],.9)
        self.assertFalse(metrics['release_gate_passed'])
        self.assertEqual(predict(model,[0]*len(FEATURES))['shot_family'],'unknown')
        self.assertEqual(predict(model,[-1]*len(FEATURES))['shot_family'],'drop')
        self.assertEqual(predict(model,[1]*len(FEATURES))['status'],'proposed')

    def test_automated_labels_need_explicit_development_mode(self):
        with self.assertRaisesRegex(ValueError,'Automated examples'):fit(synthetic_dataset())

    def test_split_leakage_duplicate_physical_contacts_nonfinite_and_missing_classes_fail(self):
        variants=[]
        data=synthetic_dataset();data['records'][-1]['source_video_sha256']='a'*64;data['records'][-1]['frame_index']=999;variants.append(data)
        data=synthetic_dataset();duplicate=deepcopy(data['records'][0]);duplicate['example_id']='new-id';data['records'].append(duplicate);variants.append(data)
        data=synthetic_dataset();data['records'][0]['features'][0]=float('nan');variants.append(data)
        data=synthetic_dataset();data['records']=[r for r in data['records'] if r['source_video_sha256']!='b'*64];variants.append(data)
        data=synthetic_dataset();data['records']=[r for r in data['records'] if r['label']=='drop'];variants.append(data)
        for data in variants:
            with self.assertRaises(ValueError):fit(data,allow_development=True)

    def test_model_manifest_detects_tampering_and_invalid_parameter_shapes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dataset=root/'data.json';dataset.write_text(json.dumps(synthetic_dataset()));out=root/'model'
            train_file(dataset,out,allow_development=True);model=load_model(out)
            broken=deepcopy(model);broken['weights']=[[1]]
            with self.assertRaisesRegex(ValueError,'shapes'):predict(broken,[0]*len(FEATURES))
            (out/'model.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed'):load_model(out)

    def test_combining_feature_files_preserves_source_splits_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);data=synthetic_dataset();paths=[]
            for split in ('train','test'):
                subset={**data,'records':[r for r in data['records'] if r['split']==split]}
                path=root/(split+'.json');path.write_text(json.dumps(subset));paths.append(path)
            out=root/'combined.json';combine_files(paths,out,allow_development=True)
            merged=json.loads(out.read_text());self.assertEqual(len(merged['records']),50)
            self.assertEqual(len(merged['source_inputs']),2)
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                combine_files([paths[0],paths[0]],root/'duplicate.json',allow_development=True)
            self.assertFalse((root/'duplicate.json').exists())


class ModelPacketTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.SessionTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups);self.fixture.run_fake()
        self.base=self.fixture.root/'analysis';prepare(self.fixture.output,self.base)

    def test_missing_eligible_shuttle_does_not_invent_shot_labels(self):
        dataset=self.fixture.root/'data.json';dataset.write_text(json.dumps(synthetic_dataset()));model=self.fixture.root/'model'
        train_file(dataset,model,allow_development=True)
        output=self.fixture.root/'classified';propose(self.fixture.output,self.base,model,output);packet=verify(self.fixture.output,output)
        self.assertEqual(packet['automatic_shot_model']['status'],'not_available')
        self.assertEqual(packet['document']['events'],[])
        self.assertIsNone(packet['reviewer_kind'])
        report=json.loads((output/'report.json').read_text());self.assertEqual(report['coaching_metrics']['status'],'not_available')

    def test_unreviewed_packet_cannot_be_exported_as_human_training_labels(self):
        output=self.fixture.root/'examples.json'
        with self.assertRaisesRegex(ValueError,'Reviewed labels'):export_examples(self.fixture.output,self.base,output)
        self.assertFalse(output.exists())

    def test_class_proposals_preserve_accepted_classes_outcomes_and_reviewer_provenance(self):
        document=example(frames=(1,3,5,7),end=9,expected=2)
        document['events'][2].update(shot_family='unknown',shot_status='unknown')
        correction={'schema_version':'2.0.0','base_sha256':sha256_file(self.base/'analysis.json'),
                    'reviewer':'Synthetic automated fixture','reviewer_kind':'automated','document':document}
        path=self.fixture.root/'correction.json';path.write_text(json.dumps(correction));reviewed=self.fixture.root/'reviewed'
        apply(self.fixture.output,self.base,path,reviewed)
        dataset=self.fixture.root/'data.json';dataset.write_text(json.dumps(synthetic_dataset()));model=self.fixture.root/'model';train_file(dataset,model,allow_development=True)
        context=(observations(),[],{'video':{'width_px':100,'height_px':80}},set(range(10)))
        output=self.fixture.root/'proposals'
        with patch('shot_model.bound_shuttle',return_value=context):propose(self.fixture.output,reviewed,model,output)
        result=verify(self.fixture.output,output)
        self.assertEqual(result['reviewer_kind'],'automated')
        self.assertEqual(result['document']['outcomes'],document['outcomes'])
        self.assertEqual(result['document']['events'][0],document['events'][0])
        self.assertIn(result['document']['events'][2]['shot_status'],('unknown','proposed'))
        self.assertIn('e3',result['automatic_shot_model']['predictions'])

    def test_reviewed_feature_export_records_skips_and_needs_explicit_development_flag(self):
        document=example(frames=(1,3,5,7),end=9,expected=2)
        correction={'schema_version':'2.0.0','base_sha256':sha256_file(self.base/'analysis.json'),
                    'reviewer':'Synthetic automated fixture','reviewer_kind':'automated','document':document}
        path=self.fixture.root/'correction.json';path.write_text(json.dumps(correction));reviewed=self.fixture.root/'reviewed'
        apply(self.fixture.output,self.base,path,reviewed)
        context=(observations(),[],{'video':{'width_px':100,'height_px':80}},set(range(10)))
        output=self.fixture.root/'features.json'
        with patch('shot_model.bound_shuttle',return_value=context):export_examples(self.fixture.output,reviewed,output,allow_development=True)
        data=json.loads(output.read_text());self.assertEqual(len(data['records']),3)
        self.assertEqual(data['skipped']['missing_observed_context'],1)
        self.assertTrue(all(r['reviewer_kind']=='automated' for r in data['records']))


if __name__=='__main__':unittest.main()
