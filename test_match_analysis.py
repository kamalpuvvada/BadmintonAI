from copy import deepcopy
import json
from pathlib import Path
import unittest

import test_session_runner as fixtures
from match_analysis import (empty_document,new_rally,new_event,new_outcome,validate_document,
                            summarize,score_timeline,prepare,apply,verify)
from match_analysis import suggest_outcomes

SOURCE={'frames':300,'fps':30,'expected_players':4}


def example(frames=(10,30,50,70),end=99,expected=4):
    doc=empty_document();doc['match']['teams']={'A':['P1','P2'],'B':['P3','P4']} if expected==4 else {'A':['P1'],'B':['P2']}
    doc['analysis_rubric']['skill_context']='Synthetic fixture; no actual coach/player judgment'
    doc['match']['initial_scores']={'1':[0,0]}
    rally=new_rally('r1',0,end);rally['status']='accepted';doc['rallies']=[rally]
    for i,(frame,side,family) in enumerate(zip(frames,['A','B','A','B'],['serve','net','smash','block'])):
        event=new_event(f'e{i+1}','r1',frame);event.update(player='P1' if side=='A' else ('P3' if expected==4 else 'P2'),
            side=side,status='accepted',shot_family=family,shot_status='accepted',execution='net' if i==3 else 'continued',
            evidence=[{'kind':'video','start_frame':frame,'end_frame':frame,'note':'Synthetic test annotation'}])
        doc['events'].append(event)
    outcome=new_outcome('r1');outcome.update(status='accepted',winner='A',terminal_cause='net_failure',terminal_status='accepted',
        finish='forced_error',finish_status='accepted',error_player='P3' if expected==4 else 'P2',last_contact_event_id='e4',
        terminal_error_event_id='e4',preceding_pressure_event_id='e3',setup_sequence_event_ids=['e1'],
        score_before=[0,0],score_after=[1,0],evidence=[{'kind':'manual','start_frame':None,'end_frame':None,'note':'Synthetic adjudicated outcome'}])
    doc['outcomes']=[outcome];return doc


class LedgerTests(unittest.TestCase):
    def test_pressure_shot_terminal_error_and_point_are_distinct(self):
        doc=example();report=summarize(doc,SOURCE,'automated')
        self.assertEqual(report['accepted_rally_points'],{'A':1})
        self.assertEqual(report['finish_counts'],{'forced_error':1})
        self.assertEqual(report['induced_forced_error_event_ids'],{'e3':1})
        self.assertEqual(report['terminal_error_rates']['block'],{'numerator':1,'denominator':1,'rate':1})
        self.assertEqual(report['direct_winner_rates']['smash']['numerator'],0)
        self.assertEqual(report['score']['latest_by_game']['1'],[1,0])
        self.assertEqual(report['coaching_metrics']['status'],'not_available')
        self.assertEqual(report['rally_results'][0]['point_winner'],'A')
        self.assertEqual(report['rally_results'][0]['terminal_error']['shot_family'],'block')
        self.assertEqual(report['rally_results'][0]['preceding_pressure_shot']['shot_family'],'smash')
        self.assertIsNone(report['rally_results'][0]['direct_winning_shot'])
        doc['events'][2]['shot_family']='drop';self.assertEqual(summarize(doc,SOURCE,'human')['finish_counts'],{'forced_error':1})

    def test_unknown_finish_does_not_erase_known_point_or_infer_forcedness(self):
        doc=example();doc['outcomes'][0]['finish_status']='proposed'
        report=summarize(doc,SOURCE,'human');self.assertEqual(report['finish_counts'],{})
        self.assertEqual(report['accepted_rally_points'],{'A':1})
        self.assertEqual(report['induced_forced_error_event_ids'],{})

    def test_forcedness_needs_explicit_skill_context_rubric(self):
        doc=example();doc['analysis_rubric']['skill_context']='unspecified'
        with self.assertRaisesRegex(ValueError,'skill/context rubric'):validate_document(doc,SOURCE)

    def test_direct_winner_and_attempt_have_separate_semantics(self):
        doc=example();doc['events'].pop();o=doc['outcomes'][0]
        o.update(finish='direct_winner',last_contact_event_id='e3',direct_winner_event_id='e3',terminal_error_event_id=None,preceding_pressure_event_id=None,error_player=None)
        self.assertEqual(summarize(doc,SOURCE,'human')['direct_winner_rates']['smash']['numerator'],1)
        doc=example();doc['events'][-1].update(kind='attempt',execution='miss')
        doc['outcomes'][0]['last_contact_event_id']='e3'
        self.assertEqual(summarize(doc,SOURCE,'human')['coverage']['accepted_attempts'],1)
        self.assertNotIn('block',summarize(doc,SOURCE,'human')['shot_counts'])
        self.assertEqual(summarize(doc,SOURCE,'human')['intended_attempt_counts'],{'block':1})
        doc['events'][-1]['kind']='contact'
        with self.assertRaisesRegex(ValueError,'must be an attempt'):validate_document(doc,SOURCE)

    def test_lets_replays_and_repeated_shots_do_not_duplicate_points(self):
        doc=example();extra=deepcopy(doc['events'][2]);extra.update(id='e-extra',start_frame=40,end_frame=40)
        doc['events'].append(extra)
        report=summarize(doc,SOURCE,'human');rate=next(r for r in report['rally_win_rates_after_shot'] if r['side']=='A' and r['shot_family']=='smash')
        self.assertEqual(rate['denominator'],1);self.assertEqual(report['shot_counts']['smash'],2)
        let=new_rally('let1',100,120);let.update(kind='let',status='accepted');doc['rallies'].append(let)
        outcome=new_outcome('let1');outcome.update(status='accepted',finish='not_applicable',score_before=[1,0],score_after=[1,0]);doc['outcomes'].append(outcome)
        replay=new_rally('replay',130,160);replay.update(kind='replay',status='accepted');doc['rallies'].append(replay)
        copied=deepcopy(doc['events'][0]);copied.update(id='replay-contact',rally_id='replay',start_frame=140,end_frame=140)
        copied['evidence'][0].update(start_frame=140,end_frame=140);doc['events'].append(copied)
        report=summarize(doc,SOURCE,'human');self.assertEqual(report['score']['latest_by_game']['1'],[1,0])
        self.assertEqual(report['shot_counts']['serve'],1)
        self.assertEqual(report['accepted_rally_points'],{'A':1})

    def test_unknown_rally_breaks_score_until_explicit_anchor(self):
        doc=example();doc['rallies'].append(new_rally('missing',100,190));doc['rallies'].append(new_rally('r3',200,290))
        outcome=new_outcome('r3');outcome.update(status='accepted',winner='B',evidence=[{'kind':'official','start_frame':None,'end_frame':None,'note':'Confirmed scorekeeper entry'}])
        doc['outcomes'].append(outcome)
        self.assertIsNone(summarize(doc,SOURCE,'human')['score']['latest_by_game']['1'])
        outcome.update(score_before=[3,2],score_after=[3,3]);self.assertEqual(summarize(doc,SOURCE,'human')['score']['latest_by_game']['1'],[3,3])
        outcome['score_after']=[3,4]
        with self.assertRaisesRegex(ValueError,'Score-after'):validate_document(doc,SOURCE)

    def test_unconfirmed_no_point_event_is_not_silently_assumed_to_award_no_point(self):
        for kind in ('let','replay','practice'):
            doc=example();segment=new_rally('possible-no-point',100,150);segment['kind']=kind;doc['rallies'].append(segment)
            self.assertIsNone(summarize(doc,SOURCE,'human')['score']['latest_by_game']['1'])
            segment['status']='accepted'
            self.assertEqual(summarize(doc,SOURCE,'human')['score']['latest_by_game']['1'],[1,0])

    def test_custom_format_deuce_cap_and_post_game_point(self):
        doc=example();doc['match']['rules']={'profile_id':'synthetic_first_to_3','target_points':3,'win_by':2,'cap_points':5,'best_of_games':3}
        doc['match']['initial_scores']={'1':[4,4]};doc['outcomes'][0].update(score_before=[4,4],score_after=[5,4])
        report=summarize(doc,SOURCE,'human');self.assertEqual(report['score']['completed_games'],{'1':'A'})
        doc['rallies'].append(new_rally('later',100,150))
        with self.assertRaisesRegex(ValueError,'completed game'):validate_document(doc,SOURCE)

    def test_match_ends_after_required_known_game_wins(self):
        doc=empty_document();doc['match']['rules']={'profile_id':'synthetic_short_match','target_points':1,'win_by':1,'cap_points':1,'best_of_games':3}
        doc['match']['initial_scores']={'1':[0,0],'2':[0,0],'3':[0,0]}
        for n in (1,2):
            rally=new_rally(f'r{n}',(n-1)*100,n*100-1);rally['game']=n;doc['rallies'].append(rally)
            outcome=new_outcome(rally['id']);outcome.update(status='accepted',winner='A',evidence=[{'kind':'manual','start_frame':None,'end_frame':None,'note':'Synthetic scorekeeper point'}]);doc['outcomes'].append(outcome)
        self.assertEqual(summarize(doc,SOURCE,'automated')['score']['match_winner_from_annotated_games'],'A')
        extra=new_rally('r3',200,299);extra['game']=3;doc['rallies'].append(extra)
        with self.assertRaisesRegex(ValueError,'completed match'):validate_document(doc,SOURCE)

    def test_invalid_identity_links_order_evidence_and_duplicate_outcomes_fail(self):
        mutations=[lambda d:d['events'][0].update(side='B'),lambda d:d['events'][0].update(evidence=[]),
          lambda d:d['outcomes'][0].update(last_contact_event_id='e3'),lambda d:d['outcomes'][0].update(error_player='P1'),
          lambda d:d['outcomes'][0].update(preceding_pressure_event_id='e1'),lambda d:d['outcomes'].append(deepcopy(d['outcomes'][0])),
          lambda d:d['events'][1].update(start_frame=10,end_frame=10),lambda d:d['events'][0].update(start_frame=True),
          lambda d:d['outcomes'][0].update(direct_winner_event_id='e3'),lambda d:d['outcomes'][0].update(setup_sequence_event_ids=['missing']),
          lambda d:d['rallies'][0].update(status='rejected')]
        for change in mutations:
            doc=example();change(doc)
            with self.subTest(change=change),self.assertRaises(ValueError):validate_document(doc,SOURCE)

    def test_administrative_points_are_not_fictional_shots_and_ambiguous_times_fail(self):
        doc=example();doc['score_events']=[{'id':'penalty','game':1,'frame':150,'side':'B','status':'accepted','reason':'Synthetic official penalty',
                                          'evidence':[{'kind':'official','start_frame':150,'end_frame':150,'note':'Synthetic official event'}]}]
        report=summarize(doc,SOURCE,'human');self.assertEqual(report['score']['latest_by_game']['1'],[1,1])
        self.assertEqual(report['coverage']['accepted_contacts'],4)
        self.assertEqual(report['accepted_administrative_points'],{'B':1})
        doc['score_events'][0]['status']='proposed';self.assertIsNone(summarize(doc,SOURCE,'human')['score']['latest_by_game']['1'])
        doc['score_events'][0].update(status='accepted',frame=99)
        with self.assertRaisesRegex(ValueError,'share a time'):validate_document(doc,SOURCE)

    def test_terminal_tag_can_only_suggest_point_and_never_forcedness(self):
        doc=example();doc['outcomes']=[]
        proposed,suggestions=suggest_outcomes(doc);outcome=proposed['outcomes'][0]
        self.assertEqual(outcome['winner'],'A');self.assertEqual(outcome['status'],'proposed')
        self.assertEqual(outcome['finish'],'undetermined');self.assertIsNone(outcome['preceding_pressure_event_id'])
        self.assertEqual(summarize(proposed,SOURCE,None)['accepted_rally_points'],{})
        accepted=example();preserved,cues=suggest_outcomes(accepted)
        self.assertEqual(preserved,accepted);self.assertEqual(cues,[])
        for mode in ('fragment','practice','unknown_execution'):
            changed=deepcopy(doc)
            if mode=='fragment':changed['rallies'][0]['coverage']='fragment'
            elif mode=='practice':changed['rallies'][0]['kind']='practice'
            else:changed['events'][-1]['execution']='unknown'
            self.assertEqual(suggest_outcomes(changed)[1],[])


class LedgerPacketTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.SessionTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups);self.fixture.run_fake()
        self.base=self.fixture.root/'analysis';prepare(self.fixture.output,self.base)

    def test_blank_actual_packet_has_no_guessed_events_or_scores(self):
        packet=verify(self.fixture.output,self.base)
        self.assertEqual(packet['revision'],0);self.assertEqual(packet['document']['events'],[])
        report=json.loads((self.base/'report.json').read_text());self.assertEqual(report['status'],'no_accepted_event_annotations')
        self.assertEqual(report['score']['latest_by_game'],{})

    def test_apply_history_integrity_provenance_and_report_reproduction(self):
        from badminton_proto import sha256_file
        patch={'schema_version':'2.0.0','base_sha256':sha256_file(self.base/'analysis.json'),'reviewer':'Synthetic reviewer','reviewer_kind':'automated',
               'document':example(frames=(1,3,5,7),end=9,expected=2)}
        path=self.fixture.root/'patch.json';path.write_text(json.dumps(patch));out=self.fixture.root/'reviewed'
        apply(self.fixture.output,self.base,path,out);packet=verify(self.fixture.output,out)
        self.assertEqual(packet['revision'],1);self.assertEqual(packet['reviewer_kind'],'automated')
        self.assertEqual((out/'parent_analysis.json').read_bytes(),(self.base/'analysis.json').read_bytes())
        self.assertEqual(json.loads((out/'report.json').read_text())['accepted_rally_points'],{'A':1})
        (out/'report.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'modified'):verify(self.fixture.output,out)

    def test_stale_patch_or_invalid_doc_creates_no_result(self):
        from badminton_proto import sha256_file
        patch={'schema_version':'2.0.0','base_sha256':'0'*64,'reviewer':'Fixture','reviewer_kind':'automated','document':empty_document()}
        path=self.fixture.root/'patch.json';out=self.fixture.root/'invalid';path.write_text(json.dumps(patch))
        with self.assertRaisesRegex(ValueError,'stale'):apply(self.fixture.output,self.base,path,out)
        self.assertFalse(out.exists());patch['base_sha256']=sha256_file(self.base/'analysis.json');patch['document']['rallies']=[new_rally('r1',0,1000)];path.write_text(json.dumps(patch))
        with self.assertRaises(ValueError):apply(self.fixture.output,self.base,path,out)
        self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
