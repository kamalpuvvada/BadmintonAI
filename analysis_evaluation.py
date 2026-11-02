"""Compare source-aligned event ledgers; metrics never approve model release."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from match_analysis import require,validate_document,verify


def match_contacts(predictions,truth,tolerance=2):
    """Order-preserving one-to-one matching, maximizing count then minimizing timing error."""
    p=sorted(predictions,key=lambda e:(e['start_frame']+e['end_frame'])/2)
    t=sorted(truth,key=lambda e:(e['start_frame']+e['end_frame'])/2)
    n,m=len(p),len(t)
    require(n*m<=2_000_000,'Split large annotation segments into rallies before event matching')
    counts=np.zeros((n+1,m+1),dtype=np.int32);cost=np.zeros((n+1,m+1));steps=np.zeros((n+1,m+1),dtype=np.uint8)
    for i in range(1,n+1):
        for j in range(1,m+1):
            candidates=[(int(counts[i-1,j]),-cost[i-1,j],1),(int(counts[i,j-1]),-cost[i,j-1],2)]
            distance=abs((p[i-1]['start_frame']+p[i-1]['end_frame']-t[j-1]['start_frame']-t[j-1]['end_frame'])/2)
            if distance<=tolerance and p[i-1]['end_frame']-p[i-1]['start_frame']<=2*tolerance:
                candidates.append((int(counts[i-1,j-1])+1,-cost[i-1,j-1]-distance,3))
            matched,negative,step=max(candidates,key=lambda v:(v[0],v[1],v[2]))
            counts[i,j]=matched;cost[i,j]=-negative;steps[i,j]=step
    pairs=[];i,j=n,m
    while i and j:
        step=steps[i,j]
        if step==3:
            pairs.append((p[i-1],t[j-1]));i-=1;j-=1
        elif step==1:i-=1
        else:j-=1
    return list(reversed(pairs))


def agreement(pairs):
    if not pairs:return {'examples':0,'agreement':None,'cohens_kappa':None}
    a=Counter(x for x,y in pairs);b=Counter(y for x,y in pairs);n=len(pairs)
    observed=sum(x==y for x,y in pairs)/n
    expected=sum(a[label]*b[label] for label in set(a)|set(b))/(n*n)
    return {'examples':n,'agreement':observed,'cohens_kappa':(observed-expected)/(1-expected) if expected<1-1e-12 else None,
            'independent_raters_verified':False}


def compare(prediction,truth,tolerance=2,allow_development=False):
    require(type(tolerance) is int and 0<=tolerance<=30,'Timing tolerance must be 0..30 native frames')
    require(prediction['source_binding']['source_video_sha256']==truth['source_binding']['source_video_sha256']
            and prediction['source']['frames']==truth['source']['frames'] and prediction['source']['fps']==truth['source']['fps'],'Analysis sources/timebases differ')
    require(truth['reviewer_kind']=='human' or (allow_development and truth['reviewer_kind']=='automated'),
            'Reference labels must be reviewed; automated fixtures need development mode')
    pd,td=prediction['document'],truth['document']
    validate_document(pd,prediction['source']);validate_document(td,truth['source'])
    require(pd['match']['teams']==td['match']['teams'],'Player/side slots must be explicitly aligned before comparison')
    require(pd['analysis_rubric']==td['analysis_rubric'],'Error rubrics/skill contexts differ')
    truth_rallies={r['id']:r for r in td['rallies'] if r['kind']!='replay' and r['status']!='rejected'}
    pred_rallies={r['id']:r for r in pd['rallies'] if r['kind']!='replay' and r['status']!='rejected'}
    tp_events=[e for e in td['events'] if e['kind']=='contact' and e['status']=='accepted' and e['rally_id'] in truth_rallies]
    uncertain=[e for e in tp_events if e['end_frame']-e['start_frame']>2*tolerance]
    truth_events=[e for e in tp_events if e not in uncertain]
    pred_events=[e for e in pd['events'] if e['kind']=='contact' and e['status'] in ('accepted','proposed') and e['rally_id'] in pred_rallies]
    pairs=[]
    for rally_id in set(e['rally_id'] for e in truth_events)|set(e['rally_id'] for e in pred_events):
        pairs.extend(match_contacts([e for e in pred_events if e['rally_id']==rally_id],
                                    [e for e in truth_events if e['rally_id']==rally_id],tolerance))
    # Predictions overlapping temporally uncertain reference events are not known
    # negatives. Exclude those from precision, and disclose the exclusion count.
    matched_pred={p['id'] for p,t in pairs}
    unscored_pred=[p for p in pred_events if p['id'] not in matched_pred and any(p['rally_id']==t['rally_id'] and p['start_frame']<=t['end_frame']+tolerance
                      and p['end_frame']>=t['start_frame']-tolerance for t in uncertain)]
    scorable_pred=[p for p in pred_events if p not in unscored_pred]
    n=len(pairs);precision=n/len(scorable_pred) if scorable_pred else None;recall=n/len(truth_events) if truth_events else None
    errors=[abs((p['start_frame']+p['end_frame']-t['start_frame']-t['end_frame'])/2) for p,t in pairs]
    pred_to_truth={p['id']:t['id'] for p,t in pairs}
    truth_to_pred={t['id']:p for p,t in pairs}
    known_hitter=[(p,t) for p,t in pairs if t['player'] is not None]
    confusion=Counter();support=Counter();proposed_counts=Counter();correct=Counter()
    for event in truth_events:
        if event['shot_status']!='accepted':continue
        guess=truth_to_pred.get(event['id'])
        label=guess['shot_family'] if guess and guess['shot_status'] in ('accepted','proposed') else 'unknown'
        confusion[(event['shot_family'],label)]+=1;support[event['shot_family']]+=1
        proposed_counts[label]+=1;correct[event['shot_family']]+=event['shot_family']==label
    for event in scorable_pred:
        if event['id'] not in matched_pred and event['shot_status'] in ('accepted','proposed') and event['shot_family']!='unknown':
            confusion[('unmatched_contact',event['shot_family'])]+=1;proposed_counts[event['shot_family']]+=1
    classes={}
    for label,count in support.items():
        p=correct[label]/proposed_counts[label] if proposed_counts[label] else 0;r=correct[label]/count
        classes[label]={'support':count,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0}
    pred_outcomes={o['rally_id']:o for o in pd['outcomes']}
    points=[];finishes=[];accepted_finish_pairs=[];pressure=[];error_people=[]
    for outcome in td['outcomes']:
        guess=pred_outcomes.get(outcome['rally_id'])
        if outcome['status']=='accepted' and outcome['winner']:
            predicted=guess['winner'] if guess and guess['status'] in ('accepted','proposed') else None
            points.append((predicted,outcome['winner']))
        if outcome['finish_status']=='accepted' and outcome['finish']!='not_applicable':
            predicted=guess['finish'] if guess and guess['finish_status'] in ('accepted','proposed') else 'undetermined'
            finishes.append((predicted,outcome['finish']))
            if guess and guess['finish_status']=='accepted':accepted_finish_pairs.append((predicted,outcome['finish']))
            if outcome['error_player'] is not None:error_people.append((guess['error_player'] if guess else None,outcome['error_player']))
            if outcome['preceding_pressure_event_id']:
                guess_link=pred_to_truth.get(guess['preceding_pressure_event_id']) if guess else None
                pressure.append(guess_link==outcome['preceding_pressure_event_id'])
    false_unscored=sum(1 for rally in td['rallies'] if rally['kind'] in ('let','practice','replay') and rally['status']=='accepted' and rally['id'] in pred_outcomes
                       and pred_outcomes[rally['id']]['winner'] is not None and pred_outcomes[rally['id']]['status'] in ('accepted','proposed'))
    return {'schema_version':'1.0.0','status':'annotation_comparison_only','source_video_sha256':truth['source_binding']['source_video_sha256'],
        'contact_tolerance_native_frames':tolerance,'contact_tolerance_ms':1000*tolerance/truth['source']['fps'],
        'contacts':{'prediction_count':len(pred_events),'reference_count':len(truth_events),'matched':n,'precision':precision,'recall':recall,
                    'reference_uncertain_excluded':len(uncertain),'predictions_unscored_near_uncertainty':len(unscored_pred),
                    'mean_absolute_frame_error':sum(errors)/len(errors) if errors else None,
                    'pairs':[{'prediction':p['id'],'reference':t['id']} for p,t in pairs]},
        'hitter':{'examples':len(known_hitter),'accuracy':sum(p['player']==t['player'] for p,t in known_hitter)/len(known_hitter) if known_hitter else None},
        'shot_classes':{'per_class':classes,'macro_f1':sum(c['f1'] for c in classes.values())/len(classes) if classes else None,
                       'confusion':[{'actual':a,'predicted':p,'count':n} for (a,p),n in sorted(confusion.items())]},
        'point_winner':{'reference_decided_rallies':len(points),'accuracy':sum(p==t for p,t in points)/len(points) if points else None,
                        'prediction_coverage':sum(p is not None for p,t in points)/len(points) if points else None,
                        'false_point_suggestions_on_unscored_rallies':false_unscored},
        'finish_labels':agreement(finishes),'jointly_accepted_finish_agreement':agreement(accepted_finish_pairs),
        'error_player':{'examples':len(error_people),'accuracy':sum(p==t for p,t in error_people)/len(error_people) if error_people else None},
        'preceding_pressure_link':{'examples':len(pressure),'accuracy':sum(pressure)/len(pressure) if pressure else None},
        'release_approved':False,'limitations':['Comparison assumes the supplied reference ledger is complete in its adjudicated intervals; this is not independently verified.',
            'Rally IDs and player/side slots must be aligned; unmatched reference rallies count as missed outcomes/contacts.',
            'Unknown or temporally uncertain reference labels are not absence. Scores do not establish independent coach agreement or academy accuracy.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True);parser.add_argument('--truth-job',type=Path)
    parser.add_argument('--prediction',type=Path,required=True);parser.add_argument('--truth',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--tolerance',type=int,default=2)
    parser.add_argument('--allow-development-labels',action='store_true');args=parser.parse_args()
    try:
        require(not args.out.exists(),'Choose a new evaluation output')
        result=compare(verify(args.job,args.prediction),verify(args.truth_job or args.job,args.truth),args.tolerance,args.allow_development_labels)
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(args.out)
    except (ValueError,OSError,KeyError,TypeError) as exc:parser.exit(1,f'Evaluation error: {exc}\n')


if __name__=='__main__':main()
