"""Observed-only shuttle recovery, conservative shot cues and large-map review."""
from __future__ import annotations

import argparse
from copy import deepcopy
import itertools
import json
import math
from pathlib import Path
import shutil

import numpy as np
import cv2

from badminton_proto import sha256_file
from broadcast_analysis import body_core, project, render_stage, write_json


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def read_rows(path):return [json.loads(l) for l in path.read_text().splitlines()]


def validate_stage(root, stage):
    metadata=read(root/f'{stage}_binding.json')
    for name,expected in metadata['artifacts_sha256'].items():
        target=(root/name).resolve()
        if not target.is_relative_to(root.resolve()) or sha256_file(target)!=expected:
            raise ValueError('Stage artifact differs from its binding')
    return metadata


def model_point(point,crop):
    a,b,c,d=crop
    return np.array([(point[0]-a)*512/(c-a),(point[1]-b)*288/(d-b)],float)


def clean_baseline(rows,court,crop,video=None):
    """Withhold source graphics and camera-stationary runs, retaining evidence."""
    result=deepcopy(rows);references={};removed=[]
    for c in court:
        if c['pixel_to_court'] is not None and c['scene_id'] not in references:references[c['scene_id']]=np.linalg.inv(c['pixel_to_court'])
    def reject(indices,reason):
        for i in indices:
            if result[i]['proposal']:
                result[i]['withheld_baseline_proposal']=result[i]['proposal'];result[i]['proposal']=None
                result[i]['status']='unobserved';result[i]['withheld_reason']=reason;removed.append(i)
    capture=cv2.VideoCapture(str(video)) if video else None
    def source_frame(index):
        capture.set(cv2.CAP_PROP_POS_FRAMES,index*2)
        ok,frame=capture.read()
        return frame if ok else None
    def actual_scoreboard(point,index):
        if capture is None:return False
        frame=source_frame(index)
        if frame is None:return False
        hsv=cv2.cvtColor(frame[350:470,:680],cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,np.array([0,0,100]),np.array([179,65,255]))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,15),np.uint8))
        _,_,stats,_=cv2.connectedComponentsWithStats(mask)
        x,y=point
        return any(w>=60 and h>=12 and area>=600 and a<=x<=a+w and b+350<=y<=b+h+350 for a,b,w,h,area in stats[1:])
    def background_present(start,end):
        if capture is None:return False
        middle=(start+end)//2;point=result[middle]['proposal']['position_px']
        matrix=court[middle//2]['pixel_to_court']
        if matrix is None:return False
        canonical=project([point],matrix)[0]
        for index in [start-15,end+15]:
            if not 0<=index<len(rows) or rows[index]['scene_id']!=rows[middle]['scene_id']:return False
            target=court[index//2]['pixel_to_court']
            if target is None:return False
            x,y=np.rint(project([canonical],np.linalg.inv(target))[0]).astype(int)
            frame=source_frame(index)
            if frame is None or not 8<=x<frame.shape[1]-8 or not 8<=y<frame.shape[0]-8:return False
            hsv=cv2.cvtColor(frame[y-7:y+8,x-7:x+8],cv2.COLOR_BGR2HSV)
            if np.count_nonzero(cv2.inRange(hsv,np.array([0,0,155]),np.array([179,75,255])))<6:return False
        return True
    for i,row in enumerate(result):
        if row['proposal']:
            x,y=row['proposal']['position_px']
            if 0<=x<=680 and 350<=y<=470 and actual_scoreboard([x,y],i):reject([i],'occluding_source_scoreboard')
    i=0
    while i<len(result):
        if not result[i]['proposal']:i+=1;continue
        start=i;segment=result[i]['proposal']['proposal_segment_id'];scene=result[i]['scene_id']
        while i<len(result) and result[i]['proposal'] and result[i]['scene_id']==scene and result[i]['proposal']['proposal_segment_id']==segment:i+=1
        if i-start<8:continue
        stable=[]
        for k in range(start,i):
            matrix=court[k//2]['pixel_to_court']
            if matrix is None:break
            point=project([result[k]['proposal']['position_px']],references[scene]@matrix)[0]
            stable.append(model_point(point,crop))
        if len(stable)==i-start and max(np.linalg.norm(a-b) for a in stable for b in stable)<8 and background_present(start,i-1):
            reject(range(start,i),'camera_stationary_texture_present_before_and_after')
    if capture is not None:capture.release()
    return result,sorted(removed)


def recover_observed_gaps(baseline, trial, players, crop, max_gap=6, court=None):
    """Recover only fully observed short paths between existing observations.

    A linear endpoint guide gates actual candidates; it never supplies an output
    coordinate. No unobserved point, cut crossing or open-ended extrapolation.
    """
    if len(baseline)!=len(trial):raise ValueError('Different shuttle timebases')
    result=deepcopy(baseline);recoveries=[];i=0;references={}
    if court:
        for row in court:
            if row['pixel_to_court'] is not None and row['scene_id'] not in references:
                references[row['scene_id']]=np.linalg.inv(row['pixel_to_court'])
    while i<len(result):
        if baseline[i]['status']!='unobserved':i+=1;continue
        start=i
        while i<len(result) and baseline[i]['status']=='unobserved':i+=1
        end=i-1;length=i-start
        if start==0 or i==len(result) or length>max_gap:continue
        left,right=baseline[start-1],baseline[i]
        if not left['proposal'] or not right['proposal']:continue
        scene=left['scene_id']
        if any(baseline[k]['scene_id']!=scene or trial[k]['scene_id']!=scene or trial[k]['status']=='excluded_view_change'
               for k in range(start,i+1)):continue
        p0=model_point(left['proposal']['position_px'],crop);p1=model_point(right['proposal']['position_px'],crop)
        options=[]
        for k in range(start,i):
            if baseline[k]['frame_index']!=trial[k]['frame_index'] or abs(baseline[k]['timestamp_seconds']-trial[k]['timestamp_seconds'])>1e-6:
                raise ValueError('Different shuttle indices/timestamps')
            people=players[min(len(players)-1,int(baseline[k]['timestamp_seconds']*15+1e-7))]['persons']
            guide=p0+(p1-p0)*(k-start+1)/(length+1)
            candidates=[]
            for candidate in trial[k]['candidates']:
                if candidate['heatmap_peak']<.35 or body_core(candidate['center_px'],people):continue
                # This source's scoreboard is an occluding graphic, not flight evidence.
                cx,cy=candidate['center_px']
                if 0<=cx<=680 and 350<=cy<=470:continue
                point=model_point(candidate['center_px'],crop)
                distance=float(np.linalg.norm(point-guide))
                if distance<=6+length*1.5:
                    candidates.append((distance+5*(1-candidate['heatmap_peak']),candidate,point))
            options.append(sorted(candidates,key=lambda c:c[0])[:4])
        if any(not group for group in options):continue
        paths=[]
        for path in itertools.product(*options):
            points=[p0]+[c[2] for c in path]+[p1]
            steps=np.diff(points,axis=0)
            if any(np.linalg.norm(step)>60 for step in steps):continue
            native_points=[left['proposal']['position_px']]+[c[1]['center_px'] for c in path]+[right['proposal']['position_px']]
            stabilized=[]
            for k,point in zip(range(start-1,i+1),native_points):
                mapping=court[k//2]['pixel_to_court'] if court else None
                if court and mapping is None:break
                point=project([point],references[scene]@mapping)[0] if court else point
                stabilized.append(model_point(point,crop))
            if len(stabilized)!=len(native_points):continue
            # Ads, court marks and scoreboard glyphs must not fill a gap merely
            # because the broadcast camera pans. A slow/ambiguous shuttle stays unknown.
            if max(np.linalg.norm(a-b) for a in stabilized for b in stabilized)<8:continue
            acceleration=float(np.linalg.norm(np.diff(steps,axis=0),axis=1).mean())
            paths.append((sum(c[0] for c in path)+.35*acceleration,path))
        paths.sort(key=lambda p:p[0])
        if not paths or len(paths)>1 and paths[1][0]-paths[0][0]<6:continue
        for k,choice in zip(range(start,i),paths[0][1]):
            row=result[k];candidate=choice[1]
            row['baseline_status']=row['status'];row['status']='proposal'
            row['recovered_candidate']=candidate
            row['proposal']={'position_px':candidate['center_px'],'verification_status':'unverified',
                             'selection_method':'two_sided_observed_candidate_path','proposal_segment_id':-1}
        recoveries.append({'start_frame':start,'end_frame':end,'samples':length,'status':'observed_candidates_only'})
    segment=-1;previous=None
    for row in result:
        if row['proposal']:
            if previous is None or not previous['proposal'] or previous['scene_id']!=row['scene_id']:segment+=1
            row['proposal']['proposal_segment_id']=segment
            row['proposal']['observation_source']='lower_threshold_recovery' if row.get('recovered_candidate') else 'baseline'
        previous=row
    return result,recoveries


def identify_hitter(point, persons):
    """Unique image-space body proximity, with a margin; never a roster identity."""
    ranked=[]
    for person in persons:
        if person['track_id'] not in (0,1):continue
        a,b,c,d=person['bbox_xyxy_px'];w,h=c-a,d-b;x,y=point
        dx=max(a-x,0,x-c);dy=max(b-y,0,y-d)
        distance=math.hypot(dx,dy)/max(1,.5*(w+h))
        if distance<=.65:ranked.append((distance,person))
    ranked.sort(key=lambda p:p[0])
    if not ranked or len(ranked)>1 and ranked[1][0]-ranked[0][0]<.18:return None
    return ranked[0][1]


def recognize_shot(point, box, following, fps=30):
    """Image-space shot hypotheses with an explicit unknown outcome."""
    if len(following)<5:return 'Unknown shot','less than five uninterrupted outgoing observations'
    a,b,c,d=box;height=max(1,d-b)
    relative=(d-point[1])/height
    points=np.array([point,*following],float)
    rise=float(point[1]-points[:,1].min())
    outgoing=points[min(5,len(points)-1)]-points[0]
    step=float(np.median(np.linalg.norm(np.diff(points[:7],axis=0),axis=1)))
    if relative<.75 and rise>.45*height:
        return 'Lift?','low contact cue followed by a substantial rise'
    if relative>1.05 and rise>.45*height:
        return 'Clear?','above-head contact cue followed by a substantial rise'
    if relative>1.05 and outgoing[1]>.55*height and step>.12*height:
        return 'Smash?','above-head cue and rapid descending observed flight'
    if relative>1.05 and outgoing[1]>.25*height and step<.10*height:
        return 'Drop?','above-head cue and short descending observed flight'
    if .4<relative<1.05 and abs(outgoing[0])>.6*height and abs(outgoing[1])<.45*abs(outgoing[0]):
        return 'Drive?','middle-height cue and predominantly lateral observed flight'
    if relative<.85 and np.linalg.norm(outgoing)<.45*height:
        return 'Net / block?','low cue and limited observed outgoing travel'
    return 'Unknown shot','image-space evidence does not resolve the shot type'


def refine_events(rows, players, court):
    from contact_proposals import find_contacts
    eligible={i for i in range(len(rows)) if court[i//2]['status']=='experimental_court'}
    cues=find_contacts(rows,players,30,15,(1080,1920),eligible)
    # Multiple turns within one short contact window are one hypothesis.
    selected=[]
    for cue in sorted(cues,key=lambda c:-c['angle_degrees']):
        if not any(s['scene_id']==cue['scene_id'] and abs(s['frame_index']-cue['frame_index'])<8 for s in selected):selected.append(cue)
    events=[]
    for cue in sorted(selected,key=lambda c:c['frame_index']):
        i=cue['frame_index'];row=rows[i];point=row['proposal']['position_px']
        person=identify_hitter(point,players[i//2]['persons'])
        following=[]
        for future in rows[i+1:i+16]:
            if not future['proposal'] or future['scene_id']!=row['scene_id'] or future['proposal']['proposal_segment_id']!=row['proposal']['proposal_segment_id']:break
            following.append(future['proposal']['position_px'])
        label,reason=recognize_shot(point,person['bbox_xyxy_px'],following) if person else ('Unknown shot','ambiguous hitter proximity')
        events.append({'timestamp_seconds':i/30,'native_frame_index':i*2,'scene_id':row['scene_id'],
                       'contact_status':'proposed_not_verified','temporary_hitter_id':person['track_id'] if person else None,
                       'shot_hint':label,'evidence':reason,'confidence':'unvalidated_heuristic','cue':cue})
    return {'status':'experimental_contextual_shot_hypotheses','events':events,'verified_shots':0,
            'winner':None,'forced_errors':None,'coaching':None,
            'limitations':['Names ending in ? are hypotheses, not trained or verified classifications.',
                           'Perspective and racket/pose uncertainty remain; fewer unknowns alone is not an accuracy improvement.']}


def viewer(output, court, events):
    template=(Path(__file__).parent/'broadcast_viewer.html').read_text(encoding='utf-8')
    payload=json.dumps({'court':court,'events':events['events'],'lines':__import__('broadcast_analysis').court_lines(),
                        'boundaries':read(output/'render_summary.json')['native_view_boundaries']},allow_nan=False).replace('<','\\u003c')
    (output/'index.html').write_text(template.replace('/*BROADCAST_DATA*/',payload),encoding='utf-8')
    shutil.copyfile(Path(__file__).parent/'broadcast_viewer.js',output/'broadcast_viewer.js')


def run(baseline,trial,job,output):
    if output.exists():raise ValueError('Choose a fresh refinement output directory')
    for stage in ['court','shuttle','relink','events','render']:validate_stage(baseline,stage)
    validate_stage(trial,'court');validate_stage(trial,'shuttle')
    base_summary=read(baseline/'shuttle_summary.json');trial_summary=read(trial/'shuttle_summary.json')
    if base_summary['source_sha256']!=trial_summary['source_sha256'] or base_summary['model']!=trial_summary['model']:
        raise ValueError('Trial recording/model differs')
    left=deepcopy(base_summary['configuration']);right=deepcopy(trial_summary['configuration'])
    left.pop('heatmap_threshold');right.pop('heatmap_threshold')
    left.pop('window_stride',None);right.pop('window_stride',None)
    if left!=right:raise ValueError('Trial geometry or sampling differs')
    original=read(job/'job.json');result=read(job/'result.json');video=Path(result['source']['path'])
    if sha256_file(video)!=base_summary['source_sha256']:raise ValueError('Video changed')
    player_path=job/result['artifacts']['observations']
    if sha256_file(player_path)!=original['stages']['players']['artifacts']['player_tracks.jsonl']:raise ValueError('Players changed')
    players=read_rows(player_path);court=read_rows(baseline/'court_frames.jsonl');config=read(baseline/'court_summary.json')['configuration']
    cleaned,withheld=clean_baseline(read_rows(baseline/'shuttle_filtered_frames.jsonl'),court,config['shuttle_crop_xyxy_px'],video)
    refined,recoveries=recover_observed_gaps(cleaned,read_rows(trial/'shuttle_frames.jsonl'),players,config['shuttle_crop_xyxy_px'],court=court)
    output.mkdir()
    for name in ['court_frames.jsonl','court_summary.json','shuttle_summary.json']:shutil.copyfile(baseline/name,output/name)
    (output/'shuttle_filtered_frames.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in refined),encoding='utf-8')
    events=refine_events(refined,players,court);write_json(output/'events.json',events)
    write_json(output/'refinement.json',{'source_sha256':base_summary['source_sha256'],'baseline_directory':str(baseline),'trial_directory':str(trial),
               'baseline_rows_sha256':sha256_file(baseline/'shuttle_filtered_frames.jsonl'),'trial_rows_sha256':sha256_file(trial/'shuttle_frames.jsonl'),
               'baseline_observations':sum(bool(r['proposal']) for r in read_rows(baseline/'shuttle_filtered_frames.jsonl')),
               'refined_observations':sum(bool(r['proposal']) for r in refined),'recoveries':recoveries,
               'withheld_baseline_frames':withheld,'withheld_baseline_samples':len(withheld),
               'interpolated_positions':0,'verified_positions':0,'player_review_modified':False,
               'method':'short two-sided candidate paths from an explicit lower-cutoff model trial; baseline observations preserved'})
    render_stage(video,config,output)
    viewer(output,court,events)
    write_json(output/'refinement_binding.json',{'code_sha256':sha256_file(Path(__file__)),
               'artifacts_sha256':{n:sha256_file(output/n) for n in ['refinement.json','court_frames.jsonl','shuttle_filtered_frames.jsonl','events.json','render_summary.json','Axelsen-Lee-All-Overlays.mp4','index.html','broadcast_viewer.js']}})
    print(str(output/'index.html'),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','trial','job','out']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();run(args.baseline,args.trial,args.job,args.out)


if __name__=='__main__':main()
