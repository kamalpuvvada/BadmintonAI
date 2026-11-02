"""Experimental contact cues from existing reviewed-eligible shuttle observations.

Direction change plus player proximity is a hypothesis, not verified contact,
shot class, point result, or error attribution. No model inference is run here.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from badminton_proto import sha256_file
from player_identity import source_snapshot, _local
from match_analysis import initial_packet,publish,new_rally,new_event,require
from session_runner import artifact_binding


def find_contacts(rows,player_rows,fps,sample_fps,image_size,eligible_frames):
    require(type(fps) in (int,float) and math.isfinite(fps) and 0<fps<=120,'Invalid native FPS')
    require(type(sample_fps) in (int,float) and math.isfinite(sample_fps) and 0<sample_fps<=fps,'Invalid player sample FPS')
    width,height=image_size
    require(width>0 and height>0,'Invalid image dimensions')
    diagonal=math.hypot(width,height)
    points={}
    for index,row in enumerate(rows):
        require(row['frame_index']==index and type(row['scene_id']) is int and row['scene_id']>=0,'Invalid shuttle frame/scene index')
        require(abs(row['timestamp_seconds']-index/fps)<=1e-4,'Shuttle timestamps differ from the native source grid')
        proposal=row.get('proposal')
        if proposal is None:continue
        point=proposal['position_px'];segment=proposal['proposal_segment_id']
        require(type(segment) is int and segment>=0 and len(point)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in point),'Malformed shuttle proposal')
        require(0<=point[0]<width and 0<=point[1]<height,'Shuttle proposal outside source image')
        if index in eligible_frames and row['status']=='proposal':points[index]=tuple(point)
    proposals=[]
    for index in range(2,len(rows)-2):
        indices=range(index-2,index+3)
        if any(i not in points for i in indices):continue
        scene=rows[index]['scene_id'];segment=rows[index]['proposal']['proposal_segment_id']
        if any(rows[i]['scene_id']!=scene or rows[i]['proposal']['proposal_segment_id']!=segment for i in indices):continue
        incoming=tuple((points[index][j]-points[index-2][j])/2 for j in (0,1))
        outgoing=tuple((points[index+2][j]-points[index][j])/2 for j in (0,1))
        speed_in,speed_out=math.hypot(*incoming),math.hypot(*outgoing)
        if min(speed_in,speed_out)<.002*diagonal:continue
        cosine=max(-1,min(1,sum(a*b for a,b in zip(incoming,outgoing))/(speed_in*speed_out)))
        angle=math.degrees(math.acos(cosine))
        if angle<55:continue
        sample_index=math.floor(index/fps*sample_fps+1e-8)
        if not 0<=sample_index<len(player_rows):continue
        x,y=points[index];nearby=[]
        for person in player_rows[sample_index]['persons']:
            left,top,right,bottom=person['bbox_xyxy_px'];w,h=right-left,bottom-top
            if left-.6*w<=x<=right+.6*w and top-.6*h<=y<=bottom+.3*h:
                nearby.append(person['track_id'])
        if not nearby:continue
        proposals.append({'frame_index':index,'scene_id':scene,'proposal_segment_id':segment,
                          'angle_degrees':round(angle,3),'incoming_step_px':round(speed_in,3),
                          'outgoing_step_px':round(speed_out,3),'nearby_track_ids':sorted(set(p for p in nearby if p is not None)),
                          'hitter_status':'unknown; proximity is not athlete identity'})
    selected=[];cooldown=max(1,round(.12*fps))
    for candidate in sorted(proposals,key=lambda p:(-p['angle_degrees'],p['frame_index'])):
        if not any(p['scene_id']==candidate['scene_id'] and p['proposal_segment_id']==candidate['proposal_segment_id']
                   and abs(p['frame_index']-candidate['frame_index'])<cooldown for p in selected):
            selected.append(candidate)
    return sorted(selected,key=lambda p:p['frame_index'])


def bound_shuttle(job_path):
    job,result,summary,players,binding=source_snapshot(job_path)
    stage=job['stages']['shuttle']
    if stage['status']!='completed':return None
    root=_local(job_path,stage['directory'])
    require(artifact_binding(root)==stage['artifacts'],'Shuttle artifacts changed since processing')
    target=_local(job_path,result['artifacts']['shuttle_summary'])
    shuttle=json.loads(target.read_text(encoding='utf-8'))
    player_root=_local(job_path,job['stages']['players']['directory'])
    review=json.loads((player_root/'tracking_review.json').read_text(encoding='utf-8'))
    current_review=sha256(json.dumps(review,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    require(shuttle['source_video_sha256']==binding['source_video_sha256'] and shuttle['player_review_snapshot_sha256']==current_review,
            'Shuttle source/review binding is stale')
    require(shuttle['sampling']['start_seconds']==0 and shuttle['sampling']['fps']==result['preview']['fps'],'Shuttle timebase differs from the job')
    rows=[json.loads(line) for line in (target.parent/'shuttle_candidates.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(rows)==result['preview']['frames'],'Shuttle frame count differs from the native preview')
    intervals=result['players']['review']['approved_intervals']
    # Never trust a saved approval after it has been revoked.
    from tracking_review import verified_snapshot
    verification=verified_snapshot(player_root)[0]
    require(intervals==verification['approved_intervals'],'Public review is stale; resume the job')
    fps,sample_fps=result['preview']['fps'],summary['sampling']['fps']
    eligible={i for i in range(len(rows)) if any(interval['start_frame']<=math.floor(i/fps*sample_fps+1e-8)<=interval['end_frame'] for interval in intervals)}
    return rows,players,summary,eligible


def prepare(job_path,output):
    packet,preview=initial_packet(job_path)
    job,result,summary,players,binding=source_snapshot(job_path)
    info={'status':'not_available','reason':'No current reviewed-eligible shuttle output',
          'method':'five-observation direction change and expanded person-box proximity',
          'proposals':[],'limitations':['Not a trained contact detector; thresholds have no measured academy accuracy.',
                                      'No automatic hitter identity, shot class, score, winner or forcedness.',
                                      'Five consecutive observed points are required; no gaps/cuts/segments bridged.']}
    context=bound_shuttle(job_path)
    if context is not None:
        rows,players,summary,eligible=context
        fps,sample_fps=packet['source']['fps'],summary['sampling']['fps']
        intervals=result['players']['review']['approved_intervals']
        target=_local(job_path,result['artifacts']['shuttle_summary'])
        contacts=find_contacts(rows,players,fps,sample_fps,(summary['video']['width_px'],summary['video']['height_px']),eligible)
        info.update(status='experimental_proposals',reason=None,proposals=contacts,shuttle_summary_sha256=sha256_file(target),
                    thresholds={'turn_angle_min_degrees':55,'minimum_step_fraction_of_image_diagonal':.002,'context_radius_frames':2,'suppression_seconds':.12})
        for n,interval in enumerate(intervals,1):
            start=math.ceil(interval['start_frame']/sample_fps*fps-1e-8)
            end=min(len(rows)-1,math.ceil((interval['end_frame']+1)/sample_fps*fps-1e-8)-1)
            selected=[p for p in contacts if start<=p['frame_index']<=end]
            if not selected:continue
            rally=new_rally(f'candidate-segment-{n}',start,end);rally['coverage']='fragment'
            rally['notes']='Provisional eligible interval, not a detected complete rally. Review boundaries and all events.'
            packet['document']['rallies'].append(rally)
            for contact in selected:
                event=new_event(f'contact-candidate-{contact["frame_index"]}',rally['id'],contact['frame_index'])
                event['notes']='Unverified direction-change/proximity cue; inspect context before accepting contact.'
                event['evidence']=[{'kind':'video','start_frame':contact['frame_index']-2,'end_frame':contact['frame_index']+2,'note':event['notes']}]
                packet['document']['events'].append(event)
    packet['automatic_proposals']=info
    require(source_snapshot(job_path)[4]==binding,'Source changed while proposing contacts')
    return publish(output,packet,preview)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try:print(prepare(args.job,args.out))
    except (ValueError,OSError,KeyError,TypeError) as exc:parser.exit(1,f'Contact proposal error: {exc}\n')


if __name__=='__main__':main()
