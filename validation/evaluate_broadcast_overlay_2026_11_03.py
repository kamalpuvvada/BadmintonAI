"""Recheck local evidence for the source-bound overlay demonstration."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import subprocess

import cv2
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'out/axelsen-lee-analysis-004'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def rows(name):return [json.loads(l) for l in (RUN/name).read_text().splitlines()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


binding=read(RUN/'analysis_binding.json')
stages={stage:read(RUN/f'{stage}_binding.json') for stage in ['court','shuttle','relink','events','render']}
for stage in stages.values():
    assert stage['inputs']==binding
    assert stage['code_sha256']==sha(ROOT/'broadcast_analysis.py')
    for name,expected in stage['artifacts_sha256'].items():assert sha(RUN/name)==expected,name
court,raw,filtered=rows('court_frames.jsonl'),rows('shuttle_frames.jsonl'),rows('shuttle_filtered_frames.jsonl')
summary,shuttle,render=read(RUN/'court_summary.json'),read(RUN/'shuttle_summary.json'),read(RUN/'render_summary.json')
assert len(court)==564 and len(raw)==len(filtered)==1128
assert render['frames']==2256 and render['fps']==60
assert render['native_view_boundaries']==[1216,1269,2131]
assert summary['cut_times_seconds']==[1216/60,1272/60,2132/60]
for row in filtered:
    if row['status']=='excluded_view_change':assert row['proposal'] is None
    assert row['verified_position'] is None
    assert row['candidates']==raw[row['frame_index']]['candidates']
    if row['proposal']:
        point=row['proposal']['position_px']
        assert any(math.dist(point,c['center_px'])<1e-7 for c in row['filtered_candidates'])
        previous=filtered[row['frame_index']-1] if row['frame_index'] else None
        if previous and previous['proposal'] and previous['proposal']['proposal_segment_id']==row['proposal']['proposal_segment_id']:
            assert previous['scene_id']==row['scene_id'] and previous['frame_index']==row['frame_index']-1
            assert previous['status']=='proposal'
seed=read(ROOT/'validation/axelsen_lee_shuttle_source_points_2026-11-03.json')
comparisons={}
for label,data in [('raw_linking',raw),('body_core_linking',filtered)]:
    cases=[]
    for case in seed['cases']:
        proposal=data[case['native_frame_index']//2]['proposal']
        error=math.dist(case['position_px'],proposal['position_px']) if proposal else None
        cases.append(dict(case,error_px=error,within_tolerance=error is not None and error<=seed['native_pixel_tolerance']))
    comparisons[label]={'matched':sum(c['within_tolerance'] for c in cases),'visible_samples':len(cases),'cases':cases}
assert comparisons['raw_linking']['matched']==4
assert comparisons['body_core_linking']['matched']==6
events=read(RUN/'events.json')
assert all(e['contact_status']=='proposed_not_verified' for e in events['events'])
assert events['winner'] is None and events['coaching'] is None and events['verified_shots']==0
original=ROOT/'out/axelsen-lee-session-001';job=read(original/'job.json')
assert sha(original/'players-001/tracking_review.json')==job['stages']['review']['review_sha256']
target=RUN/'Axelsen-Lee-All-Overlays.mp4'
probe=json.loads(subprocess.run(['ffprobe','-v','error','-show_entries','stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames,duration','-of','json',str(target)],capture_output=True,text=True,check=True).stdout)
assert any(s['codec_type']=='audio' for s in probe['streams'])
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(target),'-map','0:a:0','-f','null','-'],check=True)
cap=cv2.VideoCapture(str(target));sheet=Image.new('RGB',(1440,1356),'#111827')
for j,index in enumerate([178,480,960,1224,1440,1662,1920,2190]):
    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read();assert ok
    im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((360,640))
    x,y=j%4*360,j//4*678;sheet.paste(im,(x,y+28))
    ImageDraw.Draw(sheet).text((x+10,y+7),f'{index/60:.2f}s',fill='white')
    if index==178:Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).save(RUN/'all-overlays-example.jpg',quality=95)
cap.release();sheet.save(RUN/'all-overlays-contact-sheet.jpg',quality=94)
report={'schema_version':'0.1.0','date':'2026-11-03','status':'experimental_functional_overlay_not_validated_match_analysis',
        'source_sha256':binding['source_sha256'],'code_sha256':sha(ROOT/'broadcast_analysis.py'),'tests':{'python_passed':170},
        'stages':stages,'court':summary,'shuttle':shuttle,'filtered_shuttle':read(RUN/'shuttle_filtered_summary.json'),
        'render':render,'media_probe':probe,'sparse_source_checks':comparisons,
        'shot_hint_distribution':dict(Counter(e['shot_hint'] for e in events['events'])),
        'contact_hypotheses':len(events['events']),'verified_shots':0,'original_player_review_unchanged':True,
        'remaining':['Continuous full-rally player/shuttle labels and independent precision/recall remain unavailable.',
                     'Six source-point checks are approximate, non-held-out automated annotations; not release accuracy.',
                     'Box-bottom positions include jumps and jitter; no calibrated physical distance or speed.',
                     'Shot names are ambiguous heuristic hints, not trained or verified shot classifications.',
                     'No verified winner, error attribution or coaching score.'],
        'local_video':str(target.relative_to(ROOT))}
(ROOT/'validation/axelsen_lee_overlay_2026-11-03.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'frames':render['frames'],'shuttle_status_counts':report['filtered_shuttle']['status_counts'],
                  'source_checks':{k:v['matched'] for k,v in comparisons.items()},'shot_hints':report['shot_hint_distribution']},indent=2))
