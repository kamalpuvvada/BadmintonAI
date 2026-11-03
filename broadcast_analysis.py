"""Offline experimental broadcast overlays. Never grants a session review approval.

Court points are explicit source-bound setup, not auto-calibration. All movement,
shuttle and shot outputs are proposals; metres, winners and coaching stay null.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image

from badminton_proto import sha256_file


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def project(points, matrix):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, np.asarray(matrix, dtype=np.float64)).reshape(-1, 2)


def fit_court(pixel, court):
    pixel, court = np.asarray(pixel, float), np.asarray(court, float)
    if pixel.shape != court.shape or pixel.ndim != 2 or pixel.shape[1] != 2 or len(pixel) < 4:
        raise ValueError('At least four corresponding court intersections required')
    if not np.isfinite(pixel).all() or not np.isfinite(court).all():
        raise ValueError('Court points must be finite')
    matrix, _ = cv2.findHomography(pixel, court, 0)
    if matrix is None or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError('Degenerate court correspondences')
    return matrix


def active_difference(previous, current, roi):
    x1,y1,x2,y2 = roi
    if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
        raise ValueError('Invalid active-picture ROI')
    def thumbnail(image):
        h,w = image.shape[:2]
        return cv2.resize(image[int(y1*h):int(y2*h),int(x1*w):int(x2*w)], (96,64)).astype(float)/255
    return float(np.abs(thumbnail(previous)-thumbnail(current)).mean())


def green_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([30,45,30]), np.array([95,255,255]))


def court_lines():
    ys = [0, .76/13.4, 4.72/13.4, 8.68/13.4, 12.64/13.4, 1]
    lines = [((0,y),(1,y)) for y in ys]
    lines += [((x,0),(x,1)) for x in [0,.46/6.1,1-.46/6.1,1]]
    lines += [((.5,0),(.5,4.72/13.4)), ((.5,8.68/13.4),(.5,1))]
    return lines


def line_residual(frame, matrix):
    """Median distance of projected court-line samples to observed white pixels."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv,np.array([0,0,125]),np.array([179,100,255]))
    near_green = cv2.dilate(green_mask(frame),np.ones((13,13),np.uint8))
    white &= near_green
    distance = cv2.distanceTransform(255-white,cv2.DIST_L2,3)
    points=[]
    for a,b in court_lines():
        points.extend(np.linspace(a,b,40))
    pixels=project(points,np.linalg.inv(matrix))
    h,w=frame.shape[:2]
    errors=[float(distance[round(y),round(x)]) for x,y in pixels if 1 <= x < w-1 and .5*h < y < .87*h]
    return float(np.median(errors)) if len(errors) >= 30 else None


def court_stage(video, players, config, output):
    if (output/'court_summary.json').exists():
        raise ValueError('Court evidence already exists; choose a fresh output')
    capture=cv2.VideoCapture(str(video));fps=capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or fps <= 0:
        raise ValueError('Cannot decode source')
    stride=round(fps/config['analysis_fps'])
    if abs(fps/stride-config['analysis_fps']) > 1e-6:
        raise ValueError('Analysis FPS must divide native FPS')
    reference=fit_court(config['court_pixel_points'],config['court_normalized_points'])
    cumulative=np.eye(3);previous=None;previous_full=None;records=[];cut=False
    n=0;scene=0
    while True:
        ok,frame=capture.read()
        if not ok:break
        if n%stride:
            n+=1;continue
        small=cv2.resize(frame,(540,960));gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        score=0.;inliers=0;support=0.;mapping_valid=True;boundary=False
        if previous_full is not None:
            score=active_difference(previous_full,frame,config['active_picture_roi'])
        setup=next((s for s in config.get('additional_setups',[]) if s['native_frame_index']==n),None)
        if previous is not None and not cut:
            mask=green_mask(previous_full)
            mask=cv2.resize(mask,(540,960),interpolation=cv2.INTER_NEAREST)
            # Stable floor texture only; exclude players and portrait padding.
            mask[:480]=0;mask[805:]=0
            for person in players[max(0,n//stride-1)]['persons']:
                a,b,c,d=person['bbox_xyxy_px']
                cv2.rectangle(mask,(max(0,int(a/2)-12),max(0,int(b/2)-12)),(int(c/2)+12,int(d/2)+12),0,-1)
            corners=cv2.goodFeaturesToTrack(previous,500,.005,5,mask=mask)
            transform=None
            if corners is not None and len(corners)>=12:
                following,status,_=cv2.calcOpticalFlowPyrLK(previous,gray,corners,None,winSize=(31,31),maxLevel=4)
                back,back_status,_=cv2.calcOpticalFlowPyrLK(gray,previous,following,None,winSize=(31,31),maxLevel=4)
                keep=(status.ravel()==1)&(back_status.ravel()==1)&(np.linalg.norm(back-corners,axis=2).ravel()<1.5)
                if keep.sum()>=12:
                    transform,valid=cv2.findHomography(corners[keep]*2,following[keep]*2,cv2.RANSAC,3)
                    if valid is not None:inliers=int(valid.sum());support=float(valid.mean())
            # Abrupt active-picture changes with little coherent floor support.
            if score>=config['cut_threshold'] and (support<.55 or inliers<25):
                cut=True;scene+=1;boundary=True
            elif transform is not None and inliers>=12 and support>=.5:
                cumulative=transform@cumulative
            else:
                mapping_valid=False
        elif cut and score>=config['cut_threshold']:
            scene+=1;boundary=True
        if setup is not None:
            reference=fit_court(setup['court_pixel_points'],setup['court_normalized_points'])
            cumulative=np.eye(3);cut=False;mapping_valid=True
        matrix=reference@np.linalg.inv(cumulative)
        residual=None if cut else line_residual(frame,matrix)
        mapping_valid=bool(not cut and mapping_valid and residual is not None and residual<=config['line_residual_max_px'])
        row={'sample_index':n//stride,'native_frame_index':n,'timestamp_seconds':n/fps,'scene_id':scene,
             'status':'excluded_view_change' if cut else ('experimental_court' if mapping_valid else 'uncertain_court'),
             'active_picture_change':round(score,6),'scene_boundary':boundary,'flow_inliers':inliers,'flow_support':round(support,4),
             'line_residual_px':residual,'pixel_to_court':matrix.tolist() if mapping_valid else None,
             'players':[]}
        if mapping_valid:
            for person in players[n//stride]['persons']:
                if person['track_id'] not in (0,1):continue
                position=project([person['foot_px']],matrix)[0]
                if -.15<=position[0]<=1.15 and -.15<=position[1]<=1.15:
                    row['players'].append({'temporary_id':person['track_id'],'bbox_xyxy_px':person['bbox_xyxy_px'],
                                          'floor_proxy_px':person['foot_px'],'court_position':position.tolist(),
                                          'status':'box_bottom_proxy_not_ground_contact'})
        records.append(row);previous=gray;previous_full=frame;n+=1
    capture.release()
    result={'schema_version':'0.1.0','status':'experimental_not_review_approval','source_sha256':sha256_file(video),
            'configuration':config,'native_frames':n,'fps':fps,'sampling_stride':stride,
            'status_counts':dict(Counter(r['status'] for r in records)),
            'cut_times_seconds':[r['timestamp_seconds'] for r in records if r['scene_boundary']],
            'holdout_residual_normalized':np.linalg.norm(project(config['holdout_pixel_points'],fit_court(config['court_pixel_points'],config['court_normalized_points']))-config['holdout_normalized_points'],axis=1).tolist(),
            'limitations':['Box-bottom proxies include airborne and occluded feet; no physical distance totals.',
                           'Mapping and flow are experimental; residual is a line-consistency diagnostic, not calibrated accuracy.',
                           'A cut revokes the mapping until a new explicit setup. No athlete identities or reviews granted.']}
    output.mkdir(parents=True,exist_ok=True)
    (output/'court_frames.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records),encoding='utf-8')
    write_json(output/'court_summary.json',result)
    print(json.dumps({k:result[k] for k in ['native_frames','status_counts','cut_times_seconds','holdout_residual_normalized']}),flush=True)


def shuttle_stage(video, config, output, weights):
    """Separate opt-in experimental inference, not an approved-session export.

    Uses view exclusion from the setup; no player/shuttle review is created or
    changed. Results cannot be consumed as verified session measurements.
    """
    if (output/'shuttle_summary.json').exists():
        raise ValueError('Shuttle evidence already exists; choose a fresh output')
    from tempfile import TemporaryDirectory
    import torch
    from shuttle_tracknet import TrackNetPredictor, infer_groups
    from shuttle_linking import make_linker
    from download_tracknet import CHECKPOINT_SHA256
    court=[json.loads(l) for l in (output/'court_frames.jsonl').read_text().splitlines()]
    meta=json.loads((output/'court_summary.json').read_text())
    native_fps=meta['fps'];stride=2;fps=native_fps/stride
    torch.set_num_threads(4)
    predictor=TrackNetPredictor(weights,CHECKPOINT_SHA256)
    crop=config['shuttle_crop_xyxy_px'];x1,y1,x2,y2=crop
    contexts=[];records=[];peaks={};detections={};timer=time.perf_counter()
    with TemporaryDirectory(dir=output) as temp:
        work=Path(temp);paths=[];capture=cv2.VideoCapture(str(video));n=0
        while True:
            ok,frame=capture.read()
            if not ok:break
            if n%stride==0:
                path=work/f'{n:06d}.jpg'
                if not 0<=x1<x2<=frame.shape[1] or not 0<=y1<y2<=frame.shape[0]:
                    raise ValueError('Shuttle crop outside image')
                cv2.imwrite(str(path),cv2.resize(frame[y1:y2,x1:x2],(512,288),interpolation=cv2.INTER_AREA),[cv2.IMWRITE_JPEG_QUALITY,96])
                row=court[n//meta['sampling_stride']]
                contexts.append({'eligible':row['status']!='excluded_view_change','scene_id':row['scene_id']})
                paths.append(path)
            n+=1
        capture.release()
        # Bounded-memory groups; background and windows never cross edits.
        for start in range(0,len(paths),256):
            block_peaks={}
            block=infer_groups(paths[start:start+256],contexts[start:start+256],predictor,
                               stride=config.get('shuttle_window_stride'),
                               heatmap_threshold=config['shuttle_heatmap_threshold'],frame_peaks=block_peaks)
            detections.update({start+i:v for i,v in block.items()});peaks.update({start+i:v for i,v in block_peaks.items()})
            print(f'Shuttle model {min(start+256,len(paths))}/{len(paths)} sampled frames',flush=True)
    linker=make_linker('confidence_motion')
    for i,context in enumerate(contexts):
        candidates=detections.get(i,[])
        proposal=linker.update(i,context['scene_id'],candidates) if context['eligible'] else None
        if proposal is None:linker.reset()
        if proposal:
            # Model-space motion gate; restore every coordinate to source pixels.
            px,py=proposal['position_px']
            proposal['position_px']=[x1+px*(x2-x1)/512,y1+py*(y2-y1)/288]
        restored=[]
        for candidate in candidates:
            px,py=candidate['center_px']
            restored.append({'center_px':[x1+px*(x2-x1)/512,y1+py*(y2-y1)/288], 'heatmap_peak':candidate['heatmap_peak']})
        records.append({'frame_index':i,'native_frame_index':i*stride,'timestamp_seconds':i/fps,'scene_id':context['scene_id'],
                        'status':'excluded_view_change' if not context['eligible'] else ('proposal' if proposal else 'unobserved'),
                        'heatmap_peak':peaks.get(i),'candidates':restored,'proposal':proposal,'verified_position':None})
    (output/'shuttle_frames.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records),encoding='utf-8')
    write_json(output/'shuttle_summary.json',{'schema_version':'0.1.0','status':'experimental_independent_inference_not_review_approval',
               'source_sha256':sha256_file(video),'model':predictor.metadata,'fps':fps,'native_stride':stride,
               'configuration':{'crop_xyxy_px':crop,'model_size_px':[512,288],'heatmap_threshold':config['shuttle_heatmap_threshold'],
                                'linker':'confidence_motion_in_model_pixels','block_frames':256,
                                'window_stride':config.get('shuttle_window_stride',predictor.sequence_length),
                                'inpainting':False,'interpolation':False},
               'status_counts':dict(Counter(r['status'] for r in records)),'elapsed_seconds':time.perf_counter()-timer,
               'player_review_modified':False,'verified_positions':0})
    print('Shuttle stage complete',flush=True)


def shot_hint(contact, box, following):
    """Conservative image-space hints, explicitly NOT a trained shot classifier."""
    if len(following)<4:
        return 'Unknown shot', 'insufficient uninterrupted outgoing observations'
    x1,y1,x2,y2=box;height=max(1,y2-y1)
    relative_height=(y2-contact[1])/height
    points=np.asarray(following,float)
    rise=contact[1]-float(points[:,1].min())
    fall=float(points[-1,1]-contact[1])
    travel=float(np.linalg.norm(points[-1]-contact))
    if relative_height<.65 and rise>.45*height:
        return 'Lift / clear?', 'low contact and rising image-space flight; class ambiguous'
    if relative_height>.9 and fall>.5*height and travel>1.1*height:
        return 'Smash / drop?', 'high contact and descending image-space flight; class ambiguous'
    if relative_height>.9 and rise>.5*height:
        return 'Clear / overhead?', 'high contact and rising image-space flight; class ambiguous'
    if relative_height<.85 and travel<.8*height:
        return 'Net / block?', 'low contact and short observed image-space travel; class ambiguous'
    return 'Drive / other?', 'direction-change cue; image-space evidence does not resolve the class'


def body_core(point, persons):
    """Exclude central torso/leg texture, preserving observed candidates in JSON.

    This optional experimental mask can also remove an occluded shuttle; it is
    not a racket or contact mask and is never presented as verified absence.
    """
    x,y=point
    for person in persons:
        a,b,c,d=person['bbox_xyxy_px'];w,h=c-a,d-b
        if a+.2*w<=x<=a+.8*w and b+.25*h<=y<=b+.9*h:return True
    return False


def relink_stage(players, config, output):
    from shuttle_linking import make_linker
    if (output/'shuttle_filtered_frames.jsonl').exists():raise ValueError('Relinking evidence already exists')
    source=output/'shuttle_frames.jsonl'
    rows=[json.loads(l) for l in source.read_text().splitlines()]
    summary=json.loads((output/'shuttle_summary.json').read_text())
    linker=make_linker('confidence_motion');x1,y1,x2,y2=config['shuttle_crop_xyxy_px'];suppressed=0
    for row in rows:
        persons=players[min(len(players)-1,int(row['timestamp_seconds']*config['analysis_fps']+1e-7))]['persons']
        kept=[c for c in row['candidates'] if not body_core(c['center_px'],persons)]
        row['body_core_suppressed']=[c for c in row['candidates'] if body_core(c['center_px'],persons)]
        suppressed+=len(row['body_core_suppressed'])
        model_candidates=[{'center_px':[(c['center_px'][0]-x1)*512/(x2-x1),(c['center_px'][1]-y1)*288/(y2-y1)],'heatmap_peak':c['heatmap_peak']} for c in kept]
        proposal=linker.update(row['frame_index'],row['scene_id'],model_candidates) if row['status']!='excluded_view_change' else None
        if proposal is None:linker.reset()
        if proposal:
            px,py=proposal['position_px'];proposal['position_px']=[x1+px*(x2-x1)/512,y1+py*(y2-y1)/288]
            proposal['candidate_index_scope']='body_core_filtered_candidates'
        row['filtered_candidates']=kept;row['proposal']=proposal
        if row['status']!='excluded_view_change':row['status']='proposal' if proposal else 'unobserved'
    (output/'shuttle_filtered_frames.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
    write_json(output/'shuttle_filtered_summary.json',{'status':'experimental_body_core_exclusion','source_rows_sha256':sha256_file(source),
               'source_sha256':summary['source_sha256'],'status_counts':dict(Counter(r['status'] for r in rows)),
               'suppressed_observed_candidates':suppressed,'body_core_fractions_xyxy':[.2,.25,.8,.9],
               'limitation':'Possible shuttle observations against bodies are withheld, not labelled absent. No verified positions or contact labels.'})
    print(f'Relinked cached observations; withheld {suppressed} body-core candidates',flush=True)


def reuse_shuttle(source, config, output):
    import shutil
    binding=json.loads((source/'shuttle_binding.json').read_text())
    current=json.loads((output/'analysis_binding.json').read_text())
    if binding['inputs']!=current:raise ValueError('Cached shuttle source/configuration/players differ')
    for name,expected in binding['artifacts_sha256'].items():
        if sha256_file(source/name)!=expected:raise ValueError('Cached shuttle artifact changed')
    for name in ['shuttle_summary.json','shuttle_frames.jsonl']:
        if (output/name).exists():raise ValueError('Output shuttle evidence already exists')
        shutil.copyfile(source/name,output/name)
    write_json(output/'shuttle_reuse.json',{'source_stage_binding':binding,'source_directory':str(source),
                                          'reused_observations':True,'model_inference_repeated':False})


def event_stage(players, config, output):
    if (output/'events.json').exists():
        raise ValueError('Event evidence already exists; choose a fresh output')
    from contact_proposals import find_contacts
    rows=[json.loads(l) for l in (output/'shuttle_filtered_frames.jsonl').read_text().splitlines()]
    court=[json.loads(l) for l in (output/'court_frames.jsonl').read_text().splitlines()]
    fps=json.loads((output/'shuttle_summary.json').read_text())['fps']
    eligible={i for i in range(len(rows)) if court[i//2]['status']=='experimental_court'}
    contacts=find_contacts(rows,players,fps,config['analysis_fps'],(1080,1920),eligible)
    events=[]
    for cue in contacts:
        i=cue['frame_index'];row=rows[i];point=row['proposal']['position_px']
        nearby=[]
        for person in players[int(i/fps*config['analysis_fps'])]['persons']:
            if person['track_id'] not in (0,1):continue
            x1,y1,x2,y2=person['bbox_xyxy_px'];w,h=x2-x1,y2-y1
            if x1-.6*w<=point[0]<=x2+.6*w and y1-.6*h<=point[1]<=y2+.3*h:
                nearby.append(person)
        following=[]
        for future in rows[i+1:i+10]:
            if future['proposal'] is None or future['proposal']['proposal_segment_id']!=row['proposal']['proposal_segment_id'] or future['scene_id']!=row['scene_id']:break
            following.append(future['proposal']['position_px'])
        hint,reason=('Unknown shot','ambiguous hitter proximity') if len(nearby)!=1 else shot_hint(point,nearby[0]['bbox_xyxy_px'],following)
        events.append({'timestamp_seconds':i/fps,'native_frame_index':i*2,'scene_id':row['scene_id'],
                       'contact_status':'proposed_not_verified','temporary_hitter_id':nearby[0]['track_id'] if len(nearby)==1 else None,
                       'shot_hint':hint,'evidence':reason,'confidence':'unvalidated_heuristic','cue':cue})
    write_json(output/'events.json',{'schema_version':'0.1.0','status':'experimental_ambiguous_hints',
               'events':events,'verified_shots':0,'winner':None,'forced_errors':None,'coaching':None,
               'limitations':['Direction-change plus person proximity is not proof of racket contact.',
                              'Paired names with question marks are ambiguous hypotheses, not classified shots.',
                              'No shot model has been trained or validated on this recording.']})
    print(f'{len(events)} contact/shot hypotheses; zero verified shots',flush=True)


def contiguous(previous, current, stride):
    return previous is not None and previous['scene_id']==current['scene_id'] and current['native_frame_index']-previous['native_frame_index']==stride


def native_boundaries(video, roi, threshold):
    """Conservative native-frame guard between the slower court observations."""
    capture=cv2.VideoCapture(str(video));previous=None;indices=[];n=0
    try:
        while True:
            ok,frame=capture.read()
            if not ok:break
            small=cv2.resize(frame,(108,192))
            if previous is not None and active_difference(previous,small,roi)>=threshold:
                if not indices or n-indices[-1]>6:indices.append(n)
            previous=small;n+=1
    finally:capture.release()
    return indices


def text(frame, value, x,y,scale=.7,color=(235,240,245),thickness=1):
    cv2.putText(frame,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness,cv2.LINE_AA)


def draw_map(frame, origin, size, positions, histories, occupancy=None):
    x,y=origin;w,h=size
    if occupancy is not None:
        density=cv2.GaussianBlur(occupancy,(0,0),2)
        if density.max()>0:
            heat=cv2.applyColorMap(np.uint8(density/density.max()*255),cv2.COLORMAP_INFERNO)
            heat=cv2.resize(heat,(w,h))
            frame[y:y+h,x:x+w]=heat
    else:cv2.rectangle(frame,(x,y),(x+w,y+h),(48,77,48),-1)
    def pixel(p):return (int(x+p[0]*w),int(y+p[1]*h))
    for a,b in court_lines():cv2.line(frame,pixel(a),pixel(b),(180,195,190),1,cv2.LINE_AA)
    cv2.line(frame,pixel((0,.5)),pixel((1,.5)),(255,255,255),2)
    for tid,history in histories.items():
        color=(245,225,40) if tid==0 else (80,150,255)
        pts=[pixel(np.clip(p,0,1)) for p in history]
        if len(pts)>1:cv2.polylines(frame,[np.array(pts)],False,color,2,cv2.LINE_AA)
    for person in positions:
        color=(245,225,40) if person['temporary_id']==0 else (80,150,255)
        cv2.circle(frame,pixel(np.clip(person['court_position'],0,1)),5,color,-1,cv2.LINE_AA)


def render_stage(video, config, output):
    from tempfile import TemporaryFile
    court=[json.loads(l) for l in (output/'court_frames.jsonl').read_text().splitlines()]
    shuttle=[json.loads(l) for l in (output/'shuttle_filtered_frames.jsonl').read_text().splitlines()]
    events=json.loads((output/'events.json').read_text())['events']
    meta=json.loads((output/'court_summary.json').read_text())
    boundaries=native_boundaries(video,config['active_picture_roi'],config['cut_threshold'])
    capture=cv2.VideoCapture(str(video));fps=meta['fps'];n=0
    target=output/'Axelsen-Lee-All-Overlays.mp4'
    if target.exists():raise ValueError('Choose a fresh render output; MP4 already exists')
    args=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','1080x1920','-r',str(fps),'-i','pipe:0',
          '-i',str(video),'-map','0:v:0','-map','1:a:0?','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-t',str(meta['native_frames']/fps),'-movflags','+faststart',str(target)]
    histories={};last_court=None;last_shuttle=None;shuttle_trail=[];occupancy=np.zeros((160,80),np.float32)
    with TemporaryFile() as errors:
        process=subprocess.Popen(args,stdin=subprocess.PIPE,stderr=errors)
        try:
            while True:
                ok,frame=capture.read()
                if not ok:break
                timestamp=n/fps;c=court[n//meta['sampling_stride']];s=shuttle[n//2]
                valid=c['status']=='experimental_court' and not any(c['native_frame_index']<edge<=n for edge in boundaries)
                if n%meta['sampling_stride']==0:
                    if not valid or not contiguous(last_court,c,meta['sampling_stride']):
                        histories={};occupancy[:]=0
                    current_ids={p['temporary_id'] for p in c['players']}
                    histories={k:v for k,v in histories.items() if k in current_ids}
                    for person in c['players']:
                        tid=person['temporary_id'];point=person['court_position']
                        histories[tid]=(histories.get(tid,[])+[point])[-24:]
                        if 0<=point[0]<1 and 0<=point[1]<1:
                            occupancy[int(point[1]*160),int(point[0]*80)]+=meta['sampling_stride']/fps
                    last_court=c
                if n%2==0:
                    same=contiguous(last_shuttle,s,2)
                    if not same or not s['proposal'] or not last_shuttle['proposal'] or s['proposal']['proposal_segment_id']!=last_shuttle['proposal']['proposal_segment_id']:
                        shuttle_trail=[]
                    if s['proposal']:shuttle_trail=(shuttle_trail+[s['proposal']['position_px']])[-12:]
                    last_shuttle=s
                if valid:
                    inverse=np.linalg.inv(c['pixel_to_court'])
                    # Camera-compensated trails, never lines across gaps or edits.
                    for tid,history in histories.items():
                        color=(245,225,40) if tid==0 else (80,150,255)
                        points=project(history,inverse).astype(np.int32)
                        if len(points)>1:cv2.polylines(frame,[points],False,color,3,cv2.LINE_AA)
                    for person in c['players']:
                        a,b,d,e=map(round,person['bbox_xyxy_px']);tid=person['temporary_id'];color=(245,225,40) if tid==0 else (80,150,255)
                        cv2.rectangle(frame,(a,b),(d,e),color,2)
                        text(frame,f'T{tid}',max(0,a),max(25,b-10),.8,color,2)
                    if s['proposal'] and s['scene_id']==c['scene_id']:
                        pts=np.array(shuttle_trail,dtype=np.int32)
                        if len(pts)>1:cv2.polylines(frame,[pts],False,(50,255,255),2,cv2.LINE_AA)
                        x,y=map(round,s['proposal']['position_px']);cv2.circle(frame,(x,y),10,(50,255,255),2,cv2.LINE_AA)
                        text(frame,'Shuttle?',x+14,y,.65,(50,255,255),2)
                cv2.rectangle(frame,(16,16),(1064,218),(23,27,33),-1)
                text(frame,'BadmintonAI | Court, movement and shuttle',32,58,1.,thickness=2)
                text(frame,f'{timestamp:05.2f}s | EXPERIMENTAL - not verified measurements',32,102,.78,(120,225,250))
                active=[e for e in events if e['scene_id']==c['scene_id'] and 0<=timestamp-e['timestamp_seconds']<1.1] if valid else []
                latest=active[-1] if active else None
                label=latest['shot_hint'] if latest else ('Waiting for reliable contact cue' if valid else 'View change - overlays withheld')
                text(frame,'SHOT HINT: '+label,32,151,.87,(70,230,255),2)
                text(frame,'Paired shot names are uncertain hypotheses; T IDs are temporary',32,193,.68)
                cv2.rectangle(frame,(16,1640),(1064,1904),(23,27,33),-1)
                text(frame,'COURT MAP',34,1669,.65);text(frame,'MOVEMENT',229,1669,.65)
                if valid:
                    draw_map(frame,(53,1682),(94,206),c['players'],histories)
                    draw_map(frame,(251,1682),(94,206),c['players'],{},occupancy)
                else:
                    text(frame,'Paused',40,1770,.75);text(frame,'Paused',239,1770,.75)
                observation='observed proposal' if valid and s['proposal'] else 'no reliable observation'
                text(frame,'SHUTTLE: '+observation,405,1688,.65,(70,230,255))
                text(frame,'Near T0: cyan | Far T1: orange',405,1730,.65)
                text(frame,'Map / trails: box-bottom proxies',405,1772,.65)
                text(frame,'Heat: time observed in this segment',405,1814,.62)
                text(frame,'Physical speed / distance: unavailable',405,1856,.60)
                text(frame,'Shots / winner / errors: unverified',405,1890,.61)
                process.stdin.write(frame.tobytes())
                if n in [480,960,1224,1440,1920,2190]:cv2.imwrite(str(output/f'overlay-{n:06d}.jpg'),frame)
                n+=1
            process.stdin.close()
            if process.wait(timeout=60):
                errors.seek(0);raise RuntimeError(errors.read().decode(errors='replace')[-2500:])
        finally:
            capture.release()
            if process.stdin and not process.stdin.closed:process.stdin.close()
            if process.poll() is None:process.terminate();process.wait(timeout=10)
    check=cv2.VideoCapture(str(target));decoded=0
    while True:
        ok,frame=check.read()
        if not ok:break
        if frame.shape[:2]!=(1920,1080):raise ValueError('Output geometry changed')
        decoded+=1
    check.release()
    if decoded!=n or n!=meta['native_frames']:raise ValueError('Output frame count mismatch')
    write_json(output/'render_summary.json',{'frames':decoded,'fps':fps,'source_sha256':sha256_file(video),'output_sha256':sha256_file(target),
               'source_size_px':[1080,1920],'overlays':'experimental_court_movement_shuttle_ambiguous_shot_hints',
               'audio':'original audio re-encoded AAC','native_view_boundaries':boundaries,'verified_coaching':False,'player_review_modified':False})
    print(str(target),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['court','shuttle','relink','events','render','all'])
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--weights',type=Path,default=Path('out/models/TrackNetV3_TrackNet_best.pt'))
    parser.add_argument('--reuse-shuttle',type=Path,help='Reuse hash-bound model observations from an identical run')
    args=parser.parse_args()
    result=json.loads((args.job/'result.json').read_text(encoding='utf-8'))
    config=json.loads(args.config.read_text(encoding='utf-8'))
    video=Path(result['source']['path'])
    if sha256_file(video)!=config['source_sha256'] or result['source']['sha256']!=config['source_sha256']:
        raise ValueError('Court setup belongs to a different recording')
    players=[json.loads(l) for l in (args.job/result['artifacts']['observations']).read_text().splitlines()]
    player_path=args.job/result['artifacts']['observations']
    job=json.loads((args.job/'job.json').read_text())
    player_hash=sha256_file(player_path)
    expected=job['stages']['players']['artifacts']['player_tracks.jsonl']
    if player_hash!=expected:raise ValueError('Player observations changed since the source job')
    binding={'source_sha256':config['source_sha256'],'players_sha256':player_hash,
             'configuration_sha256':sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest(),
             'status':'experimental_no_session_approval','player_review_modified':False}
    stages=['court','shuttle','relink','events','render'] if args.stage=='all' else [args.stage]
    for stage in stages:
        if stage!='court':
            saved=json.loads((args.out/'analysis_binding.json').read_text())
            if saved!=binding:raise ValueError('Source, players or configuration differs from this analysis')
        timer=time.perf_counter()
        if stage=='court':
            court_stage(video,players,config,args.out)
            write_json(args.out/'analysis_binding.json',binding)
        elif stage=='shuttle':
            if args.reuse_shuttle:reuse_shuttle(args.reuse_shuttle,config,args.out)
            else:shuttle_stage(video,config,args.out,args.weights)
        elif stage=='relink':relink_stage(players,config,args.out)
        elif stage=='events':event_stage(players,config,args.out)
        elif stage=='render':render_stage(video,config,args.out)
        names={'court':['court_summary.json','court_frames.jsonl'],
               'shuttle':['shuttle_summary.json','shuttle_frames.jsonl'],
               'relink':['shuttle_filtered_frames.jsonl','shuttle_filtered_summary.json'],
               'events':['events.json'], 'render':['render_summary.json','Axelsen-Lee-All-Overlays.mp4']}[stage]
        write_json(args.out/f'{stage}_binding.json',{'inputs':binding,'code_sha256':sha256_file(Path(__file__)),
                   'elapsed_seconds':time.perf_counter()-timer,'artifacts_sha256':{name:sha256_file(args.out/name) for name in names}})


if __name__=='__main__':main()
