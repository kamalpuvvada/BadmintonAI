"""Independent pose-and-flight shot hypotheses for offline broadcast review."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time


LANDMARKS = {'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12,
             'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
             'left_hip': 23, 'right_hip': 24}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def extract_pose(source, video, output, runtime=None):
    if (output/'pose_frames.jsonl').exists():
        raise ValueError('Pose observations already exist. Choose a fresh directory.')
    if runtime is not None:
        sys.path.insert(0, str(runtime.resolve()))
    import cv2
    import mediapipe as mp
    from badminton_proto import sha256_file

    court = read_rows(source/'court_frames.jsonl')
    summary = read(source/'court_summary.json')
    if sha256_file(video) != summary['source_sha256']:
        raise ValueError('Video differs from the court setup')
    output.mkdir(parents=True, exist_ok=True)
    model = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1,
                                   min_detection_confidence=.5, enable_segmentation=False)
    capture = cv2.VideoCapture(str(video))
    records = []
    native_index = 0
    timer = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if native_index % summary['sampling_stride']:
                native_index += 1
                continue
            context = court[native_index//summary['sampling_stride']]
            observations = []
            if context['status'] == 'experimental_court':
                for player in context['players']:
                    left, top, right, bottom = player['bbox_xyxy_px']
                    width, height = right-left, bottom-top
                    x1 = max(0, math.floor(left-.35*width))
                    x2 = min(frame.shape[1], math.ceil(right+.35*width))
                    y1 = max(0, math.floor(top-.4*height))
                    y2 = min(frame.shape[0], math.ceil(bottom+.1*height))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    scale = 480/crop.shape[0]
                    crop = cv2.resize(crop, (max(1, round(crop.shape[1]*scale)), 480))
                    pose = model.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    if pose.pose_landmarks is None:
                        continue
                    landmarks = {}
                    for name, index in LANDMARKS.items():
                        point = pose.pose_landmarks.landmark[index]
                        landmarks[name] = {'point_px': [x1+point.x*(x2-x1), y1+point.y*(y2-y1)],
                                           'visibility': float(point.visibility)}
                    observations.append({'temporary_id': player['temporary_id'], 'landmarks': landmarks,
                                         'court_position': player['court_position'],
                                         'bbox_xyxy_px': player['bbox_xyxy_px']})
            records.append({'native_frame_index': native_index, 'timestamp_seconds': context['timestamp_seconds'],
                            'scene_id': context['scene_id'], 'status': context['status'], 'players': observations})
            if len(records) % 50 == 0:
                print(f'Pose observations {len(records)}/{len(court)}', flush=True)
            native_index += 1
    finally:
        capture.release()
        model.close()
    if len(records) != len(court) or native_index != summary['native_frames']:
        raise ValueError('Pose and source frame grids differ')
    (output/'pose_frames.jsonl').write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in records), encoding='utf-8')
    write(output/'pose_summary.json', {'source_sha256': summary['source_sha256'],
          'court_rows_sha256': sha256_file(source/'court_frames.jsonl'), 'package': 'mediapipe',
          'version': mp.__version__, 'pose_complexity': 1, 'static_image_mode': True,
          'source_crops': 'RF-DETR body box with arm/head margin, resized to height 480',
          'players_observed': sum(len(row['players']) for row in records),
          'elapsed_seconds': time.perf_counter()-timer, 'status': 'experimental_pose_observations'})
    print(f'Pose complete: {sum(len(row["players"]) for row in records)} player observations', flush=True)


def body_frame(player, arm):
    landmarks = player['landmarks']
    names = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', arm+'_wrist']
    if any(landmarks[name]['visibility'] < .6 for name in names):
        return None
    shoulder = [(landmarks['left_shoulder']['point_px'][i]+landmarks['right_shoulder']['point_px'][i])/2 for i in (0,1)]
    hip = [(landmarks['left_hip']['point_px'][i]+landmarks['right_hip']['point_px'][i])/2 for i in (0,1)]
    torso = math.dist(shoulder, hip)
    if torso < 15:
        return None
    wrist = landmarks[arm+'_wrist']['point_px']
    if not all(math.isfinite(value) for value in wrist):
        return None
    left, top, right, bottom = player['bbox_xyxy_px']
    # An expanded crop can contain the other player near the net. Require the
    # fitted torso to belong to the original detector box before using its arm.
    if any(not left <= point[0] <= right or not top <= point[1] <= bottom for point in (shoulder, hip)):
        return None
    return {'wrist': wrist, 'relative': [(wrist[i]-shoulder[i])/torso for i in (0,1)], 'torso': torso}


def swing_windows(poses):
    windows = []
    for index in range(2, len(poses)-2):
        context = poses[index-2:index+3]
        if any(row['status'] != 'experimental_court' or row['scene_id'] != context[0]['scene_id'] for row in context):
            continue
        if any(b['native_frame_index']-a['native_frame_index'] != 4 for a,b in zip(context, context[1:])):
            continue
        for current in poses[index]['players']:
            identity = current['temporary_id']
            people = [next((p for p in row['players'] if p['temporary_id'] == identity), None) for row in context]
            if any(person is None for person in people):
                continue
            for arm in ('left', 'right'):
                frames = [body_frame(person, arm) for person in people]
                if any(frame is None for frame in frames):
                    continue
                relative = [frame['relative'] for frame in frames]
                if max(frame['torso'] for frame in frames)/min(frame['torso'] for frame in frames) > 1.8:
                    continue
                times = [row['timestamp_seconds'] for row in context]
                speeds = [math.dist(a,b)/(tb-ta) for a,b,ta,tb in zip(relative, relative[1:], times, times[1:])]
                path_length = sum(math.dist(a,b) for a,b in zip(relative, relative[1:]))
                windows.append({'sample_index': index, 'native_frame_index': poses[index]['native_frame_index'],
                                'timestamp_seconds': poses[index]['timestamp_seconds'], 'scene_id': poses[index]['scene_id'],
                                'temporary_id': identity, 'arm': arm, 'speed_torso_per_second': max(speeds[1:3]),
                                'highest_hand': -min(p[1] for p in relative),
                                'vertical_sweep': relative[-1][1]-relative[0][1],
                                'horizontal_sweep': abs(relative[-1][0]-relative[0][0]),
                                'motion_coherence': math.dist(relative[0],relative[-1])/max(.001,path_length),
                                'contact_height': -relative[2][1], 'wrist_px': frames[2]['wrist'],
                                'torso_px': frames[2]['torso'], 'court_position': current['court_position']})
    return windows


def classify_motion(swing, flight):
    """Geometric stroke hypotheses. Scores and thresholds have no accuracy claim."""
    speed = swing['speed_torso_per_second']
    high = swing['highest_hand']
    vertical = swing['vertical_sweep']
    lateral = swing['horizontal_sweep']
    if swing['motion_coherence'] < .25 or speed < 4:
        return None
    if high >= .65 and swing['contact_height'] >= .25 and vertical >= .65 and speed >= 8:
        return 'Smash', 'raised wrist followed by a rapid downward swing'
    if high >= .85 and vertical < -.6 and flight['rise_torso'] > .6:
        return 'Clear', 'overhead swing with an observed rising outgoing flight'
    if swing['contact_height'] < .25 and vertical < -.35 and flight['rise_torso'] > .5:
        return 'Lift', 'underarm upward swing with observed rising flight'
    if high < .65 and lateral > .7 and abs(vertical) < .75*lateral and flight['rise_torso'] < .5:
        return 'Drive', 'coherent lateral swing below overhead height'
    if abs(swing['court_position'][1]-.5) < .2 and high < .45 and speed < 8 and lateral < .9 and flight['travel_torso'] < 1.2:
        return 'Net shot', 'short low swing near the net with limited observed flight'
    return None


def recognize(source, output):
    from badminton_proto import sha256_file
    summary = read(output/'pose_summary.json')
    if summary['court_rows_sha256'] != sha256_file(source/'court_frames.jsonl'):
        raise ValueError('Court observations changed since pose inference')
    court_summary = read(source/'court_summary.json')
    if summary['source_sha256'] != court_summary['source_sha256']:
        raise ValueError('Pose and shuttle recording differ')
    poses = read_rows(output/'pose_frames.jsonl')
    features = swing_windows(poses)
    write(output/'swing_features.json', features)
    shuttle = read_rows(source/'shuttle_filtered_frames.jsonl')
    candidates = []
    for swing in features:
        if swing['speed_torso_per_second'] < 4 or swing['motion_coherence'] < .25:
            continue
        frame = round(swing['timestamp_seconds']*30)
        nearby = [row for row in shuttle[max(0,frame-4):frame+5]
                  if row['proposal'] and row['scene_id'] == swing['scene_id']]
        if not nearby:
            continue
        aligned = []
        for row in nearby:
            pose_index = min(len(poses)-1, round(row['timestamp_seconds']*15))
            player = next((person for person in poses[pose_index]['players'] if person['temporary_id'] == swing['temporary_id']), None)
            body = body_frame(player,swing['arm']) if player else None
            if body is not None:
                distance = math.dist(row['proposal']['position_px'],body['wrist'])/body['torso']
                aligned.append((distance,row,body))
        if not aligned:
            continue
        distance, nearest, aligned_body = min(aligned,key=lambda value:value[0])
        if distance > 1.8:
            continue
        other_distances = []
        pose_index = min(len(poses)-1, round(nearest['timestamp_seconds']*15))
        for player in poses[pose_index]['players']:
            if player['temporary_id'] == swing['temporary_id']:
                continue
            for arm in ('left','right'):
                body = body_frame(player,arm)
                if body is not None:
                    other_distances.append(math.dist(nearest['proposal']['position_px'],body['wrist'])/body['torso'])
        if other_distances and min(other_distances) < distance*.7:
            continue
        # Stroke preparation and follow-through must agree on overhead context.
        # Otherwise one lateral snapshot can label an overhead stroke as a drive.
        context = [window for window in features
                   if window['scene_id'] == swing['scene_id'] and window['temporary_id'] == swing['temporary_id']
                   and window['arm'] == swing['arm'] and abs(window['timestamp_seconds']-swing['timestamp_seconds']) <= .2]
        swing = dict(swing)
        swing['highest_hand'] = max(window['highest_hand'] for window in context)
        if swing['highest_hand'] >= .65:
            swing['vertical_sweep'] = max(window['vertical_sweep'] for window in context)
        following = []
        for row in shuttle[nearest['frame_index']+1:nearest['frame_index']+11]:
            if not row['proposal'] or row['scene_id'] != nearest['scene_id'] or row['proposal']['proposal_segment_id'] != nearest['proposal']['proposal_segment_id']:
                break
            following.append(row['proposal']['position_px'])
        point = nearest['proposal']['position_px']
        flight = {'rise_torso': max([0]+[(point[1]-p[1])/swing['torso_px'] for p in following]),
                  'travel_torso': math.dist(point,following[-1])/swing['torso_px'] if following else math.inf}
        label = classify_motion(swing,flight)
        if label is None:
            continue
        candidates.append({'timestamp_seconds': nearest['timestamp_seconds'],
                           'native_frame_index': nearest['native_frame_index'], 'scene_id': swing['scene_id'],
                           'temporary_hitter_id': swing['temporary_id'], 'shot_name': label[0],
                           'shot_hint': label[0]+'?', 'evidence': label[1],
                           'evidence_source': 'pose_and_observed_shuttle', 'contact_status': 'proposed_not_verified',
                           'confidence': 'experimental_rule_based_prediction',
                           'features': swing, 'shuttle_wrist_distance_torso': distance,
                           'matching_wrist_px': aligned_body['wrist'],
                           'outgoing_observed_samples': len(following), 'flight_features': flight})
    selected = []
    for candidate in sorted(candidates,key=lambda event: (event['shuttle_wrist_distance_torso'], -event['features']['motion_coherence'])):
        if any(event['scene_id'] == candidate['scene_id'] and abs(event['timestamp_seconds']-candidate['timestamp_seconds']) < .4 for event in selected):
            continue
        selected.append(candidate)
    existing = read(source/'events.json')['events']
    events = []
    for event in existing:
        matches = [candidate for candidate in selected if candidate['scene_id'] == event['scene_id'] and abs(candidate['timestamp_seconds']-event['timestamp_seconds']) < .3]
        if matches:
            continue
        record = dict(event)
        record['evidence_source'] = 'observed_shuttle_trajectory'
        record['shot_name'] = None if event['shot_hint'] == 'Unknown shot' else event['shot_hint'].rstrip('?')
        events.append(record)
    events.extend(selected)
    events.sort(key=lambda event: event['timestamp_seconds'])
    for event in events:
        event['display_seconds'] = 1.5
    packet = {'schema_version':'0.2.0', 'status':'experimental_pose_and_flight_predictions',
              'events': events, 'verified_shots':0, 'winner':None, 'forced_errors':None, 'coaching':None,
              'method':'independent body-normalized arm motion, wrist proximity and observed shuttle context',
              'shot_distribution':dict(Counter(event['shot_name'] or 'Unknown' for event in events)),
              'limitations':['Rule-based shot predictions require coach-labelled evaluation.',
                             'Pose and shuttle association can confuse racket preparation with contact.',
                             'Perspective and gaps still limit shot classification.']}
    write(output/'events.json',packet)
    print(json.dumps(packet['shot_distribution']),flush=True)
    return packet


def render(source, output, video):
    import shutil
    from badminton_proto import sha256_file
    from broadcast_analysis import render_stage
    from broadcast_refinement import viewer

    source_binding = read(source/'refinement_binding.json')
    for name in ('court_frames.jsonl','shuttle_filtered_frames.jsonl','events.json'):
        if sha256_file(source/name) != source_binding['artifacts_sha256'][name]:
            raise ValueError('Analysis source artifacts have changed')
    config = read(source/'court_summary.json')['configuration']
    if sha256_file(video) != config['source_sha256']:
        raise ValueError('Render video differs from the analyzed recording')
    for name in ('court_frames.jsonl','court_summary.json','shuttle_filtered_frames.jsonl','shuttle_summary.json'):
        if (output/name).exists():
            raise ValueError('Render inputs already exist in this directory')
        shutil.copyfile(source/name,output/name)
    render_stage(video, config, output)
    viewer(output, read_rows(output/'court_frames.jsonl'), read(output/'events.json'))
    write(output/'shot_recognition_binding.json', {
        'source_sha256': config['source_sha256'], 'source_analysis': str(source),
        'code_sha256': sha256_file(Path(__file__)),
        'artifacts_sha256': {name: sha256_file(output/name) for name in (
            'pose_frames.jsonl','pose_summary.json','swing_features.json','events.json',
            'court_frames.jsonl','shuttle_filtered_frames.jsonl','render_summary.json',
            'Axelsen-Lee-All-Overlays.mp4','index.html','broadcast_viewer.js')},
        'status': 'experimental_predictions_require_coach_evaluation', 'verified_shots':0})
    print(str(output/'index.html'),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pose-runtime', type=Path)
    parser.add_argument('--stage', choices=['pose','recognize','render','all'], default='all')
    args = parser.parse_args()
    grid = read(args.source/'court_summary.json')
    if grid['fps'] != 60 or grid['sampling_stride'] != 4:
        parser.error('This broadcast prototype requires a 60 FPS source and 15 FPS court observations')
    if args.stage in ('pose','all'):
        extract_pose(args.source, args.video, args.out, args.pose_runtime)
    if args.stage in ('recognize','all'):
        recognize(args.source,args.out)
    if args.stage in ('render','all'):
        render(args.source,args.out,args.video)


if __name__ == '__main__':
    main()
