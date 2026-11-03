"""Summarize the local real-video test; does not approve tracking or coaching."""
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'out/axelsen-lee-session-001'
INSPECT = ROOT / 'out/axelsen-lee-inspection'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


job, result = read(RUN/'job.json'), read(RUN/'result.json')
rows = [json.loads(line) for line in (RUN/'players-001/player_tracks.jsonl').read_text().splitlines()]
seed = read(INSPECT/'anchor-seed.json')
summary = read(RUN/'players-001/tracking_summary.json')
report = {
    'schema_version': '0.1.0', 'source': result['source'],
    'inference_code_commit': '3b9c9c62cf271cc1e0acec114820696694201d82',
    'reviewer_kind': 'automated', 'ground_truth_verified': False,
    'configuration': result['configuration'], 'model': job['binding']['models']['person'],
    'player_stage_seconds': job['stages']['players']['elapsed_seconds'],
    'stage_counts': summary['diagnostics']['stage_totals'],
    'person_count_histogram': dict(Counter(len(r['persons']) for r in rows)),
    'unique_temporary_ids': summary['diagnostics']['unique_track_ids_observed'],
    'scene_review': summary['scene_review'], 'anchor_seed': seed,
    'anchor_method': 'Six sparse source-only frames selected before reading model predictions; 12 visible torso points. One-to-one box containment. Preceding nominal 15 FPS sample; FFmpeg sampling alignment approximate. Not precision/recall, IDF1 or full continuity verification.',
    'anchors': [],
    'artifacts_sha256': {p: sha(RUN/p) for p in ['job.json','result.json','players-001/player_tracks.jsonl','players-001/tracking_summary.json','render-001/preview.mp4']},
}
for name, subset in [('court_view_before_35_5s', [r for r in rows if r['timestamp_seconds'] < 35.5]),
                     ('closeup_from_35_6s', [r for r in rows if r['timestamp_seconds'] >= 35.6])]:
    report[name] = dict(sampled_frames=len(subset), person_count_histogram=dict(Counter(len(r['persons']) for r in subset)),
                        observed_temporary_ids=sorted({p['track_id'] for r in subset for p in r['persons'] if p['track_id'] is not None}))
for case in seed['cases']:
    assert sha(INSPECT/f'raw-{case["native_frame_index"]:06d}.jpg') == case['source_image_sha256']
    row = rows[case['native_frame_index']//4]
    candidates = []
    for anchor in case['anchors']:
        x,y = anchor['torso_point_px']
        candidates.append([i for i,p in enumerate(row['persons']) if p['bbox_xyxy_px'][0] <= x <= p['bbox_xyxy_px'][2] and p['bbox_xyxy_px'][1] <= y <= p['bbox_xyxy_px'][3]])
    choices = [c+[None] for c in candidates]
    assignments = [p for p in itertools.product(*choices) if len([i for i in p if i is not None]) == len(set(i for i in p if i is not None))]
    best = max(assignments, key=lambda p:sum(i is not None for i in p))
    for anchor,options,index in zip(case['anchors'], candidates, best):
        report['anchors'].append(dict(source_seconds=case['source_seconds'],appearance_tag=anchor['appearance_tag'], candidate_indices=options, covered=index is not None, proposed_temporary_id=None if index is None else row['persons'][index]['track_id']))
report['covered_anchors'] = sum(a['covered'] for a in report['anchors'])
report['visible_anchors'] = len(report['anchors'])
preview = cv2.VideoCapture(str(RUN/'render-001/preview.mp4'))
frames = 0
sheet = Image.new('RGB', (1080, 2056), '#111827')
highlights = Image.new('RGB', (1080, 552), '#111827')
sample_indices = [0,240,480,720,960,1440,1920,2100,2190]
while True:
    ok, frame = preview.read()
    if not ok:
        break
    if frames in sample_indices:
        n = sample_indices.index(frames)
        im = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((360,661))
        x,y = n%3*360, n//3*685
        sheet.paste(im,(x,y+24))
        ImageDraw.Draw(sheet).text((x+8,y+6),f'{frames/60:.2f}s | proposed tracking',fill='white')
    if frames in [480,2160]:
        column = 0 if frames == 480 else 1
        crop = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).crop((0,764,1080,1764)).resize((540,500))
        highlights.paste(crop,(column*540,52))
        title = '8s: rally | two proposed player boxes' if column == 0 else '36s: close-up | background people included'
        ImageDraw.Draw(highlights).text((column*540+10,10),title,fill='white')
        ImageDraw.Draw(highlights).text((column*540+10,30),'Cropped preview; T IDs remain unverified',fill='#67e8f9')
    frames += 1
preview.release()
assert frames == 2256, frames
report['decoded_native_preview_frames'] = frames
report['decision'] = 'Tracking remains experimental. Source changes to a close-up around 35.52 seconds; no automatic scene or athlete review is granted. No shuttle, shot, point, error or coaching measurements approved.'
report['limitations'] = ['Court crop and inclusion ROI selected for this clip, not a general accuracy benchmark.', 'Source is a portrait broadcast edit with panning and a closing close-up; not calibrated fixed academy footage.', 'Existing scoreboard and names are source graphics, not verified athlete-to-track assignments.', 'Sparse anchors cannot establish full detection recall or ID continuity.']
sheet.save(RUN/'test-contact-sheet.jpg',quality=92)
highlights.save(RUN/'test-highlights.jpg',quality=92)
(ROOT/'validation/axelsen_lee_video_test_2026-11-03.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
(RUN/'test-result.html').write_text('''<!doctype html><meta charset="utf-8"><title>Axelsen / Lee video test</title><style>body{max-width:1100px;margin:24px auto;background:#111827;color:white;font:18px system-ui}a{color:#67e8f9}video{width:100%;max-height:80vh}img{width:100%}</style><h1>Axelsen / Lee: real video test</h1><p>Full video tested: 2,256 native frames, 564 samples. Before 35.5 seconds: two temporary IDs, exactly two boxes in 516/533 sampled frames (96.8%). These are diagnostic counts, not accuracy scores.</p><p>The closing close-up around 35.52 seconds was missed by automatic scene detection and background people were included. Colored boxes and T numbers are proposed BadmintonAI tracking. Athlete identities, shots and coaching remain unverified.</p><img alt="Rally and close-up failure" src="test-highlights.jpg"><video controls preload="metadata" src="render-001/preview.mp4"></video><p><a href="index.html">Detailed session results</a></p><img alt="Sampled tracking results" src="test-contact-sheet.jpg">''',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k in ['person_count_histogram','unique_temporary_ids','covered_anchors','visible_anchors','court_view_before_35_5s','closeup_from_35_6s','scene_review','player_stage_seconds','decoded_native_preview_frames']},indent=2))
