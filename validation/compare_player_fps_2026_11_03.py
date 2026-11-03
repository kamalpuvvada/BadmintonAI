"""Reproduce this local video experiment; torso anchors are diagnostic, not ground truth."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / 'validation/player_anchor_seed_2026-11-03.json'
OUT = ROOT / 'out/academy-session-005/test-comparison'
OUT.mkdir(parents=True, exist_ok=True)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assign(anchors, persons):
    """Maximum cardinality one-to-one containment, then nearest normalized center."""
    options = []
    for anchor in anchors:
        x, y = anchor['torso_point_px']
        candidates = []
        for i, person in enumerate(persons):
            x1, y1, x2, y2 = person['bbox_xyxy_px']
            if x1 <= x <= x2 and y1 <= y <= y2:
                cost = ((x-(x1+x2)/2)/max(1,x2-x1))**2 + ((y-(y1+y2)/2)/max(1,y2-y1))**2
                candidates.append((i, cost))
        options.append(candidates)
    solutions = []

    def search(index, used, chosen, cost):
        if index == len(anchors):
            solutions.append((len(used), cost, chosen))
            return
        search(index+1, used, chosen+[None], cost)
        for i, distance in options[index]:
            if i not in used:
                search(index+1, used | {i}, chosen+[i], cost+distance)

    search(0, set(), [], 0)
    best = min(solutions, key=lambda value: (-value[0], value[1]))[2]
    return [dict(appearance_tag=a['appearance_tag'], point=a['torso_point_px'],
                 candidate_box_indices=[i for i, _ in options[k]], selected_box_index=best[k],
                 selected_temporary_id=None if best[k] is None else persons[best[k]]['track_id'])
            for k, a in enumerate(anchors)]


seed = read(SEED)
report = {
    'schema_version': '0.1.0', 'experiment': 'Same real portrait doubles clip at 15 versus 30 sampled FPS',
    'source_video_sha256': seed['source_video_sha256'], 'anchor_seed_sha256': sha(SEED),
    'reviewer_kind': 'automated', 'ground_truth_verified': False,
    'method': '23 visible torso anchors on six source frames; maximum-cardinality one-to-one box containment, then normalized center distance. One partial-occlusion anchor excluded. Sparse, non-blind internal diagnostic; not precision/recall, IDF1 or continuity approval.',
    'alignment': '30 FPS uses the exact native frame index; 15 FPS uses the preceding nominal sample at floor(native_index/2). FFmpeg fps-filter selection and approximate timestamps mean the baseline is not guaranteed to be the identical source frame.',
    'source_graphics_are_ground_truth': False, 'runs': {},
}
datasets = {}
for name, folder in [('15fps', 'academy-session-004'), ('30fps', 'academy-session-005')]:
    base = ROOT / 'out' / folder
    job, result = read(base/'job.json'), read(base/'result.json')
    assert result['source']['sha256'] == seed['source_video_sha256']
    tracks = base/'players-001/player_tracks.jsonl'
    rows = [json.loads(line) for line in tracks.read_text(encoding='utf-8').splitlines()]
    datasets[name] = rows
    histogram = Counter(len(row['persons']) for row in rows)
    previous, tracklets = set(), 0
    for row in rows:
        current = {p['track_id'] for p in row['persons'] if p['track_id'] is not None}
        tracklets += len(current - previous)
        previous = current
    cases = []
    for case in seed['cases']:
        index = case['native_frame_index'] // (2 if name == '15fps' else 1)
        visible = [a for a in case['anchors'] if a['visibility'] == 'visible']
        cases.append(dict(native_frame_index=case['native_frame_index'], sampled_frame_index=index,
                          anchors=assign(visible, rows[index]['persons'])))
    anchors = [a for c in cases for a in c['anchors']]
    report['runs'][name] = dict(
        directory=str(base.relative_to(ROOT)), tracks_sha256=sha(tracks), result_sha256=sha(base/'result.json'),
        configuration=result['configuration'], model_binding=job['binding']['models']['person'],
        frames=len(rows), person_count_histogram=dict(histogram),
        four_box_frame_fraction=histogram[4]/len(rows),
        unique_temporary_ids=result['players']['diagnostics']['unique_track_ids_observed'],
        contiguous_tracklets=tracklets,
        detector_tracker_stage_seconds=job['stages']['players']['elapsed_seconds'],
        preview=read(base/'render-001/preview.json'),
        covered_visible_anchors=sum(a['selected_box_index'] is not None for a in anchors),
        visible_anchors=len(anchors), ambiguous_containment_anchors=sum(len(a['candidate_box_indices']) > 1 for a in anchors),
        cases=cases,
    )

baseline, native = report['runs'].values()
assert baseline['model_binding'] == native['model_binding']
assert {k:v for k,v in baseline['configuration'].items() if k != 'sample_fps'} == {k:v for k,v in native['configuration'].items() if k != 'sample_fps'}
report['decision'] = 'Do not promote 30 FPS: four-box frequency is lower and more temporary IDs appear. Full identity continuity remains unverified. Inspect overlap/duplicate detections and identity association next; no coaching or outcome claims.'

for case in seed['cases']:
    frame = case['native_frame_index']
    rawpath = ROOT / f'out/academy-native-fps-test-2026-11-03/raw-{frame:06d}.png'
    assert sha(rawpath) == case['source_image_sha256']
    raw = Image.open(rawpath).convert('RGB')
    canvas = Image.new('RGB', (2160, 1320), '#111827')
    for column, name in enumerate(['Source', '15fps', '30fps']):
        panel = raw.copy()
        draw = ImageDraw.Draw(panel)
        if name != 'Source':
            row = datasets[name][frame//(2 if name == '15fps' else 1)]
            for p in row['persons']:
                box = p['bbox_xyxy_px']
                draw.rectangle(box, outline='#ff8800', width=3)
                draw.text((box[0], box[1]-12), f"T{p['track_id']}", fill='#ff8800', stroke_width=1, stroke_fill='black')
        for anchor in case['anchors']:
            x, y = anchor['torso_point_px']
            color = '#00ffff' if anchor['visibility'] == 'visible' else '#999999'
            draw.ellipse((x-4,y-4,x+4,y+4), fill=color)
        canvas.paste(panel, (column*720, 40))
        ImageDraw.Draw(canvas).text((column*720+12, 12), f'{name} | source frame {frame} | {frame/30:.3f}s', fill='white')
    canvas.save(OUT/f'comparison-{frame:06d}.jpg', quality=90)

preview = cv2.VideoCapture(str(ROOT/'out/academy-session-005/render-001/preview.mp4'))
decoded = 0
while preview.read()[0]:
    decoded += 1
preview.release()
assert decoded == 665
report['native_preview_decoded_frames'] = decoded
reportpath = ROOT/'validation/player_fps_comparison_2026-11-03.json'
reportpath.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
html = '<!doctype html><meta charset="utf-8"><title>15 / 30 FPS video test</title><style>body{background:#111827;color:white;font:18px system-ui;margin:24px}img{width:100%}a{color:#67e8f9}</style><h1>Real video: 15 versus 30 FPS</h1><p>Orange boxes and T labels: BadmintonAI proposals. Cyan points: sparse internal torso anchors. All other graphics are embedded in the source and are not ground truth. IDs and coaching remain unverified.</p><p>30 FPS was not promoted. It produced more temporary IDs and a lower four-box frame frequency. These counts are not accuracy measures.</p><p><a href="../index.html">Session results and complete preview</a></p>'
for case in seed['cases']:
    html += f'<p>Source / 15 FPS / 30 FPS</p><img alt="Source and proposed boxes" src="comparison-{case["native_frame_index"]:06d}.jpg">'
(OUT/'index.html').write_text(html, encoding='utf-8')
print(json.dumps({k:{x:v for x,v in run.items() if x in ('frames','four_box_frame_fraction','unique_temporary_ids','contiguous_tracklets','covered_visible_anchors','visible_anchors','ambiguous_containment_anchors','detector_tracker_stage_seconds')} for k,run in report['runs'].items()}, indent=2))
