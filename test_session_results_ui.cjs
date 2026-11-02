'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

// DOM doubles verify application behavior, not actual browser layout or decoding.
function harness(){
  const template=fs.readFileSync('session_results.html','utf8'), nodes=new Map();
  const element=()=>({textContent:'',children:[],value:0,currentTime:0,pause(){this.paused=true;},append(...items){this.children.push(...items);}});
  for(const match of template.matchAll(/\bid="([^"]+)"/g))nodes.set(match[1],element());
  const session={status:'needs_review',source:{video:{filename:'source <test>.mp4'}},preview:{frames:10,fps:10},
    artifacts:{preview:'render folder/preview.mp4',player_review:'players/tracking_review.html',observations:'players/rows.jsonl',detector_sheet:'players/detector.jpg',shuttle_summary:null},
    players:{sampled_frames:5,person_count_histogram:{2:5},diagnostics:{unique_track_ids_observed:2},review:{decisions:{pending:1}}},
    shuttle:{status:'not_available'},metrics:{distance_metres:{status:'not_available',reason:'Not calibrated',value:null}},limitations:['Fixture limitation']};
  const code=template.match(/<script>([\s\S]*?)<\/script>/)[1].replace('/*SESSION_DATA*/',JSON.stringify(session));
  vm.runInNewContext(code,{document:{getElementById:id=>nodes.get(id),createElement:element}});
  return nodes;
}

test('frame stepping and timeline clamp to native bounds',()=>{
  const n=harness(),video=n.get('video');
  n.get('back').onclick();assert.equal(video.currentTime,0);
  n.get('next').onclick();assert.equal(video.currentTime,.1);assert.match(n.get('timeline').textContent,/Frame 1/);
  n.get('seek').value=99;n.get('seek').oninput();assert.equal(video.currentTime,.9);
  video.currentTime=1;video.ontimeupdate();assert.match(n.get('timeline').textContent,/Frame 9/);
  assert.equal(video.paused,true);
});

test('pending review and unavailable measurements never appear as accepted or zero',()=>{
  const n=harness();assert.equal(n.get('status').textContent,'Tracking review needed');
  assert.match(n.get('reviewSummary').textContent,/pending/);
  assert.equal(n.get('metrics').children[0].children[1].textContent,'not available');
  assert.equal(n.get('links').children.length,4); // No invented shuttle link.
  assert.equal(n.get('video').src,'render%20folder/preview.mp4');
  assert.equal(n.get('source').textContent,'source <test>.mp4 · 10 frames · 10 FPS');
});

test('video loading failures give a useful local recovery message',()=>{
  const n=harness();n.get('video').onerror();assert.match(n.get('playback').textContent,/could not be loaded/);
});
