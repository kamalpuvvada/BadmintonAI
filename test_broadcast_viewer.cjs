const test=require('node:test');
const assert=require('node:assert/strict');
const {stateAt,shotAt}=require('./broadcast_viewer.js');
const row=(i,scene=0,status='experimental_court')=>({timestamp_seconds:i/15,native_frame_index:i*4,scene_id:scene,status,players:[{temporary_id:0,court_position:[.3,.7]},{temporary_id:1,court_position:[.6,.2]}]});
test('seeking recomputes movement and occupancy without retaining future data',()=>{
 const rows=[row(0),row(1),row(2)];
 assert.equal(stateAt(rows,2/15).trails['0'].length,3);
 assert.equal(stateAt(rows,0).trails['0'].length,1);
});
test('camera changes pause maps and reset heat on return',()=>{
 const rows=[row(0),row(1,1,'excluded_view_change'),row(2,2)];
 assert.equal(stateAt(rows,1/15).paused,true);
 const state=stateAt(rows,2/15);assert.equal(state.trails['0'].length,1);assert.equal(state.heat.length,2);
});
test('player filtering and missing IDs do not connect trails',()=>{
 const rows=[row(0),row(1),row(2)];rows[1].players=[];
 const state=stateAt(rows,2/15,'1');assert.deepEqual(Object.keys(state.trails),['1']);assert.equal(state.trails['1'].length,1);
});
test('shot hint expires and cannot cross camera scenes',()=>{
 const events=[{timestamp_seconds:1,scene_id:0,shot_hint:'Lift?'}];
 assert.equal(shotAt(events,1.2,0).shot_hint,'Lift?');assert.equal(shotAt(events,2.2,0),null);assert.equal(shotAt(events,1.2,1),null);
});
test('native view boundary pauses before the next sampled court row',()=>{
 const rows=[{...row(0),native_frame_index:2128,timestamp_seconds:2128/60}];
 assert.equal(stateAt(rows,2131/60,'both',[2131]).paused,true);
});
test('pose shot predictions use their declared display duration',()=>{
 const event={timestamp_seconds:2,scene_id:0,shot_hint:'Smash?',display_seconds:1.5};
 assert.equal(shotAt([event],3.3,0),event);
 assert.equal(shotAt([event],3.6,0),null);
});
test('fractional-frame seeks tolerate browser microsecond rounding',()=>{
 const event={timestamp_seconds:356/60,scene_id:0,shot_hint:'Smash?',display_seconds:1.5};
 assert.equal(shotAt([event],5.933333,0),event);
 assert.equal(shotAt([event],5.9,0),null);
});
