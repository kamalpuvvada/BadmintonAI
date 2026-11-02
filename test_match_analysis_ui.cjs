'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
function harness(storage=new Map(),seeded=false){
 const template=fs.readFileSync('match_analysis.html','utf8'),nodes=new Map(),downloads=[];
 const element=()=>{let id='';const node={value:'',textContent:'',children:[],disabled:false,currentTime:0,pause(){},play(){},append(...items){this.children.push(...items);},replaceChildren(){this.children=[];},click(){if(this.download)downloads.push(this.download);else if(!this.disabled)this.onclick?.();}};Object.defineProperty(node,'id',{get(){return id;},set(value){id=value;nodes.set(value,node);}});return node;};
 for(const match of template.matchAll(/\bid="([^"]+)"/g))nodes.set(match[1],element());
 nodes.get('initialGame').value=1;nodes.get('cursor').value=0;nodes.get('evidenceKind').value='video';
 const statuses=['unknown','proposed','accepted','rejected'],families=['unknown','serve','clear','drop','smash','drive','net','lift','block','push','kill','other'];
 const rally={id:'r1',game:1,start_frame:0,end_frame:99,kind:'play',coverage:'complete',status:'proposed',server:null,receiver:null,notes:''};
 const event={id:'e1',rally_id:'r1',start_frame:10,end_frame:10,kind:'contact',player:'P1',side:'A',status:'accepted',shot_family:'smash',shot_status:'accepted',direction:'straight',execution:'continued',evidence:[],notes:'Fixture visible contact'};
 const outcome={rally_id:'r1',status:'accepted',winner:'A',terminal_cause:'net_failure',terminal_status:'accepted',finish:'forced_error',finish_status:'accepted',error_player:'P3',last_contact_event_id:'e2',direct_winner_event_id:null,terminal_error_event_id:'e2',preceding_pressure_event_id:'e1',setup_sequence_event_ids:[],score_before:[0,0],score_after:[1,0],evidence:[],notes:'Fixture'};
 const doc={match:{teams:{A:['P1','P2'],B:['P3','P4']},rules:{profile_id:'unconfigured',target_points:null,win_by:null,cap_points:null,best_of_games:null},initial_scores:{}},rallies:seeded?[rally]:[],events:seeded?[event,{...event,id:'e2',start_frame:20,end_frame:20,player:'P3',side:'B',shot_family:'block',execution:'net'}]:[],outcomes:seeded?[outcome]:[],score_events:[],analysis_rubric:{id:'badmintonai-error-rubric-v1-draft',skill_context:'Synthetic test context'},notes:''};
 const data={packet:{schema_version:'2.0.0',revision:0,source:{fps:30,frames:300,expected_players:4},document:doc},base_sha256:'a'.repeat(64),statuses,families,finishes:['undetermined','direct_winner','forced_error','unforced_error','other_fault','not_applicable'],causes:['unknown','net_failure'],report:{playlists:[]},templates:{rally,event,outcome}};
 // Blank templates match server defaults; they never create accepted events.
 data.templates.rally={...rally,id:'rally',start_frame:0,end_frame:0,status:'proposed'};
 data.templates.event={...event,id:'event',rally_id:'rally',start_frame:0,end_frame:0,player:null,side:null,status:'proposed',shot_family:'unknown',shot_status:'unknown',direction:'unknown',execution:'unknown',evidence:[],notes:''};
 data.templates.outcome={...outcome,rally_id:'rally',status:'unknown',winner:null,terminal_cause:'unknown',terminal_status:'unknown',finish:'undetermined',finish_status:'unknown',error_player:null,last_contact_event_id:null,terminal_error_event_id:null,preceding_pressure_event_id:null,score_before:null,score_after:null,notes:''};
 vm.runInNewContext(template.match(/<script>([\s\S]*?)<\/script>/)[1].replace('/*MATCH_DATA*/',JSON.stringify(data)),{document:{getElementById:id=>nodes.get(id),createElement:element},localStorage:{getItem:key=>storage.get(key),setItem:(key,value)=>storage.set(key,value)},Blob,URL:{createObjectURL(){return 'blob:fixture';},revokeObjectURL(){}},setTimeout:fn=>fn()});
 return {nodes,storage,downloads,click:id=>nodes.get(id).click(),draft:()=>JSON.parse([...storage.values()][0])};
}
test('blank ledger invents no events and new contacts require loaded preview',()=>{
 const h=harness(),n=h.nodes;assert.equal(h.draft().events.length,0);n.get('addRally').onclick();assert.equal(h.draft().rallies.length,0);
 n.get('video').onloadedmetadata();h.click('addRally');assert.equal(h.draft().rallies.length,1);
 n.get('rallyEnd').value=99;h.click('saveRally');n.get('video').currentTime=1;h.click('addEvent');
 assert.equal(h.draft().events[0].start_frame,30);assert.equal(h.draft().events[0].status,'proposed');assert.equal(h.draft().events[0].shot_family,'unknown');
 n.get('video').onerror();assert.equal(n.get('saveEvent').disabled,true);
});
test('contact edits revoke dependent outcome acceptance and undo restores prior revision',()=>{
 const h=harness(new Map(),true),n=h.nodes;n.get('video').onloadedmetadata();n.get('eventStart').value=11;n.get('eventEnd').value=11;n.get('eventNotes').value='Corrected observation';h.click('saveEvent');
 assert.equal(h.draft().outcomes[0].status,'proposed');assert.equal(h.draft().outcomes[0].score_after,null);h.click('undo');assert.equal(h.draft().outcomes[0].status,'accepted');h.click('redo');assert.equal(h.draft().outcomes[0].status,'proposed');
});
test('deleting a terminal contact removes dangling links and draft reload preserves edits',()=>{
 let h=harness(new Map(),true),n=h.nodes;n.get('video').onloadedmetadata();n.get('eventList').value='e2';n.get('eventList').onchange();h.click('deleteEvent');
 assert.equal(h.draft().events.length,1);assert.equal(h.draft().outcomes[0].terminal_error_event_id,null);assert.equal(h.draft().outcomes[0].last_contact_event_id,null);
 h=harness(h.storage,true);assert.equal(h.draft().events.length,1);
});
test('changing only shot class preserves an independently confirmed point',()=>{
 const h=harness(new Map(),true),n=h.nodes;n.get('video').onloadedmetadata();n.get('family').value='drop';h.click('saveEvent');
 assert.equal(h.draft().events[0].shot_family,'drop');assert.equal(h.draft().outcomes[0].status,'accepted');assert.deepEqual(h.draft().outcomes[0].score_after,[1,0]);
});
test('a changed skill rubric requires new forcedness review while retaining the point',()=>{
 const h=harness(new Map(),true),n=h.nodes;n.get('video').onloadedmetadata();n.get('rules').value='unconfigured';n.get('rubricContext').value='Different academy level';h.click('saveMatch');
 assert.equal(h.draft().outcomes[0].status,'accepted');assert.equal(h.draft().outcomes[0].finish_status,'proposed');
});
test('correction download needs explicit reviewer provenance and frame navigation clamps',()=>{
 const h=harness(),n=h.nodes;h.click('download');assert.equal(h.downloads.length,0);n.get('reviewer').value='Fixture';n.get('reviewerKind').value='automated';h.click('download');assert.deepEqual(h.downloads,['match-analysis-corrections.json']);
 n.get('cursor').value=999;h.click('seek');assert.equal(n.get('video').currentTime,299/30);h.click('nextFrame');assert.equal(n.get('video').currentTime,299/30);
});
test('marking a rally at playback end uses the last native frame rather than one past it',()=>{
 const h=harness(),n=h.nodes;n.get('video').onloadedmetadata();n.get('video').currentTime=10;h.click('addRally');
 assert.equal(h.draft().rallies[0].start_frame,299);assert.equal(h.draft().rallies[0].end_frame,299);
});
