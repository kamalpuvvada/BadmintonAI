'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
function harness(storage=new Map()){
  const template=fs.readFileSync('player_identity.html','utf8'),nodes=new Map(),downloads=[];
  const element=()=>({value:'',textContent:'',children:[],currentTime:0,disabled:false,pause(){},append(...items){this.children.push(...items);},replaceChildren(){this.children=[];},click(){if(this.download)downloads.push(this.download);else if(!this.disabled)this.onclick?.();}});
  for(const match of template.matchAll(/\bid="([^"]+)"/g))nodes.set(match[1],element());
  const data={schema_version:'1.0.0',source_binding:{tracks_sha256:'a'.repeat(64)},expected_players:2,tracklets:[0,1].map(id=>({id:`S0:T${id}:F0-4`,track_id:id,scene_id:0,start_frame:0,end_frame:4,samples:[0,1,2,3,4].map(i=>({frame_index:i,source_seconds:i/5}))}))};
  vm.runInNewContext(template.match(/<script>([\s\S]*?)<\/script>/)[1].replace('/*IDENTITY_DATA*/',JSON.stringify(data)),{
    document:{getElementById:id=>nodes.get(id),createElement:element},localStorage:{getItem:key=>storage.get(key),setItem:(key,value)=>storage.set(key,value)},
    Blob,URL:{createObjectURL(){return 'blob:test';},revokeObjectURL(){}},setTimeout:fn=>fn()});
  return {nodes,storage,downloads,click:id=>nodes.get(id).click(),assignments:()=>JSON.parse([...storage.values()][0]||'[]')};
}
test('identity assignment requires loaded video and an explicit in-range slot',()=>{
  const h=harness(),n=h.nodes;assert.equal(n.get('assign').disabled,true);n.get('assign').onclick();assert.equal(h.assignments().length,0);
  n.get('video').onloadedmetadata();n.get('slot').value='P3';h.click('assign');assert.equal(h.assignments().length,0);
  n.get('slot').value='P1';h.click('assign');assert.equal(h.assignments()[0].player_slot,'P1');
  n.get('video').onerror();assert.equal(n.get('assign').disabled,true);assert.match(n.get('playback').textContent,/could not load/);
});
test('overlapping identities are blocked, range splits survive reload and clearing is scoped',()=>{
  let h=harness(),n=h.nodes;n.get('video').onloadedmetadata();n.get('slot').value='P1';n.get('end').value=1;h.click('assign');
  n.get('start').value=3;n.get('end').value=4;n.get('slot').value='P2';h.click('assign');assert.equal(h.assignments().length,2);
  h=harness(h.storage);n=h.nodes;n.get('video').onloadedmetadata();h.click('next');n.get('slot').value='P1';h.click('assign');assert.match(n.get('message').textContent,/overlaps/);assert.equal(h.assignments().length,2);
  h.click('previous');h.click('clear');assert.equal(h.assignments().length,0);
});
test('export requires reviewer provenance, frame seek uses source time',()=>{
  const h=harness(),n=h.nodes;n.get('video').onloadedmetadata();n.get('slot').value='P1';h.click('assign');h.click('download');assert.equal(h.downloads.length,0);
  n.get('reviewer').value='Fixture';n.get('kind').value='automated';h.click('download');assert.deepEqual(h.downloads,['player-identity-corrections.json']);
  n.get('frame').value=3;n.get('frame').oninput();assert.equal(n.get('video').currentTime,.6);
});
