"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const {nativePoint, correctionPatch} = require("./shuttle_annotation.js");
const fs = require("node:fs");
const vm = require("node:vm");

// Application-level tests use a small DOM double, not a browser/layout engine.
function editorHarness(storage = new Map()) {
  const data = {benchmark_sha256:"a".repeat(64),image_size_px:[640,360],
    sources:{synthetic:{start_seconds:0,fps:30,frames:3}},
    cases:[0,1,2].map(i => ({id:`frame-${i}`,source_run:"synthetic",frame_index:i,
      visibility:i === 0 ? "visible" : "uncertain",point_px:i === 0 ? [20,30] : null,
      view:"wide_court",notes:"Synthetic fixture",image:`frame-${i}.png`}))};
  const nodes = new Map(), images = [], downloads = [];
  const context = {clearRect(){},drawImage(){},beginPath(){},arc(){},stroke(){}};
  function element() { return {value:"",disabled:false,textContent:"",style:{},children:[],listeners:{},
    append(...items){this.children.push(...items);},replaceChildren(){this.children=[];},
    addEventListener(name,handler){this.listeners[name]=handler;},
    getContext(){return context;},
    getBoundingClientRect(){return {left:-100,top:-40,width:1280,height:720};},
    click(){if(this.download) downloads.push(this.download);else if(!this.disabled) this.onclick?.();}}; }
  for (const id of fs.readFileSync("shuttle_annotation.html","utf8").matchAll(/\bid="([^"]+)"/g)) nodes.set(id[1],element());
  nodes.get("annotation-data").textContent=JSON.stringify(data);
  nodes.get("zoom").value="2";
  const document={getElementById:id=>nodes.get(id),createElement:element,addEventListener(){}};
  class SourceImage { constructor(){images.push(this);} set src(value){this.path=value;} }
  const sandbox={document,Image:SourceImage,structuredClone,Blob,URL:{createObjectURL(){return "blob:synthetic";},revokeObjectURL(){}},
    localStorage:{getItem:key=>storage.get(key),setItem:(key,value)=>storage.set(key,value)},setTimeout:fn=>fn()};
  vm.runInNewContext(fs.readFileSync("shuttle_annotation.js","utf8"),sandbox);
  return {nodes,images,storage,downloads,data,click:id=>nodes.get(id).click(),
    draft:()=>JSON.parse([...storage.values()][0])};
}

test("native points account for zoom and a scrolled canvas rectangle", () => {
  assert.deepEqual(nativePoint(540,320,{left:-100,top:-40,width:1280,height:720},[640,360]),[320,180]);
  assert.deepEqual(nativePoint(125,75,{left:100,top:50,width:640,height:360},[640,360]),[25,25]);
  assert.deepEqual(nativePoint(639.99999,359.99999,{left:0,top:0,width:640,height:360},[640,360]),[639.999,359.999]);
  for(const [x,y] of [[-1,0],[640,0],[0,360],[NaN,1]]) assert.throws(() => nativePoint(x,y,{left:0,top:0,width:640,height:360},[640,360]));
  assert.throws(() => nativePoint(1,1,{left:0,top:0,width:0,height:360},[640,360]));
});

test("export requires explicit reviewer type and cannot turn uncertainty into a point", () => {
  const data={benchmark_sha256:"a".repeat(64),image_size_px:[640,360],cases:[{id:"frame-1"}]};
  const change={case_id:"frame-1",visibility:"uncertain",point_px:null,view:"wide_court",notes:"Unclear"};
  const patch=correctionPatch(data,[change]," Synthetic reviewer ","automated");
  assert.equal(patch.reviewer,"Synthetic reviewer");assert.equal(patch.reviewer_kind,"automated");
  assert.equal(patch.benchmark_sha256,data.benchmark_sha256);assert.equal(patch.changes[0].point_px,null);
  assert.throws(() => correctionPatch(data,[change],"","human"));
  assert.throws(() => correctionPatch(data,[change],"Name",""));
  assert.throws(() => correctionPatch(data,[],"Name","automated"));
  assert.throws(() => correctionPatch(data,[change,change],"Name","automated"));
  assert.throws(() => correctionPatch(data,[{...change,point_px:[1,1]}],"Name","automated"));
  assert.throws(() => correctionPatch(data,[{...change,visibility:"visible",point_px:[Infinity,0]}],"Name","automated"));
  assert.throws(() => correctionPatch(data,[{...change,source_video_sha256:"changed"}],"Name","automated"));
});

test("unloaded, failed and stale source images cannot approve frame labels", () => {
  const h=editorHarness(), n=h.nodes;
  assert.equal(n.get("confirm").disabled,true);
  n.get("confirm").onclick(); // Also guard programmatic events.
  n.get("markUncertain").onclick();
  assert.equal(h.draft().changes.length,0);
  h.click("next");
  h.images[0].onload(); // Previous frame finishes after navigation.
  assert.equal(n.get("confirm").disabled,true);
  h.images[1].onerror();
  assert.match(n.get("message").textContent,/could not be loaded/);
  n.get("usePoint").onclick();
  assert.equal(h.draft().changes.length,0);
  h.click("next");h.images.at(-1).onload();h.click("confirm");
  assert.equal(h.draft().changes[0].case_id,"frame-2");
  assert.equal(h.draft().changes[0].point_px,null);
});

test("native point drafts survive reload, export needs reviewer and reset restores source", () => {
  let h=editorHarness();h.images[0].onload();
  h.nodes.get("frame").listeners.click({clientX:540,clientY:320});
  assert.deepEqual(h.draft().changes[0].point_px,[320,180]);
  h.click("next");h.images.at(-1).onload();
  assert.equal(h.nodes.get("pointX").value,""); // Never carry a location forward.
  h.nodes.get("pointX").value="640";h.nodes.get("pointY").value="20";h.click("usePoint");
  assert.equal(h.draft().changes.length,1); // Invalid point rejected.
  h.nodes.get("pointX").value="40";h.click("usePoint");
  assert.deepEqual(h.draft().changes[1].point_px,[40,20]);
  h.click("markUncertain");assert.equal(h.draft().changes[1].point_px,null);
  h=editorHarness(h.storage);h.images[0].onload();
  assert.equal(h.draft().changes.length,2);
  assert.equal(h.nodes.get("pointX").value,320);
  h.click("download");assert.equal(h.downloads.length,0);
  assert.match(h.nodes.get("message").textContent,/name and choose/);
  h.nodes.get("reviewer").value="Synthetic automated test";
  h.nodes.get("reviewerKind").value="automated";
  h.click("download");assert.deepEqual(h.downloads,["shuttle-corrections.json"]);
  h.click("reset");assert.equal(h.draft().changes.length,1);
  assert.equal(h.nodes.get("pointX").value,20);
  h.click("uncheckedNext");assert.match(h.nodes.get("frameTitle").textContent,/source frame 2/);
  h.images.at(-1).onload();h.click("previous");h.images.at(-1).onload();h.click("reset");
  assert.equal(h.draft().changes.length,0);
  assert.equal(h.nodes.get("download").disabled,true);
});
