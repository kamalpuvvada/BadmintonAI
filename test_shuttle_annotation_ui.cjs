"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const {nativePoint, correctionPatch} = require("./shuttle_annotation.js");

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
