"use strict";

function nativePoint(clientX, clientY, rect, size) {
  if (!Array.isArray(size) || size.length !== 2 || !size.every(v => Number.isInteger(v) && v > 0)) throw new Error("Invalid image size");
  if (![clientX, clientY, rect.left, rect.top, rect.width, rect.height, ...size].every(Number.isFinite) || rect.width <= 0 || rect.height <= 0) throw new Error("Invalid image coordinates");
  const x = (clientX - rect.left) * size[0] / rect.width;
  const y = (clientY - rect.top) * size[1] / rect.height;
  if (x < 0 || y < 0 || x >= size[0] || y >= size[1]) throw new Error("Click inside the source image");
  // Rounding at the right/bottom edge must remain within the native image.
  return [Math.min(size[0] - .001, Math.round(x * 1000) / 1000), Math.min(size[1] - .001, Math.round(y * 1000) / 1000)];
}

function correctionPatch(data, entries, reviewer, kind) {
  if (typeof reviewer !== "string" || !reviewer.trim() || !["human", "automated"].includes(kind)) throw new Error("Enter your name and choose a review type");
  if (!Array.isArray(entries) || entries.length === 0) throw new Error("Mark or confirm at least one frame");
  const ids = new Set(data.cases.map(c => c.id));
  const seen = new Set();
  for (const c of entries) {
    if (!ids.has(c.case_id) || seen.has(c.case_id) || Object.keys(c).sort().join(",") !== "case_id,notes,point_px,view,visibility") throw new Error("Invalid corrected frame");
    seen.add(c.case_id);
    if (typeof c.notes !== "string" || !["wide_court", "closeup", "unknown", "unspecified"].includes(c.view)) throw new Error("Invalid review note or view");
    if (c.visibility === "uncertain") {
      if (c.point_px !== null) throw new Error("An uncertain frame cannot have a point");
    } else if (c.visibility === "visible") {
      if (!Array.isArray(c.point_px) || c.point_px.length !== 2 || !c.point_px.every((v, i) => Number.isFinite(v) && v >= 0 && v < data.image_size_px[i])) throw new Error("Enter a point inside the source image");
    } else throw new Error("Invalid visibility");
  }
  return {schema_version: "0.1.0", benchmark_sha256: data.benchmark_sha256, reviewer: reviewer.trim(), reviewer_kind: kind, changes: entries};
}

if (typeof module !== "undefined") module.exports = {nativePoint, correctionPatch};

if (typeof document !== "undefined") {
  const data = JSON.parse(document.getElementById("annotation-data").textContent);
  const $ = id => document.getElementById(id);
  const canvas = $("frame"), ctx = canvas.getContext("2d"), size = data.image_size_px;
  canvas.width = size[0]; canvas.height = size[1];
  $("pointX").max = size[0] - .001; $("pointY").max = size[1] - .001;
  const changes = new Map();
  const cacheKey = "badminton-shuttle-annotation:" + data.benchmark_sha256;
  let index = 0, current, sourceImage = null, generation = 0;
  const labelControls = ["usePoint", "markUncertain", "confirm", "view", "notes"];
  function allowLabels(allowed) { labelControls.forEach(id => { $(id).disabled = !allowed; }); }
  function frameReady() {
    if (sourceImage) return true;
    message("Wait for the source frame to load before reviewing its label.");
    return false;
  }
  const labelOf = c => ({case_id: c.id, visibility: c.visibility, point_px: c.point_px, view: c.view || "unspecified", notes: c.notes || ""});
  function message(text) { $("message").textContent = text; }
  function persist() {
    try { localStorage.setItem(cacheKey, JSON.stringify({benchmark_sha256: data.benchmark_sha256, changes: [...changes.values()], reviewer: $("reviewer").value, reviewer_kind: $("reviewerKind").value})); }
    catch { message("Browser storage is unavailable. Download corrections to keep your work."); }
    $("progress").textContent = `${changes.size} of ${data.cases.length} frames marked or confirmed in this draft.`;
    $("download").disabled = changes.size === 0;
  }
  try {
    const draft = JSON.parse(localStorage.getItem(cacheKey) || "null");
    if (draft && draft.benchmark_sha256 === data.benchmark_sha256 && draft.changes.length) {
      // Validate restored edits even when reviewer details have not been entered yet.
      correctionPatch(data, draft.changes, "Draft validation", "automated");
      draft.changes.forEach(c => changes.set(c.case_id, c));
      $("reviewer").value = typeof draft.reviewer === "string" ? draft.reviewer : "";
      $("reviewerKind").value = draft.reviewer_kind || "";
    }
  } catch { message("Saved draft could not be restored; the source labels are unchanged."); }
  const ordered = data.cases.map((c, i) => ({c, i})).sort((a,b) => a.c.source_run.localeCompare(b.c.source_run) || a.c.frame_index - b.c.frame_index);
  for (const {c, i} of ordered) {
    const option = document.createElement("option"); option.value = i; option.textContent = `${c.source_run} · frame ${c.frame_index}`; $("frameSelect").append(option);
  }
  function draw() {
    ctx.clearRect(0, 0, ...size);
    if (sourceImage) ctx.drawImage(sourceImage, 0, 0, ...size);
    if (current.point_px) {
      const [x,y] = current.point_px;
      ctx.strokeStyle = "#38ff93"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x,y,6,0,Math.PI*2); ctx.stroke();
    }
    const zoom = Number($("zoom").value);
    canvas.style.width = `${size[0]*zoom}px`; canvas.style.height = `${size[1]*zoom}px`;
  }
  function showLabel() {
    $("labelStatus").textContent = current.visibility === "visible" ? `Visible · (${current.point_px.join(", ")}) px` : "Uncertain · no location";
    $("pointX").value = current.point_px ? current.point_px[0] : "";
    $("pointY").value = current.point_px ? current.point_px[1] : "";
    draw();
  }
  function record() {
    if (!frameReady()) return false;
    current.view = $("view").value; current.notes = $("notes").value;
    changes.set(current.case_id, structuredClone(current)); persist(); showLabel();
    return true;
  }
  function load(next) {
    index = next; const c = data.cases[index];
    current = structuredClone(changes.get(c.id) || labelOf(c));
    $("view").value = current.view; $("notes").value = current.notes; $("frameSelect").value = index;
    const sampling = data.sources[c.source_run];
    $("frameTitle").textContent = `${c.source_run} · source frame ${c.frame_index} · ${(sampling.start_seconds+c.frame_index/sampling.fps).toFixed(3)} seconds`;
    sourceImage = null; allowLabels(false); showLabel(); const ticket = ++generation;
    const image = new Image(); image.onload = () => { if (ticket === generation) { sourceImage = image; allowLabels(true); draw(); message(""); } };
    image.onerror = () => { if (ticket === generation) message("Source image could not be loaded. Label editing is unavailable for this frame."); };
    image.src = c.image;
    $("context").replaceChildren();
    for (const offset of [-1,1]) {
      const adjacent = data.cases.find(a => a.source_run === c.source_run && a.frame_index === c.frame_index + offset);
      if (!adjacent) continue;
      const figure = document.createElement("figure"), thumb = document.createElement("img"), caption = document.createElement("figcaption");
      thumb.src = adjacent.image; thumb.alt = `${offset < 0 ? "Previous" : "Next"} source frame ${adjacent.frame_index}`;
      caption.textContent = `${offset < 0 ? "Previous" : "Next"} · frame ${adjacent.frame_index} · ${(sampling.start_seconds+adjacent.frame_index/sampling.fps).toFixed(3)}s`;
      figure.append(thumb,caption); $("context").append(figure);
    }
  }
  function move(offset) {
    const position = ordered.findIndex(x => x.i === index);
    load(ordered[(position+offset+ordered.length)%ordered.length].i);
  }
  function findNext(predicate) {
    const position = ordered.findIndex(x => x.i === index);
    for (let k=1;k<=ordered.length;k++) { const item=ordered[(position+k)%ordered.length]; if (predicate(item.c)) { load(item.i); return; } }
    message("No matching frames remain.");
  }
  canvas.addEventListener("click", event => {
    if (!sourceImage) return;
    try { current.point_px=nativePoint(event.clientX,event.clientY,canvas.getBoundingClientRect(),size); current.visibility="visible"; record(); message("Location saved in the browser draft."); }
    catch (error) { message(error.message); }
  });
  $("usePoint").onclick = () => {
    if (!frameReady()) return;
    const values=[$("pointX").value,$("pointY").value];
    if (values.some(v => !v.trim())) { message("Enter both native-pixel coordinates."); return; }
    const candidate={...current, visibility:"visible", point_px:values.map(Number), view:$("view").value, notes:$("notes").value};
    try { correctionPatch(data,[candidate],"Draft validation","automated"); current=candidate; record(); message("Location saved in the browser draft."); }
    catch (error) { message(error.message); }
  };
  $("markUncertain").onclick = () => { if (!frameReady()) return; current.visibility="uncertain"; current.point_px=null; record(); message("Uncertain label saved; no absence claim."); };
  $("confirm").onclick = () => { if (record()) message("Current label marked as reviewed in this draft."); };
  $("reset").onclick = () => { changes.delete(current.case_id); load(index); persist(); message("This frame restored to its original packet label."); };
  $("notes").oninput = record; $("view").onchange = record;
  $("reviewer").oninput = persist; $("reviewerKind").onchange = persist;
  $("previous").onclick = () => move(-1); $("next").onclick = () => move(1);
  $("uncertainNext").onclick = () => findNext(c => (changes.get(c.id) || c).visibility === "uncertain");
  $("uncheckedNext").onclick = () => findNext(c => !changes.has(c.id));
  $("frameSelect").onchange = () => load(Number($("frameSelect").value));
  $("zoom").onchange = draw;
  document.addEventListener("keydown", event => {
    if (["INPUT","TEXTAREA","SELECT","BUTTON"].includes(event.target.tagName) || event.target.isContentEditable) return;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") { event.preventDefault(); move(event.key === "ArrowLeft" ? -1 : 1); }
  });
  $("download").onclick = () => {
    try {
      const patch=correctionPatch(data,[...changes.values()],$("reviewer").value,$("reviewerKind").value);
      const url=URL.createObjectURL(new Blob([JSON.stringify(patch,null,2)+"\n"],{type:"application/json"}));
      const link=document.createElement("a"); link.href=url; link.download="shuttle-corrections.json"; link.click();
      setTimeout(() => URL.revokeObjectURL(url),1000); message("Corrections downloaded. Import them into a new packet with the project tool.");
    } catch (error) { message(error.message); }
  };
  load(ordered[0].i); persist();
}
