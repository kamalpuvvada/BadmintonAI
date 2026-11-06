(function(root){
  'use strict';
  function stateAt(rows,time,player='both',boundaries=[]){
    let trails={},heat=[],current=null,previous=null;
    for(const row of rows){
      if(row.timestamp_seconds>time+1e-7)break;
      const valid=row.status==='experimental_court';
      if(!valid || !previous || previous.scene_id!==row.scene_id || row.native_frame_index-previous.native_frame_index!==4){trails={};heat=[];}
      current=row;
      if(valid){
        const visible=new Set(row.players.map(p=>String(p.temporary_id)));
        for(const id of Object.keys(trails))if(!visible.has(id))delete trails[id];
        for(const p of row.players){
          const id=String(p.temporary_id);
          if(player!=='both' && player!==id)continue;
          trails[id]=[...(trails[id]||[]),p.court_position].slice(-24);
          if(p.court_position.every(v=>v>=0&&v<=1))heat.push({point:p.court_position,seconds:4/60});
        }
      }
      previous=row;
    }
    const between=!!current&&boundaries.some(edge=>edge>current.native_frame_index&&edge<=Math.floor(time*60+1e-7));
    return {current,trails,heat,paused:!current || current.status!=='experimental_court'||between};
  }
  function shotAt(events,time,scene){return events.filter(e=>e.scene_id===scene&&time>=e.timestamp_seconds&&time-e.timestamp_seconds<1.1).at(-1)||null;}
  const api={stateAt,shotAt};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  root.BroadcastMaps=api;
  if(typeof document==='undefined')return;
  const data=JSON.parse(document.getElementById('broadcast-data').textContent);
  const el=id=>document.getElementById(id),video=el('video'),dialog=el('map-dialog');
  let largeMode='movement',lastKey='';
  function draw(canvas,state,mode){
    const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
    ctx.fillStyle='#101923';ctx.fillRect(0,0,w,h);
    if(state.paused){ctx.fillStyle='#c9d7e5';ctx.font='32px system-ui';ctx.fillText('View change · map paused',35,h/2);return;}
    const mh=h-100,mw=mh*6.1/13.4,x=(w-mw)/2,y=50;
    ctx.fillStyle='#234636';ctx.fillRect(x,y,mw,mh);
    const pixel=p=>[x+p[0]*mw,y+p[1]*mh];
    if(mode==='heatmap'){
      const bins=new Float64Array(32*64);let peak=0;
      for(const sample of state.heat){
        const gx=Math.min(31,Math.floor(sample.point[0]*32)),gy=Math.min(63,Math.floor(sample.point[1]*64));
        bins[gy*32+gx]+=sample.seconds;peak=Math.max(peak,bins[gy*32+gx]);
      }
      for(let j=0;j<64;j++)for(let i=0;i<32;i++){
        const strength=bins[j*32+i]/(peak||1);if(!strength)continue;
        ctx.fillStyle=`rgba(255,${Math.round(80+175*strength)},55,${.3+.7*strength})`;
        ctx.fillRect(x+i*mw/32,y+j*mh/64,mw/32+1,mh/64+1);
      }
    }
    ctx.strokeStyle='#d1e5d8';ctx.lineWidth=2;
    for(const [a,b] of data.lines){ctx.beginPath();ctx.moveTo(...pixel(a));ctx.lineTo(...pixel(b));ctx.stroke();}
    ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(...pixel([0,.5]));ctx.lineTo(...pixel([1,.5]));ctx.stroke();
    for(const [id,points] of Object.entries(state.trails)){
      const color=id==='0'?'#66e4ef':'#ffae73';ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=5;
      if(mode==='movement'&&points.length){ctx.beginPath();ctx.moveTo(...pixel(points[0]));for(const p of points.slice(1))ctx.lineTo(...pixel(p));ctx.stroke();}
      if(points.length){const p=pixel(points.at(-1));ctx.beginPath();ctx.arc(...p,10,0,Math.PI*2);ctx.fill();ctx.font='24px system-ui';ctx.fillText('T'+id,p[0]+14,p[1]);}
    }
  }
  function refresh(force=false){
    const key=Math.floor(video.currentTime*15)+'|'+el('player').value+'|'+dialog.open+'|'+largeMode;
    if(!force&&key===lastKey)return;lastKey=key;
    const state=stateAt(data.court,video.currentTime,el('player').value,data.boundaries||[]);
    draw(el('movement'),state,'movement');draw(el('heatmap'),state,'heatmap');
    if(dialog.open)draw(el('large-map'),state,largeMode);
    const message=video.currentTime.toFixed(2)+'s · '+(state.paused?'View change: maps paused':'Experimental movement · box-bottom positions');
    el('status').textContent=message;el('large-status').textContent=message;
    const shot=state.current&&!state.paused?shotAt(data.events,video.currentTime,state.current.scene_id):null;
    el('shot').textContent=shot?'Shot hypothesis: '+shot.shot_hint:'No resolved shot hypothesis at this time';
  }
  function enlarge(mode){largeMode=mode;el('dialog-title').textContent=mode==='movement'?'Large movement map':'Large occupancy heatmap';dialog.showModal();refresh(true);}
  el('large-movement').addEventListener('click',()=>enlarge('movement'));
  el('large-heatmap').addEventListener('click',()=>enlarge('heatmap'));
  el('close-map').addEventListener('click',()=>dialog.close());
  el('player').addEventListener('change',()=>refresh(true));
  video.addEventListener('timeupdate',()=>refresh());video.addEventListener('seeked',()=>refresh(true));
  let animating=false;
  function animate(){refresh();if(!video.paused&&!video.ended)requestAnimationFrame(animate);else animating=false;}
  video.addEventListener('play',()=>{if(!animating){animating=true;requestAnimationFrame(animate);}});
  function jumpToShot(event){
    video.pause();
    video.currentTime=event.timestamp_seconds;
    refresh(true);
  }
  const namedShots=data.events.filter(event=>event.shot_hint && event.shot_hint!=='Unknown shot');
  if(!namedShots.length)el('shot-moments-help').textContent='This clip has no named shot hypotheses.';
  for(const event of namedShots){
    const button=document.createElement('button');
    button.type='button';
    button.textContent=`${event.shot_hint} · ${event.timestamp_seconds.toFixed(2)}s`;
    button.addEventListener('click',()=>jumpToShot(event));
    el('shot-moments').appendChild(button);
  }
  for(const event of data.events){
    const tr=document.createElement('tr'),td=document.createElement('td'),button=document.createElement('button');
    button.textContent=event.timestamp_seconds.toFixed(2)+'s';button.addEventListener('click',()=>{video.currentTime=event.timestamp_seconds;refresh(true);});
    td.appendChild(button);tr.appendChild(td);
    for(const value of [event.shot_hint,event.evidence]){const cell=document.createElement('td');cell.textContent=value;tr.appendChild(cell);}
    el('events').appendChild(tr);
  }
  refresh(true);
})(typeof globalThis!=='undefined'?globalThis:this);
