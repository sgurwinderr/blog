/* =====================================================================
   scenes.js — teaching-oriented WebGL scenes (Three.js r128)
   "How vLLM Scales Across GPUs"

   Every scene DEMONSTRATES a multi-GPU mechanism with real (small)
   computations and a live read-out panel — not decorative motion.
   Scenes step through phases so the viewer watches state change:
   partial sums combining, pipeline bubbles filling, tokens routed to
   experts, KV cache shipped prefill→decode, kernel launches collapsing
   into one graph replay.

   Markup per scene:
     <div class="scene-stage" data-scene="ID">
       <canvas class="scene-canvas"></canvas>
       <div class="scene-loading">…</div>
       <div class="scene-readout"></div>
       <div class="scene-controls"> …[data-scene-act="step|reset"] </div>
       <div class="scene-caption">…</div>
     </div>

   Scene contract: factory(THREE, scene, ctx) -> {
       phases:Number, setPhase(i):fn, update(dt,t):fn }
   ctx = { camera, controls, C, setReadout(html) }

   Depends on globals THREE (r128) + THREE.OrbitControls. Degrades to a
   text notice if WebGL/THREE is unavailable.
   ===================================================================== */
(function () {
  'use strict';

  var SCENES = {};
  function registerScene(id, f) { SCENES[id] = f; }

  var C = {
    stage: 0x0d1017,
    grid:  0x263042,
    white: '#e8ecf5',
    dim:   '#8b93a7',
    violet:0xa78bfa, violetH:'#a78bfa',
    blue:  0x38bdf8, blueH:'#38bdf8',
    green: 0x34d399, greenH:'#34d399',
    orange:0xfb923c, orangeH:'#fb923c',
    yellow:0xfbbf24, yellowH:'#fbbf24',
    red:   0xf43f5e, redH:'#f43f5e',
    gray:  0x475069
  };
  function hx(n){ return '#' + ('000000'+n.toString(16)).slice(-6); }
  function fmt(x, d){ d = (d==null)?2:d; return (x>=0?' ':'') + x.toFixed(d); }
  function ease(t){ return t<0.5 ? 2*t*t : 1-Math.pow(-2*t+2,2)/2; }
  function lerp(a,b,t){ return a+(b-a)*t; }
  function clamp01(x){ return Math.max(0, Math.min(1, x)); }

  /* ---------- updatable text sprite ---------- */
  function textSprite(THREE, text, color, scale) {
    var cv = document.createElement('canvas'); cv.width = 512; cv.height = 128;
    var ctx = cv.getContext('2d');
    var tex = new THREE.CanvasTexture(cv); tex.minFilter = THREE.LinearFilter;
    var mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false, depthWrite: false });
    var sp = new THREE.Sprite(mat);
    sp.scale.set(scale || 1.8, (scale || 1.8) * 0.25, 1);
    sp.setText = function (t, col) {
      ctx.clearRect(0, 0, 512, 128);
      ctx.font = '600 58px "JetBrains Mono", monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillStyle = col || color || C.white;
      ctx.fillText(t, 256, 64);
      tex.needsUpdate = true;
    };
    sp.setText(text, color);
    return sp;
  }

  /* ---------- a labelled GPU box ---------- */
  function gpuBox(THREE, x, y, z, colorHex, label) {
    var g = new THREE.Group();
    var mat = new THREE.MeshStandardMaterial({ color: colorHex, metalness: 0.25, roughness: 0.55, transparent: true, opacity: 0.92 });
    var box = new THREE.Mesh(new THREE.BoxGeometry(1.15, 0.7, 0.9), mat);
    g.add(box);
    var edges = new THREE.LineSegments(new THREE.EdgesGeometry(box.geometry), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.25 }));
    g.add(edges);
    g.position.set(x, y, z);
    g._box = box; g._mat = mat;
    return g;
  }

  /* ---------- a flying packet (small glowing sphere) ---------- */
  function packet(THREE, colorHex) {
    var m = new THREE.Mesh(new THREE.SphereGeometry(0.13, 14, 14), new THREE.MeshBasicMaterial({ color: colorHex }));
    m.visible = false;
    return m;
  }

  /* =====================================================================
     1) allreduce — Tensor Parallelism: partial sums combine to one value,
        then the reduce-scatter/all-gather OVERLAP decomposition.
     ===================================================================== */
  registerScene('allreduce', function (THREE, scene, ctx) {
    ctx.camera.position.set(0, 3.2, 8.2);
    var cols = [C.violet, C.blue, C.green, C.orange];
    var colsH = [C.violetH, C.blueH, C.greenH, C.orangeH];
    // 4 GPUs on a ring, each holding a partial sum of the same activation
    var partials = [2.0, -1.0, 3.0, 1.0];   // real numbers; true sum = 5.0
    var sum = partials.reduce(function(a,b){return a+b;}, 0);
    var R = 2.6;
    var gpus = partials.map(function(p, i){
      var ang = Math.PI/2 - i * (Math.PI*2/4);
      var g = gpuBox(THREE, Math.cos(ang)*R, 0, -Math.sin(ang)*R, cols[i], 'G'+i);
      scene.add(g);
      var lab = textSprite(THREE, 'GPU'+i, colsH[i], 1.0);
      lab.position.set(g.position.x, 0.85, g.position.z); scene.add(lab);
      var val = textSprite(THREE, fmt(p), C.white, 1.0);
      val.position.set(g.position.x, -0.85, g.position.z); scene.add(val);
      return { g:g, lab:lab, val:val, partial:p };
    });
    var center = textSprite(THREE, '', C.greenH, 1.5);
    center.position.set(0, 0, 0); center.material.opacity = 0; scene.add(center);
    var pkts = gpus.map(function(_, i){ var p = packet(THREE, cols[i]); scene.add(p); return p; });

    function say(p){
      if (p===0) ctx.setReadout('<span class="rk">Tensor parallelism: each GPU computed a PARTIAL result for the same activation.</span> GPU0='+fmt(partials[0])+'  GPU1='+fmt(partials[1])+'  GPU2='+fmt(partials[2])+'  GPU3='+fmt(partials[3])+'. <span class="rk">Press Next.</span>');
      else if (p===1) ctx.setReadout('<span class="rk">all-reduce: every partial is summed and the result broadcast back.</span> &Sigma; = '+fmt(sum)+' <span class="rk">— this happens TWICE per transformer layer, every step.</span>');
      else if (p===2) ctx.setReadout('<span class="rk">The overlap trick (PR #16155):</span> split the all-reduce into <b style="color:'+C.blueH+'">reduce-scatter</b> + <b style="color:'+C.greenH+'">all-gather</b>. RMSNorm runs on the smaller sharded slice in between.');
      else ctx.setReadout('<span class="rk">async TP (PR #17882):</span> the decomposed comm now OVERLAPS the next matmul — the wire time is partly hidden. <span class="rk">4×H200 Llama-70B: 0.590s → 0.526s per step.</span>');
    }
    return {
      phases: 4, _p: 0, setPhase: function(i){ this._p = i; say(i); if(i===0){ center.material.opacity=0; } },
      update: function(dt, t){
        var p = this._p;
        // packets fly to center during phase 1 (all-reduce)
        gpus.forEach(function(gp, i){
          var pulse = 1 + 0.05*Math.sin(t*3 + i);
          gp.g.scale.setScalar(pulse);
          var op = (p>=1) ? 1 : 0;
          var k = clamp01((t % 2)/1.2);
          if (p===1){
            pkts[i].visible = true;
            pkts[i].position.lerpVectors(gp.g.position, new THREE.Vector3(0,0,0), ease(k));
          } else if (p===2){
            // reduce-scatter: neighbour exchange (ring)
            pkts[i].visible = true;
            var nb = gpus[(i+1)%4].g.position;
            pkts[i].position.lerpVectors(gp.g.position, nb, ease(k));
          } else if (p===3){
            // all-gather: packets circulate the ring quickly, overlapping
            pkts[i].visible = true;
            var ang = (t*1.5 + i*Math.PI/2);
            pkts[i].position.set(Math.cos(ang)*R*0.6, 0, -Math.sin(ang)*R*0.6);
          } else {
            pkts[i].visible = false;
          }
        });
        if (p>=1){ center.material.opacity = lerp(center.material.opacity, 1, dt*3); center.setText('&Sigma;='+fmt(sum), C.greenH); center.setText('Σ = '+fmt(sum), C.greenH); }
        else { center.material.opacity = lerp(center.material.opacity, 0, dt*5); }
      }
    };
  });

  /* =====================================================================
     2) pipeline — Pipeline Parallelism: stages × microbatches over time.
        Watch the bubble (idle grey) shrink as scheduling improves.
     ===================================================================== */
  registerScene('pipeline', function (THREE, scene, ctx) {
    ctx.camera.position.set(0.5, 2.4, 9.0); if (ctx.controls) ctx.controls.enableRotate = true;
    var STAGES = 4, SLOTS = 7;
    var cellW = 1.1, cellH = 0.62, x0 = -((SLOTS-1)*cellW)/2, y0 = (STAGES-1)*0.85/2;
    var cells = [];
    for (var s=0; s<STAGES; s++){
      for (var t=0; t<SLOTS; t++){
        var m = new THREE.Mesh(new THREE.BoxGeometry(cellW*0.9, cellH, 0.3),
          new THREE.MeshStandardMaterial({ color: C.gray, metalness:0.1, roughness:0.7, transparent:true, opacity:0.9 }));
        m.position.set(x0 + t*cellW, y0 - s*0.85, 0);
        scene.add(m); cells.push({ m:m, s:s, t:t });
      }
    }
    // stage labels
    for (var s2=0; s2<STAGES; s2++){
      var sl = textSprite(THREE, 'Stage'+s2, C.dim, 0.9); sl.position.set(x0 - 1.5, y0 - s2*0.85, 0); scene.add(sl);
    }
    var axis = textSprite(THREE, 'time →', C.dim, 1.0); axis.position.set(0, y0 - STAGES*0.85 + 0.1, 0); scene.add(axis);

    // schedule: naive (diagonal, big bubbles) vs vLLM (packed)
    // occupied[s][t] = microbatch id or -1
    function naive(){ var g=[]; for(var s=0;s<STAGES;s++){g[s]=[];for(var t=0;t<SLOTS;t++){ var mb=t-s; g[s][t]=(mb>=0&&mb<SLOTS-STAGES+1)?mb:-1; }} return g; }
    function packed(){ var g=[]; for(var s=0;s<STAGES;s++){g[s]=[];for(var t=0;t<SLOTS;t++){ var mb=t-s; g[s][t]=(mb>=0)?((mb)%(SLOTS)):-1; if(g[s][t]>=SLOTS-1) g[s][t]=-1; }} return g; }
    var mbCols = [C.violet, C.blue, C.green, C.orange, C.yellow, C.violet, C.blue];

    function utilization(g){ var busy=0,total=STAGES*SLOTS; for(var s=0;s<STAGES;s++)for(var t=0;t<SLOTS;t++) if(g[s][t]>=0) busy++; return busy/total; }

    function say(p){
      var g = p===0 ? naive() : packed();
      var u = Math.round(utilization(g)*100);
      if (p===0) ctx.setReadout('<span class="rk">Naive pipeline: each stage waits for the one before it.</span> Grey = idle "bubble". <b style="color:'+C.redH+'">GPU utilization &asymp; '+u+'%</b>.');
      else ctx.setReadout('<span class="rk">vLLM V1 (PRs #32618, #42187): schedule ahead, overlap the sampled-token broadcast on its own stream.</span> <b style="color:'+C.greenH+'">utilization &asymp; '+u+'%</b> — bubbles shrink. <span class="rk">GB200 PP=4: up to 3.17&times; tok/s.</span>');
    }
    var curGrid = naive();
    function applyGrid(g){ curGrid = g; }
    return {
      phases: 2, _p:0, setPhase:function(i){ this._p=i; applyGrid(i===0?naive():packed()); say(i); },
      update: function(dt, t){
        cells.forEach(function(c){
          var mb = curGrid[c.s][c.t];
          var targetCol = mb>=0 ? mbCols[mb % mbCols.length] : C.gray;
          var col = new THREE.Color(targetCol);
          c.m.material.color.lerp(col, dt*6);
          var targetOp = mb>=0 ? 0.95 : 0.28;
          c.m.material.opacity = lerp(c.m.material.opacity, targetOp, dt*6);
          // active cell gently pulses
          var pulse = mb>=0 ? (1 + 0.05*Math.sin(t*3 + c.t + c.s)) : 1;
          c.m.scale.set(pulse, pulse, 1);
        });
      }
    };
  });

  /* =====================================================================
     3) expert — Wide EP: tokens dispatched to their expert GPUs via
        all-to-all, computed, then combined back. Idle rank runs a dummy.
     ===================================================================== */
  registerScene('expert', function (THREE, scene, ctx) {
    ctx.camera.position.set(0, 3.0, 8.6);
    // 4 expert GPUs across the top, 4 DP-rank token sources across the bottom
    var EXP = 4;
    var cols = [C.violet, C.blue, C.green, C.orange];
    var colsH = [C.violetH, C.blueH, C.greenH, C.orangeH];
    var experts = [], sources = [];
    for (var i=0;i<EXP;i++){
      var ex = gpuBox(THREE, -3.6 + i*2.4, 1.7, 0, cols[i]); scene.add(ex);
      var el = textSprite(THREE, 'E'+i, colsH[i], 0.9); el.position.set(ex.position.x, 2.5, 0); scene.add(el);
      experts.push(ex);
      var sc = gpuBox(THREE, -3.6 + i*2.4, -1.7, 0, C.gray); scene.add(sc);
      var slb = textSprite(THREE, 'rank'+i, C.dim, 0.85); slb.position.set(sc.position.x, -2.5, 0); scene.add(slb);
      sources.push(sc);
    }
    // tokens: each token routes to its top-1 expert (real routing table)
    var routing = [ {rank:0,exp:2}, {rank:1,exp:2}, {rank:2,exp:0}, {rank:3,exp:1}, {rank:0,exp:2} ]; // note E2 is HOT (3 tokens)
    var toks = routing.map(function(r, i){
      var p = packet(THREE, cols[r.exp]); scene.add(p); return { p:p, r:r };
    });

    function say(p){
      if (p===0) ctx.setReadout('<span class="rk">MoE: only a few experts fire per token. Each rank holds its own tokens (DP attention).</span> Routing: rank0,1,4&rarr;E2 · rank2&rarr;E0 · rank3&rarr;E1. <span class="rk">Press Next.</span>');
      else if (p===1) ctx.setReadout('<span class="rk">Dispatch (all-to-all):</span> each token flies to the GPU owning its expert. <b style="color:'+C.greenH+'">E2 receives 3 tokens</b> — a hot expert, while E3 gets none.');
      else if (p===2) ctx.setReadout('<span class="rk">Skew hurts:</span> E2 is a straggler, E3 idle. <b style="color:'+C.yellowH+'">EPLB (--enable-eplb)</b> replicates hot experts to rebalance. <span class="rk">Faster dispatch backends: #23964 ~1.97&times;, MoRI #28664 up to 2.68&times;.</span>');
      else ctx.setReadout('<span class="rk">Combine (all-to-all):</span> results gather back to each token’s home rank. <span class="rk">All ranks must step together — idle ranks run empty DUMMY passes so the collective never deadlocks.</span>');
    }
    return {
      phases: 4, _p:0, setPhase:function(i){ this._p=i; say(i); },
      update: function(dt, t){
        var p = this._p;
        var k = clamp01((t % 2.2)/1.4);
        toks.forEach(function(tok, i){
          var src = sources[tok.r.rank].position;
          var dst = experts[tok.r.exp].position;
          var off = new THREE.Vector3((i-2)*0.12, 0, 0.35);
          if (p===0){ tok.p.visible=true; tok.p.position.copy(src.clone().add(off)); }
          else if (p===1){ tok.p.visible=true; tok.p.position.lerpVectors(src.clone().add(off), dst.clone().add(off), ease(k)); }
          else if (p===2){ tok.p.visible=true; tok.p.position.copy(dst.clone().add(off)); }
          else { tok.p.visible=true; tok.p.position.lerpVectors(dst.clone().add(off), src.clone().add(off), ease(k)); }
        });
        // highlight hot expert E2 / idle E3
        experts.forEach(function(ex, i){
          var hot = (i===2), idle = (i===3);
          var target = (p>=2 && hot) ? 1.18 : (p>=2 && idle ? 0.86 : 1.0);
          ex.scale.setScalar(lerp(ex.scale.x, target, dt*5));
          ex._mat.emissive = ex._mat.emissive || new THREE.Color(0x000000);
        });
      }
    };
  });

  /* =====================================================================
     4) disagg — Disaggregated prefill/decode: KV cache produced by the
        prefill instance, shipped once over the connector to decode.
     ===================================================================== */
  registerScene('disagg', function (THREE, scene, ctx) {
    ctx.camera.position.set(0, 2.4, 9.0);
    var prefill = gpuBox(THREE, -3.4, 0, 0, C.orange); prefill._mat.color.set(C.orange); scene.add(prefill);
    var decode  = gpuBox(THREE,  3.4, 0, 0, C.blue);   scene.add(decode);
    var pL = textSprite(THREE, 'Prefill', C.orangeH, 1.2); pL.position.set(-3.4, 1.0, 0); scene.add(pL);
    var dL = textSprite(THREE, 'Decode', C.blueH, 1.2); dL.position.set(3.4, 1.0, 0); scene.add(dL);
    var pSub = textSprite(THREE, 'compute-bound', C.dim, 0.8); pSub.position.set(-3.4, -1.0, 0); scene.add(pSub);
    var dSub = textSprite(THREE, 'memory-bound', C.dim, 0.8); dSub.position.set(3.4, -1.0, 0); scene.add(dSub);
    // KV blocks that build up on prefill then transfer
    var kvs = [];
    for (var i=0;i<5;i++){
      var m = new THREE.Mesh(new THREE.BoxGeometry(0.28,0.28,0.28), new THREE.MeshStandardMaterial({color:C.green, metalness:0.2, roughness:0.5}));
      m.position.set(-3.4, -0.5 + i*0.26, 0.6); m.visible=false; scene.add(m); kvs.push(m);
    }
    var connLabel = textSprite(THREE, 'NIXL', C.greenH, 0.9); connLabel.position.set(0, 0.5, 0); connLabel.material.opacity=0; scene.add(connLabel);

    function say(p){
      if (p===0) ctx.setReadout('<span class="rk">Two phases, two machines. Prefill is compute-bound; decode is memory-bound.</span> Press Next to run the prefill.');
      else if (p===1) ctx.setReadout('<span class="rk">Prefill builds the KV cache</span> (green blocks) for the prompt on its own instance, tuned for <b style="color:'+C.orangeH+'">TTFT</b>.');
      else if (p===2) ctx.setReadout('<span class="rk">Transfer once</span> over a KV connector (<b style="color:'+C.greenH+'">NIXL #17751 / LMCache #16625</b>) — prefill&rarr;decode. LMCache 2&times;H100 1P1D: ~40% tok/s, ~8&times; better tail ITL.');
      else ctx.setReadout('<span class="rk">Decode streams tokens</span>, tuned independently for <b style="color:'+C.blueH+'">inter-token latency</b>. <span class="rk">Disaggregation buys goodput under a latency SLO — NOT raw throughput.</span>');
    }
    return {
      phases: 4, _p:0, setPhase:function(i){ this._p=i; say(i); },
      update: function(dt, t){
        var p = this._p;
        var k = clamp01((t % 2.4)/1.5);
        kvs.forEach(function(m, i){
          if (p===0){ m.visible=false; m.position.x = -3.4; }
          else if (p===1){ m.visible = (t*2 > i); m.position.x = -3.4; }
          else if (p===2){ m.visible=true; m.position.x = lerp(-3.4, 3.4, ease(clamp01(k - i*0.05))); }
          else { m.visible=true; m.position.x = 3.4; }
        });
        connLabel.material.opacity = lerp(connLabel.material.opacity, p===2?1:0, dt*4);
        prefill.scale.setScalar(1 + (p===1?0.06*Math.sin(t*5):0));
        decode.scale.setScalar(1 + (p>=3?0.06*Math.sin(t*5):0));
      }
    };
  });

  /* =====================================================================
     5) cudagraph — many CPU kernel launches collapse into ONE graph replay.
     ===================================================================== */
  registerScene('cudagraph', function (THREE, scene, ctx) {
    ctx.camera.position.set(0, 2.2, 9.2);
    var N = 8;
    // top row: individual CPU launches (each its own overhead block)
    var launches = [];
    for (var i=0;i<N;i++){
      var m = new THREE.Mesh(new THREE.BoxGeometry(0.5,0.5,0.5), new THREE.MeshStandardMaterial({color:C.orange, metalness:0.2, roughness:0.6}));
      m.position.set(-((N-1)*0.72)/2 + i*0.72, 1.4, 0); scene.add(m); launches.push(m);
    }
    // the single replay block (hidden until phase 1)
    var replay = new THREE.Mesh(new THREE.BoxGeometry(N*0.72, 0.6, 0.6), new THREE.MeshStandardMaterial({color:C.green, metalness:0.2, roughness:0.5, transparent:true, opacity:0}));
    replay.position.set(0, -1.2, 0); scene.add(replay);
    var rLab = textSprite(THREE, 'one graph replay', C.greenH, 1.4); rLab.position.set(0, -2.1, 0); rLab.material.opacity=0; scene.add(rLab);
    var tLab = textSprite(THREE, '8 CPU launches', C.orangeH, 1.4); tLab.position.set(0, 2.3, 0); scene.add(tLab);

    function say(p){
      if (p===0) ctx.setReadout('<span class="rk">Without CUDA graphs: the CPU launches every kernel separately, every step.</span> Each orange block is per-op launch overhead — paid by every rank. ~56 ms on A100 (PR #20059).');
      else ctx.setReadout('<span class="rk">CUDA graph: capture the launch sequence once, then REPLAY it as a single op.</span> <b style="color:'+C.greenH+'">~28 ms — roughly halved.</b> In TP/PP this compounds: a slow rank stalls the next collective. <span class="rk">Cutlass MLA #22763: +6% E2E, P99 TTFT 1818&rarr;1002 ms.</span>');
    }
    return {
      phases: 2, _p:0, setPhase:function(i){ this._p=i; say(i); },
      update: function(dt, t){
        var p = this._p;
        launches.forEach(function(m, i){
          if (p===0){
            m.material.opacity = 1; m.material.transparent = true;
            // sequential launch pulse travelling left→right
            var active = (Math.floor(t*4) % N) === i;
            m.scale.setScalar(active ? 1.25 : 1.0);
            m.position.y = 1.4;
          } else {
            m.material.transparent = true;
            m.material.opacity = lerp(m.material.opacity, 0.12, dt*4);
            m.position.y = lerp(m.position.y, 0.2, dt*3);
            m.scale.setScalar(1.0);
          }
        });
        replay.material.opacity = lerp(replay.material.opacity, p===1?0.95:0, dt*4);
        rLab.material.opacity = lerp(rLab.material.opacity, p===1?1:0, dt*4);
        if (p===1){ replay.scale.setScalar(1 + 0.03*Math.sin(t*4)); }
        tLab.material.opacity = lerp(tLab.material.opacity, p===0?1:0.3, dt*4);
      }
    };
  });

  /* =====================================================================
     6) fusion — collective fusion: all-reduce + RMSNorm + quant collapse
        from 3 kernels into 1 fused FlashInfer op.
     ===================================================================== */
  registerScene('fusion', function (THREE, scene, ctx) {
    ctx.camera.position.set(0, 2.0, 9.0);
    var names = ['all-reduce', 'RMSNorm', 'quant'];
    var cols = [C.blue, C.violet, C.orange];
    var colsH = [C.blueH, C.violetH, C.orangeH];
    var blocks = [], labs = [];
    for (var i=0;i<3;i++){
      var m = new THREE.Mesh(new THREE.BoxGeometry(1.5,0.8,0.6), new THREE.MeshStandardMaterial({color:cols[i], metalness:0.2, roughness:0.55}));
      m.position.set(-3.0 + i*3.0, 0, 0); scene.add(m); blocks.push(m);
      var l = textSprite(THREE, names[i], colsH[i], 1.3); l.position.set(-3.0 + i*3.0, 0.85, 0); scene.add(l);
      labs.push(l);
    }
    var fused = new THREE.Mesh(new THREE.BoxGeometry(4.4,0.9,0.7), new THREE.MeshStandardMaterial({color:C.green, metalness:0.25, roughness:0.45, transparent:true, opacity:0}));
    fused.position.set(0,0,0); scene.add(fused);
    var fLab = textSprite(THREE, 'one fused FlashInfer op', C.greenH, 1.6); fLab.position.set(0,-1.1,0); fLab.material.opacity=0; scene.add(fLab);

    function say(p){
      if (p===0) ctx.setReadout('<span class="rk">In the TP path these three run back-to-back — three kernel launches, three round-trips to memory.</span> Press Next.');
      else ctx.setReadout('<span class="rk">Collective fusion (PR #21069):</span> a torch.compile pass pattern-matches the trio and emits <b style="color:'+C.greenH+'">one fused FlashInfer op</b>. B200 TP=2: ~7&ndash;8% TPOT vs custom ops. <span class="rk">(Read the baseline — vs default compiled ops the FP8 gain can vanish.)</span>');
    }
    return {
      phases: 2, _p:0, setPhase:function(i){ this._p=i; say(i); },
      update: function(dt, t){
        var p = this._p;
        blocks.forEach(function(m, i){
          var targetX = p===1 ? 0 : (-3.0 + i*3.0);
          m.position.x = lerp(m.position.x, targetX, dt*4);
          m.material.transparent = true;
          m.material.opacity = lerp(m.material.opacity, p===1?0.0:1.0, dt*4);
          labs[i].material.opacity = lerp(labs[i].material.opacity, p===1?0.0:1.0, dt*4);
        });
        fused.material.opacity = lerp(fused.material.opacity, p===1?0.95:0, dt*4);
        fLab.material.opacity = lerp(fLab.material.opacity, p===1?1:0, dt*4);
        if (p===1) fused.scale.setScalar(1 + 0.03*Math.sin(t*4));
      }
    };
  });

  /* =====================================================================
     ENGINE  (identical contract to the transformers course)
     ===================================================================== */
  function hasWebGL(){ try{ var c=document.createElement('canvas'); return !!(window.WebGLRenderingContext&&(c.getContext('webgl')||c.getContext('experimental-webgl'))); }catch(e){ return false; } }

  function mountStage(stage) {
    var id = stage.dataset.scene, factory = SCENES[id];
    var canvas = stage.querySelector('.scene-canvas');
    if (!factory || !canvas) return;
    if (typeof THREE === 'undefined' || !hasWebGL()) {
      var n = stage.querySelector('.scene-loading'); if (n) n.textContent = 'This 3D scene needs WebGL — the caption below describes what it shows.';
      return;
    }
    var readoutEl = stage.querySelector('.scene-readout');
    function setReadout(html){ if (readoutEl) readoutEl.innerHTML = html; }

    var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    var scene = new THREE.Scene(); scene.background = new THREE.Color(C.stage);
    var camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100); camera.position.set(4,3,7);
    scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    var key=new THREE.DirectionalLight(0xffffff,0.8); key.position.set(4,6,4); scene.add(key);
    var rim=new THREE.DirectionalLight(0xa78bfa,0.35); rim.position.set(-5,2,-4); scene.add(rim);
    var grid=new THREE.GridHelper(16,16,C.grid,C.grid); grid.material.opacity=0.3; grid.material.transparent=true; grid.position.y=-2.9; scene.add(grid);

    var controls=null;
    if (THREE.OrbitControls){ controls=new THREE.OrbitControls(camera, renderer.domElement); controls.enableDamping=true; controls.dampingFactor=0.08; controls.enablePan=false; controls.minDistance=4; controls.maxDistance=18; }

    var inst = factory(THREE, scene, { camera: camera, controls: controls, C: C, setReadout: setReadout });
    if (inst.setPhase) inst.setPhase(0);
    stage.classList.add('ready');

    function resize(){ var w=canvas.clientWidth,h=canvas.clientHeight; if(!w||!h) return; renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix(); }
    resize(); window.addEventListener('resize', resize);

    var t=0, playing=true, visible=false, last=null;
    function frame(now){
      requestAnimationFrame(frame);
      if(last===null) last=now; var dt=Math.min(0.05,(now-last)/1000); last=now;
      if(visible&&playing){ t+=dt; if(inst.update) inst.update(dt,t); }
      if(controls) controls.update();
      renderer.render(scene,camera);
    }
    requestAnimationFrame(frame);

    new IntersectionObserver(function(es){ es.forEach(function(e){ visible=e.isIntersecting; }); }, {threshold:0.12}).observe(stage);

    var phase=0, progressEl=stage.querySelector('.scene-progress');
    function setProg(){ if(progressEl&&inst.phases>1) progressEl.textContent='step '+(phase+1)+' / '+inst.phases; }
    setProg();
    stage.querySelectorAll('[data-scene-act]').forEach(function(btn){
      btn.addEventListener('click', function(){
        var a=btn.dataset.sceneAct;
        if(a==='play') playing=true;
        else if(a==='pause') playing=false;
        else if(a==='reset'){ t=0; phase=0; if(inst.setPhase) inst.setPhase(0); playing=true; setProg(); }
        else if(a==='step'){ if(inst.phases>1){ phase=(phase+1)%inst.phases; if(inst.setPhase) inst.setPhase(phase); setProg(); } }
      });
    });
  }

  function init(){
    var stages=[].slice.call(document.querySelectorAll('.scene-stage[data-scene]'));
    if(!stages.length) return;
    var mounted=new WeakSet();
    var io=new IntersectionObserver(function(entries){
      entries.forEach(function(e){ if(e.isIntersecting&&!mounted.has(e.target)){ mounted.add(e.target); mountStage(e.target); io.unobserve(e.target); } });
    }, {rootMargin:'200px 0px'});
    stages.forEach(function(s){ io.observe(s); });
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
