/* =====================================================================
   scenes.js — teaching-oriented WebGL scenes (Three.js r128)
   "Transformers & Attention on Multi-GPU"

   Every scene DEMONSTRATES a mechanism with real (small) computations
   and a live read-out panel — not decorative motion. Scenes step through
   phases so the viewer watches numbers change: dot products, softmax
   weights, gradient averages, output slices, pipeline utilization.

   Markup per scene:
     <div class="scene-stage" data-scene="ID">
       <canvas class="scene-canvas"></canvas>
       <div class="scene-loading">…</div>
       <div class="scene-readout"></div>   (optional live numbers panel)
       <div class="scene-controls"> …[data-scene-act="play|pause|step|reset"] </div>
       <div class="scene-caption">…</div>
     </div>

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
      ctx.font = '600 60px "JetBrains Mono", monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillStyle = col || color || C.white;
      ctx.fillText(t, 256, 64);
      tex.needsUpdate = true;
    };
    sp.setText(text, color);
    return sp;
  }

  function arrow(THREE, from, to, colorHex, headFrac) {
    var g = new THREE.Group();
    var d = to.clone().sub(from); var len = d.length();
    if (len < 1e-4) len = 1e-4;
    var dir = d.clone().normalize();
    var a = new THREE.ArrowHelper(dir, from, len, colorHex, len * (headFrac || 0.22), len * (headFrac || 0.22) * 0.6);
    g.add(a);
    var geo = new THREE.CylinderGeometry(0.035, 0.035, len, 10);
    var mat = new THREE.MeshBasicMaterial({ color: colorHex, transparent: true, opacity: 0.35 });
    var cyl = new THREE.Mesh(geo, mat);
    cyl.position.copy(from.clone().add(dir.clone().multiplyScalar(len / 2)));
    cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    g.add(cyl);
    return g;
  }

  function ease(t){ return t<0.5 ? 2*t*t : 1-Math.pow(-2*t+2,2)/2; }
  function lerp(a,b,t){ return a+(b-a)*t; }
  function softmax(a){ var m=Math.max.apply(null,a); var e=a.map(function(x){return Math.exp(x-m);}); var s=e.reduce(function(p,q){return p+q;},0); return e.map(function(x){return x/s;}); }

  /* =====================================================================
     1) tokens — embedding ARITHMETIC: king - man + woman ≈ queen
     ===================================================================== */
  registerScene('tokens', function (THREE, scene, ctx) {
    ctx.camera.position.set(2.5, 2.2, 6.5);
    var P = {
      man:   new THREE.Vector3(-1.4, -0.6, 0),
      woman: new THREE.Vector3(-1.4,  1.0, 0),
      king:  new THREE.Vector3( 1.6, -0.6, 0.3)
    };
    P.queen = P.king.clone().sub(P.man).add(P.woman); // exact by construction
    var cols = { man: C.blue, woman: C.violet, king: C.orange, queen: C.green };
    var dots = {};
    Object.keys(P).forEach(function (name) {
      var m = new THREE.Mesh(new THREE.SphereGeometry(0.13, 20, 20), new THREE.MeshBasicMaterial({ color: cols[name] }));
      m.position.copy(P[name]); scene.add(m); dots[name] = m;
      var lab = textSprite(THREE, name, hx(cols[name]), 1.5);
      lab.position.copy(P[name]).add(new THREE.Vector3(0, 0.32, 0)); scene.add(lab);
    });
    // the two "royalty direction" arrows (parallelogram)
    var aRoyal = arrow(THREE, P.man, P.king, C.gray); aRoyal.visible = false; scene.add(aRoyal);
    var bRoyal = arrow(THREE, P.woman, P.queen, C.gray); bRoyal.visible = false; scene.add(bRoyal);
    // travelling result dot
    var travel = new THREE.Mesh(new THREE.SphereGeometry(0.16, 20, 20), new THREE.MeshBasicMaterial({ color: C.green, transparent: true, opacity: 0.9 }));
    travel.visible = false; scene.add(travel);

    function say(p){
      if (p === 0) ctx.setReadout('<span class="rk">Four token embeddings.</span> Direction encodes meaning — watch the relationships.');
      else if (p === 1) ctx.setReadout('<span class="rk">The "royalty" direction:</span> man→king equals woman→queen (parallel, same length). Relationships are consistent directions.');
      else ctx.setReadout('<span class="rk">Arithmetic:</span> king − man + woman = ('+fmt(P.queen.x)+','+fmt(P.queen.y)+','+fmt(P.queen.z)+') → lands on <b style="color:'+C.greenH+'">queen</b>. distance ≈ 0.00');
    }
    return {
      phases: 3, _p: 0, setPhase: function(i){ this._p=i; say(i); },
      update: function (dt, t) {
        var p = this._p;
        aRoyal.visible = bRoyal.visible = (p >= 1);
        if (p >= 2) {
          travel.visible = true;
          var k = (t % 3) / 3; var e = ease(k);
          // king -> (king-man) -> (king-man+woman)=queen, shown as two legs
          var mid = P.king.clone().sub(P.man);
          if (e < 0.5) travel.position.lerpVectors(P.king, mid, e/0.5);
          else travel.position.lerpVectors(mid, P.queen, (e-0.5)/0.5);
        } else travel.visible = false;
      },
      _sayInit: say(0)
    };
  });

  /* =====================================================================
     2) qkv — one x, three matrices, three directions (with numbers)
     ===================================================================== */
  registerScene('qkv', function (THREE, scene, ctx) {
    ctx.camera.position.set(3, 2.4, 6.5);
    var x = new THREE.Vector3(1.2, 0.6, 0.4);
    var origin = new THREE.Vector3(0,0,0);
    var xm = new THREE.Mesh(new THREE.SphereGeometry(0.15,20,20), new THREE.MeshBasicMaterial({color:0xffffff}));
    scene.add(xm);
    var xl = textSprite(THREE,'x',C.white,1.2); xl.position.set(0,-0.35,0); scene.add(xl);
    // three "matrices" as small rotated planes, each maps x to a different direction
    var defs = [
      { name:'q = W_Q·x', dir:new THREE.Vector3(2.0,1.1,0.2),  c:C.violet, ch:C.violetH, delay:0.0 },
      { name:'k = W_K·x', dir:new THREE.Vector3(-0.8,1.8,1.0), c:C.blue,   ch:C.blueH,   delay:0.6 },
      { name:'v = W_V·x', dir:new THREE.Vector3(0.5,-0.8,2.0), c:C.green,  ch:C.greenH,  delay:1.2 }
    ];
    defs.forEach(function(d){
      d.arrow = arrow(THREE, origin, d.dir, d.c); d.arrow.scale.setScalar(0.001); scene.add(d.arrow);
      d.lab = textSprite(THREE, d.name.split(' ')[0], d.ch, 1.1);
      d.lab.position.copy(d.dir.clone().multiplyScalar(1.08)); d.lab.material.opacity=0; scene.add(d.lab);
    });
    ctx.setReadout('<span class="rk">Same input</span> x=('+fmt(x.x)+','+fmt(x.y)+','+fmt(x.z)+') <span class="rk">→ three learned matrices → three different directions Q, K, V.</span>');
    return {
      phases: 1,
      update: function (dt, t) {
        defs.forEach(function(d){
          var k = Math.min(1, Math.max(0,(t-d.delay)/0.9));
          d.arrow.scale.setScalar(0.001 + ease(k));
          d.lab.material.opacity = k;
        });
      }
    };
  });

  /* =====================================================================
     3) attention — THE scene: dot-product → softmax → weighted sum
     ===================================================================== */
  registerScene('attention', function (THREE, scene, ctx) {
    ctx.camera.position.set(3.4, 2.6, 7.2);
    var O = new THREE.Vector3(-1.8,-1.0,0);
    var q = new THREE.Vector3(2.6,1.7,0.4);
    var keys = [
      { name:'the', v:new THREE.Vector3(1.0, 2.2, 0.2) },
      { name:'cat', v:new THREE.Vector3(2.7, 1.4, 0.3) },
      { name:'sat', v:new THREE.Vector3(2.4,-0.6,-0.4) }
    ];
    var vals = [ new THREE.Vector3(-1.4,1.2,1.0), new THREE.Vector3(1.6,1.0,0.6), new THREE.Vector3(0.8,-1.4,1.2) ];
    var qA = arrow(THREE, O, O.clone().add(q), C.violet); scene.add(qA);
    var ql = textSprite(THREE,'Q(sat)',C.violetH,1.5); ql.position.copy(O.clone().add(q).add(new THREE.Vector3(0,0.3,0))); scene.add(ql);
    keys.forEach(function(kd,i){
      kd.arrow = arrow(THREE, O, O.clone().add(kd.v), C.blue);
      kd.arrow.children.forEach(function(c){ if(c.material){c.material.transparent=true; c.material.opacity=0.4;} });
      scene.add(kd.arrow);
      kd.lab = textSprite(THREE,'K('+kd.name+')',C.blueH,1.3); kd.lab.position.copy(O.clone().add(kd.v).add(new THREE.Vector3(0,0.28,0))); scene.add(kd.lab);
    });
    // compute real numbers
    var qn = q.clone().normalize();
    var scores = keys.map(function(kd){ return qn.dot(kd.v.clone().normalize()) * 3.0; }); // scaled
    var weights = softmax(scores);
    var outArrow = null, outLab = null;

    // weight bars (drawn as boxes standing on a shelf)
    var bars = keys.map(function(kd,i){
      var b = new THREE.Mesh(new THREE.BoxGeometry(0.34,1,0.34), new THREE.MeshStandardMaterial({color:C.green, metalness:0.2, roughness:0.5}));
      b.position.set(-1.0 + i*0.7, -2.0, 2.2); b.scale.y = 0.001; scene.add(b);
      var bl = textSprite(THREE, kd.name, C.dim, 0.8); bl.position.set(-1.0+i*0.7, -2.35, 2.2); scene.add(bl);
      return b;
    });

    function say(p){
      if (p===0) ctx.setReadout('<span class="rk">Step 1 — score by alignment.</span> Press <b>Next step</b>. Each key gets a score = Q·K.');
      else if (p===1) ctx.setReadout('<span class="rk">scores (Q·K):</span> the '+fmt(scores[0])+'   cat '+fmt(scores[1])+'   sat '+fmt(scores[2])+'  <span class="rk">— "cat" aligns best.</span>');
      else if (p===2) ctx.setReadout('<span class="rk">softmax →</span> weights: the '+fmt(weights[0])+'  cat '+fmt(weights[1])+'  sat '+fmt(weights[2])+'   <span class="rk">(sum = 1.00)</span> — bars grow to the weights.');
      else ctx.setReadout('<span class="rk">output =</span> '+fmt(weights[0])+'·V(the) + '+fmt(weights[1])+'·V(cat) + '+fmt(weights[2])+'·V(sat) <span class="rk">→ the orange vector. Attention = weighted average of values.</span>');
    }
    return {
      phases: 4, _p:0, setPhase:function(i){ this._p=i; say(i); },
      update: function (dt, t) {
        var p = this._p;
        // phase>=2: bars grow to weights, best key brightens
        bars.forEach(function(b,i){
          var target = (p>=2) ? (0.05 + weights[i]*2.6) : 0.001;
          b.scale.y = lerp(b.scale.y, target, Math.min(1, dt*4));
          b.position.y = -2.0 + b.scale.y/2;
        });
        keys.forEach(function(kd,i){
          var op = (p>=1) ? (0.2 + (weights[i])*0.8) : 0.4;
          kd.arrow.children.forEach(function(c){ if(c.material&&c.material.transparent) c.material.opacity=op; });
        });
        // phase 3: build weighted output
        if (p>=3 && !outArrow){
          var out = new THREE.Vector3();
          vals.forEach(function(v,i){ out.add(v.clone().multiplyScalar(weights[i])); });
          outArrow = arrow(THREE, O, O.clone().add(out), C.orange); scene.add(outArrow);
          outLab = textSprite(THREE,'out',C.orangeH,1.2); outLab.position.copy(O.clone().add(out).add(new THREE.Vector3(0,0.3,0))); scene.add(outLab);
        }
        if (p<3 && outArrow){ scene.remove(outArrow); scene.remove(outLab); outArrow=null; outLab=null; }
      },
      _init: say(0)
    };
  });

  /* =====================================================================
     4) multihead — different heads = different attention EDGES
     ===================================================================== */
  registerScene('multihead', function (THREE, scene, ctx) {
    ctx.camera.position.set(0, 1.5, 8.5); ctx.controls && (ctx.controls.enableRotate = true);
    var toks = ['The','cat','that','chased','the','mouse','sat'];
    var nodes = toks.map(function(w,i){
      var x = (i - (toks.length-1)/2) * 1.35;
      var m = new THREE.Mesh(new THREE.SphereGeometry(0.16,18,18), new THREE.MeshBasicMaterial({color:0xffffff}));
      m.position.set(x,0,0); scene.add(m);
      var l = textSprite(THREE,w,C.white,1.1); l.position.set(x,-0.42,0); scene.add(l);
      return m.position.clone();
    });
    // head patterns: [from,to,weight]
    var heads = [
      { name:'Head 1 — local (attend to previous token)', c:C.violet, ch:C.violetH,
        edges: toks.map(function(_,i){ return i>0?[i,i-1,0.9]:null; }).filter(Boolean) },
      { name:'Head 2 — syntactic (verb "sat" → subject "cat"; "chased" → "cat")', c:C.blue, ch:C.blueH,
        edges: [[6,1,0.95],[3,1,0.8],[5,3,0.5]] },
      { name:'Head 3 — broad / positional (attend back to sentence start)', c:C.green, ch:C.greenH,
        edges: toks.map(function(_,i){ return i>0?[i,0,0.5]:null; }).filter(Boolean) }
    ];
    var arcGroup = new THREE.Group(); scene.add(arcGroup);
    function drawHead(h){
      while(arcGroup.children.length) arcGroup.remove(arcGroup.children[0]);
      h.edges.forEach(function(e){
        var a = nodes[e[0]], b = nodes[e[1]];
        var mid = a.clone().add(b).multiplyScalar(0.5); mid.y += 1.0 + Math.abs(e[0]-e[1])*0.25;
        var curve = new THREE.QuadraticBezierCurve3(a, mid, b);
        var geo = new THREE.TubeGeometry(curve, 24, 0.02 + e[2]*0.05, 8, false);
        var mat = new THREE.MeshBasicMaterial({ color: h.c, transparent:true, opacity: 0.35 + e[2]*0.55 });
        arcGroup.add(new THREE.Mesh(geo, mat));
      });
      ctx.setReadout('<b style="color:'+h.ch+'">'+h.name+'</b> — <span class="rk">same tokens, different learned connections. Press Next head.</span>');
    }
    drawHead(heads[0]);
    return {
      phases: heads.length, _p:0,
      setPhase: function(i){ this._p=i; drawHead(heads[i]); },
      update: function(){}
    };
  });

  /* =====================================================================
     5) scorematrix — the T×T score grid: dots sized by Q·K, softmax a col,
        causal mask (FUTURE = masked). This is the O(T^2) object.
     ===================================================================== */
  registerScene('scorematrix', function (THREE, scene, ctx) {
    ctx.camera.position.set(0, 0.5, 9);
    var toks = ['The','cat','sat','on','mat'];
    var n = toks.length, sp = 1.15, x0 = -(n-1)/2*sp, y0 = (n-1)/2*sp;
    // pseudo scores: higher near diagonal + a couple learned spikes
    function rawScore(i,j){ return Math.exp(-Math.abs(i-j)*0.6) + (i===2&&j===1?0.7:0) + (i===4&&j===2?0.5:0); }
    var dots = [];
    for (var i=0;i<n;i++) for (var j=0;j<n;j++){
      var m = new THREE.Mesh(new THREE.CircleGeometry(0.18,24), new THREE.MeshBasicMaterial({color:C.blue}));
      m.position.set(x0 + j*sp, y0 - i*sp, 0);
      m.userData = {i:i, j:j};
      scene.add(m); dots.push(m);
    }
    // axis labels
    toks.forEach(function(w,j){ var l=textSprite(THREE,w,C.dim,0.8); l.position.set(x0+j*sp, y0+0.8, 0); scene.add(l); });
    toks.forEach(function(w,i){ var l=textSprite(THREE,w,C.dim,0.8); l.position.set(x0-1.4, y0-i*sp, 0); l.scale.set(1.2,0.3,1); scene.add(l); });
    var qlab = textSprite(THREE,'query i →',C.violetH,1.4); qlab.position.set(x0-1.6, y0+0.9,0); scene.add(qlab);
    var klab = textSprite(THREE,'key j',C.blueH,1.2); klab.position.set(0, y0+1.6,0); scene.add(klab);
    function say(p){
      if(p===0) ctx.setReadout('<span class="rk">Each cell (i,j) = Q<sub>i</sub>·K<sub>j</sub>. Bigger dot = stronger match.</span> This grid is n×n → the <b>O(T²)</b> cost. Press Next.');
      else if(p===1) ctx.setReadout('<span class="rk">softmax runs along each ROW (over keys j):</span> row i becomes a probability distribution summing to 1 — token i\'s "attention pattern".');
      else ctx.setReadout('<span class="rk">causal mask:</span> a query may not see the <b>future</b> — cells with j &gt; i are set to −∞ (greyed), so each token attends only to itself and earlier tokens.');
    }
    say(0);
    return {
      phases:3, _p:0, setPhase:function(k){ this._p=k; say(k); },
      update:function(dt,t){
        var p=this._p;
        // per-row softmax when p>=1
        dots.forEach(function(m){
          var i=m.userData.i, j=m.userData.j;
          var masked = (p>=2 && j>i);
          var val;
          if(masked){ val=0.02; m.material.color.setHex(C.gray); }
          else if(p>=1){
            // softmax over row i (unmasked keys)
            var denom=0; for(var jj=0;jj<n;jj++){ if(!(p>=2&&jj>i)) denom+=rawScore(i,jj); }
            val=rawScore(i,j)/denom; m.material.color.setHex(C.green);
          } else { val=rawScore(i,j)*0.5; m.material.color.setHex(C.blue); }
          var target=0.06 + val*0.9;
          m.scale.setScalar(lerp(m.scale.x, target, Math.min(1,dt*5)));
        });
      }
    };
  });

  /* =====================================================================
     6) kvcache — MHA→MQA→GQA→MLA: KV heads collapse; KV size readout
     ===================================================================== */
  registerScene('kvcache', function (THREE, scene, ctx) {
    ctx.camera.position.set(0, 0.5, 9);
    var H = 8; // query heads
    var qy = 1.7, ky = -0.6, sp = 1.0, x0 = -(H-1)/2*sp;
    // query heads (fixed row, violet)
    for(var i=0;i<H;i++){
      var q=new THREE.Mesh(new THREE.BoxGeometry(0.7,0.5,0.2), new THREE.MeshStandardMaterial({color:C.violet,metalness:0.3,roughness:0.5}));
      q.position.set(x0+i*sp, qy, 0); scene.add(q);
    }
    var qlab=textSprite(THREE,'8 query heads',C.violetH,2.0); qlab.position.set(0, qy+0.7,0); scene.add(qlab);
    // KV head boxes (mutable)
    var kv=[];
    for(var j=0;j<H;j++){
      var b=new THREE.Mesh(new THREE.BoxGeometry(0.7,0.5,0.2), new THREE.MeshStandardMaterial({color:C.blue,metalness:0.3,roughness:0.5}));
      b.position.set(x0+j*sp, ky, 0); scene.add(b); kv.push(b);
    }
    var kvlab=textSprite(THREE,'KV heads',C.blueH,1.6); kvlab.position.set(0, ky-0.7,0); scene.add(kvlab);
    var latent=new THREE.Mesh(new THREE.CylinderGeometry(0.35,0.35,0.5,20), new THREE.MeshStandardMaterial({color:C.green,metalness:0.4,roughness:0.4}));
    latent.rotation.x=Math.PI/2; latent.position.set(0,ky,0); latent.visible=false; scene.add(latent);
    // config per phase: KV head count (relative KV size) + label
    var modes=[
      {name:'MHA', kvHeads:8, note:'one KV head per query head — biggest KV cache (baseline 1.0×)', size:'1.00×'},
      {name:'MQA', kvHeads:1, note:'a single shared KV head — ~8× smaller KV cache, fastest decode, some quality loss', size:'0.125×'},
      {name:'GQA', kvHeads:2, note:'a few KV groups (here 2) — interpolates: near-MHA quality at MQA-like speed (Llama-2/3, Mistral)', size:'0.25×'},
      {name:'MLA', kvHeads:0, note:'compress K and V into one shared latent vector (DeepSeek) — the smallest KV footprint', size:'latent'}
    ];
    function say(p){ var m=modes[p]; ctx.setReadout('<b style="color:'+C.greenH+'">'+m.name+'</b> — '+m.note+'. <span class="rk">relative KV size ≈</span> <b>'+m.size+'</b>. Press Next.'); }
    say(0);
    return {
      phases:modes.length, _p:0, setPhase:function(p){ this._p=p; say(p); },
      update:function(dt,t){
        var mode=modes[this._p]; var kvh=mode.kvHeads;
        latent.visible=(kvh===0);
        kv.forEach(function(b,j){
          if(kvh===0){ b.visible=false; return; }
          b.visible=true;
          // group j into kvh buckets: target x = center of its group
          var groupSize=H/kvh; var group=Math.floor(j/groupSize);
          var groupCenter=x0 + (group*groupSize + (groupSize-1)/2)*sp;
          var tx = (kvh===8) ? (x0+j*sp) : groupCenter;
          b.position.x = lerp(b.position.x, tx, dt*4);
          // fade duplicates that collapsed onto the group leader
          var isLeader = (j % groupSize)===0 || kvh===8;
          b.material.opacity = lerp(b.material.opacity!=null?b.material.opacity:1, isLeader?1:0.15, dt*4);
          b.material.transparent=true;
        });
        if(kvh===0){ latent.scale.setScalar(1+0.08*Math.sin(t*3)); }
      }
    };
  });

  /* =====================================================================
     7) sparsity — full vs sliding-window vs strided: which cells compute
     ===================================================================== */
  registerScene('sparsity', function (THREE, scene, ctx) {
    ctx.camera.position.set(0, 0, 9.5);
    var n=9, sp=0.82, x0=-(n-1)/2*sp, y0=(n-1)/2*sp, W=2;
    var cells=[];
    for(var i=0;i<n;i++) for(var j=0;j<n;j++){
      var m=new THREE.Mesh(new THREE.PlaneGeometry(0.68,0.68), new THREE.MeshBasicMaterial({color:C.blue, transparent:true, opacity:0.15, side:THREE.DoubleSide}));
      m.position.set(x0+j*sp, y0-i*sp, 0); m.userData={i:i,j:j}; scene.add(m); cells.push(m);
    }
    function computed(mode,i,j){
      if(j>i) return false;                        // causal
      if(mode===0) return true;                    // full
      if(mode===1) return (i-j)<W;                 // sliding window (width W)
      return (i-j)<W || j%3===0;                   // strided sparse: window + every 3rd key
    }
    var modes=['full (causal)','sliding window (w=2)','strided sparse'];
    function count(mode){ var c=0; for(var i=0;i<n;i++)for(var j=0;j<n;j++) if(computed(mode,i,j)) c++; return c; }
    function say(p){
      var c=count(p), full=count(0);
      var big = p===0 ? 'grows as O(T²)' : (p===1?'grows as O(T·w) — linear in T':'stays sparse — far fewer cells');
      ctx.setReadout('<b style="color:'+C.greenH+'">'+modes[p]+'</b>: <b>'+c+'</b> of '+(n*n)+' cells computed ('+Math.round(100*c/full)+'% of full). Cost '+big+'. Press Next.');
    }
    say(0);
    return {
      phases:modes.length, _p:0, setPhase:function(p){ this._p=p; say(p); },
      update:function(dt,t){
        var mode=this._p;
        cells.forEach(function(m){
          var on=computed(mode,m.userData.i,m.userData.j);
          m.material.opacity=lerp(m.material.opacity, on?0.9:0.06, dt*5);
          m.material.color.setHex(on?C.green:C.gray);
        });
      }
    };
  });

  /* =====================================================================
     8) linear — associativity: (QKᵀ)V  vs  Q(KᵀV). Big T×T vs small d×d.
     ===================================================================== */
  registerScene('linear', function (THREE, scene, ctx) {
    ctx.camera.position.set(0, 0, 9);
    var T=6, d=3;
    function block(w,h,color,cx,cy){
      var g=new THREE.Group();
      var m=new THREE.Mesh(new THREE.PlaneGeometry(w,h), new THREE.MeshBasicMaterial({color:color,transparent:true,opacity:0.35,side:THREE.DoubleSide}));
      g.add(m);
      var e=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(w,h)), new THREE.LineBasicMaterial({color:color}));
      g.add(e);
      g.position.set(cx,cy,0); g.userData.plane=m; return g;
    }
    // Path A (top): (QKᵀ) = T×T big, then ·V
    var big = block(2.6,2.6,C.red,-1.8,1.4); scene.add(big);
    var bigL=textSprite(THREE,'QKᵀ : T×T',C.redH,1.6); bigL.position.set(-1.8,3.0,0); scene.add(bigL);
    // Path B (bottom): (KᵀV) = d×d small, then Q·
    var small = block(1.1,1.1,C.green,-2.4,-1.6); scene.add(small);
    var smallL=textSprite(THREE,'KᵀV : d×d',C.greenH,1.4); smallL.position.set(-2.4,-2.6,0); scene.add(smallL);
    function say(p){
      if(p===0) ctx.setReadout('<span class="rk">Standard order</span> (QKᵀ)V builds a <b style="color:'+C.redH+'">T×T</b> matrix first → <b>O(T²)</b> in time & memory.');
      else ctx.setReadout('<span class="rk">Linear attention reorders to</span> Q(KᵀV): build the tiny <b style="color:'+C.greenH+'">d×d</b> state first → <b>O(T)</b>. Same result, associativity of matmul — no T×T ever formed.');
    }
    say(0);
    return {
      phases:2, _p:0, setPhase:function(p){ this._p=p; say(p); },
      update:function(dt,t){
        var showSmall=this._p>=1;
        big.userData.plane.material.opacity=lerp(big.userData.plane.material.opacity, showSmall?0.08:0.4, dt*4);
        big.scale.setScalar(lerp(big.scale.x, showSmall?0.9:1.0+0.02*Math.sin(t*2), dt*4));
        small.userData.plane.material.opacity=lerp(small.userData.plane.material.opacity, showSmall?0.55:0.15, dt*4);
        small.scale.setScalar(lerp(small.scale.x, showSmall?1.0+0.05*Math.sin(t*3):0.85, dt*4));
      }
    };
  });

  /* =====================================================================
     9) flashattn — tile K/V into SRAM, online softmax (running m,l),
        T×T matrix never materialized. Still O(T²) compute.
     ===================================================================== */
  registerScene('flashattn', function (THREE, scene, ctx) {
    ctx.camera.position.set(0, 0.3, 9);
    // greyed T×T matrix in the back = "never stored"
    var n=6, sp=0.5, x0=-(n-1)/2*sp, y0=(n-1)/2*sp;
    var ghost=[];
    for(var i=0;i<n;i++)for(var j=0;j<n;j++){
      var m=new THREE.Mesh(new THREE.PlaneGeometry(0.42,0.42), new THREE.MeshBasicMaterial({color:C.gray,transparent:true,opacity:0.1,side:THREE.DoubleSide}));
      m.position.set(x0+j*sp-2.4, y0-i*sp+0.3, -0.5); scene.add(m); ghost.push(m);
    }
    var ghostL=textSprite(THREE,'T×T — never written to HBM',C.dim,2.4); ghostL.position.set(-2.4, y0+0.9,0); scene.add(ghostL);
    // SRAM tile on the right
    var tile=new THREE.Mesh(new THREE.BoxGeometry(1.6,1.0,0.3), new THREE.MeshStandardMaterial({color:C.orange,metalness:0.3,roughness:0.5,transparent:true,opacity:0.85}));
    tile.position.set(2.6,0.3,0); scene.add(tile);
    var tileL=textSprite(THREE,'SRAM tile',C.orangeH,1.4); tileL.position.set(2.6,1.1,0); scene.add(tileL);
    // running stats sprites
    var mSp=textSprite(THREE,'m = -inf',C.white,1.6); mSp.position.set(2.6,-0.1,0.3); scene.add(mSp);
    var lSp=textSprite(THREE,'l = 0',C.white,1.6); lSp.position.set(2.6,-0.5,0.3); scene.add(lSp);
    // a K/V block that streams in
    var blk=new THREE.Mesh(new THREE.BoxGeometry(0.5,1.0,0.3), new THREE.MeshStandardMaterial({color:C.blue,metalness:0.3,roughness:0.5}));
    scene.add(blk);
    var mRun=-1e9, lRun=0, blockIdx=0, nblocks=n;
    var stageNames=['naive: write full T×T (3 passes over HBM)','safe: subtract row-max, still 3 passes','online: fuse into 2 passes (running m,l)','FlashAttention: 1 pass, tiled, never materialize'];
    function say(p){ ctx.setReadout('<b style="color:'+C.orangeH+'">'+stageNames[p]+'</b>. <span class="rk">Online softmax tracks m (running max) and l (running Σexp) so a tile needs no full row. Still O(T²) compute, but O(1)-in-T memory. Press Next.</span>'); }
    say(0);
    return {
      phases:4, _p:0, setPhase:function(p){ this._p=p; say(p); if(p===3){ mRun=-1e9; lRun=0; blockIdx=0; } },
      update:function(dt,t){
        var p=this._p;
        // ghost matrix visible only in naive/safe; greyed in online/flash
        ghost.forEach(function(m){ m.material.opacity=lerp(m.material.opacity, (p<=1?0.35:0.08), dt*3); m.material.color.setHex(p<=1?C.blue:C.gray); });
        if(p>=3){
          // animate K/V block streaming into the SRAM tile, updating m & l
          var phase=(t*0.6)%nblocks;
          blockIdx=Math.floor(phase);
          var frac=phase-blockIdx;
          var startX=-2.4 + (blockIdx%n)*sp - 0.5;
          blk.position.set(lerp(-4.5,2.6,frac), 0.3, 0);
          blk.visible=true;
          // fake running stats: converge upward
          var newMax = -0.5 + blockIdx*0.15;
          if(newMax>mRun){ mRun=newMax; }
          lRun = lRun*0.9 + 0.4;
          mSp.setText('m = '+mRun.toFixed(2), C.greenH);
          lSp.setText('l = '+lRun.toFixed(2), C.greenH);
          tile.scale.setScalar(1+0.06*Math.sin(t*4));
        } else {
          blk.visible=false;
          mSp.setText('m = -inf', C.white); lSp.setText('l = 0', C.white);
        }
      }
    };
  });

  /* =====================================================================
     ENGINE
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
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    var key=new THREE.DirectionalLight(0xffffff,0.8); key.position.set(4,6,4); scene.add(key);
    var rim=new THREE.DirectionalLight(0xa78bfa,0.35); rim.position.set(-5,2,-4); scene.add(rim);
    var grid=new THREE.GridHelper(16,16,C.grid,C.grid); grid.material.opacity=0.3; grid.material.transparent=true; grid.position.y=-2.6; scene.add(grid);

    var controls=null;
    if (THREE.OrbitControls){ controls=new THREE.OrbitControls(camera, renderer.domElement); controls.enableDamping=true; controls.dampingFactor=0.08; controls.enablePan=false; controls.minDistance=3; controls.maxDistance=16; }

    var inst = factory(THREE, scene, { camera: camera, controls: controls, C: C, setReadout: setReadout });
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
