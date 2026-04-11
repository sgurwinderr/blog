---
author: Gurwinder
categories:
- AI
- Quantization
date: '2026-03-31T00:00:00Z'
slug: 'turboquant'
aliases:
- /post/2026-03-31-turboquant/
featured: false
hidden: true
image: assets/images/turboquant.svg
title: 'TurboQuant from First Principles: One Example to the End'
---

TurboQuant is easiest to understand when we stop talking in abstractions and run one vector pair from start to finish.

The practical motivation is simple: take a key or value vector stored in 16-bit floating point and compress it down to 4 bits, without destroying the attention score that the model actually uses.

This post uses one tiny 4D example and keeps the same numbers all the way from the original 16-bit-style vector to a 4-bit-style compressed approximation and then to the final recovered score.

$$
q = [3.0,\; 0.5,\; -1.0,\; 2.0], \qquad
k = [1.0,\; 2.0,\; 0.5,\; -0.5]
$$

Ground-truth attention score:

$$
\langle q, k \rangle = 3(1) + 0.5(2) + (-1)(0.5) + 2(-0.5) = 2.5
$$

That number, $2.5$, is the anchor for the whole post. You can think of $q$ and $k$ as starting life in a normal high-precision format such as FP16, and the whole goal is to approximate that same score after squeezing the stored key down to a much lower bit budget.

Every compression step below is judged by one question:

1. If we compress the key vector, how much does the score move away from $2.5$?
2. If the score moves, is that movement random noise or systematic bias?
3. Can we keep most of the memory savings while still recovering the original score?

TurboQuant answers those questions by splitting the job into two pieces: a cheap coarse approximation and a small unbiased correction.

Reference inspiration: [toooold.com TurboQuant write-up](https://toooold.com/2026/03/28/turboquant.html).

---

## 1. First Principle: Attention is Just a Dot Product

If compression perturbs this score too much, attention picks the wrong token.

$$
\text{score}(q,k) = \langle q,k \rangle = \sum_i q_i k_i
$$

For our vectors, term-by-term:

$$
[3.0,\;1.0,\;-0.5,\;-1.0] \;\Rightarrow\; 2.5
$$

There is no mystery here. Each coordinate pair contributes one signed amount to the final attention score:

1. The first two coordinates push the score up.
2. The last two coordinates push the score down.
3. Attention is the balance of those positive and negative contributions.

That is why compression is dangerous. If one coordinate gets rounded up and another gets rounded down, the final score can drift even when each individual rounding looks small in isolation.

<div class="tq-block">
  <canvas id="tq-s1" class="tq-canvas"></canvas>
  <p class="tq-cap">Animation: term-by-term accumulation of the true score 2.500.</p>
</div>

---

## 2. Why Naive Quantization Fails (Bias)

Suppose we directly quantize $k$ with a tiny scalar quantizer. In the toy example below I use 2 bits so the bins are easy to see by hand, but the production motivation is the same idea scaled up: compressing from 16-bit storage to 4-bit storage.

A representative quantized key is:

$$
\hat{k}_{\text{naive}} = [1.167,\;2.0,\;0.333,\;-0.5]
$$

Then:

$$
\langle q,\hat{k}_{\text{naive}}\rangle = 3.168
$$

Error:

$$
3.168 - 2.5 = 0.668
$$

This is exactly the dangerous part: deterministic rounding introduces directional bias.

To see why, write down the 4 scalar levels. The range of $k$ is from $-0.5$ to $2.0$, so a 2-bit uniform quantizer uses 4 levels:

$$
-0.5,\;0.333,\;1.167,\;2.0
$$

Now each coordinate snaps to the nearest level:

$$
1.0 \to 1.167,\qquad 2.0 \to 2.0,\qquad 0.5 \to 0.333,\qquad -0.5 \to -0.5
$$

The key detail is that the rounding error does not cancel for this query:

$$
\langle q,\hat{k}_{\text{naive}}\rangle = 3(1.167) + 0.5(2.0) + (-1)(0.333) + 2(-0.5)
$$

$$
= 3.501 + 1.0 - 0.333 - 1.0 = 3.168
$$

The first coordinate alone contributes an extra $3 \times 0.167 \approx 0.501$ to the score. That is already most of the total bias. So the failure is not just "rounding error exists." The failure is "rounding error interacts with the query in a directional way."

<div class="tq-block">
  <canvas id="tq-s2" class="tq-canvas"></canvas>
  <p class="tq-cap">Animation: naive 2-bit snapping causes score drift from 2.500 to 3.168.</p>
</div>

---

## 3. Polarization Identity: Dot Product from Distances

TurboQuant leans on this bridge:

$$
\langle q,k \rangle = \frac{1}{4}\left(\|q+k\|_2^2 - \|q-k\|_2^2\right)
$$

For our example:

$$
q+k=[4,\;2.5,\;-0.5,\;1.5], \quad \|q+k\|_2^2 = 24.75
$$

$$
q-k=[2,\;-1.5,\;-1.5,\;2.5], \quad \|q-k\|_2^2 = 14.75
$$

$$
\frac{1}{4}(24.75 - 14.75)=2.5
$$

So if distances are preserved, the dot product is preserved.

This identity matters because distance is much easier to reason about under random projections than raw dot product. The logic is:

1. Attention depends on a dot product.
2. A dot product can be rewritten using two squared distances.
3. If a projection preserves those distances, it preserves attention scores approximately.

That is the whole conceptual bridge from “geometry of attention” to “JL-style compression.” TurboQuant is not preserving some mysterious Transformer quantity. It is preserving geometry well enough that the geometry-derived score stays accurate.

<div class="tq-block">
  <canvas id="tq-s3" class="tq-canvas"></canvas>
  <p class="tq-cap">Animation: distance bars combine to recover the same score.</p>
</div>

---

## 4. Rotation Step with FWHT (Explicit Numbers)

TurboQuant uses a random orthogonal transform. For a first-principles toy demo, use normalized Hadamard:

$$
H_4 = \frac{1}{2}
\begin{bmatrix}
1&1&1&1\\
1&-1&1&-1\\
1&1&-1&-1\\
1&-1&-1&1
\end{bmatrix}
$$

Compute rotated vectors:

$$
q' = H_4 q = [2.25,\;-0.25,\;1.25,\;2.75]
$$

$$
k' = H_4 k = [1.5,\;0.0,\;1.5,\;-1.0]
$$

Dot product is preserved by orthogonal transforms:

$$
\langle q',k'\rangle = 2.5
$$

It is worth computing one row by hand so this does not feel magical.

For the first row of $H_4$:

$$
q'_0 = \frac{1}{2}(3 + 0.5 - 1 + 2) = 2.25
$$

For the second row:

$$
q'_1 = \frac{1}{2}(3 - 0.5 - 1 - 2) = -0.25
$$

Doing the same for all rows gives the full rotated vector $q' = [2.25,-0.25,1.25,2.75]$. Likewise,

$$
k'_0 = \frac{1}{2}(1 + 2 + 0.5 - 0.5) = 1.5
$$

and so on until we get $k' = [1.5,0,1.5,-1.0]$.

The important first-principles fact is this: an orthogonal transform only changes coordinates, not geometry. It is a change of basis. Lengths stay the same, angles stay the same, and therefore the dot product stays the same.

For this example:

$$
\langle q',k'\rangle = 2.25(1.5) + (-0.25)(0) + 1.25(1.5) + 2.75(-1.0)
$$

$$
= 3.375 + 0 + 1.875 - 2.75 = 2.5
$$

<div class="tq-block">
  <canvas id="tq-s4" class="tq-canvas"></canvas>
  <p class="tq-cap">Animation: FWHT mixes coordinates while keeping the dot product invariant.</p>
</div>

---

## 5. Quantize in Rotated Space, Then Invert

Now quantize $k'$ with 2-bit levels. Using range $[-1.0, 1.5]$:

$$
\hat{k}' = [1.5,\;-0.167,\;1.5,\;-1.0]
$$

Invert transform ($H_4^{-1}=H_4$):

$$
\hat{k}_{\text{rot}} = H_4\hat{k}' \approx [0.917,\;2.084,\;0.417,\;-0.417]
$$

Score after rotated quantization:

$$
\langle q,\hat{k}_{\text{rot}}\rangle \approx 2.543
$$

Now error is only:

$$
2.543 - 2.5 = 0.043
$$

This is already much better than naive bias $0.668$.

Why did this help? Not because quantization disappeared. Quantization is still there. What changed is the shape of the thing we quantized.

In raw coordinates, the score was sensitive to a specific axis getting pushed the wrong way. In rotated coordinates, information is mixed across coordinates first. That makes the scalar quantizer’s mistakes less aligned with the original basis and less damaging to the final score.

You can verify the recovered score directly:

$$
\langle q,\hat{k}_{\text{rot}}\rangle = 3(0.917) + 0.5(2.084) + (-1)(0.417) + 2(-0.417)
$$

$$
= 2.751 + 1.042 - 0.417 - 0.834 = 2.542 \approx 2.543
$$

So the same 2-bit budget behaves very differently depending on whether we quantize before or after a geometry-preserving rotation.

In this toy example, the absolute score error drops from:

$$
0.668 \quad \text{to} \quad 0.043
$$

which is about a $15.5\times$ reduction in error.

<div class="tq-block">
  <canvas id="tq-s5" class="tq-canvas"></canvas>
  <p class="tq-cap">Animation: compare naive error vs rotated-space quantization error.</p>
</div>

---

## 6. Residual Correction (QJL Intuition)

Residual after rotated pipeline:

$$
r = k - \hat{k}_{\text{rot}} \approx [0.083,\;-0.084,\;0.083,\;-0.083]
$$

Its exact contribution to score is:

$$
\langle q,r\rangle \approx -0.042
$$

Then:

$$
2.543 + (-0.042) \approx 2.501
$$

QJL stores residual information in a 1-bit randomized form, and the estimator is unbiased in expectation:

$$
\mathbb{E}[\widehat{\langle q,r\rangle}_{\text{QJL}}] = \langle q,r\rangle
$$

So expected corrected score lands back at the true score.

This section is the second half of the design. The rotated coarse quantizer gave us low variance, but it still left a small error. TurboQuant does not ignore that error. It isolates it.

Because

$$
k = \hat{k}_{\text{rot}} + r
$$

we automatically have

$$
\langle q,k\rangle = \langle q,\hat{k}_{\text{rot}}\rangle + \langle q,r\rangle
$$

For our concrete residual,

$$
\langle q,r\rangle = 3(0.083) + 0.5(-0.084) + (-1)(0.083) + 2(-0.083)
$$

$$
= 0.249 - 0.042 - 0.083 - 0.166 \approx -0.042
$$

So the residual is exactly the missing piece that pulls the coarse estimate back down toward the true score.

The reason QJL matters is not that it stores the residual perfectly. It does not. The reason it matters is that its error is unbiased. Over repeated random projections, the correction term is centered on the right answer instead of consistently drifting in one direction.

<div class="tq-block">
  <canvas id="tq-s6" class="tq-canvas"></canvas>
  <p class="tq-cap">Animation: many randomized residual corrections center around the true value 2.500.</p>
</div>

---

## 7. One Example, End-to-End Ledger

Single-pair ledger for this post:

1. True score: $\langle q,k\rangle = 2.500$
2. Naive 2-bit on raw key: $3.168$ (bias $+0.668$)
3. Rotate -> quantize -> inverse: $2.543$ (error $+0.043$)
4. Add residual correction: $\approx 2.501$

That is TurboQuant in one line:

$$
\text{coarse low-bit estimate} + \text{unbiased residual correction}
$$

That compact formula hides the full story, so here it is in plain language:

1. Start with the true score we care about.
2. Naive raw-space quantization breaks it badly.
3. Rotating first makes coarse low-bit quantization much safer.
4. A small residual still remains.
5. The residual is corrected with a randomized unbiased estimator.
6. The final estimate comes back to the original score.

This is the main first-principles lesson of TurboQuant: it is not trying to make quantization error disappear. It is trying to make the large part of the error small and the remaining part unbiased.

<div class="tq-block">
  <canvas id="tq-s7" class="tq-canvas"></canvas>
  <p class="tq-cap">Animation: complete pipeline score moves from true -> biased -> corrected.</p>
</div>

---

## Final Mental Model

TurboQuant succeeds because it separates two jobs:

1. Low variance with coarse quantization.
2. Zero bias in expectation with randomized residual coding.

If you want a one-sentence summary, it is this:

$$
  ext{Rotate to make coarse quantization safer, then randomize the residual so the remaining error has zero mean.}
$$

Randomness is not noise here. It is the mechanism that protects attention geometry under aggressive compression.

<style>
.tq-block {
  margin: 1.5rem 0 2rem 0;
  text-align: center;
}

.tq-canvas {
  width: 100%;
  max-width: 920px;
  height: auto;
  border-radius: 10px;
  border: 1px solid #2a2f3d;
  background: linear-gradient(180deg, #0f1220 0%, #111827 100%);
}

.tq-cap {
  margin-top: 0.55rem;
  font-size: 0.88rem;
  color: #667085;
}
</style>

<script>
(function () {
  'use strict';

  var C = {
    bg: '#0f1220',
    panel: '#151d30',
    grid: '#293247',
    text: '#e5e7eb',
    dim: '#94a3b8',
    good: '#34d399',
    bad: '#f43f5e',
    blue: '#38bdf8',
    orange: '#fb923c',
    violet: '#a78bfa',
    yellow: '#fbbf24'
  };

  var q = [3.0, 0.5, -1.0, 2.0];
  var k = [1.0, 2.0, 0.5, -0.5];
  var kNaive = [1.167, 2.0, 0.333, -0.5];
  var qRot = [2.25, -0.25, 1.25, 2.75];
  var kRot = [1.5, 0.0, 1.5, -1.0];
  var kRotQ = [1.5, -0.167, 1.5, -1.0];
  var kBack = [0.917, 2.084, 0.417, -0.417];

  function dot(a, b) {
    var s = 0;
    for (var i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  var trueScore = dot(q, k);
  var naiveScore = dot(q, kNaive);
  var rotScore = dot(q, kBack);

  function setupCanvas(id) {
    var cv = document.getElementById(id);
    if (!cv) return null;
    cv.width = 920;
    cv.height = 320;
    return { cv: cv, ctx: cv.getContext('2d'), w: cv.width, h: cv.height };
  }

  function clear(g) {
    g.ctx.fillStyle = C.bg;
    g.ctx.fillRect(0, 0, g.w, g.h);
  }

  function text(g, s, x, y, size, color, align, bold) {
    g.ctx.font = (bold ? 'bold ' : '') + size + 'px system-ui, sans-serif';
    g.ctx.fillStyle = color || C.text;
    g.ctx.textAlign = align || 'left';
    g.ctx.fillText(s, x, y);
    g.ctx.textAlign = 'left';
  }

  function mono(g, s, x, y, size, color, align) {
    g.ctx.font = size + 'px ui-monospace, SFMono-Regular, Menlo, monospace';
    g.ctx.fillStyle = color || C.text;
    g.ctx.textAlign = align || 'left';
    g.ctx.fillText(s, x, y);
    g.ctx.textAlign = 'left';
  }

  function panel(g, x, y, w, h) {
    g.ctx.fillStyle = C.panel;
    g.ctx.fillRect(x, y, w, h);
    g.ctx.strokeStyle = C.grid;
    g.ctx.lineWidth = 1;
    g.ctx.strokeRect(x, y, w, h);
  }

  function barPanel(g, x, y, w, h, vals, color, maxAbs) {
    panel(g, x, y, w, h);
    var ctx = g.ctx;
    var z = y + h * 0.62;
    ctx.strokeStyle = '#334155';
    ctx.beginPath();
    ctx.moveTo(x, z);
    ctx.lineTo(x + w, z);
    ctx.stroke();

    var n = vals.length;
    var pad = 8;
    var bw = (w - pad * (n + 1)) / n;
    var sc = (h * 0.45) / maxAbs;

    for (var i = 0; i < n; i++) {
      var v = vals[i];
      var bh = Math.abs(v) * sc;
      var bx = x + pad + i * (bw + pad);
      var by = v >= 0 ? z - bh : z;
      ctx.fillStyle = Array.isArray(color) ? color[i] : color;
      ctx.fillRect(bx, by, bw, Math.max(2, bh));
      mono(g, v.toFixed(3), bx + bw / 2, v >= 0 ? by - 6 : by + bh + 13, 10, C.text, 'center');
    }
  }

  function pulse(t, speed) {
    return 0.5 + 0.5 * Math.sin(t * speed);
  }

  function loop(drawFn) {
    var t0 = performance.now();
    function frame(t) {
      drawFn((t - t0) / 1000);
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  // S1 Dot product accumulator
  var g1 = setupCanvas('tq-s1');
  if (g1) {
    loop(function (t) {
      clear(g1);
      text(g1, 'True Dot Product Build', 20, 30, 18, C.text, 'left', true);
      barPanel(g1, 20, 48, 190, 150, q, C.blue, 3.5);
      text(g1, 'q', 115, 216, 12, C.blue, 'center', true);
      barPanel(g1, 710, 48, 190, 150, k, C.orange, 2.5);
      text(g1, 'k', 805, 216, 12, C.orange, 'center', true);

      var terms = [3.0, 1.0, -0.5, -1.0];
      var names = ['3.0*1.0', '0.5*2.0', '-1.0*0.5', '2.0*(-0.5)'];
      var show = Math.floor((t % 8) / 2) + 1;

      text(g1, 'Element terms:', 250, 78, 13, C.dim, 'left', true);
      var sum = 0;
      for (var i = 0; i < show; i++) {
        sum += terms[i];
        mono(g1, names[i] + ' = ' + terms[i].toFixed(3), 250, 108 + i * 30, 14, terms[i] >= 0 ? C.good : C.bad);
      }
      text(g1, 'Running sum = ' + sum.toFixed(3), 250, 252, 15, C.yellow, 'left', true);
      if (show === 4) {
        text(g1, '<q, k> = 2.500', 250, 284, 22, C.good, 'left', true);
      }
    });
  }

  // S2 Naive bias
  var g2 = setupCanvas('tq-s2');
  if (g2) {
    loop(function (t) {
      clear(g2);
      text(g2, 'Naive 2-bit Quantization Bias', 20, 30, 18, C.text, 'left', true);

      var a = pulse(t, 1.8);
      var mix = [];
      for (var i = 0; i < k.length; i++) mix.push(k[i] * (1 - a) + kNaive[i] * a);

      barPanel(g2, 20, 48, 240, 150, k, C.orange, 2.5);
      text(g2, 'k (original)', 140, 216, 12, C.orange, 'center', true);

      text(g2, '->', 282, 130, 24, C.dim, 'center', true);

      barPanel(g2, 320, 48, 240, 150, mix, C.bad, 2.5);
      text(g2, 'k naive-quantized', 440, 216, 12, C.bad, 'center', true);

      var s = dot(q, mix);
      mono(g2, '<q,k> true   = 2.500', 590, 90, 16, C.good);
      mono(g2, '<q,k_hat>    = ' + s.toFixed(3), 590, 126, 16, C.bad);
      mono(g2, 'bias         = ' + (s - trueScore).toFixed(3), 590, 162, 16, C.yellow);
      text(g2, 'Deterministic rounding shifts attention score.', 590, 196, 12, C.dim);
    });
  }

  // S3 Polarization identity
  var g3 = setupCanvas('tq-s3');
  if (g3) {
    loop(function (t) {
      clear(g3);
      text(g3, 'Polarization Identity', 20, 30, 18, C.text, 'left', true);
      mono(g3, '<q,k> = 1/4 ( ||q+k||^2 - ||q-k||^2 )', 20, 58, 15, C.violet);

      var a = 24.75;
      var b = 14.75;
      var left = 20, top = 90, w = 870, h = 34;

      panel(g3, left, top, w, h);
      g3.ctx.fillStyle = 'rgba(52,211,153,0.5)';
      g3.ctx.fillRect(left, top, w * (a / 30), h);
      text(g3, '||q+k||^2 = 24.75', left + 10, top + 22, 13, C.text);

      panel(g3, left, top + 62, w, h);
      g3.ctx.fillStyle = 'rgba(244,63,94,0.5)';
      g3.ctx.fillRect(left, top + 62, w * (b / 30), h);
      text(g3, '||q-k||^2 = 14.75', left + 10, top + 84, 13, C.text);

      var est = 0.25 * (a - b);
      text(g3, 'Recovered score: 1/4 * (24.75 - 14.75) = ' + est.toFixed(3), 20, 212, 19, C.good, 'left', true);
      g3.ctx.globalAlpha = 0.4 + 0.6 * pulse(t, 2.4);
      text(g3, 'same as true score 2.500', 20, 248, 14, C.yellow, 'left', true);
      g3.ctx.globalAlpha = 1;
    });
  }

  // S4 FWHT rotation
  var g4 = setupCanvas('tq-s4');
  if (g4) {
    loop(function (t) {
      clear(g4);
      text(g4, 'FWHT Rotation (H4)', 20, 30, 18, C.text, 'left', true);
      mono(g4, "q' = [2.25, -0.25, 1.25, 2.75]", 20, 60, 14, C.blue);
      mono(g4, "k' = [1.50,  0.00, 1.50,-1.00]", 20, 84, 14, C.orange);

      var m = pulse(t, 1.6);
      var qMix = q.map(function (v, i) { return v * (1 - m) + qRot[i] * m; });
      var kMix = k.map(function (v, i) { return v * (1 - m) + kRot[i] * m; });

      barPanel(g4, 20, 100, 250, 150, qMix, C.blue, 3.5);
      text(g4, 'q -> q\'', 145, 268, 12, C.blue, 'center', true);
      barPanel(g4, 300, 100, 250, 150, kMix, C.orange, 2.5);
      text(g4, 'k -> k\'', 425, 268, 12, C.orange, 'center', true);

      var s1 = dot(qMix, kMix);
      mono(g4, '<q,k>           = 2.500', 590, 130, 16, C.good);
      mono(g4, '<q_mixed,k_mixed> = ' + s1.toFixed(3), 590, 164, 16, C.yellow);
      mono(g4, '<q\',k\'>         = 2.500', 590, 198, 16, C.good);
      text(g4, 'Orthogonal transform preserves inner product.', 590, 228, 12, C.dim);
    });
  }

  // S5 rotated quantization improvement
  var g5 = setupCanvas('tq-s5');
  if (g5) {
    loop(function () {
      clear(g5);
      text(g5, 'Quantize in Rotated Space: Much Smaller Error', 20, 30, 18, C.text, 'left', true);

      barPanel(g5, 20, 52, 200, 150, kRot, C.orange, 2.0);
      text(g5, "k'", 120, 220, 12, C.orange, 'center', true);
      text(g5, '->', 242, 130, 24, C.dim, 'center', true);
      barPanel(g5, 280, 52, 200, 150, kRotQ, C.violet, 2.0);
      text(g5, "k'_quant", 380, 220, 12, C.violet, 'center', true);
      text(g5, '-> H4^-1 ->', 515, 130, 12, C.dim, 'center', true);
      barPanel(g5, 570, 52, 200, 150, kBack, C.good, 2.5);
      text(g5, 'k_hat_rot', 670, 220, 12, C.good, 'center', true);

      mono(g5, 'naive score:   3.168   (error +0.668)', 20, 266, 15, C.bad);
      mono(g5, 'rotated score: 2.543   (error +0.043)', 20, 294, 15, C.good);
    });
  }

  // S6 residual correction distribution
  var g6 = setupCanvas('tq-s6');
  if (g6) {
    loop(function (t) {
      clear(g6);
      text(g6, 'Residual Correction is Unbiased in Expectation', 20, 30, 18, C.text, 'left', true);

      var base = rotScore;
      var trueLine = trueScore;
      var x0 = 40, y0 = 250, w = 840, h = 160;
      panel(g6, x0, 72, w, h);

      g6.ctx.strokeStyle = C.good;
      g6.ctx.setLineDash([6, 4]);
      var yt = y0 - trueLine * 40;
      g6.ctx.beginPath();
      g6.ctx.moveTo(x0 + 5, yt);
      g6.ctx.lineTo(x0 + w - 5, yt);
      g6.ctx.stroke();
      g6.ctx.setLineDash([]);
      text(g6, 'true = 2.500', x0 + w - 110, yt - 8, 11, C.good);

      for (var i = 0; i < 30; i++) {
        var phase = t * 1.2 + i * 0.6;
        var corr = -0.042 + 0.09 * Math.sin(phase) * Math.cos(phase * 0.7);
        var est = base + corr;
        var x = x0 + 15 + i * 27;
        var y = y0 - est * 40;
        g6.ctx.fillStyle = 'rgba(167,139,250,0.75)';
        g6.ctx.fillRect(x, y, 8, 8);
      }

      mono(g6, 'coarse estimate: 2.543', 20, 280, 14, C.yellow);
      mono(g6, 'mean corrected estimate ~ 2.500', 20, 304, 14, C.good);
    });
  }

  // S7 end-to-end ledger
  var g7 = setupCanvas('tq-s7');
  if (g7) {
    loop(function (t) {
      clear(g7);
      text(g7, 'End-to-End Score Path (Same Example)', 20, 30, 18, C.text, 'left', true);

      var steps = [
        ['true', 2.500, C.good],
        ['naive', 3.168, C.bad],
        ['rotated', 2.543, C.yellow],
        ['corrected', 2.501, C.good]
      ];

      var xStart = 80;
      var yBase = 270;
      var sc = 45;
      for (var i = 0; i < steps.length; i++) {
        var x = xStart + i * 210;
        panel(g7, x - 55, 70, 110, 200);
        var h = steps[i][1] * sc;
        g7.ctx.fillStyle = steps[i][2];
        g7.ctx.fillRect(x - 22, yBase - h, 44, h);
        text(g7, steps[i][0], x, 292, 12, C.dim, 'center', true);
        mono(g7, steps[i][1].toFixed(3), x, yBase - h - 8, 12, C.text, 'center');

        if (i < steps.length - 1) {
          var pulseAlpha = 0.35 + 0.65 * pulse(t + i, 2.2);
          g7.ctx.globalAlpha = pulseAlpha;
          text(g7, '->', x + 105, 170, 26, C.violet, 'center', true);
          g7.ctx.globalAlpha = 1;
        }
      }

      mono(g7, '2.500 -> 3.168 -> 2.543 -> 2.501', 20, 56, 15, C.violet);
      text(g7, 'Single pair carried to the end: bias appears, then is corrected.', 20, 314, 12, C.dim);
    });
  }
})();
</script>
