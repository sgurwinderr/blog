---
author: Gurwinder
categories:
- AI
- Triton
- GPU
date: '2026-06-24T00:00:00Z'
slug: 'stage-large-fma-dots-via-slm'
featured: false
draft: true
image: assets/images/triton.png
imageAlt: 'TTGIR transformation diagram: tt.dot K-loop staged through ttg.local_alloc and ttg.memdesc_subslice into SLM on Intel iGPU'
description: 'A TTGIR pass that stages tt.dot operands in SLM and tiles K — recovers 9/9 PTSS-overflow regressions on Intel ARL-S iGPU at 1.86× geomean speedup.'
title: 'Staging Large FMA Dots via SLM: A TTGIR Pass for Non-DPAS Intel GPUs'
---

<style>
.post-diagram { background: #fff; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 12px rgba(44,42,40,0.08); margin: 2rem 0; overflow-x: auto; border: 1px solid #eeebe5; }
.post-diagram svg { display: block; margin: 0 auto; max-width: 100%; height: auto; }
.post-diagram-caption { text-align: center; font-size: 0.875rem; color: #6b6560; font-style: italic; margin-top: 1rem; }
.mem-text { font-family: 'DM Sans', system-ui, sans-serif; font-size: 13px; fill: #2c2a28; }
.mem-mono { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px; fill: #6b6560; }
.mem-title { font-family: 'Bricolage Grotesque', Georgia, serif; font-size: 14px; font-weight: 700; fill: #2c2a28; }
.post-step-cards { display: flex; flex-direction: column; gap: 0.75rem; margin: 1.5rem 0; }
.post-step-card { display: flex; align-items: flex-start; gap: 1rem; padding: 1rem 1.25rem; background: #fafbfc; border-radius: 12px; border-left: 3px solid #0071c5; box-shadow: 0 1px 2px rgba(44,42,40,0.05); }
.post-step-num { width: 32px; height: 32px; border-radius: 50%; background: #0071c5; color: white; font-weight: 700; display: flex; align-items: center; justify-content: center; font-family: 'Bricolage Grotesque', Georgia, serif; flex-shrink: 0; }
.post-step-body strong { display: block; font-weight: 600; margin-bottom: 0.25rem; color: #2c2a28; }
.post-step-body p { margin: 0.25rem 0 0; color: #4a4540; font-size: 0.95rem; line-height: 1.55; }
.post-callout { display: flex; gap: 1rem; padding: 1.25rem; border-radius: 12px; margin: 1.5rem 0; border-left: 4px solid; }
.post-callout-info { background: #e4f2f7; border-color: #2a7b9b; }
.post-callout-warn { background: #fde8e8; border-color: #c93b3b; }
.post-callout-accent { background: #e1f0fa; border-color: #0071c5; }
.post-callout-icon { font-size: 1.25rem; flex-shrink: 0; }
.post-callout-title { display: block; font-weight: 700; margin-bottom: 0.25rem; color: #2c2a28; }
.post-callout-content p { font-size: 0.95rem; margin: 0; color: #2c2a28; line-height: 1.55; }
.post-bench { width: 100%; border-collapse: collapse; margin: 1.5rem 0; font-size: 0.9rem; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(44,42,40,0.1); border: 1px solid #d8d2c8; }
.post-bench th { background: #2c2a28; color: #fff; font-weight: 700; padding: 0.75rem 1rem; text-align: left; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
.post-bench td { padding: 0.75rem 1rem; border-top: 1px solid #eeebe5; color: #2c2a28; }
.post-bench tr:nth-child(even) td { background: #fafbfc; }
.post-bench .pass-fail { color: #c93b3b; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
.post-bench .pass-ok { color: #2d8b55; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.post-bench .speedup { color: #0071c5; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.post-bench tbody tr:hover td { background: #e1f0fa; }
</style>

A tt.dot over a BlockedEncoding on non-DPAS Intel GPUs lowers to a fused-multiply-add (FMA) loop nest. If the K dimension is large, the LLVM-level unroller produces a register schedule wider than the 4 KB GRF, the surplus spills into Per-Thread Scratch Space (PTSS), and on integrated Xe-LPG hardware (ARL-S, A770, MTL) we blow past the 256 KB PTSS limit — the IGC build fails before the kernel ever runs.

This post walks through tritonintelgpu-stage-large-fma-dots-via-slm — a TTGIR pass that fixes the cliff by staging operands in SLM and emitting K-tile chunks against a memdesc_subslice view. On a 24-case ARL-S sweep it recovers **9/9 PTSS regressions** and delivers a **1.86× geomean speedup** on cases where both builds compile.

> Note: this is a deep walk through compiler internals. If you're new to Triton's execution model, start with [Triton Kernels from First Principles](/triton-kernel-first-principles/). For SIMD32/workgroup/SLM hardware background, [Intel GPU Scheduling](/gpu-kernel-scheduling/) is the primer.

The post is structured as a linear build: first the problem, then the idea, then the algorithm, then the IR, then the math, then the gates, then where it sits in the pipeline, then design notes, then numbers, then prior art, then risk.

---

## 1. The Problem: A Kernel That Won't Build

### 1.1 First, the playing field — where bytes can live

Every GPU optimization argument is, at heart, an argument about *where* a value sits when you need it. Get the location wrong and either the memory bus or the register file bottlenecks you. Three locations matter for this story:

<div class="post-diagram">
<svg viewBox="0 0 720 380" xmlns="http://www.w3.org/2000/svg" aria-label="Memory hierarchy on Intel Xe-LPG">
  <rect x="20" y="290" width="680" height="70" rx="8" fill="#e8e0d0" stroke="#6b6560" stroke-width="2"/>
  <text x="360" y="316" class="mem-title" text-anchor="middle">DRAM / system memory</text>
  <text x="360" y="338" class="mem-mono" text-anchor="middle">~50 GB/s · GBs of capacity · ~hundreds of cycles · PTSS lives here</text>

  <rect x="80" y="200" width="560" height="70" rx="8" fill="#f0e6f6" stroke="#7b6daa" stroke-width="2"/>
  <text x="360" y="226" class="mem-title" text-anchor="middle">L3 cache (shared across the GPU)</text>
  <text x="360" y="248" class="mem-mono" text-anchor="middle">few MB · ~tens of cycles</text>

  <rect x="180" y="110" width="360" height="70" rx="8" fill="#d6eaf8" stroke="#0071c5" stroke-width="2"/>
  <text x="360" y="136" class="mem-title" text-anchor="middle">SLM — Shared Local Memory</text>
  <text x="360" y="158" class="mem-mono" text-anchor="middle">64 KB per workgroup · ~few cycles · software-managed</text>

  <rect x="260" y="20" width="200" height="70" rx="8" fill="#fde4d0" stroke="#d94f30" stroke-width="2"/>
  <text x="360" y="46" class="mem-title" text-anchor="middle">GRF — General Register File</text>
  <text x="360" y="68" class="mem-mono" text-anchor="middle">4 KB per thread · 1 cycle</text>

  <path d="M 360 90 L 360 110" stroke="#6b6560" stroke-width="2" fill="none" marker-end="url(#arr1)"/>
  <path d="M 360 180 L 360 200" stroke="#6b6560" stroke-width="2" fill="none" marker-end="url(#arr1)"/>
  <path d="M 360 270 L 360 290" stroke="#6b6560" stroke-width="2" fill="none" marker-end="url(#arr1)"/>
  <defs>
    <marker id="arr1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6b6560"/>
    </marker>
  </defs>
</svg>
<div class="post-diagram-caption">Smaller, faster, more private as you go up. The pass we're studying moves bytes between these levels to dodge a specific disaster.</div>
</div>

<div class="post-step-cards">
  <div class="post-step-card">
    <div class="post-step-num">1</div>
    <div class="post-step-body">
      <strong>GRF — the register file</strong>
      <p>Each hardware thread on Xe-LPG owns 128 registers, each 32 bytes wide, for a total of <strong>4 KB</strong>. That's the entire universe of "things this thread can name in a single cycle." Run out of GRF and the compiler has to spill.</p>
    </div>
  </div>
  <div class="post-step-card">
    <div class="post-step-num">2</div>
    <div class="post-step-body">
      <strong>SLM — Shared Local Memory</strong>
      <p>A scratchpad shared across all threads of one workgroup. <strong>64 KB</strong> on Xe-LPG iGPUs. Software-controlled; one or two cycles to access. (CUDA calls it "shared memory"; OpenCL/SYCL call it "local memory.")</p>
    </div>
  </div>
  <div class="post-step-card">
    <div class="post-step-num">3</div>
    <div class="post-step-body">
      <strong>PTSS — Per-Thread Scratch Space</strong>
      <p>Not really a separate physical place — a region in DRAM (backed by L3) the driver allocates per thread, where the compiler dumps spilled registers. <strong>Capped at 256 KB per thread</strong> on Xe-LPG / Xe-HPG / Xe-HPC. Spill more than that and your kernel doesn't even build.</p>
    </div>
  </div>
</div>

### 1.2 Two paths from a tt.dot

When you write `tl.dot(a, b, acc)` in a Triton kernel, the compiler picks one of two lowering paths depending on the hardware:

- **DPAS path** — On hardware with a systolic matrix unit (PVC, BMG, ARL-H Xe2), the dot lowers to a few `dpas` instructions per K step. Hardware does the heavy lifting; per-thread register pressure stays bounded. **This pass is a no-op on DPAS hardware.**
- **FMA path** — On non-DPAS Intel iGPUs (ARL-S, MTL, A770, and other integrated Xe-LPG GPUs) there is no DPAS unit. The dot falls back to a hand-coded FMA loop nest, fully unrolled along K. **This is the path that breaks.**

### 1.3 The arithmetic of "fully unrolled K"

Take a concrete shape: a 64×128×128 f16 GEMM where one thread covers an 8×8 micro-tile of the output.

<div class="post-diagram">
<svg viewBox="0 0 700 300" xmlns="http://www.w3.org/2000/svg" aria-label="One thread covers an 8x8 micro-tile">
  <rect x="40" y="60" width="80" height="160" fill="#e1f0fa" stroke="#0071c5" stroke-width="2"/>
  <text x="80" y="50" class="mem-title" text-anchor="middle">A: 64×128</text>
  <rect x="40" y="80" width="80" height="20" fill="#0071c5" opacity="0.5"/>

  <rect x="180" y="60" width="160" height="80" fill="#e1f0fa" stroke="#0071c5" stroke-width="2"/>
  <text x="260" y="50" class="mem-title" text-anchor="middle">B: 128×128</text>
  <rect x="200" y="60" width="20" height="80" fill="#0071c5" opacity="0.5"/>

  <text x="370" y="115" class="mem-title" font-size="22" text-anchor="middle">=</text>

  <rect x="420" y="60" width="160" height="80" fill="#fde4d0" stroke="#d94f30" stroke-width="2"/>
  <text x="500" y="50" class="mem-title" text-anchor="middle">C: 64×128</text>
  <rect x="440" y="80" width="20" height="20" fill="#d94f30"/>

  <text x="350" y="255" class="mem-text" text-anchor="middle">For my 8×8 chunk of C, I need C[i][j] = Σ A[i][k] × B[k][j] for k = 0..127</text>
  <text x="350" y="280" class="mem-text" text-anchor="middle">That's 64 output cells × 128 K-steps = <tspan font-weight="700" fill="#0071c5">8192 FMAs per thread, all unrolled flat</tspan></text>
</svg>
<div class="post-diagram-caption">One thread is responsible for 64 output cells × 128 multiply-add steps each = 8192 FMAs. Highlighted strips are the operand slices it actually consumes.</div>
</div>

The Triton lowering, plus IGC's backend, fully unrolls that K loop. There is no TTGIR-level K-tile knob in upstream's pipeliner.[^triton-pipe] IGC has a register-pressure-aware unroll heuristic and a `DisableLoopUnroll` switch,[^igc-unroll] but they don't fire on the Triton-emitted IR pattern at the pressure regime that bites here.

### 1.4 The cliff: 345 KB spill, 256 KB cap

A SIMD16 thread on Xe-LPG has a **4 KB GRF** in the default *small* register mode (128 × 32-byte registers). Xe-LPG also supports a *large* register mode that doubles capacity to 8 KB, but it halves thread occupancy per XVE — and Intel's `-ftarget-register-alloc-mode` flag is currently restricted to PVC, so flipping into it isn't a portable knob.[^xe-grf]

The fully-unrolled schedule wants more than 4 KB. IGC dutifully spills the surplus into PTSS. ARL-S, A770, and MTL iGPUs cap PTSS at **256 KB per thread**, enforced by Intel's compute-runtime (NEO) user-mode driver in `gfx_core_helper_xehp_and_later.inl::getMaxScratchSize`.[^neo-ptss] An over-budget kernel is rejected at `zeKernelCreate` time with `ZE_RESULT_ERROR_INVALID_NATIVE_BINARY` — the binary never loads.

On the 64×128×128 f16 GEMM, the spilled schedule wants **345 KB**:

<div class="post-diagram">
<svg viewBox="0 0 700 240" xmlns="http://www.w3.org/2000/svg" aria-label="PTSS overflow: 345 KB requested, 256 KB ceiling">
  <line x1="500" y1="40" x2="500" y2="120" stroke="#c93b3b" stroke-width="2" stroke-dasharray="6,3"/>
  <text x="500" y="32" class="mem-title" fill="#c93b3b" text-anchor="middle">256 KB ceiling</text>

  <rect x="40" y="65" width="460" height="40" rx="6" fill="#e8f5ee" stroke="#2d8b55" stroke-width="2"/>
  <rect x="500" y="65" width="160" height="40" rx="6" fill="#fde0e0" stroke="#c93b3b" stroke-width="2"/>
  <text x="270" y="90" class="mem-title" text-anchor="middle" fill="#2d8b55">"in-budget" zone</text>
  <text x="580" y="90" class="mem-title" text-anchor="middle" fill="#c93b3b">overshoot 89 KB</text>

  <text x="350" y="135" class="mem-mono" text-anchor="middle">FMA-unrolled spill estimate per thread: 345 KB</text>

  <text x="40" y="175" class="mem-text">↳ IGC reports: <tspan font-family="JetBrains Mono, monospace" fill="#c93b3b">out of scratch space</tspan></text>
  <text x="40" y="195" class="mem-text">↳ Driver returns: <tspan font-family="JetBrains Mono, monospace" fill="#c93b3b">ZE_RESULT_ERROR_INVALID_NATIVE_BINARY</tspan></text>
  <text x="40" y="215" class="mem-text">↳ <tspan font-weight="700">No SPIR-V is emitted. The kernel never gets to run.</tspan></text>
</svg>
<div class="post-diagram-caption">Full-K unroll asks for 345 KB; the driver caps you at 256 KB. There is no compiler flag to flip.</div>
</div>

This is not a soft regression. The kernel does not run at all.

---

## 2. The Idea: Stage in SLM, Tile K

The pass detects this regime in TTGIR — *before* the LLVM unroller sees the IR — and rewrites the dot so K is no longer fully unrolled. The mechanism: park the operands in SLM, then replace the monolithic dot with a short ladder of K-tile dots over `memdesc_subslice` views.

<div class="post-diagram">
<svg viewBox="0 0 720 320" xmlns="http://www.w3.org/2000/svg" aria-label="Before vs after: monolithic dot vs four K-tiled dots">
  <text x="180" y="25" class="mem-title" text-anchor="middle" fill="#c93b3b">BEFORE — one monolithic K=128 dot</text>
  <rect x="40" y="40" width="280" height="80" rx="8" fill="#fde0e0" stroke="#c93b3b" stroke-width="2"/>
  <text x="180" y="72" class="mem-title" text-anchor="middle">tt.dot (full K=128)</text>
  <text x="180" y="98" class="mem-mono" text-anchor="middle">8192 FMAs unrolled flat</text>
  <text x="180" y="148" class="mem-mono" text-anchor="middle" fill="#c93b3b">→ 345 KB spill → fail</text>

  <text x="540" y="25" class="mem-title" text-anchor="middle" fill="#2d8b55">AFTER — four K=32 dots fed from SLM</text>
  <rect x="380" y="40" width="78" height="80" rx="6" fill="#e8f5ee" stroke="#2d8b55" stroke-width="2"/>
  <rect x="468" y="40" width="78" height="80" rx="6" fill="#e8f5ee" stroke="#2d8b55" stroke-width="2"/>
  <rect x="556" y="40" width="78" height="80" rx="6" fill="#e8f5ee" stroke="#2d8b55" stroke-width="2"/>
  <rect x="644" y="40" width="56" height="80" rx="6" fill="#e8f5ee" stroke="#2d8b55" stroke-width="2"/>
  <text x="419" y="76" class="mem-mono" text-anchor="middle">K=32</text>
  <text x="507" y="76" class="mem-mono" text-anchor="middle">K=32</text>
  <text x="595" y="76" class="mem-mono" text-anchor="middle">K=32</text>
  <text x="672" y="76" class="mem-mono" text-anchor="middle">K=32</text>
  <text x="419" y="96" class="mem-mono" text-anchor="middle" fill="#2d8b55">~1 KB</text>
  <text x="507" y="96" class="mem-mono" text-anchor="middle" fill="#2d8b55">~1 KB</text>
  <text x="595" y="96" class="mem-mono" text-anchor="middle" fill="#2d8b55">~1 KB</text>
  <text x="672" y="96" class="mem-mono" text-anchor="middle" fill="#2d8b55">~1 KB</text>
  <text x="540" y="148" class="mem-mono" text-anchor="middle" fill="#2d8b55">→ each fits in GRF, no spill</text>

  <rect x="380" y="200" width="320" height="60" rx="8" fill="#d6eaf8" stroke="#0071c5" stroke-width="2"/>
  <text x="540" y="226" class="mem-title" text-anchor="middle">SLM: full A and B staged once (16 KB)</text>
  <text x="540" y="248" class="mem-mono" text-anchor="middle">memdesc_subslice + local_load per K-tile</text>

  <path d="M 419 200 L 419 168" stroke="#0071c5" stroke-width="2" fill="none" marker-end="url(#arr2)"/>
  <path d="M 507 200 L 507 168" stroke="#0071c5" stroke-width="2" fill="none" marker-end="url(#arr2)"/>
  <path d="M 595 200 L 595 168" stroke="#0071c5" stroke-width="2" fill="none" marker-end="url(#arr2)"/>
  <path d="M 672 200 L 672 168" stroke="#0071c5" stroke-width="2" fill="none" marker-end="url(#arr2)"/>
  <text x="540" y="290" class="mem-text" text-anchor="middle">Same total work. Fewer registers live at once.</text>
  <defs>
    <marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#0071c5"/>
    </marker>
  </defs>
</svg>
<div class="post-diagram-caption">Before: one big dot, full unroll, spill exceeds the cap. After: SLM holds A and B, four small dots pull K-tiles back, each one fits in GRF.</div>
</div>

**Why SLM specifically?** We need a place that's bigger than the GRF (so the full K-extent of A and B fits) but faster than DRAM (so the K-tile loads aren't a regression). On Xe-LPG, SLM is 64 KB per workgroup and one or two cycles to access — exactly the right rung.

**A trick to flag now.** When an operand is f16 but the dot promotes it to f32, the pass stages the **pre-promotion f16** bytes in SLM (16 KB instead of 32 KB) and replays the cast on each tile load. We'll see why that matters once we get to the SLM-fit gate.

---

## 3. The Algorithm in Five Steps

Per dot, the pass does five things in order. No IR yet — just the verbal sketch:

<div class="post-step-cards">
  <div class="post-step-card">
    <div class="post-step-num">1</div>
    <div class="post-step-body">
      <strong>Walk back to the staging boundary</strong>
      <p>From each dot operand, walk back through optional fp_to_fp / arith.extf casts. Land on either a ttg.convert_layout (fresh boundary, allocate SLM) or an existing ttg.local_load (reuse SLM).</p>
    </div>
  </div>
  <div class="post-step-card">
    <div class="post-step-num">2</div>
    <div class="post-step-body">
      <strong>Stage at the smallest representation</strong>
      <p>If the walk passed through a promoting cast (e.g. f16 → f32), allocate SLM with the <em>pre-cast</em> type. The cast is replayed per K-tile load. For f16 → f32 that's 2× SLM savings, often the difference between fitting and not.</p>
    </div>
  </div>
  <div class="post-step-card">
    <div class="post-step-num">3</div>
    <div class="post-step-body">
      <strong>Emit the K-tile chunks</strong>
      <p>For each K-tile of size K_TILE: emit memdesc_subslice (slice along K) → local_load (materialize the per-tile tensor) → replayed cast → tt.dot with running accumulator. The last K-tile dot replaces all uses of the original dot.</p>
    </div>
  </div>
  <div class="post-step-card">
    <div class="post-step-num">4</div>
    <div class="post-step-body">
      <strong>Erase orphans</strong>
      <p>The original convert_layout and any walked-through fp_to_fp ops have no users now. Erase them so downstream passes don't see dead IR.</p>
    </div>
  </div>
  <div class="post-step-card">
    <div class="post-step-num">5</div>
    <div class="post-step-body">
      <strong>SLM reuse via findStagedSmem</strong>
      <p>Before allocating SLM in step 2, check whether the operand is already a local_load from an earlier dot in the same block. If so, reuse the source memdesc instead of allocating again. This makes the pass <em>idempotent</em> under software pipelining — without it, double-staging would blow the SLM-fit gate.</p>
    </div>
  </div>
</div>

Step 1 walks back through casts because AccelerateMatmul's `decomposeMixedModeDotOp` runs a `promoteOperand` helper between convert_layout and the dot — emitting tt.fp_to_fp for float8 element types and arith.extf otherwise.[^promote]

---

## 4. The IR Rewrite, Concretely

Now the actual MLIR. **Source IR** (after AccelerateMatmul, before this pass):

```mlir
%a_blk = ttg.convert_layout %a : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #dot_a>
%b_blk = ttg.convert_layout %b : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #dot_b>
%a_f32 = tt.fp_to_fp %a_blk : tensor<64x128xf16, #dot_a> -> tensor<64x128xf32, #dot_a>
%b_f32 = tt.fp_to_fp %b_blk : tensor<128x128xf16, #dot_b> -> tensor<128x128xf32, #dot_b>
%c     = tt.dot %a_f32, %b_f32, %c_init
         : tensor<64x128xf32, #dot_a> * tensor<128x128xf32, #dot_b> -> tensor<64x128xf32, #blocked>
```

The dot is monolithic. K = 128 is implicit in the operand shapes; the LLVM unroller will see this as one big multiply-accumulate and unroll the inner K loop completely.

**Target IR** (after this pass, K_TILE = 32):

```mlir
// Stage at the smallest representation — pre-fp_to_fp f16 sources.
%a_smem = ttg.local_alloc %a : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared>
%b_smem = ttg.local_alloc %b : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared>

// Walk K in tiles of 32. Replay fp_to_fp per partial-K load to keep SLM cost bounded.
%c0 = ttg.memdesc_subslice %a_smem [0, 0]   [64, 32]  : !ttg.memdesc<64x32xf16,  #shared>
%a0 = ttg.local_load %c0                    : tensor<64x32xf16,  #dot_a>
%a0f = tt.fp_to_fp %a0                      : tensor<64x32xf32,  #dot_a>
%d0 = ttg.memdesc_subslice %b_smem [0, 0]   [32, 128] : !ttg.memdesc<32x128xf16, #shared>
%b0 = ttg.local_load %d0                    : tensor<32x128xf16, #dot_b>
%b0f = tt.fp_to_fp %b0                      : tensor<32x128xf32, #dot_b>
%c0_acc = tt.dot %a0f, %b0f, %c_init        // K-tile 0

%c1 = ttg.memdesc_subslice %a_smem [0, 32]  [64, 32]  : ...
%a1 = ttg.local_load %c1
%a1f = tt.fp_to_fp %a1
%d1 = ttg.memdesc_subslice %b_smem [32, 0]  [32, 128] : ...
%b1 = ttg.local_load %d1
%b1f = tt.fp_to_fp %b1
%c1_acc = tt.dot %a1f, %b1f, %c0_acc        // K-tile 1, threading the accumulator

// ... K-tiles 2 and 3 ...
// %c3_acc is the final result; replaces all uses of %c.
```

Two structural observations:

1. The operand pulled into SLM is **f16**, not the post-fp_to_fp f32. The cast is replayed on each partial-K local_load. That cuts the SLM footprint in half and is what keeps the staging cost under the SLM-fit gate (next section).
2. Each K-tile dot threads its result into the next dot's accumulator. After the pass, the LLVM unroller sees four small K-32 dots instead of one K-128 dot — register pressure per dot drops by ~4×, no PTSS spill.

---

## 5. Picking K_TILE — A Per-Thread Byte Budget

K_TILE isn't hardcoded — it's derived from a per-thread byte budget. This is the math behind the K=32 in the IR snippet above.

Aim to use half the GRF for accumulator + operand registers per K-tile, leaving the rest for instruction-scheduling slack:

$$
B_{\text{per-thread}} \;=\; \frac{B_{\text{GRF}}}{2} \;=\; \frac{4096}{2} \;=\; 2048 \text{ B}
$$

For one K-tile of size $K_t$, on a thread covering an $M_t \times N_t$ micro-tile, with operand element-size $s_{\text{op}}$ and accumulator element-size $s_{\text{acc}}$:

$$
B_{\text{tile}}(K_t) \;=\; \underbrace{M_t N_t \cdot s_{\text{acc}}}_{\text{accumulator}} \;+\; \underbrace{(M_t + N_t) \cdot K_t \cdot s_{\text{op}}}_{\text{A and B operand stripes}}
$$

For the canonical case — $M_t = N_t = 8$, $s_{\text{op}} = 2$ B (f16), $s_{\text{acc}} = 4$ B (f32):

$$
B_{\text{tile}}(K_t) \;=\; 64 \cdot 4 \;+\; 16 K_t \cdot 2 \;=\; 256 + 32 K_t
$$

Set $B_{\text{tile}}(K_t) \le 2048$:

$$
256 + 32 K_t \;\le\; 2048 \;\;\Longrightarrow\;\; K_t \;\le\; 56
$$

Pick the largest power-of-two ≤ 56 that divides K=128: **K_TILE = 32**. The same formula handles f32 operands by setting $s_{\text{op}} = 4$ — no separate code path needed.

---

## 6. The Four Gates: When the Pass Fires

K_TILE selection is one of four checks. All four must pass; otherwise the pass leaves the dot alone, byte-for-byte unchanged.

<div class="post-step-cards">
  <div class="post-step-card">
    <div class="post-step-num">1</div>
    <div class="post-step-body">
      <strong>DPAS exclusion</strong>
      <p>If the module has the ttig.support_subgroup_matrix_multiply_accumulate attribute, exit. DPAS hardware doesn't have the FMA-unroll cliff, so staging is pure overhead. See <a href="/dpas/">the DPAS post</a> for why DPAS sidesteps the GRF problem.</p>
    </div>
  </div>
  <div class="post-step-card">
    <div class="post-step-num">2</div>
    <div class="post-step-body">
      <strong>Per-thread pressure threshold</strong>
      <p>estimatePerThreadBytes(dotOp) &gt; 4096. We only stage when projected register footprint exceeds the GRF. Below threshold, the LLVM unroller produces fine schedules and SLM staging is a regression.</p>
    </div>
  </div>
  <div class="post-step-card">
    <div class="post-step-num">3</div>
    <div class="post-step-body">
      <strong>SLM fit</strong>
      <p>slmStagingBytes ≤ 56 KB. ARL-S iGPU caps workgroup SLM at 64 KB; we leave 8 KB headroom for shared metadata and downstream allocations. If staging won't fit, skip the dot.</p>
    </div>
  </div>
  <div class="post-step-card">
    <div class="post-step-num">4</div>
    <div class="post-step-body">
      <strong>K_TILE feasibility</strong>
      <p>selectKTile() must return a power-of-two ≥ 16 that divides K. If no such tile exists, skip.</p>
    </div>
  </div>
</div>

<div class="post-callout post-callout-info">
  <span class="post-callout-icon">🛡️</span>
  <div class="post-callout-content">
    <span class="post-callout-title">Why so conservative?</span>
    <p>Every gate is "is the pass <em>definitely</em> a win?" If our staging hurts a workload that didn't need help, we can't reclaim the time. So when in doubt, the pass produces byte-identical IR and exits — and that's empirically verified in §9.</p>
  </div>
</div>

---

## 7. Where the Pass Sits in the Pipeline

```
add_accelerate_matmul →
add_stage_large_fma_dots_via_slm →   ← new
add_materialize_block_pointer →
add_pipeline (software pipelining)
```

Two non-obvious choices about position:

**Why immediately after AccelerateMatmul.** AccelerateMatmul is what inserts the ttg.convert_layout ops between the operand producers and the dot. That convert_layout is exactly the boundary the new pass anchors on (step 1 of the algorithm). Run the pass earlier, before AccelerateMatmul, and there's no boundary to find.

**Why before pipelining and materialize_block_pointer.** Software pipelining and materialize_block_pointer collapse K into one CTA-stride load — by that point, the K-loop structure we need to slice has already been folded away. Run the pass after pipelining and it's a no-op.

The window is narrow on purpose: post-AccelerateMatmul, pre-pipelining.

---

## 8. Five Differences from the Original Sketch

The pass is a redesign of a closed PR #7276. The architectural sketch [in the review](https://github.com/intel/intel-xpu-backend-for-triton/pull/7276#issuecomment-4774793921) read:

> We can simply convert the ttg.convert_layout to the ops pair ttg.local_alloc and ttg.local_load. And the return type of the ttg.local_alloc is addressable by the loop var index.

Implementing that sketch literally would not have worked. Five places where the implementation diverges:

1. **SLM reuse via findStagedSmem.** A literal read allocates fresh SLM at every convert_layout. Under software pipelining you'd allocate twice for the same operand and overshoot the SLM-fit gate. The reuse path makes the pass idempotent under pipelining (Test 8 in the lit suite locks this in).

2. **Walking past promotion casts.** AccelerateMatmul's decomposeMixedModeDotOp inserts tt.fp_to_fp between the convert_layout and the dot. Staging at the post-promotion (larger) value would trip the SLM-fit gate. Staging pre-promotion and replaying the cast per partial-K load keeps it bounded.

3. **Adaptive K_TILE.** A hardcoded {16, 32, 64} table works for one shape and fails for the next. The byte-budget derivation in §5 generalizes to any element-size combination including f32-ieee.

4. **Explicit SLM-fit gate.** Letting add_allocate_shared_memory fail later would generate confusing error messages and waste compile time. The 56 KB cap is checked here, with a no-op fallback that logs *dot SLM-staging cost N B exceeds cap 57344 B, skipping*.

5. **Real-kernel-shape lit coverage.** Test 1 in the lit suite is the loads-inside-outer-scf.for case the closed PR's traceToLoad couldn't reach. That kernel shape is what triggered the regression in production; without it, the test suite would pass while the real kernel failed.

---

## 9. ARL-S iGPU Validation

**Hardware.** Intel(R) Graphics, arch=arl_s, no DPAS, **64 KB SLM/workgroup**, **256 KB PTSS/thread**. (Forward-looking note: Xe2 iGPUs — Lunar Lake and ARL-H — raise SLM/workgroup to 128 KB[^xe2-slm] and add DPAS, so this pass is a no-op there.)

**Method.** 24-case sweep, baseline = stock `main`, optimized = this PR. Same kernel, inputs, launch params. Mean of 5 trials × 50 iters; std ≤ 7% for all reported cases.

### 9.1 Group A — Cases that previously didn't compile

These are the cases the pass exists for. Baseline produces no SPIR-V binary at all.

<table class="post-bench">
  <thead>
    <tr><th>Shape</th><th>Tile</th><th>grid</th><th>Baseline</th><th>Optimized</th><th>TFLOPS</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>64×128×128</strong></td><td>64×128×128</td><td>1×1</td><td class="pass-fail">PTSS overflow (345 KB &gt; 256 KB)</td><td class="pass-ok">0.891 ms</td><td>0.002</td></tr>
    <tr><td>64×128×256</td><td>64×128×128</td><td>1×1</td><td class="pass-fail">PTSS overflow</td><td class="pass-ok">1.372 ms</td><td>0.003</td></tr>
    <tr><td>64×128×512</td><td>64×128×128</td><td>1×1</td><td class="pass-fail">PTSS overflow</td><td class="pass-ok">2.479 ms</td><td>0.003</td></tr>
    <tr><td>64×128×1024</td><td>64×128×128</td><td>1×1</td><td class="pass-fail">PTSS overflow</td><td class="pass-ok">4.800 ms</td><td>0.003</td></tr>
    <tr><td>128×128×128</td><td>64×128×128</td><td>2×1</td><td class="pass-fail">PTSS overflow</td><td class="pass-ok">0.894 ms</td><td>0.005</td></tr>
    <tr><td>128×128×512</td><td>64×128×128</td><td>2×1</td><td class="pass-fail">PTSS overflow</td><td class="pass-ok">2.633 ms</td><td>0.006</td></tr>
    <tr><td>128×256×256</td><td>64×128×128</td><td>2×2</td><td class="pass-fail">PTSS overflow</td><td class="pass-ok">1.532 ms</td><td>0.011</td></tr>
    <tr><td>128×256×512</td><td>64×128×128</td><td>2×2</td><td class="pass-fail">PTSS overflow</td><td class="pass-ok">2.587 ms</td><td>0.013</td></tr>
    <tr><td>256×256×512</td><td>64×128×128</td><td>4×2</td><td class="pass-fail">PTSS overflow</td><td class="pass-ok">4.275 ms</td><td>0.016</td></tr>
  </tbody>
</table>

**9 / 9 regression cases recovered.** Every case that previously failed at IGC now compiles, runs, and produces correct results.

### 9.2 Group B — Both build, the pass produces a faster schedule

The gate fires; baseline still compiles but the schedule is bad. Optimized version is straightforwardly faster.

<table class="post-bench">
  <thead>
    <tr><th>Shape</th><th>Tile</th><th>grid</th><th>Baseline ms</th><th>Optimized ms</th><th>Speedup</th></tr>
  </thead>
  <tbody>
    <tr><td>64×64×128</td><td>64×64×128</td><td>1×1</td><td>0.999</td><td>0.769</td><td class="speedup">1.30×</td></tr>
    <tr><td>64×64×256</td><td>64×64×128</td><td>1×1</td><td>1.797</td><td>1.124</td><td class="speedup">1.60×</td></tr>
    <tr><td>64×64×512</td><td>64×64×128</td><td>1×1</td><td>3.399</td><td>1.918</td><td class="speedup">1.77×</td></tr>
    <tr><td>128×128×128</td><td>64×64×128</td><td>2×2</td><td>1.161</td><td>0.611</td><td class="speedup">1.90×</td></tr>
    <tr><td>128×128×256</td><td>64×64×128</td><td>2×2</td><td>2.129</td><td>1.146</td><td class="speedup">1.86×</td></tr>
    <tr><td>128×128×512</td><td>64×64×128</td><td>2×2</td><td>4.027</td><td>2.017</td><td class="speedup">2.00×</td></tr>
    <tr><td>64×128×256</td><td>64×128×64</td><td>1×1</td><td>3.040</td><td>1.373</td><td class="speedup">2.21×</td></tr>
    <tr><td>64×128×512</td><td>64×128×64</td><td>1×1</td><td>5.854</td><td>2.158</td><td class="speedup">2.71×</td></tr>
  </tbody>
</table>

**Geomean speedup ≈ 1.86×.** Speedup grows with K — larger K means more wasted FMA unrolling avoided. The 2.71× outlier is the K=512 case where baseline's spilled schedule is doing the most damage.

### 9.3 Group C — Should be no-op, was apparently no-op

The 24-case sweep flagged 6–17% regressions on f32-ieee shapes that should have been no-op. Two checks rule out a real regression:

1. **IR-equivalence proof.** Running `triton-opt --tritonintelgpu-stage-large-fma-dots-via-slm` on the f32 64×64×128 case produces **byte-identical** output. The pass logs *dot SLM-staging cost 65536 B exceeds cap 57344 B, skipping* and exits without touching IR. Identical IR → identical SPIR-V binary.
2. **Larger sample reverses the apparent regression.** 10 trials × 100 iters (vs the wider sweep's 3 × 30) flips the f32 cases to 12–15% **faster** optimized — well within sample noise of "no change." The original Group C numbers were thermal/scheduling artifacts from running f32 cases late in a sequential sweep.

When the gate skips, IR is byte-for-byte unchanged. There is no third bucket where the pass legitimately makes things worse.

### 9.4 Correctness

All 23 successful Optimized runs report max |result − torch.matmul| < 1e-1 (f16 GEMM tolerance); most report exact match (0).

---

## 10. Prior Art and the Architectural Novelty

GEMM kernels staging operands in shared memory and tiling K is not a new idea. The canonical playbooks:

- **CUTLASS** structures GEMM as a hierarchical loop nest where the *outer* cta_k mainloop is **not** unrolled (one iteration = one pipeline stage), while *inner* warp_k and per-thread loops are fully unrolled. SMEM is double-buffered (CUTLASS 1.x/2.x) or N-stage circularly buffered via cutlass::PipelineAsync (3.x), with producer warps loading the next K-tile while consumer warps multiply the current one.[^cutlass]
- **ROCm Composable Kernel** (GemmPipelineAGmemBGmemCRegV1) stages both A and B in LDS with PrefetchStages = 1, PrefillStages = 1, and walks K via a runtime while-loop driven by num_loop advancing the DRAM tile window by kKPerBlock per iteration — no K_TILE constant; the K quantum is the block-tile shape.[^ck]
- **Upstream Triton's pipeliner** allocates a multi-buffered LocalAllocOp whose buffer count equals the producer-to-consumer stage distance (getDefUseStageDiff + 1 for WGMMA), then carves per-iteration slices via createSingleBufferView indexed by modulo-incremented insertIdx/extractIdx — circular cross-iteration multi-buffering, sized by schedule stage distances rather than per-dot K-tile.[^triton-pipe]

All three share the same shape: **circular cross-iteration multi-buffering**. SMEM is sliced across iterations of an outer K loop — stage N's load overlaps stage N-1's compute.

This pass does something different. Inside *one* iteration:

<div class="post-diagram">
<svg viewBox="0 0 720 280" xmlns="http://www.w3.org/2000/svg" aria-label="Cross-iteration circular vs intra-iteration sub-slice">
  <text x="180" y="25" class="mem-title" text-anchor="middle">CUTLASS / CK / upstream Triton</text>
  <text x="180" y="45" class="mem-mono" text-anchor="middle">circular multi-buffer across iterations</text>

  <rect x="40" y="65" width="60" height="50" fill="#fde4d0" stroke="#d94f30" stroke-width="2"/>
  <rect x="105" y="65" width="60" height="50" fill="#e8f5ee" stroke="#2d8b55" stroke-width="2"/>
  <rect x="170" y="65" width="60" height="50" fill="#e1f0fa" stroke="#0071c5" stroke-width="2"/>
  <rect x="235" y="65" width="60" height="50" fill="#f0e6f6" stroke="#7b6daa" stroke-width="2"/>
  <text x="70" y="95" class="mem-mono" text-anchor="middle">Iter 0</text>
  <text x="135" y="95" class="mem-mono" text-anchor="middle">Iter 1</text>
  <text x="200" y="95" class="mem-mono" text-anchor="middle">Iter 2</text>
  <text x="265" y="95" class="mem-mono" text-anchor="middle">Iter 3</text>
  <text x="70" y="135" class="mem-mono" text-anchor="middle">smem[0]</text>
  <text x="135" y="135" class="mem-mono" text-anchor="middle">smem[1]</text>
  <text x="200" y="135" class="mem-mono" text-anchor="middle">smem[0]</text>
  <text x="265" y="135" class="mem-mono" text-anchor="middle">smem[1]</text>

  <text x="540" y="25" class="mem-title" text-anchor="middle">This pass</text>
  <text x="540" y="45" class="mem-mono" text-anchor="middle">single SMEM, sub-slice within one iteration</text>

  <rect x="380" y="65" width="320" height="50" fill="#d6eaf8" stroke="#0071c5" stroke-width="2"/>
  <text x="540" y="95" class="mem-mono" text-anchor="middle" fill="#0071c5">one allocation: full A and B in SLM</text>
  <line x1="460" y1="65" x2="460" y2="115" stroke="#0071c5" stroke-dasharray="3,3"/>
  <line x1="540" y1="65" x2="540" y2="115" stroke="#0071c5" stroke-dasharray="3,3"/>
  <line x1="620" y1="65" x2="620" y2="115" stroke="#0071c5" stroke-dasharray="3,3"/>
  <text x="420" y="135" class="mem-mono" text-anchor="middle">K-tile 0</text>
  <text x="500" y="135" class="mem-mono" text-anchor="middle">K-tile 1</text>
  <text x="580" y="135" class="mem-mono" text-anchor="middle">K-tile 2</text>
  <text x="660" y="135" class="mem-mono" text-anchor="middle">K-tile 3</text>

  <text x="360" y="200" class="mem-text" text-anchor="middle">Trade-off: this pass gives up cross-iteration prefetch overlap</text>
  <text x="360" y="222" class="mem-text" text-anchor="middle">in exchange for a hard cap on per-thread register pressure.</text>
  <text x="360" y="252" class="mem-text" text-anchor="middle" font-style="italic" fill="#6b6560">The two patterns compose — software pipelining still buys the overlap.</text>
</svg>
<div class="post-diagram-caption">Same SMEM tool, used differently: across iterations vs within one iteration.</div>
</div>

The architecturally novel piece is **a single SMEM allocation per operand for the full K-extent, sub-sliced per K-tile via memdesc_subslice within one iteration**. Software pipelining still buys cross-iteration overlap; the SMEM-reuse path in §3 step 5 keeps K-tile sub-slicing idempotent under it.

---

## 11. Composability and Risk

**No-op on DPAS hardware.** PVC, BMG, and ARL-H Xe2 all carry the ttig.support_subgroup_matrix_multiply_accumulate module attribute. The pass exits at gate 1. There is no chance of accidentally regressing systolic-unit kernels.

**Composes with software pipelining.** Test 8 in the lit suite covers the case where the pipelining pass has already inserted a local_load. findStagedSmem reuses the existing memdesc instead of allocating fresh SLM. Without this, double-staging would trip the 56 KB SLM-fit gate.

**f32 cases verified equivalent under IR diff, not just timing.** The triton-opt byte-identity check is stronger than wall-clock benchmarking. It eliminates the "did I actually no-op?" question from the validation surface.

**No new runtime dependencies.** Pure TTGIR rewrite. No host-side changes, no kernel ABI changes.

**Files touched** (stats: +989 / -0):

<table class="post-bench">
  <thead><tr><th>File</th><th>Purpose</th></tr></thead>
  <tbody>
    <tr><td>third_party/intel/lib/TritonIntelGPUTransforms/StageLargeFMADotsViaSLM.cpp</td><td>Pass implementation</td></tr>
    <tr><td>third_party/intel/include/Dialect/TritonIntelGPU/Transforms/Passes.td</td><td>Pass declaration</td></tr>
    <tr><td>third_party/intel/lib/TritonIntelGPUTransforms/CMakeLists.txt</td><td>Build registration</td></tr>
    <tr><td>third_party/intel/triton_xpu.cc</td><td>Python binding</td></tr>
    <tr><td>third_party/intel/backend/compiler.py</td><td>Pipeline insertion</td></tr>
    <tr><td>test/TritonIntelGPU/stage-large-fma-dots-via-slm.mlir</td><td>8 lit cases</td></tr>
    <tr><td>python/test/unit/intel/test_stage_large_fma_dots_via_slm.py</td><td>E2E pytest</td></tr>
  </tbody>
</table>

---

## References

- PR: [intel/intel-xpu-backend-for-triton#7291](https://github.com/intel/intel-xpu-backend-for-triton/pull/7291) — *Add StageLargeFMADotsViaSLM: SLM-staged K-tiling on non-DPAS hw*
- Closed sketch PR: [#7276](https://github.com/intel/intel-xpu-backend-for-triton/pull/7276) — original architectural review thread
- Background: [Triton Kernels from First Principles](/triton-kernel-first-principles/) · [Intel GPU Scheduling](/gpu-kernel-scheduling/) · [DPAS](/dpas/)

[^xe-grf]: Intel oneAPI Optimization Guide for GPU 2025-2, [*Intel Xe GPU Architecture*](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html) and [*Small Register Mode vs Large Register Mode*](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/small-register-mode-vs-large-register-mode.html).

[^neo-ptss]: [compute-runtime/shared/source/helpers/gfx_core_helper_xehp_and_later.inl](https://github.com/intel/compute-runtime/blob/master/shared/source/helpers/gfx_core_helper_xehp_and_later.inl) lines 133-135; enforced at [kernel_helpers.cpp](https://github.com/intel/compute-runtime/blob/master/shared/source/helpers/kernel_helpers.cpp) lines 83-86. Xe3p+ has an opt-in isAvailableExtendedScratch path that can lift the cap to 16 MB, but no shipping product flips that default today.

[^igc-unroll]: IGC [GenTTI.cpp::getUnrollingPreferences](https://github.com/intel/intel-graphics-compiler/blob/master/IGC/Compiler/GenTTI.cpp) factors register pressure into unrolling via SetRegisterPressureThresholdForLoopUnroll, scaled by the platform's GRF count.

[^triton-pipe]: Upstream [Pipeliner/LowerLoops.cpp](https://github.com/triton-lang/triton/blob/main/lib/Dialect/TritonGPU/Transforms/Pipeliner/LowerLoops.cpp) sizes LocalAllocOp by getDefUseStageDiff and indexes them circularly via createSingleBufferView. There is no K_TILE constant or full-K-unroll logic anywhere in the file.

[^promote]: Upstream [AccelerateMatmul.cpp](https://github.com/triton-lang/triton/blob/main/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp) lines 933-985; identical logic exists in the Intel fork's parallel [third_party/intel/lib/TritonIntelGPUTransforms/AccelerateMatmul.cpp](https://github.com/intel/intel-xpu-backend-for-triton/blob/main/third_party/intel/lib/TritonIntelGPUTransforms/AccelerateMatmul.cpp).

[^xe2-slm]: Intel oneAPI Optimization Guide for GPU 2025-2, [*Intel Xe GPU Architecture*](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html): Max SLM per Work-Group is 64 KB on Xe-LPG/Xe-HPG/Xe-LP and 128 KB on Xe2-HPG/Xe2-LPG/Xe-HPC.

[^cutlass]: CUTLASS [efficient_gemm.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md): the GEMM mainloop iterates cta_k by CtaTileK with no unrolling. Pipeline: [pipeline.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/pipeline.md). Background: NVIDIA Developer Blog, [*CUTLASS: Fast Linear Algebra in CUDA C++*](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/) (2017): "All loops except the outermost main loop have constant iteration counts and can be fully unrolled by the compiler."

[^ck]: ROCm Composable Kernel [gemm_pipeline_agmem_bgmem_creg_v1.hpp](https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1.hpp).
