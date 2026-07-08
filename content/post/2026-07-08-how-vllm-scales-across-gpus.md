---
author: Gurwinder
categories:
- AI
- GPU
date: '2026-07-08T00:00:00Z'
slug: 'how-vllm-scales-across-gpus'
featured: false
draft: false
image: assets/images/cudamapping.png
imageAlt: 'GPU compute-and-memory mapping, illustrating the multi-GPU execution vLLM coordinates across tensor, pipeline, expert, and data parallelism'
description: 'How vLLM V1 scales inference across GPUs — tensor/pipeline/expert/data parallelism, the custom all-reduce fast path, Wide EP, and disaggregated prefill — traced through the merged PRs that made each faster.'
title: 'How vLLM Scales Across GPUs: Parallelism, Collectives, and the PRs Behind Them'
---

<style>
.post-diagram { background: #fff; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 12px rgba(44,42,40,0.08); margin: 2rem 0; overflow-x: auto; border: 1px solid #eeebe5; }
.post-diagram svg { display: block; margin: 0 auto; max-width: 100%; height: auto; }
.post-diagram-caption { text-align: center; font-size: 0.875rem; color: #6b6560; font-style: italic; margin-top: 1rem; }
.mem-text { font-family: 'DM Sans', system-ui, sans-serif; font-size: 13px; fill: #2c2a28; }
.mem-mono { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px; fill: #6b6560; }
.mem-title { font-family: 'Bricolage Grotesque', Georgia, serif; font-size: 14px; font-weight: 700; fill: #2c2a28; }
.post-callout { display: flex; gap: 1rem; padding: 1.25rem; border-radius: 12px; margin: 1.5rem 0; border-left: 4px solid; }
.post-callout-info { background: #e4f2f7; border-color: #2a7b9b; }
.post-callout-warn { background: #fdf3e8; border-color: #d98a2b; }
.post-callout-accent { background: #ece7f6; border-color: #764ba2; }
.post-callout-icon { font-size: 1.25rem; flex-shrink: 0; }
.post-callout-title { display: block; font-weight: 700; margin-bottom: 0.25rem; color: #2c2a28; }
.post-callout-content p { font-size: 0.95rem; margin: 0; color: #2c2a28; line-height: 1.55; }
</style>

A large language model that does not fit on one GPU is not, primarily, a compute problem. The matrix multiplies are embarrassingly parallel; you can always buy more FLOPs. What you cannot buy your way out of is the *communication* the moment you split a model across devices. Every tensor-parallel layer ends in an all-reduce. Every pipeline stage hands its activations to the next over a link. Every Mixture-of-Experts layer shuffles tokens to wherever their experts live and shuffles the results back. The GPUs spend a startling fraction of their wall-clock time waiting on each other.

vLLM's multi-GPU story — especially since the V1 engine landed in early 2025 — is the story of attacking that communication tax from four different directions at once, and then chipping away at the residual overheads (kernel launches, recompilation, redundant syncs) that show up once the collectives are fast. This post traces how each of the four parallelism dimensions actually works, and grounds every mechanism in the merged `vllm-project/vllm` pull request that implemented or optimized it — with the measured numbers, and with honest notes on which numbers to trust.

> Note: this is a systems-level walk through distributed inference. If you want the single-GPU foundations first — how a kernel is scheduled, how PyTorch drives it — [Intel GPU Scheduling](/gpu-kernel-scheduling/) and [How PyTorch Sees Your Triton Kernel](/triton-custom-kernels-pytorch/) are good primers. For the attention-kernel background that MoE and disaggregation both lean on, see [PyTorch SDPA Kernels](/pytorch-sdpa-kernel/).

Every performance figure below was checked against the actual PR page — merge status, benchmark table, and hardware — before it went in. Where a number rests on a weak baseline, an unstated GPU, or a best-case workload, it carries a caveat. The full catalog of 38 verified PRs is tabulated at the end.

The build is linear: first the process model that everything else sits inside, then the four dimensions (tensor, pipeline, expert+data, distributed/KV), then the two cross-cutting multipliers (CUDA graphs and `torch.compile`), then a decision guide.

---

## 1. The Playing Field: A Communication Budget

Before any parallelism, fix the picture of *where* bytes travel and *how fast*. Four tiers matter for distributed inference, and they span five orders of magnitude in bandwidth.

<div class="post-diagram">
<svg viewBox="0 0 720 360" xmlns="http://www.w3.org/2000/svg" aria-label="Interconnect hierarchy for multi-GPU inference">
  <rect x="20" y="280" width="680" height="64" rx="8" fill="#e8e0d0" stroke="#6b6560" stroke-width="2"/>
  <text x="360" y="306" class="mem-title" text-anchor="middle">Cross-node network (Ethernet / InfiniBand / RDMA)</text>
  <text x="360" y="328" class="mem-mono" text-anchor="middle">~25–400 Gb/s · µs latency · pipeline &amp; disaggregation live here</text>

  <rect x="80" y="196" width="560" height="64" rx="8" fill="#f0e6f6" stroke="#7b6daa" stroke-width="2"/>
  <text x="360" y="222" class="mem-title" text-anchor="middle">PCIe (GPU ↔ GPU within a node, no NVLink)</text>
  <text x="360" y="244" class="mem-mono" text-anchor="middle">~32–64 GB/s · custom all-reduce disabled beyond 2 GPUs</text>

  <rect x="180" y="112" width="360" height="64" rx="8" fill="#d6eaf8" stroke="#0071c5" stroke-width="2"/>
  <text x="360" y="138" class="mem-title" text-anchor="middle">NVLink / XGMI (intra-node fabric)</text>
  <text x="360" y="160" class="mem-mono" text-anchor="middle">~300–900 GB/s · the tensor-parallel comfort zone</text>

  <rect x="260" y="28" width="200" height="64" rx="8" fill="#fde4d0" stroke="#d94f30" stroke-width="2"/>
  <text x="360" y="54" class="mem-title" text-anchor="middle">HBM (on-package)</text>
  <text x="360" y="76" class="mem-mono" text-anchor="middle">~2–8 TB/s · 1 GPU</text>

  <path d="M 360 92 L 360 112" stroke="#6b6560" stroke-width="2" fill="none" marker-end="url(#arr1)"/>
  <path d="M 360 176 L 360 196" stroke="#6b6560" stroke-width="2" fill="none" marker-end="url(#arr1)"/>
  <path d="M 360 260 L 360 280" stroke="#6b6560" stroke-width="2" fill="none" marker-end="url(#arr1)"/>
  <defs>
    <marker id="arr1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6b6560"/>
    </marker>
  </defs>
</svg>
<div class="post-diagram-caption">Each step down is roughly an order of magnitude slower. The whole art of multi-GPU serving is keeping the hottest, most frequent communication on the fastest tier — and picking a parallelism strategy that matches the fabric you actually have.</div>
</div>

The single most important consequence, which recurs in every section below: **the right parallelism strategy is a function of your interconnect.** Tensor parallelism hammers the fabric every layer, so it belongs on NVLink. Pipeline parallelism communicates once per stage boundary, so it tolerates PCIe or even the network. Expert parallelism trades an all-reduce for a sparser all-to-all. Disaggregation moves the KV cache over the network exactly once per request. Keep that mapping in mind; the PRs are all, in one way or another, about making a particular tier of this diagram hurt less.

---

## 2. The V1 Process Model

vLLM V1 (the re-architected core, [announced January 2025](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)) runs multiple GPUs as a small constellation of OS processes. Getting this mental model right makes every later section legible, because the parallelism flags are really just instructions for how many of each process to spawn and how to wire them.

The rules, straight from vLLM's architecture documentation:

- **One worker process per accelerator.** The total number of GPU workers is $N = \text{TP} \times \text{PP} \times \text{DP}$ — tensor-parallel size times pipeline-parallel size times data-parallel size. A `TP=2, PP=2` launch is 4 workers.
- **One engine-core process per data-parallel rank.** The engine core runs the scheduler and KV-cache manager; it owns `TP × PP` workers beneath it.
- **One DP Coordinator process — but only when data parallelism is enabled.** It exists to keep data-parallel ranks in lockstep, which turns out to be essential for MoE models (§4).

<div class="post-callout post-callout-info">
  <div class="post-callout-icon">🧩</div>
  <div class="post-callout-content">
    <span class="post-callout-title">Executor backends</span>
    <p>How those processes are launched depends on scope. vLLM defaults to <strong>native Python multiprocessing for single-node</strong> and <strong>Ray for multi-node</strong>, overridable with <code>--distributed-executor-backend {mp,ray}</code>. Single box: multiprocessing. Spanning machines: Ray.</p>
  </div>
</div>

A defining V1 design choice for multi-GPU efficiency: workers are *stateful and symmetric*. Each worker caches request state locally, and the scheduler ships only the per-step diff rather than re-broadcasting the whole batch every iteration. That keeps the scheduler→worker chatter — which is pure overhead that grows with GPU count — small. It is also the substrate that makes the CUDA-graph and `torch.compile` work in §7 pay off: less Python on the critical path per step means the captured graphs dominate.

---

## 3. Tensor Parallelism: Sharding a Layer, Paying an All-Reduce

Tensor parallelism (TP), set with `--tensor-parallel-size`, implements Megatron-LM's algorithm: each weight matrix is split across GPUs, every device computes a partial result, and an **all-reduce** sums the partials back to a coherent activation. It is the default for splitting a model within one node — `TP=4` on a 4-GPU box — and composes with pipeline parallelism across nodes (`TP=8, PP=2` for two 8-GPU nodes).

The cost is brutal in its regularity: **two all-reduces per transformer layer**, one after attention and one after the MLP, every single forward step. On NVLink this is tolerable. On anything slower it dominates. So the TP optimization story is really two stories — *make the all-reduce faster*, and *make it overlap with compute so it's partly free*.

### 3.1 The custom all-reduce fast path — and its gates

vLLM ships a hand-written NVLink-optimized all-reduce kernel that beats NCCL for the small-to-medium tensors typical of inference. But it is *heavily* gated, and reading the gates (straight from `custom_all_reduce.py` on `main`) tells you exactly when you get the fast path and when you silently fall back:

- Supported world sizes are **exactly `[2, 4, 6, 8]`**. A single GPU skips it entirely.
- For **more than 2 GPUs, full NVLink connectivity is required** — on more than two PCIe-only GPUs it disables itself with a warning.
- It only fires when the input is **smaller than `max_size` (default 8 MB = `8192 × 1024` bytes)** and the byte size is a **multiple of 16**.

That last condition is a genuine dispatch decision made per call. In shorthand, the kernel runs only when

$$
\text{world\_size} \in \{2,4,6,8\} \;\wedge\; (\text{world\_size}=2 \;\vee\; \text{fully\_connected}) \;\wedge\; \big(\text{inp\_bytes} \bmod 16 = 0\big) \;\wedge\; \big(\text{inp\_bytes} < 8\,\text{MB}\big).
$$

<div class="post-callout post-callout-warn">
  <div class="post-callout-icon">⚠️</div>
  <div class="post-callout-content">
    <span class="post-callout-title">The PCIe cliff</span>
    <p>On a box with more than two PCIe-only GPUs, the custom all-reduce turns itself off and you fall back to NCCL. This is a concrete reason TP underperforms on such machines — and a direct argument for pipeline parallelism instead (§5). The interconnect isn't a footnote; it's the gate condition in the source.</p>
  </div>
</div>

### 3.2 Swapping in faster collectives

Beyond the built-in kernel, vLLM added pluggable high-performance all-reduce backends:

- **PyTorch symmetric memory** ([#20759](https://github.com/vllm-project/vllm/pull/20759), later default-on via [#24111](https://github.com/vllm-project/vllm/pull/24111)) picks a strategy by world size — two-shot for 2/4 GPUs, multimem for 6/8 — and targets mid-size TP inputs. Measured on Llama-3.1-70B: **up to ~7–10% lower TTFT and 5–7% lower TPOT at TP=8** on B200, and up to ~10% TTFT on H100.
- **NCCL symmetric memory** ([#24532](https://github.com/vllm-project/vllm/pull/24532)) registers buffers with NCCL so it dispatches its symmetric kernels (`AllReduce [Symmetric]`), which win at large messages — micro-benchmarks show **up to ~2.0× at 8 MB** on TP=8.

> Note: the NCCL-symmetric end-to-end gain on DeepSeek-R1 (TP=8) was a more modest 4.50 → 4.66 req/s (~3.5%), and the PR doesn't state the exact GPU. The 2× is a micro-benchmark at a favorable message size; the request-level number is the one to quote for real serving.

On AMD, the equivalent play is **quantized** all-reduce: QuickReduce ([#19744](https://github.com/vllm-project/vllm/pull/19744)) symmetrically quantizes activations to INT8/6/4 before reducing, cutting the bytes on the wire. On MI300 with Llama-3.1-70B it reports **TP=4 TTFT ~2.2× / TPOT ~2×** with INT4. That is a large win, though partly against a weak RCCL baseline above 16 MB — expected for a technique whose whole point is sending fewer bytes.

### 3.3 Overlapping the all-reduce with compute

The deeper idea is to stop treating the all-reduce as a blocking barrier. Two compilation passes make this happen:

- **Sequence parallelism** ([#16155](https://github.com/vllm-project/vllm/pull/16155)) rewrites the FX graph so the post-attention/MLP all-reduce becomes a *reduce-scatter + all-gather* pair, with the RMSNorm in between running on the smaller, sharded tensor. This is the structural prerequisite for overlap.
- **Async tensor parallelism** ([#17882](https://github.com/vllm-project/vllm/pull/17882)) then overlaps the decomposed communication with the surrounding matmul. On 4×H200 Llama-3.1-70B it moved a step from 0.590s → 0.526s; on 4×A100 8B, ~10%.

<div class="post-callout post-callout-accent">
  <div class="post-callout-icon">📐</div>
  <div class="post-callout-content">
    <span class="post-callout-title">Overlap is not free at every scale</span>
    <p>The async-TP PR is refreshingly honest: on <strong>2×A100 with an 8B model it showed little to no gain</strong>, sometimes a hair slower. Overlap pays off as GPU count and model size grow and the collective gets big enough to hide real work behind. A follow-up (<a href="https://github.com/vllm-project/vllm/pull/28672">#28672</a>) gates sequence parallelism by token count for exactly this reason — SP wins at large token counts, fused all-reduce+RMSNorm+quant wins at small ones.</p>
  </div>
</div>

Finally, **collective fusion** ([#21069](https://github.com/vllm-project/vllm/pull/21069)) folds the all-reduce, the following RMSNorm, and the quantization into a *single* FlashInfer op, cutting kernel launches and memory traffic on the TP critical path. On B200 TP=2 it shows ~7–8% TPOT for FP8 — but read the baseline carefully: the gain is against *custom ops*, and against default `torch.compile`d ops FP8 TTFT can actually regress up to 8%. NVFP4 fares better (5% TPOT/TTFT vs compiled).

---

## 4. Expert + Data Parallelism: "Wide EP" for MoE

Mixture-of-Experts models break the TP assumption. Only a handful of experts fire per token, so sharding every expert's weights across all GPUs (plain TP) wastes memory and bandwidth. vLLM's answer, sometimes called **Wide EP**, pairs two ideas:

1. **Data-parallel attention.** Each DP rank owns its *own* attention computation, KV cache, and request stream. Attention weights are replicated across DP ranks (when `TP=1`) or TP-sharded within each DP group (when `TP>1`).
2. **Expert-parallel MoE.** The expert layers are sharded across an expert-parallel group whose size is computed automatically as

$$
\text{EP\_SIZE} = \text{TP\_SIZE} \times \text{DP\_SIZE}.
$$

   Turn it on with `--enable-expert-parallel`. Each expert lives on some GPU; tokens are routed to wherever their top-k experts are.

The routing pattern *changes character* mid-layer. Attention is **request-based** (each rank handles its own sequences); the MoE block is **expert-based** (tokens scatter to expert owners and gather back). Between the two sits a **dispatch/combine** all-to-all — a token permutation across GPUs — and every rank must keep its forward pass aligned with the others.

<div class="post-diagram">
<svg viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg" aria-label="Wide EP: DP attention, dispatch, expert compute, combine">
  <rect x="20" y="30" width="150" height="60" rx="8" fill="#d6eaf8" stroke="#0071c5" stroke-width="2"/>
  <text x="95" y="55" class="mem-title" text-anchor="middle">DP attention</text>
  <text x="95" y="74" class="mem-mono" text-anchor="middle">per-rank KV</text>

  <rect x="230" y="30" width="120" height="60" rx="8" fill="#f0e6f6" stroke="#764ba2" stroke-width="2"/>
  <text x="290" y="48" class="mem-title" text-anchor="middle">Dispatch</text>
  <text x="290" y="68" class="mem-mono" text-anchor="middle">all-to-all</text>
  <text x="290" y="82" class="mem-mono" text-anchor="middle">(token scatter)</text>

  <rect x="410" y="30" width="120" height="60" rx="8" fill="#fde4d0" stroke="#d94f30" stroke-width="2"/>
  <text x="470" y="55" class="mem-title" text-anchor="middle">Experts</text>
  <text x="470" y="74" class="mem-mono" text-anchor="middle">sharded / EP</text>

  <rect x="590" y="30" width="110" height="60" rx="8" fill="#f0e6f6" stroke="#764ba2" stroke-width="2"/>
  <text x="645" y="48" class="mem-title" text-anchor="middle">Combine</text>
  <text x="645" y="68" class="mem-mono" text-anchor="middle">all-to-all</text>
  <text x="645" y="82" class="mem-mono" text-anchor="middle">(gather back)</text>

  <path d="M 170 60 L 230 60" stroke="#6b6560" stroke-width="2" marker-end="url(#arr2)"/>
  <path d="M 350 60 L 410 60" stroke="#6b6560" stroke-width="2" marker-end="url(#arr2)"/>
  <path d="M 530 60 L 590 60" stroke="#6b6560" stroke-width="2" marker-end="url(#arr2)"/>

  <rect x="20" y="150" width="680" height="110" rx="8" fill="#fdf9f3" stroke="#e5dfd6" stroke-width="1.5"/>
  <text x="360" y="176" class="mem-title" text-anchor="middle">The alignment tax</text>
  <text x="360" y="202" class="mem-text" text-anchor="middle">All DP ranks must step together. If any rank has a live request, idle ranks run empty</text>
  <text x="360" y="222" class="mem-text" text-anchor="middle">"dummy" forward passes so the collective still has a partner — coordinated by the DP Coordinator.</text>
  <text x="360" y="246" class="mem-mono" text-anchor="middle">EPLB rebalances hot experts on top of this to keep the load even.</text>
  <defs>
    <marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6b6560"/>
    </marker>
  </defs>
</svg>
<div class="post-diagram-caption">A token's journey through a Wide-EP layer: attend locally, scatter to expert owners, compute, gather results. The two all-to-alls are the new bottleneck that replaces TP's all-reduce.</div>
</div>

### 4.1 The alignment tax and load balancing

Because the dispatch/combine collective needs every rank to participate, **when any DP rank has work, the idle ranks must run empty dummy forward passes** — otherwise the all-to-all has no partner and deadlocks. The DP Coordinator process orchestrates this. It is a real efficiency cost of MoE serving, and it is why data parallelism, not raw replication, is the coordination unit here.

The second problem is *skew*: real routing sends far more tokens to some experts than others, so the GPUs holding hot experts become stragglers. The **Expert Parallel Load Balancer** (EPLB, `--enable-eplb`) collects routing statistics on every forward pass and periodically rebalances expert placement, including **replicating hot experts** across GPUs. It's tunable via `--eplb-config` (`window_size` default 1000, `step_interval` default 3000).

### 4.2 Making the dispatch/combine faster

The all-to-all is where the PRs pile up:

- **AllGather + ReduceScatter backend** ([#23964](https://github.com/vllm-project/vllm/pull/23964)) replaces the naive all-to-all with allgather-for-dispatch and reduce-scatter-for-combine. On 4-GPU DeepSeek-R1-FP4 (DP=4, EP): **~1.97× total token throughput**, mean TPOT 228.78 → 116.10 ms. This one is CUDA-graph compatible and the discussion trended toward making it a default.
- **MoRI-EP** ([#28664](https://github.com/vllm-project/vllm/pull/28664)), an AMD all-to-all backend that quantizes to FP8 before dispatch: on MI300X+CX7 DeepSeek-R1, EP=8 gave 1.33× and **EP=16 gave 2.68×** throughput. The gain grows with scale precisely because naive all-to-all degrades so badly at EP=16.
- **EPLB zero-copy transfers** ([#41633](https://github.com/vllm-project/vllm/pull/41633)) uses one-sided RDMA READ to move expert weights during rebalancing with no staging buffers — on DeepSeek-V3.2 EP=8, transfer time 0.9 → 0.7s and **staging memory 4.4 GiB → 0**.

<div class="post-callout post-callout-warn">
  <div class="post-callout-icon">⏱️</div>
  <div class="post-callout-content">
    <span class="post-callout-title">A moving target</span>
    <p>The EP kernel landscape churns fast. The <strong>PPLX backend was removed in early 2026</strong> (<a href="https://github.com/vllm-project/vllm/pull/33724">#33724</a>), and vLLM now carries several all-to-all backends (allgather_reducescatter, DeepEP, FlashInfer, NIXL, MoRI). Treat any specific "backend X beats Y" claim as version-stamped; verify against the release you actually run.</p>
  </div>
</div>

---

## 5. Pipeline Parallelism: Layers Across the Network

Pipeline parallelism (PP), `--pipeline-parallel-size`, splits the model along its layers and hands activations from one stage to the next over a point-to-point link. It **supports uneven splits**, and — crucially — it is the recommended choice when GPUs *lack* fast NVLINK (e.g. L40S), because it communicates far less than TP: one activation handoff per stage boundary instead of two all-reduces per layer. That maps it onto the PCIe/network tiers of the §1 diagram, where TP would stall.

The classic weakness of pipelining is the **bubble**: while stage 0 works on microbatch $n$, stages 1..k sit idle unless you keep them fed. vLLM's PP journey is a march against that idleness.

The original implementation ([#4412](https://github.com/vllm-project/vllm/pull/4412), 2024) established the shape: one scheduler and cache engine per stage, blocking send/recv between stages (no all-reduce needed), and the last stage's TP group does the sampling. It even concatenated residuals and hidden states into one message to dodge a PyTorch NCCL bug. It shipped with only a functional sanity check, not a speedup — so treat it as the foundation, not a benchmark.

The V1-era gains came from letting the scheduler run *ahead* of the pipeline:

- **Async scheduling + PP** ([#32618](https://github.com/vllm-project/vllm/pull/32618)) removed the restriction that blocked async scheduling under PP. The obstacle was that only the last stage produces sampled tokens, so it now broadcasts them from the last PP rank to all others *directly on the GPU* (via `torch.distributed.broadcast`, no CPU round-trip). On Qwen3-30B-A3B pp=4: **30.8% E2E throughput and 31.8% TPOT improvement** (5999 vs 4585 out-tok/s).
- **Bubble removal in ModelRunnerV2** ([#42187](https://github.com/vllm-project/vllm/pull/42187)) reorganized scheduling so decode and chunked prefill actually pipeline: decode tokens scheduled every `pp_size` steps, prefill chunks pipelined across consecutive steps, and the sampled-token broadcast moved onto a *separate CUDA stream and process group* so it overlaps the stage-to-stage P2P instead of serializing behind it. On a GB200 node at PP=4 it reports up to **3.17× total tok/s and 3.15× TTFT reduction**.

> Note: those PP multipliers are best-case shapes. The 3.17× for #42187 is the `128000/1` case (huge prefill, one output token) on a single GB200 node with prefix caching off; more balanced shapes land at 1.24×–2.28×. Likewise the chunked-PP stall fix ([#38726](https://github.com/vllm-project/vllm/pull/38726)) shows ~3.3× (1392 → 4578 tok/s at pp=4 tp=2) but with `output-len=1`, a near-pure-prefill workload that maximally flatters a prefill-side fix. Expect smaller gains once real decode is in the mix.

---

## 6. Distributed Serving and KV Transfer

The fourth dimension is orchestration across instances, and its headline technique is **disaggregated prefill/decode (P/D)**: run the compute-bound prefill phase and the memory-bound decode phase on *different* vLLM instances, connected by a KV-cache transfer.

The counterintuitive framing, stated plainly in vLLM's docs: **disaggregation does not improve raw throughput.** What it buys is *independent control* of the two phases — you can tune TTFT without disturbing inter-token latency, and reliably bound tail ITL — which improves **goodput under latency SLOs**. If your only metric is tokens/sec on an unconstrained batch, disaggregation is not your tool; if you have a p99 latency contract, it is.

All the disaggregation machinery lives under `vllm/distributed/kv_transfer`, built from three abstractions — **Connector, LookupBuffer, and Pipe** (a single-direction FIFO with `send_tensor`/`recv_tensor`). The connector is the pluggable part, and the ecosystem grew quickly:

- **KV Connector API V1** ([#15960](https://github.com/vllm-project/vllm/pull/15960)) is the foundation: per-process connectors split into a scheduler-side role (prefill picks which tokens' KV to send; decode reports what's already computed) and worker-side roles that store/load KV layer-by-layer alongside attention. It deliberately leaves xPyD orchestration to external infrastructure. Ships mechanism, not benchmarks.
- **NIXL integration** ([#17751](https://github.com/vllm-project/vllm/pull/17751)) is the direct high-performance transport: fully async send/recv from prefill to decode workers over NVIDIA's NIXL library (UCX/GDS backends), with a `MultiConnector` abstraction so several connectors coexist. Supports xPyD and homogeneous TP>1.
- **LMCache connector** ([#16625](https://github.com/vllm-project/vllm/pull/16625)) adds CPU KV offload, cross-instance KV pooling, and P/D over NIXL. Its benchmark — 2×H100, Llama-3.1-8B, 1P1D — claims **~40% higher tok/s and ~8× better tail inter-token latency**.
- **MooncakeStore XpYd** ([#12957](https://github.com/vllm-project/vllm/pull/12957)) offers a shared-object-store topology so any decode instance can fetch any prefill instance's KV via per-TP-rank key prefixes.

> Note: the LMCache ~40% / ~8× figures are against a baseline of *"two separate vLLM instances"* rather than a tuned co-located deployment — an unusual comparison — measured at a single request rate. The direction (disaggregation helps tail ITL) is sound and matches the DistServe research framing; the exact multipliers are baseline-dependent.

There's also a quieter but important correctness fix: **hybrid allocator + connector** ([#30166](https://github.com/vllm-project/vllm/pull/30166)) made vLLM's hybrid KV-cache manager allocate only in-window tokens on sliding-window layers, which unblocked combining long-context external KV stores (LMCache) with the allocator — previously the allocate-then-free pattern caused GPU memory pressure and cross-worker data contention. It's parity on throughput but enabled a 69,888-token retrieval that previously failed outright.

---

## 7. The Multipliers: CUDA Graphs and `torch.compile`

Once the collectives are fast, the next bottleneck is everything *around* them — Python dispatch, per-op kernel launches, recompilation. Two subsystems attack that, and they matter *more* in multi-GPU settings because launch overhead multiplies across ranks and any per-rank stall stalls the whole collective.

**CUDA graphs** capture a sequence of kernel launches once and replay them, eliminating per-step CPU launch cost. The V1 framework here ([#20059](https://github.com/vllm-project/vllm/pull/20059)) decoupled graph capture from `torch.compile`'s piecewise compilation with a "nested wrapper" design and a `cudagraph_mode` flag (`NONE / PIECEWISE / FULL / FULL_DECODE_ONLY / FULL_AND_PIECEWISE`). Measured impact on A100: FA2 output throughput ~5%, and — the number that shows why it matters at scale — CPU launch time of **~56ms flattened vs ~28ms piecewise**, with Triton ITL down up to 38%. Full-graph capture for Cutlass MLA ([#22763](https://github.com/vllm-project/vllm/pull/22763)) added ~6% E2E on DeepSeek-V2-Lite and roughly halved P99 TTFT (1818 → 1002 ms).

**`torch.compile`** integration ([#9715](https://github.com/vllm-project/vllm/pull/9715) established the piecewise-compile levels; [#10528](https://github.com/vllm-project/vllm/pull/10528) enabled CUDA graphs by default in V1) uses attention ops as graph-split boundaries so each segment captures cleanly. Much of the follow-up work is unglamorous but essential cache correctness — making custom Inductor passes picklable so the code cache actually works ([#10273](https://github.com/vllm-project/vllm/pull/10273)), and keying the compile cache on traced source including distributed ops ([#11614](https://github.com/vllm-project/vllm/pull/11614)) so a change to communication code correctly invalidates compiled artifacts.

<div class="post-callout post-callout-info">
  <div class="post-callout-icon">🔁</div>
  <div class="post-callout-content">
    <span class="post-callout-title">Why these are "multipliers"</span>
    <p>None of these are multi-GPU features per se — they run per rank. But in a TP or PP deployment, CPU launch overhead on the critical path is paid by every rank every step, and a slow rank makes every other rank wait at the next collective. Removing launch overhead therefore compounds: it's the difference between the fast collectives of §3–6 being the bottleneck (good) or the Python dispatch being the bottleneck (waste).</p>
  </div>
</div>

Chunked prefill deserves a mention alongside these as a scheduler-level multiplier. Landed originally in [#3884](https://github.com/vllm-project/vllm/pull/3884) (SARATHI-style token-budget chunking), it splits long prefills so they interleave with decode and bound peak activation memory — later auto-enabled for long-context ([#6666](https://github.com/vllm-project/vllm/pull/6666)) and made a first-class citizen of the V1 scheduler ([#15419](https://github.com/vllm-project/vllm/pull/15419)). Its multi-GPU relevance is indirect (it shapes batch composition and memory pressure, not cross-device comms), but it's the reason mixed prefill/decode batching works at all — which is what the PP bubble-removal work in §5 exploits.

---

## 8. Choosing a Configuration

Pulling the four dimensions together into a decision, driven — as §1 promised — by your interconnect and your workload:

| Situation | Reach for | Why |
|---|---|---|
| Single node, fast NVLink, dense model | **Tensor parallelism** (`TP` = GPUs/node) | All-reduce is cheap on NVLink; simplest scaling. |
| GPUs lack NVLink (PCIe-only, L40S), or spanning nodes | **Pipeline parallelism** (add `PP` = #nodes) | One handoff per stage beats two all-reduces per layer; the custom all-reduce disables itself on >2 PCIe GPUs anyway. |
| Large MoE model (DeepSeek-scale) | **Wide EP** (`--enable-expert-parallel` + `--enable-eplb`) | DP attention + expert sharding replaces a wasteful all-reduce with a sparse all-to-all; EPLB fixes routing skew. |
| A p99 latency / TTFT SLO to hit | **Disaggregated P/D** (NIXL or LMCache connector) | Tune prefill and decode independently; control tail ITL. Won't raise raw throughput. |
| Any of the above | **Turn on CUDA graphs + `torch.compile`** | Per-rank launch-overhead removal compounds across ranks. Default-on in V1. |

These compose. A production DeepSeek deployment might run `TP` within a node, `PP` across nodes, `--enable-expert-parallel` for the MoE layers with EPLB balancing them, and a disaggregated front-end with a NIXL connector — every dimension in this post active at once, each targeting a different tier of the §1 hierarchy.

The throughline: **vLLM's multi-GPU performance work is overwhelmingly communication engineering.** Faster collectives (§3, §4), cheaper topologies (§5), moving the KV cache exactly once (§6), and clearing the Python and launch overhead that sits between the collectives (§7). The FLOPs were never the hard part.

---

## References

**Background posts**

- [Intel GPU Scheduling](/gpu-kernel-scheduling/) — SIMD/workgroup/memory-hierarchy foundations.
- [How PyTorch Sees Your Triton Kernel](/triton-custom-kernels-pytorch/) — Dynamo, AOT Autograd, the compile path.
- [PyTorch SDPA Kernels](/pytorch-sdpa-kernel/) — attention-kernel background for MoE and disaggregation.

**Primary sources**

- vLLM docs: [Parallelism and Scaling][d1], [Architecture Overview][d2], [Data-Parallel Deployment][d3], [Expert-Parallel Deployment][d4], [Disaggregated Prefilling][d5].
- vLLM source: [`custom_all_reduce.py`][s1] (the all-reduce gates in §3.1).
- [vLLM V1 alpha release][d6] (Jan 2025); [Scaling DeepSeek-style MoEs with Wide EP (Red Hat)][d7].

**Verified PR catalog (38 PRs)**

Every PR below was checked against its GitHub page for merge status, mechanism, and reported numbers. "Caveat" flags a benchmark to read carefully (weak baseline, unstated hardware, or a best-case workload).

| Theme | PR | What it does | Measured | Caveat |
|---|---|---|---|---|
| Comm/overlap | [#20759](https://github.com/vllm-project/vllm/pull/20759) | PyTorch symm-mem all-reduce | B200 TP=8 ~7–10% TTFT | — |
| Comm/overlap | [#24532](https://github.com/vllm-project/vllm/pull/24532) | NCCL symm-mem all-reduce | DeepSeek-R1 TP=8 +3.5% req/s | GPU unstated; 2× is micro-bench |
| Comm/overlap | [#17882](https://github.com/vllm-project/vllm/pull/17882) | Async tensor parallelism | 4×H200 70B 0.590→0.526s | ~0 gain at 2×A100/8B |
| Comm/overlap | [#16155](https://github.com/vllm-project/vllm/pull/16155) | Sequence parallelism pass | mechanism-only | no perf reported |
| Comm/overlap | [#28672](https://github.com/vllm-project/vllm/pull/28672) | SP token-count threshold | 70B-FP8 TP=4 50.3→18.9s | headline vs baseline; SP-on-vs-off ~4% |
| Comm/overlap | [#42993](https://github.com/vllm-project/vllm/pull/42993) | symm_mem cap-equal fix + AR logging | mechanism-only | correctness/observability |
| All-reduce kernels | [#19744](https://github.com/vllm-project/vllm/pull/19744) | ROCm QuickReduce (quantized) | MI300 TP=4 TTFT ~2.2× | weak RCCL baseline >16MB |
| All-reduce kernels | [#23964](https://github.com/vllm-project/vllm/pull/23964) | allgather+reduce-scatter All2All | 4-GPU DeepSeek-R1-FP4 ~1.97× | — |
| All-reduce kernels | [#41675](https://github.com/vllm-project/vllm/pull/41675) | ROCm QR min-size + codec knobs | 4×MI355X up to −12.7% TPOT | combined config only; knob alone regresses |
| All-reduce kernels | [#46065](https://github.com/vllm-project/vllm/pull/46065) | AITER custom AR in CudaCommunicator | neutral (±1%) | consolidation/correctness, not perf |
| All-reduce kernels | [#46703](https://github.com/vllm-project/vllm/pull/46703) | NCCL symm-mem → AllGather/ReduceScatter | TP4 810→287µs | numbers from commit msg, not test section |
| All-reduce kernels | [#46392](https://github.com/vllm-project/vllm/pull/46392) | FlashInfer fused AR @ WS=16 (GB300) | fused AR+RMSNorm 2.4–5.0× | baseline NVLS disabled → inflated; no E2E |
| MoE / EP | [#28664](https://github.com/vllm-project/vllm/pull/28664) | MoRI-EP all2all (AMD) | MI300X EP=16 2.68× | — |
| MoE / EP | [#41633](https://github.com/vllm-project/vllm/pull/41633) | EPLB Nixl zero-copy transfers | EP=8 0.9→0.7s; 4.4GiB→0 | modest latency; memory is the win |
| MoE / EP | [#33724](https://github.com/vllm-project/vllm/pull/33724) | Removes PPLX backend | — | context: EP backends churn |
| MoE / EP | [#18343](https://github.com/vllm-project/vllm/pull/18343) | EPLB (expert load balancer) | mechanism | rebalances hot experts |
| CUDA graphs | [#20059](https://github.com/vllm-project/vllm/pull/20059) | Full cudagraph orthogonal to compile | A100 FA2 ~5%; launch 56→28ms | per-rank; compounds in TP/PP |
| CUDA graphs | [#22763](https://github.com/vllm-project/vllm/pull/22763) | Full cudagraph for Cutlass MLA | +6% E2E; P99 TTFT 1818→1002ms | single-GPU decode path |
| CUDA graphs | [#22594](https://github.com/vllm-project/vllm/pull/22594) | Full cudagraph default for hybrid/mamba | V1 8.38→3.40s | GPU unstated; bs=1 latency |
| CUDA graphs | [#23035](https://github.com/vllm-project/vllm/pull/23035) | Full+piecewise for Mamba1 | Jamba FCG 3.10s vs 5.00s | GPU unstated; bs=1 |
| torch.compile | [#21069](https://github.com/vllm-project/vllm/pull/21069) | FlashInfer AR+RMSNorm+quant fusion | B200 TP=2 ~7–8% TPOT | vs custom ops; FP8 TTFT can regress vs compiled |
| torch.compile | [#9715](https://github.com/vllm-project/vllm/pull/9715) | Compile levels + piecewise cudagraph | mechanism-only | foundational |
| torch.compile | [#10528](https://github.com/vllm-project/vllm/pull/10528) | Enable CUDA graph by default (V1) | mechanism-only | — |
| torch.compile | [#11614](https://github.com/vllm-project/vllm/pull/11614) | Code-aware compile cache | mechanism-only | correctness |
| torch.compile | [#10273](https://github.com/vllm-project/vllm/pull/10273) | PostGradPassManager + Inductor cache fix | mechanism-only | correctness |
| Chunked prefill | [#3884](https://github.com/vllm-project/vllm/pull/3884) | Chunked prefill e2e (SARATHI-style) | 1×A10 dry-run only | unvalidated at merge |
| Chunked prefill | [#6666](https://github.com/vllm-project/vllm/pull/6666) | Auto-enable for >32K context | mechanism-only | OOM avoidance |
| Chunked prefill | [#8342](https://github.com/vllm-project/vllm/pull/8342) | Coexist with prefix caching | mechanism-only | — |
| Chunked prefill | [#7874](https://github.com/vllm-project/vllm/pull/7874) | Restore decode-over-prefill ordering | mechanism-only | scheduler fix |
| Chunked prefill | [#15419](https://github.com/vllm-project/vllm/pull/15419) | `long_prefill_token_threshold` (V1) | mechanism-only | — |
| Disagg / KV | [#15960](https://github.com/vllm-project/vllm/pull/15960) | KV Connector API V1 (foundation) | mechanism-only | perf deferred |
| Disagg / KV | [#16625](https://github.com/vllm-project/vllm/pull/16625) | LMCache connector | 2×H100 1P1D ~40% tok/s, ~8× tail ITL | vs "2 separate instances" baseline |
| Disagg / KV | [#17751](https://github.com/vllm-project/vllm/pull/17751) | NIXL integration | mechanism-only | gsm8k correctness only |
| Disagg / KV | [#12957](https://github.com/vllm-project/vllm/pull/12957) | MooncakeStore XpYd | mechanism-only | early-stage (no prefix reuse) |
| Disagg / KV | [#30166](https://github.com/vllm-project/vllm/pull/30166) | Hybrid allocator + connector | parity + 69,888-tok retrieval | throughput parity, not gain |
| Pipeline | [#4412](https://github.com/vllm-project/vllm/pull/4412) | Original PP support | functional sanity only | not a speedup measurement |
| Pipeline | [#32618](https://github.com/vllm-project/vllm/pull/32618) | Async scheduling + PP | Qwen3-30B pp=4 30.8% E2E | GPU unstated |
| Pipeline | [#42187](https://github.com/vllm-project/vllm/pull/42187) | Avoid PP bubbles (ModelRunnerV2) | GB200 PP=4 up to 3.17× | best-case 128k/1; prefix-cache off |
| Pipeline | [#38726](https://github.com/vllm-project/vllm/pull/38726) | Fix stuck chunked PP | GLM-4.7 pp4tp2 1392→4578 tok/s | output-len=1 flatters it |

[d1]: https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html
[d2]: https://docs.vllm.ai/en/latest/design/arch_overview.html
[d3]: https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html
[d4]: https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html
[d5]: https://docs.vllm.ai/en/latest/features/disagg_prefill.html
[d6]: https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
[d7]: https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep
[s1]: https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/custom_all_reduce.py
