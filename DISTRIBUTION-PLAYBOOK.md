# Distribution Playbook — sgurwinderr.com

A practical guide for getting deep-technical GPU/ML posts in front of the right
audience (peer engineers + researchers) and building authority over time. This is a
reference doc, not a published page.

## The positioning (keep it sharp)
Site is now **"Gurwinder - AI & GPU"** — *"Making large-scale AI fast, one kernel at a
time."* Everything you publish should reinforce the niche: **GPU programming, PyTorch/
Triton internals, ML compilers, kernel-level performance.** Depth in a narrow lane beats
breadth. The Unity/game-dev posts are legacy — fine to keep, but the forward identity is
low-level GPU/ML systems.

## Cadence (realistic > aspirational)
- Your history: 16 posts in 2024, then 2 (2025) and 4 (2026). The dropoff hurts authority
  more than low volume does — **consistency signals reliability.**
- **Target: 1 substantial post / month** (or 1 every 6 weeks). One rigorous piece beats
  four shallow ones for this audience.
- Batch: keep a running ideas list; draft in the gaps; ship on a predictable rhythm.

## Which channel for which post
| Channel | Best for | How |
|---|---|---|
| **Hacker News** | Novel, rigorous, "I built/measured X" posts (e.g. the TTGIR SLM pass, custom all-reduce internals) | Submit with the plain title, no editorializing. Post Tue–Thu ~8–10am ET. Reply substantively in comments. Don't ask for upvotes. |
| **r/MachineLearning** | Research-adjacent, results-driven (attention kernels, quantization, compiler passes) | Use the `[D]`/`[P]` flair. Lead with the finding. Engage technically. |
| **r/CUDA, r/LocalLLaMA** | GPU-kernel + inference-optimization posts | Niche but high-signal; great for the vLLM/Triton content. |
| **X/Twitter** | Threads that tease a post's key insight (1 diagram + 1 result + link) | Build in public; tag relevant tools (PyTorch, Triton, vLLM). *(Note: no Twitter handle in config yet — add one.)* |
| **LinkedIn** | Career-authority framing; reaches recruiters + Intel network | Post the "why it matters" angle, link the deep dive. You already have a Follow button. |
| **lobste.rs** | Same as HN but smaller/higher-signal (needs invite) | Tag `compilers`/`ml`/`performance`. |

## Per-post checklist (before you hit publish)
- [ ] `description` set (≤160 chars, keyword-rich) — used in search + OG + Twitter cards
- [ ] `image` + `imageAlt` set (OG card art — decides click-through on socials)
- [ ] `categories` set; consider adding `tags` for topic clustering (see below)
- [ ] Internal links to 2–3 related older posts (builds topic clusters + keeps readers)
- [ ] Featured image is LCP-optimized (already: eager + fetchpriority, no lazy)
- [ ] Ran locally, checked light + dark, mobile
- [ ] After deploy: **GSC → URL Inspection → Request Indexing** for the new URL
- [ ] Cross-post plan: pick 1–2 channels above, don't spray all at once

## Topic clusters + internal linking (authority lever)
Your content already forms three strong spines — lean into them:
1. **PyTorch internals** (profiler, Dynamo/AOT autograd, SDPA kernels, quant)
2. **GPU kernels / low-level** (DPAS, SYCL/CUDA, Intel GPU scheduling)
3. **Triton / compilers** (first principles, Triton-through-PyTorch, TTGIR SLM pass) ← the growth area

Actions:
- Cross-link within each spine (every Triton post links the others).
- Consider 2–3 **pillar pages** (or use the `/learn-ai/` + `/pr-walkthroughs/` sections as hubs)
  that index a cluster and link out to every post in it.
- **Activate `tags`** (the template plumbing exists but posts don't set tags, and `/tags/` is
  currently robots-blocked). Decision to make: either (a) use tags as internal-nav clusters and
  allow `/tags/` to be indexed, or (b) keep them on-site only. Recommend (a) for GPU/Triton/
  PyTorch/CUDA tags — they're strong long-tail SEO landing pages.

## Retention / engagement
- **Social-first distribution** — reach is driven by the channels above (X, LinkedIn, HN,
  Reddit), not on-site subscriptions. Post consistently and let the platforms do the fan-out.
- **Claps** — on-site engagement mechanic exists; consider adding real comments (giscus =
  GitHub-issue-backed, fits a dev audience) later.
- Deliberately not using RSS or an email newsletter.

## Quick wins still open (from the audit)
- Set the **Google Search Console verification** code in `hugo.toml` (`[params.seo]`).
- Add a **Twitter/X handle** to `[params.social]` so `twitter:site`/`creator` cards attribute you.
- Consider **self-hosting fonts** and **dropping render-blocking Bootstrap CSS** (Phase-2
  follow-up) for a faster LCP.
