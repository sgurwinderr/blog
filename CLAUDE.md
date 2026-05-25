# Gurwinder's Technical Blog - Claude Context

## Project Overview

This is a **Hugo static site** for publishing deep technical content on AI, GPU programming, graphics, and game development.

**Live site:** https://sgurwinderr.com/  
**Author:** Gurwinder (GPU SDE @ Intel)  
**Tech stack:** Hugo (static site generator), GitHub Pages, custom Medium theme

---

## Core Focus Areas

### Content Categories

1. **AI & Machine Learning**
   - PyTorch internals (profiler, autograd, graph acquisition, SDPA kernels)
   - LLM optimization (quantization, inference)
   - Vision transformers

2. **GPU Programming**
   - Triton kernels (integration with PyTorch, first principles)
   - CUDA/OpenCL/SYCL
   - GPU kernel scheduling
   - Memory optimization

3. **Graphics & Ray Tracing**
   - DirectX 12, HLSL
   - DirectML
   - Gaussian splatting

4. **Game Development**
   - Unity tutorials
   - Unreal Engine
   - AR/VR

### Writing Style

**Target audience:** GPU developers, ML engineers, systems programmers

**Depth level:**
- Mathematical rigor (complete proofs, derivations)
- Production code examples (Triton, CUDA, PyTorch)
- First principles explanations
- Compiler/runtime internals

**NOT surface-level tutorials** - aim for publication-quality technical depth.

**Examples of depth:**
- "How PyTorch Sees Your Triton Kernel" - shows Dynamo FX graphs, AOT Autograd output
- "Online Softmax Algorithm" - complete mathematical derivation with proofs
- Posts typically 8,000-12,000 words

### SEO and Image Guidelines

**Image alt text:**
- Always add `imageAlt` field to post frontmatter for featured images
- Be descriptive and specific, not generic
- For diagrams: Describe what the diagram shows
  - ✅ "Architecture diagram showing PyTorch Triton kernel integration flow"
  - ❌ "Diagram"
- For code screenshots: Describe the code's purpose
  - ✅ "Code snippet demonstrating PyTorch autograd hook registration"
- For illustrations: Describe the concept
  - ✅ "Conceptual illustration of GPU memory hierarchy with L1/L2 caches"

**Meta descriptions:**
- Add `description` field to post frontmatter (max 160 characters)
- Focus on what the reader will learn or the problem being solved
- Should entice clicks from search results
- If omitted, first 160 chars of content used automatically

**Internal linking:**
- Link to related posts for context and background
- Example: "For background on PyTorch's autograd system, see [Understanding PyTorch Autograd](/pytorch-autograd-internals)"
- Build knowledge graph by connecting related topics
- Link from new posts to older foundational posts

---

## Repository Structure

```
blog/
├── content/
│   ├── about.md              # About page
│   ├── courses.md            # Interactive courses landing page
│   └── post/                 # Blog posts (26+ articles)
│       └── 2026-*.md
│
├── layouts/
│   ├── _default/
│   │   ├── single.html       # Blog post template (has MathJax config)
│   │   ├── list.html         # List page template
│   │   └── _markup/
│   └── partials/
│       ├── _shared/          # Common components (navbar, footer, head)
│       └── list-partials/    # Post grid components
│
├── static/
│   ├── assets/
│   │   ├── images/           # Post images, avatars
│   │   └── videos/           # Video content (when used)
│   ├── courses/              # Interactive HTML courses
│   │   └── README.md
│   └── css/
│       └── custom.css        # Custom styling
│
├── themes/medium/            # Base theme (customized via layouts/)
│
├── .skills/                  # Content creation tools
│   ├── README.md
│   └── codebase-to-course/   # Generates interactive courses
│
├── hugo.toml                 # Main configuration
└── CLAUDE.md                 # This file
```

---

## Hugo Configuration

**File:** `hugo.toml`

### Key Settings

```toml
baseURL = "https://sgurwinderr.com/"
title = "Gurwinder's Blog: AI & Graphics"
theme = "medium"

# Permalinks: posts use /:slug/ (no date prefix)
[permalinks]
  post = "/:slug/"

# LaTeX support (critical for math-heavy posts)
[markup.goldmark.extensions.passthrough]
  enable = true
  
[markup.goldmark.extensions.passthrough.delimiters]
  block = [["$$", "$$"]]
  inline = [["\\(", "\\)"], ["$", "$"]]

# Syntax highlighting
[markup.highlight]
  style = "github-dark"
  lineNos = false
  noClasses = true
```

### Menu Structure

```
Blog (/) → Courses (/courses) → AI → Game Dev → PyTorch → About
```

Menu items defined in `hugo.toml` under `[[menu.main]]`

---

## MathJax Configuration

**Location:** `layouts/_default/single.html` (lines 101-109)

```javascript
MathJax = {
    tex: {
        inlineMath: [['\\(', '\\)'], ['$', '$']],
        displayMath: [['$$', '$$']],
        processEscapes: true
    }
};
```

**Critical:** Supports both `$...$` and `\(...\)` for inline math.

---

## Creating Blog Posts

### Post Frontmatter Template

```yaml
---
author: Gurwinder
categories:
- AI
- PyTorch
date: '2026-05-22T00:00:00Z'
slug: 'post-slug-here'
featured: false
draft: false
image: assets/images/post-cover.jpg
imageAlt: 'Descriptive alt text for the featured image (for accessibility and SEO)'
description: 'Brief summary of what readers will learn (max 160 chars, shown in search results)'
title: 'Post Title: Subtitle'
---
```

**Note on new fields:**
- `imageAlt`: Required for accessibility and SEO. Be specific about what the image shows.
- `description`: Recommended for SEO. If omitted, Hugo uses the first 160 characters of content.

### Writing Guidelines

**Structure:**
1. **Problem/motivation** (why should the reader care?)
2. **Mathematical foundation** (if applicable, with complete proofs)
3. **Algorithm/technique** (step-by-step, with derivations)
4. **Implementation** (production code, not pseudocode)
5. **Practical application** (how it's used in real systems)
6. **Historical context** (papers, timeline, impact)

**LaTeX:**
- Inline: `$e^{x}$` or `\(e^{x}\)`
- Display: `$$..$$` (newline before/after)
- Multi-line: Use `align` environment

**Code blocks:**
```python
# Always specify language
def example():
    pass
```

**Images:**
- Store in `static/assets/images/`
- Reference as `/assets/images/filename.jpg`
- Hugo handles path resolution

**NO videos in posts** (prefer text + code + math)

### Featured Posts

Set `featured: true` in frontmatter to show on homepage featured section.

Limit to 2-3 most important posts at a time.

---

## Development Workflow

### Local Preview

```bash
cd ~/blog
hugo server -D --bind 0.0.0.0 --port 1313

# Visit: http://localhost:1313
```

**Live reload:** Changes auto-refresh browser

### Adding a New Post

```bash
# Create post file
touch content/post/YYYY-MM-DD-slug.md

# Add frontmatter and content

# Preview locally
hugo server -D

# When ready: set draft: false
```

### Hugo Commands

```bash
# Build site (outputs to public/)
hugo

# List all pages
hugo list all

# List draft pages
hugo list drafts

# Check config
hugo config
```

---

## Interactive Courses

### Skill: codebase-to-course

**Location:** `.skills/codebase-to-course/`

**Purpose:** Generate beautiful, single-page HTML courses from codebases.

**Usage:**
```
"Turn [codebase/repo/directory] into a course"
```

**Output:** Self-contained HTML with:
- Scroll-based navigation
- Animated diagrams
- Interactive quizzes
- Plain-English code explanations
- Group chat animations (components talking)

**Add to blog:**
```bash
cp -r generated-course static/courses/
# Update content/courses.md
```

**Target audience:** "Vibe coders" - people who build with AI tools but want to understand internals.

---

## Common Tasks

### Add a Blog Post

1. Create `content/post/YYYY-MM-DD-title.md`
2. Write content with LaTeX math
3. Preview: `hugo server -D`
4. Publish: set `draft: false`

### Add an Interactive Course

1. Tell Claude: `"Turn X into a course"`
2. Copy output: `cp -r course-name static/courses/`
3. Update `content/courses.md` to list it
4. Link from relevant blog posts

### Update LaTeX Rendering

Edit `layouts/_default/single.html` MathJax config (line ~102)

### Customize Theme

- Override templates in `layouts/` (Hugo checks layouts/ first, then themes/)
- Add custom CSS in `static/css/custom.css`

### Check Build Locally

```bash
hugo --cleanDestinationDir
# Outputs to public/
# Check public/index.html
```

---

## Technical Constraints

### Hugo Version

Using **Hugo v0.123.7** (Extended)

Check version: `hugo version`

### Dependencies

- **Required:** Hugo (extended version for SCSS processing)
- **Optional:** Git (for version control)

### Browser Support

- Modern browsers (Chrome, Firefox, Safari, Edge)
- Mobile responsive (Hugo medium theme)

### External Dependencies

- **Google Fonts CDN** (for typography)
- **MathJax 3 CDN** (for LaTeX rendering)
- **No other CDNs** - everything else self-hosted

---

## Content Guidelines

### What to Write About

**Good topics:**
- GPU programming (CUDA, Triton, kernels)
- ML optimization (inference, quantization, attention)
- Graphics algorithms (ray tracing, splatting)
- Compiler internals (PyTorch, Triton)
- Systems programming (memory, caching, performance)

**Avoid:**
- Surface-level tutorials
- Beginner content
- "Top 10" listicles
- Generic advice

### Writing Quality Bar

**Ask before publishing:**
- [ ] Is this deeper than what's already available?
- [ ] Did I derive equations from first principles?
- [ ] Is the code production-ready (not pseudocode)?
- [ ] Would this be useful to GPU/ML engineers?
- [ ] Is the math rigorous (complete proofs)?

**Benchmarks:**
- Your Triton posts
- Your Online Softmax post
- Papers like Flash Attention (Dao et al.)

---

## Maintenance

### Regular Updates

- Check Hugo version: `hugo version`
- Update theme: `cd themes/medium && git pull`
- Verify LaTeX rendering still works
- Test on mobile devices

### Performance

- Keep images optimized (<200KB per image)
- No videos in posts (heavy, not worth it)
- LaTeX loads asynchronously (MathJax CDN)

### Backups

- Git repo (origin: GitHub)
- Regular commits for new posts
- Consider GitHub Actions for auto-deploy

---

## Deployment

**Currently:** Manual (content pushed to GitHub)

**Future:** Could set up GitHub Actions to auto-build and deploy on push to main.

**Build output:** `public/` directory (contains full static site)

---

## SEO Health Checks

### Monthly Checklist

Run these checks at the start of each month to maintain SEO health:

```bash
# 1. Check for missing alt text
cd ~/blog
grep -r "^image:" content/post/*.md | while read -r line; do
    file=$(echo "$line" | cut -d: -f1)
    if ! grep -q "^imageAlt:" "$file"; then
        echo "Missing imageAlt: $file"
    fi
done

# 2. Check for missing meta descriptions
grep -l "^draft: false" content/post/*.md | while read -r file; do
    if ! grep -q "^description:" "$file"; then
        echo "Missing description: $file"
    fi
done

# 3. Validate sitemap.xml is up to date
hugo && ls -lh public/sitemap.xml

# 4. Check for broken internal links (requires htmltest or similar)
# Manual: Scan recent posts for [[...]] style links

# 5. Review Google Search Console
# - Check for crawl errors
# - Review new indexed pages
# - Check mobile usability issues
```

**Action items:**
- Fix any missing imageAlt fields (add descriptive alt text)
- Add meta descriptions to posts without them
- Update old posts with internal links to new related content
- Fix any broken links found

**Frequency:** First Monday of each month (15-30 minutes)

---

## Getting Help

**Hugo docs:** https://gohugo.io/documentation/  
**Medium theme:** Check `themes/medium/` for templates  
**MathJax docs:** https://docs.mathjax.org/  

**For blog issues:**
- Check `hugo server` output for errors
- Verify frontmatter YAML syntax
- Test LaTeX with minimal example
- Check browser console for JS errors

---

## Author Context

**Gurwinder**
- GPU SDE at Intel
- Focus: AI & Graphics
- Expertise: PyTorch, Triton, CUDA, GPU kernels, DirectX
- Writing style: Deep technical, first principles, production code

**Social:**
- LinkedIn: sgurwinderr
- GitHub: sgurwinderr

---

## Quick Reference

**Start server:** `cd ~/blog && hugo server -D`  
**Build site:** `hugo`  
**New post:** `content/post/YYYY-MM-DD-slug.md`  
**LaTeX config:** `layouts/_default/single.html` line 102  
**Courses:** `static/courses/` + update `content/courses.md`  
**Skill:** `.skills/codebase-to-course/SKILL.md`

---

**Last updated:** 2026-05-22  
**Hugo version:** 0.123.7  
**Posts:** 26+  
**Status:** Active, regularly updated
