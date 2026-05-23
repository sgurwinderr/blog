---
name: blog-management
description: Use when creating blog posts, managing Hugo blog, or working with technical content and LaTeX
---

# Hugo Blog Management

## Create Post
```bash
touch content/post/YYYY-MM-DD-slug.md
```

Frontmatter:
```yaml
---
author: Gurwinder
categories: [AI, PyTorch]
date: 'YYYY-MM-DD'
slug: 'post-slug'
featured: false
draft: false
image: assets/images/cover.jpg
title: 'Title: Subtitle'
---
```

## Hugo Commands
```bash
cd ~/blog
hugo server -D --bind 0.0.0.0 --port 1313  # Start (localhost:1313)
hugo                                        # Build
hugo list drafts                            # List drafts
```

## LaTeX
Inline: `$e^{i\pi}$` or `\(x^2\)`  
Display: `$$\frac{e^{x_i}}{\sum_j e^{x_j}}$$`

Config: `layouts/_default/single.html` (lines 101-109)

## Writing Style
- **Depth:** Publication-quality, complete proofs, 8-12K words
- **Code:** Production-ready (Triton, CUDA, PyTorch)
- **Math:** Full derivations from first principles
- **Avoid:** Videos, listicles, surface-level content

## Locations
```
content/post/              # Posts
static/assets/images/      # Images (use /assets/images/ in posts)
layouts/_default/single.html  # MathJax config
layouts/section/learn-ai.html # Course catalog
hugo.toml                  # Site config, menu
```

## Publish
```bash
# 1. Set draft: false
# 2. Commit
git add content/post/YYYY-MM-DD-slug.md
git commit -m "feat: add post on [topic]"
```

## Common Issues
- **No LaTeX:** Check `$` delimiters, restart server
- **Post hidden:** Verify `draft: false`, date not future
- **Image broken:** Use `/assets/images/` not relative path
- **Featured post:** Set `featured: true`

## File Structure
```
blog/
├── content/post/           # Blog posts
├── content/learn-ai/       # Course catalog
├── static/assets/images/   # Images
├── static/learn-ai/        # Courses
└── layouts/               # Templates
```

Menu: Blog → Learn AI → AI → Game Dev → PyTorch → About (edit `hugo.toml`)
