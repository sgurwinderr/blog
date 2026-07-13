---
name: blog-management
description: Manage this Hugo technical blog. Use when creating or updating posts, editing blog/catalog data, running Hugo build/preview checks, fixing blog navigation or content presentation, or preparing local publish steps for sgurwinderr.com.
---

# Blog Management

Use the repo's existing Hugo structure and data-driven catalog patterns. Keep edits scoped to the requested content, layouts, or data files.

## Core Workflow

1. Inspect `hugo.toml`, relevant layouts, and existing nearby content before editing.
2. Make the smallest content/layout/data change that satisfies the request.
3. Run `hugo --minify --cleanDestinationDir` before finishing when the change can affect output.
4. Report drafts, build warnings, and any checks that could not run.
5. Commit only when the user explicitly asks. Never push without explicit approval.

## Blog Posts

Create posts under `content/post/YYYY-MM-DD-slug.md`. Match existing frontmatter style and include SEO fields for published posts:

```yaml
---
author: Gurwinder
categories:
- AI
- PyTorch
date: 'YYYY-MM-DDT00:00:00Z'
slug: 'post-slug'
featured: false
draft: false
image: assets/images/cover.jpg
imageAlt: 'Specific description of the featured image'
description: 'Search-result summary under 160 characters.'
title: 'Post Title'
---
```

Use `/assets/images/name.ext` in Markdown content for images stored under `static/assets/images/`. Add meaningful Markdown alt text for inline images.

## Learn AI And PR Walkthrough Catalogs

Course cards are data-driven:

- `data/courses.json` feeds `/learn-ai/` and the Learn AI homepage section.
- `data/pr_walkthroughs.json` feeds `/pr-walkthroughs/` and the homepage PR Walkthroughs section.

Catalog entry shape:

```json
{
  "title": "Course Title",
  "url": "/learn-ai/course-slug/",
  "category": "Category",
  "summary": "Brief description.",
  "gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
  "icon": "GPU",
  "iconType": "text",
  "modules": "8 Modules",
  "duration": "1-2 Hours"
}
```

Use `iconType: "text"` for short labels such as `GPU`, `SLM`, or `vLLM`; use `emoji` only when the file already uses a compatible emoji and encoding is preserved.

## Hugo Commands

Run from the repo root:

```bash
hugo server -D --bind 0.0.0.0 --port 1313
hugo --minify --cleanDestinationDir
hugo list drafts
hugo list all
```

If port `1313` is busy, use another port and tell the user the URL.

## Common Fixes

- Course card missing: add or correct the entry in `data/courses.json` or `data/pr_walkthroughs.json`.
- Featured image missing: verify `image:` maps to a real file under `static/`.
- Weak SEO snippet: add or tighten `description:` and `imageAlt:` in frontmatter.
- Math not rendering: keep inline math as `$...$` or `\(...\)` and display math as `$$...$$`; verify `layouts/_default/single.html` before changing MathJax.
- Hugo deprecation warnings: update config/templates only if compatible with the deployed Hugo version in `.github/workflows/*.yml`.
