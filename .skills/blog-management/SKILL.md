---
name: blog-management
description: Use when asked to create blog posts, add courses to catalog, fix course styling issues, or publish blog content
---

# Hugo Blog Management

## When to Use

**Use this skill when:**
- Creating new blog posts with Hugo frontmatter
- Adding courses to the Learn AI catalog (courses.json)
- Fixing module number visibility in course styling
- Running Hugo commands (server, build, list drafts)
- Publishing workflow (test → build → commit)

**Don't use for:**
- Theme customization (see CLAUDE.md layouts/ section)
- LaTeX configuration (see CLAUDE.md MathJax section)
- Codebase architecture questions (see CLAUDE.md)

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
hugo server -D --bind 0.0.0.0 --port 1313  # Start (default port)
hugo --cleanDestinationDir                 # Build
hugo list drafts                           # List drafts
```

## Course Management

### Add to Catalog
Edit `data/courses.json`:
```json
{
  "title": "Course Title",
  "url": "/learn-ai/course-name/",
  "category": "Category",
  "summary": "Description",
  "gradient": "linear-gradient(135deg, #color1 0%, #color2 100%)",
  "icon": "⚡",
  "iconType": "emoji",
  "modules": "N Modules",
  "duration": "X Hours"
}
```

### Module Number Visibility Fix
All courses need this in `styles.css`:
```css
.module-number {
  color: var(--color-text);  /* NOT var(--color-accent) */
  opacity: 0.8;               /* NOT 0.15 */
}
```

## LaTeX
Inline: `$e^{i\pi}$` or `\(x^2\)`  
Display: `$$\frac{e^{x_i}}{\sum_j e^{x_j}}$$`

## Key Locations
```
content/post/             # Blog posts
data/courses.json         # Course metadata (homepage + catalog)
static/learn-ai/          # Courses
layouts/index.html        # Homepage
hugo.toml                 # Config
```

## Publish Workflow
```bash
# 1. Test locally
hugo server -D --bind 0.0.0.0 --port 1313
# 2. Build
hugo --cleanDestinationDir
# 3. Commit (DON'T auto-push)
git add -A && git commit -m "message"
# 4. Get approval, then push
```

## Common Issues
- **Course not showing:** Add to `data/courses.json`
- **Numbers invisible:** Fix opacity + color in styles.css
- **No LaTeX:** Check `$` delimiters, restart server
