---
name: blog-management
description: Use when creating blog posts, managing Hugo blog, working with technical content and LaTeX, or managing Learn AI courses
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
hugo server -D --bind 0.0.0.0 --port 8080  # Start (localhost:8080)
hugo --cleanDestinationDir                 # Clean build
hugo list drafts                           # List drafts

# Reload server
pkill -f "hugo server"; sleep 1; hugo server -D --bind 0.0.0.0 --port 8080
```

## Course Management

### Add New Course
1. **Create course HTML/CSS/JS** in `static/learn-ai/course-name/`
2. **Add to courses.json:**
```json
{
  "title": "Course Title",
  "url": "/learn-ai/course-name/",
  "category": "Category",
  "summary": "Short description",
  "gradient": "linear-gradient(135deg, #color1 0%, #color2 100%)",
  "icon": "emoji or text",
  "iconType": "emoji" or "text",
  "modules": "N Modules",
  "duration": "X Hours"
}
```
3. **Verify module numbers are visible** - Check `styles.css` has:
```css
.module-number {
  color: var(--color-text);  /* Use text color, not accent */
  opacity: 0.8;               /* Dark enough to see (not 0.15) */
}
```

### Course Data (data/courses.json)
Central source for course cards shown on:
- Homepage (`layouts/index.html`) - Learn AI section
- Course catalog (`layouts/section/learn-ai.html`)

Both pages render from the same JSON data for consistency.

### Course Styling Standards
All courses must have:
- **White backgrounds** with smooth scrolling
- **Module numbers visible** (opacity 0.8, text color)
- **Consistent structure:** module-content → module-header → module-body
- **Proper HTML nesting:** All divs must close properly
- **Responsive navigation:** Dot navigation at top
- **Module height:** `min-height: 100dvh` with `100vh` fallback

### Course Style Files
Each course has its own `styles.css`:
- `static/learn-ai/vllm-triton-optimization/styles.css`
- `static/learn-ai/gpu-optimization-prs/styles.css`
- `static/learn-ai/pytorch-optimization-prs/styles.css`

**Module number visibility fix:**
```css
.module-number {
  color: var(--color-text);  /* NOT var(--color-accent) */
  opacity: 0.8;               /* NOT 0.15 */
}
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
content/post/                    # Blog posts
content/learn-ai/                # Course catalog page
data/courses.json                # Course metadata (central source)
static/assets/images/            # Images (use /assets/images/)
static/learn-ai/                 # Course HTML/CSS/JS
layouts/_default/single.html     # MathJax config
layouts/section/learn-ai.html    # Course catalog template
layouts/index.html               # Homepage (includes Learn AI section)
hugo.toml                        # Site config, menu
```

## Publish
```bash
# 1. Set draft: false in posts
# 2. Test locally first
hugo server -D --bind 0.0.0.0 --port 8080
# 3. Build
hugo --cleanDestinationDir
# 4. Commit (DO NOT push without approval)
git add .
git commit -m "feat: add [feature]"
# 5. Get approval, then push
git push origin master
```

## Common Issues
- **No LaTeX:** Check `$` delimiters, restart server
- **Post hidden:** Verify `draft: false`, date not future
- **Image broken:** Use `/assets/images/` not relative path
- **Featured post:** Set `featured: true` in frontmatter
- **Course numbers invisible:** Check opacity and color in styles.css
- **Layout broken:** Verify all module divs close properly
- **Course not showing:** Add to `data/courses.json`

## File Structure
```
blog/
├── content/
│   ├── post/                # Blog posts
│   └── learn-ai/            # Course catalog page
├── data/
│   └── courses.json         # Course metadata (NEW - central source)
├── static/
│   ├── assets/images/       # Images
│   ├── learn-ai/            # Course directories
│   │   ├── vllm-triton-optimization/
│   │   ├── gpu-optimization-prs/
│   │   └── pytorch-optimization-prs/
│   └── css/
│       └── learn-ai.css     # Course catalog styles
└── layouts/
    ├── index.html           # Homepage (includes Learn AI section)
    ├── section/learn-ai.html # Course catalog
    └── _default/single.html # Blog post template
```

Menu: Blog → Learn AI → AI → Game Dev → PyTorch → About (edit `hugo.toml`)

## Development Workflow
1. **Make changes locally**
2. **Build and test:** `hugo --cleanDestinationDir`
3. **Review in browser:** http://localhost:8080
4. **Commit locally:** `git add -A && git commit -m "message"`
5. **Wait for approval before pushing**
6. **Never auto-push** - always ask first
