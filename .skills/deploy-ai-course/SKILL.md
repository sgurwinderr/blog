---
name: deploy-ai-course
description: Use when deploying interactive AI courses to Hugo blog Learn AI section with navigation and catalog
---

# Deploy AI Course to Hugo Blog

## Quick Workflow

### 1. Clone & Generate Course
```bash
cd /tmp
git clone --depth 1 https://github.com/org/repo.git repo-name

# Create prompt: /tmp/course-prompt.md with module structure
# Use: "Using codebase-to-course skill, generate course from /tmp/repo-name following /tmp/course-prompt.md"
```

### 2. Move to Blog
```bash
mkdir -p ~/blog/static/learn-ai/course-slug/
cp -r [generated]/* ~/blog/static/learn-ai/course-slug/
```

### 3. Add Back Navigation
Edit `static/learn-ai/course-slug/index.html` after `<body>`:
```html
<div class="back-to-blog">
  <a href="/" class="back-link">
    <svg width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
      <path fill-rule="evenodd" d="M15 8a.5.5 0 0 0-.5-.5H2.707l3.147-3.146a.5.5 0 1 0-.708-.708l-4 4a.5.5 0 0 0 0 .708l4 4a.5.5 0 0 0 .708-.708L2.707 8.5H14.5A.5.5 0 0 0 15 8z"/>
    </svg>
    <span>Back to Blog</span>
  </a>
  <a href="/learn-ai/" class="back-link ml-auto"><span>All Courses</span></a>
</div>
```

Append to `static/learn-ai/course-slug/styles.css`:
```css
.back-to-blog { position: fixed; top: 0; left: 0; right: 0; height: 50px; background: var(--color-surface); border-bottom: 1px solid var(--color-border); display: flex; align-items: center; padding: 0 var(--space-6); z-index: 999; font-family: var(--font-body); }
.back-link { display: flex; align-items: center; gap: var(--space-2); color: var(--color-text-secondary); text-decoration: none; font-size: var(--text-sm); font-weight: 500; transition: color 0.2s; }
.back-link:hover { color: var(--color-accent); }
.ml-auto { margin-left: auto; }
.nav { top: 50px; }
body { padding-top: 50px; }
```

### 4. Add to Catalog
Edit `layouts/section/learn-ai.html`, duplicate existing card, update:
- All `/learn-ai/course-slug/` URLs
- Title, description
- `<span class="badge">N Modules</span>` counts
- Topics list

### 5. Test
```bash
# Visit http://localhost:1313/learn-ai/ - check card
# Visit http://localhost:1313/learn-ai/course-slug/ - check navigation
```

### 6. Commit
```bash
git add static/learn-ai/course-slug/ layouts/section/learn-ai.html
git commit -m "feat: add [Course Title]

- N modules on [topic]
- 15+ animations, quizzes, code walkthroughs
- Target: [audience]"
rm -rf /tmp/repo-name /tmp/course-prompt.md
```

## File Structure
```
blog/
├── layouts/section/learn-ai.html        # Catalog
├── static/
│   ├── css/learn-ai.css
│   └── learn-ai/
│       └── course-slug/
│           ├── index.html               # Add back nav here
│           └── styles.css               # Append nav CSS here
```

## Common Issues
- **404 on course**: Check `ls static/learn-ai/course-slug/index.html`
- **No back nav**: Verify HTML/CSS added after generation
- **Wrong catalog URL**: Use `/learn-ai/` not `/courses/`
- **Style mismatch**: Copy existing card block exactly, only change text/URLs

## Example
See: http://localhost:1313/learn-ai/vllm-triton-optimization/
