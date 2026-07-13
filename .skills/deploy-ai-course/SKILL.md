---
name: deploy-ai-course
description: Deploy or update interactive Learn AI courses in this Hugo blog. Use when moving generated course files into `static/learn-ai`, adding course metadata to `data/courses.json`, fixing course navigation/styling, or validating course pages in the blog.
---

# Deploy AI Course

Deploy courses as static assets under `static/learn-ai/<course-slug>/` and register them in `data/courses.json`. Keep generated course internals intact unless a blog integration fix is required.

## Workflow

1. Confirm the course slug, title, category, module count, duration, and source directory.
2. Copy or update files under `static/learn-ai/<course-slug>/`.
3. Ensure these files exist:

```text
static/learn-ai/<course-slug>/index.html
static/learn-ai/<course-slug>/styles.css
static/learn-ai/<course-slug>/main.js    # optional, only if the course needs JS
```

4. Add or update `data/courses.json`:

```json
{
  "title": "Course Title",
  "url": "/learn-ai/course-slug/",
  "category": "Category Name",
  "summary": "Brief description of what the course teaches.",
  "gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
  "icon": "GPU",
  "iconType": "text",
  "modules": "8 Modules",
  "duration": "1-2 Hours"
}
```

5. Add top navigation inside the course if missing:

```html
<div class="back-to-blog">
  <a href="/" class="back-link">Back to Blog</a>
  <a href="/learn-ai/" class="back-link ml-auto">All Courses</a>
</div>
```

6. Add compatible CSS if missing:

```css
.back-to-blog { position: fixed; top: 0; left: 0; right: 0; height: 50px; background: var(--color-surface); border-bottom: 1px solid var(--color-border); display: flex; align-items: center; padding: 0 var(--space-6); z-index: 999; font-family: var(--font-body); }
.back-link { display: flex; align-items: center; gap: var(--space-2); color: var(--color-text-secondary); text-decoration: none; font-size: var(--text-sm); font-weight: 500; }
.back-link:hover { color: var(--color-accent); }
.ml-auto { margin-left: auto; }
.nav { top: 50px; }
body { padding-top: 50px; }
```

7. Fix module number visibility:

```css
.module-number {
  color: var(--color-text);
  opacity: 0.8;
}
```

8. Build and preview:

```bash
hugo --minify --cleanDestinationDir
hugo server -D --bind 0.0.0.0 --port 1313
```

Check:

- `/` shows the course card.
- `/learn-ai/` shows the course card.
- `/learn-ai/<course-slug>/` loads without layout breakage.
- Module numbers are visible.
- Back links work.
- Navigation dots, quizzes, canvases, and animations still work.

## Course Structure Standards

Expected module structure:

```html
<section class="module" id="module-1">
  <div class="module-content">
    <header class="module-header">
      <span class="module-number">01</span>
      <h1 class="module-title">Title</h1>
    </header>
    <div class="module-body">
      <!-- module content -->
    </div>
  </div>
</section>
```

Use `min-height: 100dvh` with a `100vh` fallback for full-screen course modules. Preserve existing course typography, spacing, and interaction conventions unless the user asks for a redesign.

## Common Problems

- Course missing from catalog: fix `data/courses.json`.
- Module numbers invisible: use `color: var(--color-text)` and `opacity: 0.8`.
- Content shifted or broken: check unclosed module/body/content divs.
- Back navigation hidden: check `z-index`, fixed top offset, and `body` padding.
- Static assets 404: use paths relative to `/learn-ai/<course-slug>/` or absolute `/learn-ai/<course-slug>/asset.ext` paths.

Commit only when the user explicitly asks. Never push without explicit approval.
