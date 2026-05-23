---
name: deploy-ai-course
description: Use when deploying interactive AI courses to Hugo blog Learn AI section with navigation, catalog, and data-driven course cards
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

### 4. Fix Module Number Visibility (CRITICAL)
**REQUIRED for all courses** - Module numbers must be visible.

Edit `static/learn-ai/course-slug/styles.css` and find `.module-number` block (around line 262):

```css
.module-number {
  display: block;
  font-family: var(--font-display);
  font-size: var(--text-6xl);
  font-weight: 800;
  color: var(--color-text);      /* ← Use text color, NOT accent */
  opacity: 0.8;                   /* ← Dark enough to see (NOT 0.15) */
  line-height: 1;
  margin-bottom: var(--space-2);
}
```

**Why:** Light accent colors (especially cyan/blue) with 15% opacity are invisible on white backgrounds. All courses must use dark text color at 80% opacity for module numbers (01, 02, etc.) to be readable.

### 5. Add to Course Data (NEW - Centralized)
Edit `data/courses.json` and add course entry:

```json
{
  "title": "Course Title",
  "url": "/learn-ai/course-slug/",
  "category": "Category Name",
  "summary": "Brief description of what the course teaches.",
  "gradient": "linear-gradient(135deg, #color1 0%, #color2 100%)",
  "icon": "emoji or text",
  "iconType": "emoji",
  "modules": "N Modules",
  "duration": "X Hours"
}
```

**Gradient colors by course type:**
- vLLM/Triton: `#667eea 0%, #764ba2 100%` (purple)
- GPU Optimization: `#f093fb 0%, #f5576c 100%` (pink/magenta)
- PyTorch: `#4facfe 0%, #00f2fe 100%` (cyan/blue)
- New courses: Pick from these or create new gradient

**Icon types:**
- `"iconType": "text"` - For text like "vLLM", "GPU"
- `"iconType": "emoji"` - For emoji like ⚡, 🔥, 📚

This JSON file is used by:
- `layouts/index.html` - Learn AI section on homepage
- `layouts/section/learn-ai.html` - Course catalog page

**Both pages render from the same data** - no need to update HTML templates manually.

### 6. Verify Course Structure
Check these files exist:
```bash
ls static/learn-ai/course-slug/index.html    # Main course
ls static/learn-ai/course-slug/styles.css    # Styling
ls static/learn-ai/course-slug/main.js       # Interactivity (optional)
```

Verify HTML structure:
- Each module: `<section class="module" id="module-N">`
- Module header: `<div class="module-content">` → `<header class="module-header">` or `<div class="module-header">`
- Module content: `<div class="module-body">` contains all content
- All divs close properly (no orphaned tags)

### 7. Test
```bash
cd ~/blog
hugo server -D --bind 0.0.0.0 --port 8080

# Test these URLs:
# http://localhost:8080/ - Check Learn AI section on homepage
# http://localhost:8080/learn-ai/ - Check catalog page
# http://localhost:8080/learn-ai/course-slug/ - Check course
```

**Checklist:**
- [ ] Course card appears on homepage
- [ ] Course card appears on /learn-ai/ catalog
- [ ] Module numbers (01, 02...) are visible and dark
- [ ] Back navigation works (Blog and All Courses links)
- [ ] Navigation dots at top work
- [ ] Smooth scrolling between modules
- [ ] All interactive elements work (quizzes, animations)

### 8. Commit (Local Only - No Auto-Push)
```bash
git add static/learn-ai/course-slug/ data/courses.json
git commit -m "feat: add [Course Title] to Learn AI

- N modules on [topic]
- Verified module numbers visible
- Added to courses.json for homepage/catalog
- Target: [audience]"

# DO NOT push without approval
# User will review and approve before git push origin master
```

## File Structure
```
blog/
├── data/
│   └── courses.json                     # Central course metadata (NEW)
├── layouts/
│   ├── index.html                       # Homepage (renders from courses.json)
│   └── section/learn-ai.html            # Catalog (renders from courses.json)
├── static/
│   ├── css/learn-ai.css                 # Catalog page styles
│   └── learn-ai/
│       ├── vllm-triton-optimization/
│       ├── gpu-optimization-prs/
│       ├── pytorch-optimization-prs/
│       └── course-slug/                 # New course
│           ├── index.html               # Course content
│           ├── styles.css               # Course styles (check module-number!)
│           └── main.js                  # Interactive elements (optional)
```

## Course Styling Standards

All courses must follow these standards:

### 1. Module Numbers Visibility
```css
.module-number {
  color: var(--color-text);  /* Dark text color */
  opacity: 0.8;               /* Visible opacity */
}
```

### 2. White Background with Smooth Scrolling
```css
:root {
  --color-bg: #FFFFFF;
}

html {
  scroll-behavior: smooth;
}
```

### 3. Module Structure
```html
<section class="module" id="module-N">
  <div class="module-content">
    <header class="module-header">
      <span class="module-number">0N</span>
      <h1 class="module-title">Title</h1>
    </header>
    <div class="module-body">
      <!-- All module content here -->
    </div>
  </div>
</section>
```

### 4. Module Height
```css
.module {
  min-height: 100dvh;
  min-height: 100vh;  /* Fallback */
}
```

## Common Issues

### Module Numbers Not Visible
**Problem:** Numbers appear too light or invisible
**Fix:** Edit `styles.css` line ~262:
```css
.module-number {
  color: var(--color-text);  /* NOT var(--color-accent) */
  opacity: 0.8;               /* NOT 0.15 */
}
```

### Course Not Showing on Homepage/Catalog
**Problem:** Card missing from Learn AI section
**Fix:** Add course to `data/courses.json`

### Layout Broken/Content Pushed Left
**Problem:** Improper HTML div nesting
**Fix:** Verify all module divs close in correct order:
```html
</div>  <!-- close pattern-grid or content -->
</div>  <!-- close module-body -->
</div>  <!-- close module-content -->
</section>  <!-- close module -->
```

### Back Navigation Missing
**Problem:** No way to return to catalog
**Fix:** Add back-to-blog HTML and CSS after generation

### Style Inconsistency Between Courses
**Problem:** Different fonts, colors, spacing
**Fix:** Copy `styles.css` from reference course (vLLM) and customize only:
- `:root` accent colors
- Course-specific content (not structure)

## Examples
- vLLM Triton: http://localhost:8080/learn-ai/vllm-triton-optimization/
- GPU Optimization: http://localhost:8080/learn-ai/gpu-optimization-prs/
- PyTorch Optimization: http://localhost:8080/learn-ai/pytorch-optimization-prs/

## Deployment Workflow
1. **Generate course** using codebase-to-course skill
2. **Move to blog** static directory
3. **Add back navigation** HTML/CSS
4. **Fix module numbers** in styles.css
5. **Add to courses.json** with metadata
6. **Test locally** on all pages
7. **Commit** (do not push)
8. **Get approval** from user
9. **Push** only after approval
