---
name: deploying-hugo-courses
description: Use when deploying interactive courses to Hugo blog with Learn AI section, codebase-to-course skill, and back navigation
---

# Deploying Interactive Courses to Hugo Blog

## Overview

Complete workflow for generating and deploying interactive courses to a Hugo blog's Learn AI section. Uses codebase-to-course skill to generate courses from codebases, then integrates them with proper navigation and catalog listing.

## When to Use

Use this skill when:
- Deploying a new interactive course to the blog
- You have a codebase to transform into a course
- Need to add course to the Learn AI catalog
- Want course pages to have back-to-blog navigation

## Prerequisites

- Hugo blog with Learn AI section at `/learn-ai/`
- codebase-to-course skill installed in `.skills/`
- Course catalog layout at `layouts/section/learn-ai.html`
- Hugo server accessible for testing

## Deployment Workflow

### Step 1: Clone Source Repository

```bash
cd /tmp
git clone --depth 1 https://github.com/org/repo.git repo-name
```

Verify repo structure:
```bash
ls -la /tmp/repo-name/
```

### Step 2: Create Course Generation Prompt

Create `/tmp/course-prompt.md` with:

```markdown
# [Course Title] Course Generation

## Overview
[One paragraph describing the course goal]

## Repository
Source: /tmp/repo-name

## Course Structure

### Module 1: [Title]
**Files to analyze:**
- path/to/relevant/file.py
- path/to/relevant/file2.py

**Key concepts:**
- Concept 1
- Concept 2

**Interactive elements:**
- Animation: [description]
- Code display: [what to show]
- Quiz: [example question]

### Module 2-6: [Repeat structure]

## Output Requirements
1. Single HTML file (or directory with index.html)
2. Scroll-based navigation (4-6 modules)
3. 15+ animations
4. 15+ quizzes
5. 15+ code displays
6. Interactive tools

## Style
- Target audience: [audience description]
- Depth: [level of detail]
- Tone: Technical, rigorous, educational
```

### Step 3: Generate Course

Use codebase-to-course skill:

```
"Using the codebase-to-course skill, generate an interactive course from 
the repository at /tmp/repo-name. Follow the structure in 
/tmp/course-prompt.md. Generate a comprehensive course with [N] modules, 
15+ animations, 15+ quizzes, and full code displays."
```

The skill will output course files (HTML, CSS, JS, modules).

### Step 4: Move Course to Blog

```bash
# Create course directory
mkdir -p /home/gurwinde/blog/static/learn-ai/course-slug/

# Copy generated course
cp -r [generated-course-path]/* /home/gurwinde/blog/static/learn-ai/course-slug/

# Verify
ls -la /home/gurwinde/blog/static/learn-ai/course-slug/
```

### Step 5: Add Back-to-Blog Navigation

Edit `/home/gurwinde/blog/static/learn-ai/course-slug/index.html`:

Add immediately after `<body>` tag:

```html
  <!-- Back to Blog Header -->
  <div class="back-to-blog">
    <a href="/" class="back-link">
      <svg width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
        <path fill-rule="evenodd" d="M15 8a.5.5 0 0 0-.5-.5H2.707l3.147-3.146a.5.5 0 1 0-.708-.708l-4 4a.5.5 0 0 0 0 .708l4 4a.5.5 0 0 0 .708-.708L2.707 8.5H14.5A.5.5 0 0 0 15 8z"/>
      </svg>
      <span>Back to Blog</span>
    </a>
    <a href="/learn-ai/" class="back-link ml-auto">
      <span>All Courses</span>
    </a>
  </div>
```

Add CSS to `/home/gurwinde/blog/static/learn-ai/course-slug/styles.css`:

```css
/* Back to Blog Header */
.back-to-blog {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 50px;
  background: var(--color-surface);
  border-bottom: 1px solid var(--color-border);
  display: flex;
  align-items: center;
  padding: 0 var(--space-6);
  z-index: 999;
  font-family: var(--font-body);
}

.back-link {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  color: var(--color-text-secondary);
  text-decoration: none;
  font-size: var(--text-sm);
  font-weight: 500;
  transition: color 0.2s ease;
}

.back-link:hover {
  color: var(--color-accent);
}

.back-link svg {
  transition: transform 0.2s ease;
}

.back-link:hover svg {
  transform: translateX(-2px);
}

.ml-auto {
  margin-left: auto;
}

/* Adjust nav positioning */
.nav {
  top: 50px;
}

/* Adjust main content padding */
body {
  padding-top: 50px;
}
```

### Step 6: Add Course to Catalog

Edit `/home/gurwinde/blog/layouts/section/learn-ai.html`.

Add course card to the grid (duplicate existing card and modify):

```html
<!-- [Course Name] -->
<div class="col-xl-4 col-lg-4 col-md-6 mb-30px card-group">
    <div class="card h-100">
        <div class="maxthumb">
            <a href="/learn-ai/course-slug/">
                <div class="course-thumb" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); height: 200px; display: flex; align-items: center; justify-content: center; color: white; font-size: 48px; font-weight: bold;">
                    LOGO
                </div>
            </a>
        </div>
        <div class="card-body">
            <div class="featured-card-cat">
                <a href="#">Category Name</a>
            </div>
            <h2 class="card-title">
                <a class="text-dark" href="/learn-ai/course-slug/">
                    Course Title
                </a>
            </h2>
            <h4 class="card-text card-summary">
                One-sentence course description here.
            </h4>
            <div class="course-meta mt-3">
                <span class="badge badge-secondary">N Modules</span>
                <span class="badge badge-secondary">X Hours</span>
                <span class="badge badge-secondary">Level</span>
            </div>
            <div class="course-topics mt-3">
                <small class="text-muted">
                    <strong>Topics:</strong> Topic1, Topic2, Topic3
                </small>
            </div>
        </div>
        <div class="card-footer bg-white">
            <div class="post-card-meta">
                {{ if isset .Site.Params.author "thumbnail" }}
                <img class="author-thumb" src="{{ .Site.Params.author.thumbnail | urlize | relURL }}" alt="{{ .Site.Params.author.name }}">
                {{ end }}
                <div class="post-card-meta-text">
                    {{ if isset .Site.Params.author "name" }}
                    <span class="post-name">{{ .Site.Params.author.name }}</span>
                    {{ end }}
                    <span class="post-date">Date &middot; Interactive</span>
                </div>
                <span class="post-read-more">
                    <a href="/learn-ai/course-slug/" title="Launch Course">
                        <svg class="svgIcon-use" width="25" height="25" viewbox="0 0 25 25">
                            <path d="M19 6c0-1.1-.9-2-2-2H8c-1.1 0-2 .9-2 2v14.66h.012c.01.103.045.204.12.285a.5.5 0 0 0 .706.03L12.5 16.85l5.662 4.126a.508.508 0 0 0 .708-.03.5.5 0 0 0 .118-.285H19V6zm-6.838 9.97L7 19.636V6c0-.55.45-1 1-1h9c.55 0 1 .45 1 1v13.637l-5.162-3.668a.49.49 0 0 0-.676 0z" fill-rule="evenodd"></path>
                        </svg>
                    </a>
                </span>
            </div>
        </div>
    </div>
</div>
<!-- End Course -->
```

Customize:
- `course-slug` - URL-friendly course name
- `LOGO` - Short text or emoji for thumbnail
- Gradient colors (optional)
- Title, description, metadata
- Topics list

### Step 7: Test Locally

Start Hugo server (if not running):
```bash
cd /home/gurwinde/blog
hugo server -D --bind 0.0.0.0 --port 1313
```

Test:
1. **Catalog page:** http://localhost:1313/learn-ai/
   - Verify new course card appears
   - Check title, description, badges

2. **Course page:** http://localhost:1313/learn-ai/course-slug/
   - Verify course loads
   - Check back navigation (Back to Blog, All Courses links)
   - Test module navigation
   - Verify animations work
   - Test quizzes

3. **Navigation:**
   - Click "Back to Blog" → should go to homepage
   - Click "All Courses" → should go to catalog
   - From catalog, click course card → loads course

### Step 8: Commit Changes

```bash
cd /home/gurwinde/blog

# Check what changed
git status

# Stage course files
git add static/learn-ai/course-slug/
git add layouts/section/learn-ai.html

# Commit
git commit -m "feat: add [Course Title] course

- N-module course on [topic]
- Focus on [key aspects]
- Covers: [topic1, topic2, topic3]
- 15+ animations, 15+ quizzes, full code walkthroughs
- Target audience: [audience]

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Step 9: Cleanup

```bash
# Remove temporary files
rm -rf /tmp/repo-name
rm -f /tmp/course-prompt.md
```

## Quick Reference

| Step | Command/Action |
|------|----------------|
| Clone repo | `git clone --depth 1 [url] /tmp/repo-name` |
| Generate course | Use codebase-to-course skill with prompt |
| Move course | `cp -r [source]/* static/learn-ai/course-slug/` |
| Add navigation | Edit index.html + styles.css |
| Add to catalog | Edit layouts/section/learn-ai.html |
| Test | Visit http://localhost:1313/learn-ai/course-slug/ |
| Commit | `git add ... && git commit` |
| Cleanup | `rm -rf /tmp/repo-name /tmp/course-prompt.md` |

## File Locations Reference

```
blog/
├── content/
│   └── learn-ai/
│       └── _index.md                    # Catalog page content
├── layouts/
│   └── section/
│       └── learn-ai.html                # Catalog page layout
├── static/
│   ├── css/
│   │   └── learn-ai.css                 # Catalog styles
│   └── learn-ai/
│       └── course-slug/
│           ├── index.html               # Course main file
│           ├── styles.css               # Course styles
│           ├── main.js                  # Course JS
│           └── modules/                 # Course modules
│               ├── 01-module.html
│               ├── 02-module.html
│               └── ...
```

## Common Mistakes

### Missing Back Navigation
**Problem:** Course pages don't have back links to blog/catalog

**Solution:** Always add back-to-blog div + CSS after generating course. Verify by checking course HTML in browser.

### Wrong URL in Catalog
**Problem:** Course card links to `/courses/` instead of `/learn-ai/`

**Solution:** Use `/learn-ai/course-slug/` for all course URLs (catalog links, back navigation).

### Course Not Accessible
**Problem:** 404 when visiting course URL

**Solution:** 
- Verify course directory exists: `ls static/learn-ai/course-slug/`
- Verify index.html exists
- Check Hugo server output for errors
- Restart Hugo server if needed

### Navigation Header Not Styled
**Problem:** Back links appear but look broken

**Solution:** Ensure CSS was added to styles.css. Check browser console for CSS errors. Verify CSS uses correct variable names.

### Catalog Card Doesn't Match Style
**Problem:** New course card looks different from existing cards

**Solution:** Duplicate an existing card block exactly, then only modify:
- URLs (`href` attributes)
- Text content (title, description, topics)
- Gradient colors (optional)
- Badge text (modules, hours, level)

## Customization Options

### Course Thumbnail Gradient

Change gradient in catalog card:
```html
style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);"
```

Popular gradients:
- Purple: `#667eea → #764ba2`
- Blue: `#2A7B9B → #1F6280`
- Green: `#2D8B55 → #1F6B3F`
- Orange: `#D94F30 → #C4432A`

### Course Badge Levels
- `Beginner` - Introductory content
- `Intermediate` - Requires some background
- `Advanced` - Expert-level content

### Module Count
Update based on actual course structure:
```html
<span class="badge badge-secondary">6 Modules</span>
```

## Integration with Blog

Courses integrate seamlessly with blog:
- **Navigation:** Learn AI menu item in header
- **Styling:** Uses same fonts, colors, spacing
- **Layout:** Card-based catalog matches blog post grid
- **Author info:** Shows same author thumbnail and name

## Real-World Example

See the vLLM Triton Optimization course at:
- **Catalog:** http://localhost:1313/learn-ai/
- **Course:** http://localhost:1313/learn-ai/vllm-triton-optimization/
- **Files:** `/home/gurwinde/blog/static/learn-ai/vllm-triton-optimization/`
