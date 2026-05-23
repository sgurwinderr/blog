# Interactive Courses

This directory contains interactive, single-page HTML courses generated from codebases.

## How to Generate a Course

### Using the codebase-to-course skill:

```bash
# From Claude Code, with a codebase directory open:
"Turn this codebase into a course"

# Or specify a codebase:
"Create a course from ~/path/to/project"

# Or from a GitHub repo:
"Make a course from https://github.com/user/repo"
```

The skill will:
1. Analyze the codebase deeply
2. Design a 4-6 module curriculum
3. Generate interactive HTML with animations
4. Create quizzes and plain-English explanations
5. Output to `course-name/index.html`

### Adding to the Blog

Once generated, copy the course to this directory:

```bash
# Move the generated course
cp -r /path/to/generated/course-name ~/blog/static/courses/

# The course will be accessible at:
# https://sgurwinderr.com/courses/course-name/
```

### Update the Courses Page

Edit `~/blog/content/courses.md` to add a link to your new course:

```markdown
### [Course Title](/courses/course-name/)
Brief description of what the course teaches.
- Module 1: ...
- Module 2: ...
```

## Course Structure

Each course is self-contained:

```
course-name/
├── index.html         # Main course (contains everything)
├── styles.css         # Course styling
├── main.js            # Interactive elements
└── modules/           # Optional: module breakdown
    ├── 01-intro.html
    └── ...
```

## Examples of Good Course Topics

From your blog's technical focus:

**GPU/Kernel Topics:**
- "How CUDA Kernels Work" (from a simple CUDA project)
- "Triton Compiler Internals" (from Triton source)
- "GPU Memory Hierarchy" (from a memory-bound kernel)

**ML Infrastructure:**
- "PyTorch Autograd from Inside" (from PyTorch core)
- "Flash Attention Implementation" (from the paper's code)
- "Quantization Techniques" (from ONNX runtime)

**Graphics:**
- "Ray Tracing Pipeline" (from a ray tracer codebase)
- "Gaussian Splatting Explained" (from 3DGS repo)

## Target Audience

Courses are designed for **"vibe coders"**:
- People who build with AI tools but want to understand internals
- Practical learners (not academic)
- Need enough knowledge to steer AI, debug, and make decisions

**NOT for:** Teaching people to write code from scratch

## Course Quality Checklist

Before publishing a course:
- [ ] All interactive elements work (quizzes, animations)
- [ ] Plain-English explanations for every technical concept
- [ ] At least one "aha!" insight per module
- [ ] Works offline (self-contained HTML)
- [ ] Mobile responsive
- [ ] Loads in <3 seconds

## Technical Details

**Dependencies:**
- Only external: Google Fonts CDN
- Everything else: embedded in HTML

**Browser support:**
- Modern browsers (Chrome, Firefox, Safari, Edge)
- Mobile: iOS Safari, Chrome Android

**File size:**
- Target: <500KB per course
- Includes: CSS, JS, and all content

---

**Skill location:** `~/.claude/skills/codebase-to-course/`

For skill documentation, see: `~/.claude/skills/codebase-to-course/SKILL.md`
