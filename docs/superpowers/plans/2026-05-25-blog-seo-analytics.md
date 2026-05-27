# Blog SEO and Analytics Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add comprehensive SEO enhancements (meta tags, structured data, breadcrumbs) and GA4 custom event tracking to Hugo blog for better search visibility and engagement insights.

**Architecture:** Layer SEO enhancements into Hugo's template hierarchy without modifying the base theme. Add partials for structured data (JSON-LD schemas), analytics events, and breadcrumb navigation. Update head.html with enhanced meta tags and resource hints. All changes are backward compatible with intelligent defaults.

**Tech Stack:** Hugo static site generator, HTML/Hugo templates, JSON-LD structured data, Google Analytics 4, vanilla JavaScript

---

## File Structure

**Files to create:**
- `layouts/partials/_shared/breadcrumb-schema.html` - BreadcrumbList JSON-LD schema
- `layouts/partials/_shared/organization-schema.html` - Person/Organization JSON-LD schema
- `layouts/partials/_shared/analytics-events.html` - GA4 custom event tracking script
- `layouts/partials/_shared/breadcrumb-nav.html` - Visual breadcrumb navigation component

**Files to modify:**
- `layouts/partials/_shared/head.html` - Add enhanced meta tags, resource hints, schema includes
- `layouts/partials/_shared/schema.html` - Upgrade Article to TechArticle with extended metadata
- `layouts/_default/single.html` - Add alt text to featured image, include breadcrumb nav and analytics
- `assets/css/custom.css` - Add breadcrumb styling
- `CLAUDE.md` - Add SEO guidelines and monitoring checklist

**Files to verify (no changes needed unless issues found):**
- `hugo.toml` - Verify sitemap and analytics config
- `static/robots.txt` - Verify current setup

---

### Task 1: Enhanced Meta Tags and Resource Hints

**Files:**
- Modify: `layouts/partials/_shared/head.html`

- [ ] **Step 1: Read current head.html file**

```bash
cat layouts/partials/_shared/head.html
```

Expected: See existing meta tags, OG tags, and CSS/JS includes

- [ ] **Step 2: Add resource hints after opening `<head>` tag**

Add after line 1 (`<head data-baseurl="{{ .Site.BaseURL }}">`), before theme-color meta:

```html
<head data-baseurl="{{ .Site.BaseURL }}">
	<meta charset="utf-8">
	<meta http-equiv="X-UA-Compatible" content="IE=edge">
	<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
	<meta name="theme-color" content="#038252">
	
	<!-- Preconnect to external domains for performance -->
	<link rel="preconnect" href="https://fonts.googleapis.com">
	<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
	<link rel="preconnect" href="https://cdn.jsdelivr.net">
	<link rel="dns-prefetch" href="https://www.googletagmanager.com">
	<link rel="dns-prefetch" href="https://www.google-analytics.com">
	
	<!-- Apply dark mode before first paint to avoid flash -->
	<script>
		(function(){
			var t=localStorage.getItem('color-scheme')||'light';
			if(t==='dark')document.documentElement.setAttribute('data-theme','dark');
		})();
	</script>
```

- [ ] **Step 3: Replace existing description and keywords meta tags**

Find the section with description and keywords meta tags (around line 20-21). Replace:

```html
{{ with .Site.Params.meta.description }}<meta name="description" content="{{ . }}">{{ end }}
{{ with .Site.Params.meta.keywords }}<meta name="keywords" content="{{ . }}">{{ end }}
```

With:

```html
<!-- SEO meta tags -->
{{ if .IsPage }}
  <meta name="description" content="{{ with .Description }}{{ . }}{{ else }}{{ .Content | plainify | truncate 160 }}{{ end }}">
  <meta name="author" content="{{ .Site.Params.author.name }}">
  {{ with .Params.categories }}
    <meta name="keywords" content="{{ delimit . ", " }}{{ with $.Params.tags }}, {{ delimit . ", " }}{{ end }}">
  {{ end }}
{{ else }}
  <meta name="description" content="{{ .Site.Params.description }}">
{{ end }}
```

- [ ] **Step 4: Add enhanced Open Graph article tags**

Find the existing Open Graph section (after line 32). After the existing OG tags and before the Twitter tags, add:

```html
{{ if .IsPage }}
  <meta property="article:published_time" content="{{ .Date.Format "2006-01-02T15:04:05Z07:00" }}" />
  {{ with .Lastmod }}
    <meta property="article:modified_time" content="{{ .Format "2006-01-02T15:04:05Z07:00" }}" />
  {{ end }}
  <meta property="article:author" content="{{ .Site.Params.author.name }}" />
  {{ range .Params.categories }}
    <meta property="article:section" content="{{ . }}" />
  {{ end }}
  {{ range .Params.tags }}
    <meta property="article:tag" content="{{ . }}" />
  {{ end }}
{{ end }}

<!-- Site-level OG tags -->
<meta property="og:site_name" content="{{ .Site.Title }}" />
```

- [ ] **Step 5: Add organization schema include before closing `</head>`**

Before the closing `</head>` tag (at the end of the file), add:

```html
	<!-- Organization/Person structured data -->
	{{- partial "_shared/organization-schema.html" . -}}
</head>
```

- [ ] **Step 6: Build site and verify no errors**

```bash
cd /home/gurwinde/blog
hugo server -D --bind 0.0.0.0 --port 1313
```

Expected: Server starts without errors, site builds successfully

- [ ] **Step 7: Commit changes**

```bash
git add layouts/partials/_shared/head.html
git commit -m "feat(seo): add enhanced meta tags and resource hints

- Add preconnect/dns-prefetch for external domains
- Enhanced description and keywords meta tags with fallbacks
- Add article-specific Open Graph tags (published_time, author, section, tags)
- Add og:site_name for brand consistency
- Include organization schema partial

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2: Organization Schema (JSON-LD)

**Files:**
- Create: `layouts/partials/_shared/organization-schema.html`

- [ ] **Step 1: Create organization schema partial**

```bash
touch layouts/partials/_shared/organization-schema.html
```

- [ ] **Step 2: Write organization schema content**

Write to `layouts/partials/_shared/organization-schema.html`:

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Person",
  "name": "Gurwinder",
  "url": "{{ .Site.BaseURL }}",
  "description": "{{ .Site.Params.description }}",
  "jobTitle": "GPU Software Development Engineer",
  "worksFor": {
    "@type": "Organization",
    "name": "Intel Corporation"
  },
  "sameAs": [
    "https://linkedin.com/in/{{ .Site.Params.social.linkedin }}",
    "https://github.com/{{ .Site.Params.social.github }}"
  ],
  "image": "{{ .Site.BaseURL }}{{ .Site.Params.author.thumbnail }}"
}
</script>
```

- [ ] **Step 3: Build site and check for errors**

```bash
hugo server -D --bind 0.0.0.0 --port 1313
```

Expected: No build errors, organization schema included in all pages

- [ ] **Step 4: Test in browser**

Open http://localhost:1313/ in browser, view source, search for "schema.org" and verify Person schema appears in `<head>`

Expected: JSON-LD script with Person type and correct data

- [ ] **Step 5: Commit**

```bash
git add layouts/partials/_shared/organization-schema.html
git commit -m "feat(seo): add organization/person structured data

- JSON-LD schema with Person type
- Include job title, affiliation, social profiles
- Site-wide inclusion via head.html

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 3: Enhanced Article Schema (TechArticle)

**Files:**
- Modify: `layouts/partials/_shared/schema.html`

- [ ] **Step 1: Read current schema.html**

```bash
cat layouts/partials/_shared/schema.html
```

Expected: See current Article schema implementation

- [ ] **Step 2: Replace entire schema.html content**

Replace the entire content with enhanced TechArticle schema:

```html
{{- if .IsPage -}}
<!-- Schema.org TechArticle Structured Data -->
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": {{ .Title | jsonify }},
  "description": {{ with .Description }}{{ . | jsonify }}{{ else }}{{ .Summary | plainify | jsonify }}{{ end }},
  "image": {{- if .Params.image }}{{ .Params.image | absURL | jsonify }}{{- else }}"{{ .Site.BaseURL }}assets/images/logo.png"{{- end }},
  "datePublished": "{{ .Date.Format "2006-01-02T15:04:05Z07:00" }}",
  {{ with .Lastmod }}
  "dateModified": "{{ .Format "2006-01-02T15:04:05Z07:00" }}",
  {{ end }}
  "author": {
    "@type": "Person",
    "name": {{ .Site.Params.author.name | jsonify }},
    "url": "{{ .Site.BaseURL }}",
    "jobTitle": "GPU Software Development Engineer",
    "affiliation": {
      "@type": "Organization",
      "name": "Intel Corporation"
    }
  },
  "publisher": {
    "@type": "Person",
    "name": {{ .Site.Params.author.name | jsonify }},
    "url": "{{ .Site.BaseURL }}"
  },
  "mainEntityOfPage": {
    "@type": "WebPage",
    "@id": {{ .Permalink | jsonify }}
  },
  "proficiencyLevel": "Expert",
  {{ with .Params.categories }}
  "articleSection": {{ index . 0 | jsonify }},
  {{ end }}
  {{ with .Params.tags }}
  "keywords": {{ delimit . ", " | jsonify }},
  {{ end }}
  "wordCount": {{ .WordCount }},
  "timeRequired": "PT{{ .ReadingTime }}M"
}
</script>
{{- end -}}
```

- [ ] **Step 3: Build and verify**

```bash
hugo server -D --bind 0.0.0.0 --port 1313
```

Expected: No errors, site builds successfully

- [ ] **Step 4: Test schema in browser**

Open a blog post (e.g., http://localhost:1313/turboquant/), view source, verify TechArticle schema with proficiencyLevel, articleSection, wordCount, timeRequired

Expected: Complete TechArticle JSON-LD with all fields

- [ ] **Step 5: Validate schema online**

Copy a post URL and test at https://validator.schema.org/

Expected: Valid TechArticle schema with no errors

- [ ] **Step 6: Commit**

```bash
git add layouts/partials/_shared/schema.html
git commit -m "feat(seo): upgrade Article schema to TechArticle

- Change @type from Article to TechArticle
- Add proficiencyLevel field (Expert)
- Add articleSection from primary category
- Include wordCount and timeRequired
- Enhanced author with jobTitle and affiliation
- Description fallback to Summary if not in frontmatter

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 4: Breadcrumb Schema (JSON-LD)

**Files:**
- Create: `layouts/partials/_shared/breadcrumb-schema.html`

- [ ] **Step 1: Create breadcrumb schema partial**

```bash
touch layouts/partials/_shared/breadcrumb-schema.html
```

- [ ] **Step 2: Write breadcrumb schema content**

Write to `layouts/partials/_shared/breadcrumb-schema.html`:

```html
{{- if .IsPage -}}
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "BreadcrumbList",
  "itemListElement": [
    {
      "@type": "ListItem",
      "position": 1,
      "name": "Home",
      "item": "{{ .Site.BaseURL }}"
    },
    {{- if .Params.categories -}}
    {
      "@type": "ListItem",
      "position": 2,
      "name": {{ (index .Params.categories 0) | jsonify }},
      "item": "{{ .Site.BaseURL }}categories/{{ (index .Params.categories 0) | urlize }}/"
    },
    {{- end -}}
    {
      "@type": "ListItem",
      "position": {{ if .Params.categories }}3{{ else }}2{{ end }},
      "name": {{ .Title | jsonify }},
      "item": {{ .Permalink | jsonify }}
    }
  ]
}
</script>
{{- end -}}
```

- [ ] **Step 3: Include in single.html head section**

Edit `layouts/_default/single.html`. Find the `{{ define "header"}}` section and after it (before `{{ define "main" }}`), add:

```html
{{ define "header"}}
    {{- partial "_shared/navbar.html" . -}}
    {{- partial "_shared/breadcrumb-schema.html" . -}}
{{ end }}
```

- [ ] **Step 4: Build and test**

```bash
hugo server -D --bind 0.0.0.0 --port 1313
```

Expected: No errors

- [ ] **Step 5: Verify breadcrumb schema in browser**

Open a post, view source, search for "BreadcrumbList"

Expected: JSON-LD with Home → Category → Post breadcrumb structure

- [ ] **Step 6: Validate with Rich Results Test**

Test a post URL at https://search.google.com/test/rich-results

Expected: BreadcrumbList detected and valid

- [ ] **Step 7: Commit**

```bash
git add layouts/partials/_shared/breadcrumb-schema.html layouts/_default/single.html
git commit -m "feat(seo): add breadcrumb structured data

- Create BreadcrumbList JSON-LD schema
- Dynamic position based on category presence
- Include in single.html for all posts
- Enables breadcrumb rich results in search

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 5: Visual Breadcrumb Navigation

**Files:**
- Create: `layouts/partials/_shared/breadcrumb-nav.html`
- Modify: `layouts/_default/single.html`
- Modify: `assets/css/custom.css`

- [ ] **Step 1: Create breadcrumb nav partial**

```bash
touch layouts/partials/_shared/breadcrumb-nav.html
```

- [ ] **Step 2: Write breadcrumb nav content**

Write to `layouts/partials/_shared/breadcrumb-nav.html`:

```html
<nav aria-label="breadcrumb" class="breadcrumb-nav">
  <ol class="breadcrumb">
    <li class="breadcrumb-item"><a href="{{ .Site.BaseURL }}">Home</a></li>
    {{ if .Params.categories }}
      {{ $category := index .Params.categories 0 }}
      <li class="breadcrumb-item"><a href="{{ .Site.BaseURL }}categories/{{ $category | urlize }}/">{{ $category }}</a></li>
    {{ end }}
    <li class="breadcrumb-item active" aria-current="page">{{ .Title }}</li>
  </ol>
</nav>
```

- [ ] **Step 3: Add breadcrumb CSS styling**

Edit `assets/css/custom.css` and add at the end:

```css
/* Breadcrumb navigation */
.breadcrumb-nav {
  padding: 1rem 0;
  font-size: 0.9rem;
}

.breadcrumb {
  background: none;
  padding: 0;
  margin: 0;
  list-style: none;
  display: flex;
  flex-wrap: wrap;
}

.breadcrumb-item + .breadcrumb-item::before {
  content: "›";
  padding: 0 0.5rem;
  color: #6c757d;
}

.breadcrumb-item a {
  color: #038252;
  text-decoration: none;
}

.breadcrumb-item a:hover {
  text-decoration: underline;
}

.breadcrumb-item.active {
  color: #6c757d;
}
```

- [ ] **Step 4: Include breadcrumb in single.html**

Edit `layouts/_default/single.html`. Find `<div class="container">` in the main section (around line 8), and add breadcrumb partial right after:

```html
{{ define "main" }}
    <div class="main-content">
        <!-- Begin Article -->
        <div class="container">
            {{- partial "_shared/breadcrumb-nav.html" . -}}
            <div class="row">
```

- [ ] **Step 5: Build and test in browser**

```bash
hugo server -D --bind 0.0.0.0 --port 1313
```

Open a blog post, verify breadcrumb appears above the post content: "Home › AI › Post Title"

Expected: Breadcrumb navigation visible and clickable

- [ ] **Step 6: Test breadcrumb links**

Click on "Home" and category links in breadcrumb

Expected: Navigation works correctly

- [ ] **Step 7: Commit**

```bash
git add layouts/partials/_shared/breadcrumb-nav.html layouts/_default/single.html assets/css/custom.css
git commit -m "feat(seo): add visual breadcrumb navigation

- Create breadcrumb-nav.html partial
- Display Home > Category > Post breadcrumbs
- Add CSS styling matching site theme
- Include in single.html above post content
- Improves navigation and SEO

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 6: Featured Image Alt Text Enhancement

**Files:**
- Modify: `layouts/_default/single.html`

- [ ] **Step 1: Find featured image code in single.html**

```bash
grep -n "featured-image" layouts/_default/single.html
```

Expected: Shows line number of featured image (around line 58)

- [ ] **Step 2: Update featured image with alt text and fetchpriority**

Find this code:

```html
{{ with .Params.image }}
    <img class="featured-image img-fluid" src="{{ . | relURL }}" alt="thumbnail for this post" loading="lazy" decoding="async">
{{ end }}
```

Replace with:

```html
{{ with .Params.image }}
    <img class="featured-image img-fluid" 
         src="{{ . | relURL }}" 
         alt="{{ with $.Params.imageAlt }}{{ . }}{{ else }}Featured image for {{ $.Title }}{{ end }}" 
         loading="lazy" 
         decoding="async"
         fetchpriority="high">
{{ end }}
```

- [ ] **Step 3: Build and verify**

```bash
hugo server -D --bind 0.0.0.0 --port 1313
```

Expected: No errors

- [ ] **Step 4: Test in browser**

Open a blog post with featured image, inspect image element, verify alt attribute uses intelligent fallback

Expected: alt="Featured image for [Post Title]"

- [ ] **Step 5: Commit**

```bash
git add layouts/_default/single.html
git commit -m "feat(seo): enhance featured image alt text

- Add imageAlt frontmatter field support
- Intelligent fallback to 'Featured image for [Title]'
- Add fetchpriority='high' for LCP optimization
- Improves accessibility and SEO

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 7: GA4 Custom Events Tracking

**Files:**
- Create: `layouts/partials/_shared/analytics-events.html`
- Modify: `layouts/_default/single.html`

- [ ] **Step 1: Create analytics events partial**

```bash
touch layouts/partials/_shared/analytics-events.html
```

- [ ] **Step 2: Write analytics events script**

Write to `layouts/partials/_shared/analytics-events.html`:

```html
<script defer>
(function() {
  // Respect Do Not Track
  if (navigator.doNotTrack === "1" || window.doNotTrack === "1") {
    return;
  }

  // Wait for GA4 to load
  window.addEventListener('load', function() {
    if (typeof gtag === 'undefined') return;

    // Track outbound link clicks
    document.querySelectorAll('a[href^="http"]').forEach(function(link) {
      if (!link.href.includes(location.hostname)) {
        link.addEventListener('click', function(e) {
          var label = 'unknown';
          if (link.href.includes('linkedin.com')) label = 'linkedin_profile';
          else if (link.href.includes('github.com')) label = 'github_profile';
          else if (link.href.includes('arxiv.org')) label = 'arxiv_paper';
          else label = link.hostname;

          gtag('event', 'click', {
            'event_category': 'outbound',
            'event_label': label,
            'value': link.href
          });
        });
      }
    });

    // Track course page visits
    var learnAiLinks = document.querySelectorAll('a[href*="/learn-ai"]');
    learnAiLinks.forEach(function(link) {
      link.addEventListener('click', function() {
        gtag('event', 'course_visit', {
          'event_category': 'navigation',
          'event_label': 'learn_ai_click'
        });
      });
    });

    // Track scroll depth
    var scrollDepths = [25, 50, 75, 100];
    var scrollFired = {};
    
    window.addEventListener('scroll', function() {
      var scrollPercent = (window.scrollY / (document.documentElement.scrollHeight - window.innerHeight)) * 100;
      
      scrollDepths.forEach(function(depth) {
        if (scrollPercent >= depth && !scrollFired[depth]) {
          scrollFired[depth] = true;
          gtag('event', 'scroll', {
            'event_category': 'engagement',
            'event_label': 'scroll_depth',
            'value': depth
          });
        }
      });
    });

    // Track deep reading (5+ minutes on page)
    setTimeout(function() {
      gtag('event', 'deep_read', {
        'event_category': 'engagement',
        'event_label': 'time_on_page',
        'value': 5
      });
    }, 300000); // 5 minutes

  });
})();
</script>
```

- [ ] **Step 3: Include analytics events in single.html footer**

Edit `layouts/_default/single.html`. Find the `{{ define "footer"}}` section (around line 98). After the MathJax script and before the post.js script, add:

```html
{{ define "footer"}}
    {{- partial "_shared/footer.html" . -}}
    <script>
        MathJax = {
            tex: {
                inlineMath: [['\\(', '\\)'], ['$', '$']],
                displayMath: [['$$', '$$']],
                processEscapes: true
            }
        };
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
    
    <!-- Analytics custom events -->
    {{- partial "_shared/analytics-events.html" . -}}
    
    {{ $postJS := resources.Get "js/post.js" | resources.Minify | resources.Fingerprint }}
    <script src="{{ $postJS.RelPermalink }}" integrity="{{ $postJS.Data.Integrity }}" crossorigin="anonymous" defer></script>
```

- [ ] **Step 4: Build and test**

```bash
hugo server -D --bind 0.0.0.0 --port 1313
```

Expected: No errors

- [ ] **Step 5: Test in browser console**

Open a post, open browser DevTools console, check for errors

Expected: No JavaScript errors

- [ ] **Step 6: Test event firing manually**

In browser on a post page:
1. Open DevTools → Network tab → Filter "collect"
2. Scroll down page to 25%, 50%, 75%
3. Click an outbound link (LinkedIn, GitHub)
4. Check Network tab for GA4 event requests

Expected: See POST requests to google-analytics.com/g/collect with event parameters

- [ ] **Step 7: Commit**

```bash
git add layouts/partials/_shared/analytics-events.html layouts/_default/single.html
git commit -m "feat(analytics): add GA4 custom event tracking

- Track outbound link clicks (LinkedIn, GitHub, arXiv)
- Track course page visits (/learn-ai)
- Track scroll depth (25%, 50%, 75%, 100%)
- Track deep reading (5+ minutes on page)
- Respect Do Not Track preference
- Non-blocking, privacy-respectful implementation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 8: Documentation Updates (CLAUDE.md)

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add SEO section to CLAUDE.md**

Edit `CLAUDE.md`. After the "Writing Style" section (around line 40) and before "Repository Structure", add:

```markdown
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
```

- [ ] **Step 2: Add monthly SEO checklist section**

Add after the "Deployment" section (near end of file):

```markdown
---

## Monthly SEO Health Check

**Search Console monitoring:**
- [ ] Check for indexing errors
- [ ] Review Coverage report (indexed vs excluded pages)
- [ ] Check Mobile Usability issues
- [ ] Review Core Web Vitals report
- [ ] Analyze Search Performance (queries, impressions, clicks)
- [ ] Check for manual actions or security issues
- [ ] Review top-performing pages and queries

**Analytics review:**
- [ ] Check GA4 for outbound link click patterns
- [ ] Review scroll depth distribution
- [ ] Analyze deep read rate
- [ ] Check course page visit conversions
- [ ] Review traffic sources and trends

**Content audit:**
- [ ] Verify new posts have `imageAlt` and `description` fields
- [ ] Check for broken internal links
- [ ] Review and update older posts if needed
```

- [ ] **Step 3: Update post frontmatter template**

Find the "Post Frontmatter Template" section in CLAUDE.md (around line 105). Update the template to:

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
imageAlt: 'Descriptive alt text for featured image'
description: 'SEO-friendly description (max 160 characters) - appears in search results'
title: 'Post Title: Subtitle'
---
```

Add after the template:

```markdown
**New optional fields:**
- `imageAlt`: Descriptive alt text for featured image (recommended for accessibility)
- `description`: SEO meta description, max 160 chars (recommended for search visibility)

Both fields are optional - intelligent defaults generated if omitted.
```

- [ ] **Step 4: Build and verify**

```bash
hugo server -D --bind 0.0.0.0 --port 1313
```

Expected: No errors, documentation updated

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add SEO guidelines and monitoring checklist

- Add image alt text guidelines with examples
- Add meta description guidelines
- Add internal linking best practices
- Add monthly SEO health check checklist
- Update post frontmatter template with imageAlt and description fields

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 9: Verification and Testing

**Files:**
- None (testing only)

- [ ] **Step 1: Build production site**

```bash
cd /home/gurwinde/blog
hugo --minify --cleanDestinationDir
```

Expected: Site builds successfully with no errors

- [ ] **Step 2: Start local server for testing**

```bash
hugo server -D --bind 0.0.0.0 --port 1313
```

- [ ] **Step 3: Test meta tags in browser**

Open http://localhost:1313/ and a blog post. Right-click → View Page Source. Verify:
- [ ] Resource hints present (preconnect, dns-prefetch)
- [ ] Meta description present
- [ ] Meta keywords present (if post has categories/tags)
- [ ] Article OG tags present (published_time, author, section, tag)
- [ ] og:site_name present

Expected: All meta tags present with correct content

- [ ] **Step 4: Test structured data**

View source of a blog post. Verify three JSON-LD scripts present:
- [ ] TechArticle schema with proficiencyLevel, articleSection, wordCount
- [ ] BreadcrumbList schema with Home → Category → Post
- [ ] Person schema with jobTitle, worksFor, sameAs

Expected: All three schemas present with valid JSON

- [ ] **Step 5: Validate schemas online**

Copy a post URL and test at:
1. https://validator.schema.org/
2. https://search.google.com/test/rich-results

Expected: All schemas valid, rich results eligible

- [ ] **Step 6: Test Open Graph preview**

Copy a post URL and test at https://www.opengraph.xyz/

Expected: Rich preview with image, title, description, and all OG tags

- [ ] **Step 7: Test breadcrumb navigation**

Open a blog post:
- [ ] Breadcrumb visible above post content
- [ ] Breadcrumb shows: Home › Category › Post Title
- [ ] Links are clickable and work correctly
- [ ] Styling matches site theme

Expected: Breadcrumb navigation working and styled correctly

- [ ] **Step 8: Test featured image alt text**

Open a blog post with featured image. Right-click image → Inspect Element. Verify:
- [ ] Alt attribute present
- [ ] Alt text is descriptive (not "thumbnail for this post")
- [ ] fetchpriority="high" attribute present

Expected: Image has proper alt text and performance attributes

- [ ] **Step 9: Test GA4 custom events**

Open a blog post in browser with DevTools:
1. Open DevTools → Console tab
2. Check for JavaScript errors
3. Open Network tab, filter by "collect"
4. Scroll down page to trigger scroll depth events
5. Click an external link (LinkedIn, GitHub)
6. Verify POST requests to google-analytics.com/g/collect

Expected: Events fire without errors, GA4 receives events

- [ ] **Step 10: Verify performance**

Run Lighthouse audit in Chrome DevTools:
1. Open DevTools → Lighthouse tab
2. Select "Performance", "Accessibility", "SEO"
3. Run audit

Expected: 
- Performance: 90+ score
- Accessibility: 90+ score (improved with alt text)
- SEO: 100 score (enhanced meta tags and structured data)

- [ ] **Step 11: Document test results**

Record any issues found during testing in a comment or note

---

### Task 10: Google Search Console Setup Instructions

**Files:**
- Modify: `hugo.toml`
- Modify: `layouts/partials/_shared/head.html`

- [ ] **Step 1: Add SEO config section to hugo.toml**

Edit `hugo.toml`. After the `[params.social]` section (around line 83), add:

```toml
[params.seo]
  # Add your Google Search Console verification code here after setup
  # Get code from: https://search.google.com/search-console
  # googleSiteVerification = "YOUR_VERIFICATION_CODE"
```

- [ ] **Step 2: Add verification tag support to head.html**

Edit `layouts/partials/_shared/head.html`. After the viewport meta tag and before the theme-color, add:

```html
	<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
	{{ with .Site.Params.seo.googleSiteVerification }}
	<meta name="google-site-verification" content="{{ . }}" />
	{{ end }}
	<meta name="theme-color" content="#038252">
```

- [ ] **Step 3: Create Search Console setup doc**

Create `docs/google-search-console-setup.md`:

```markdown
# Google Search Console Setup

## Step 1: Add Property

1. Visit https://search.google.com/search-console
2. Click "Add property"
3. Enter: `https://sgurwinderr.com`
4. Choose "URL prefix" method

## Step 2: Verify Ownership

1. Choose "HTML tag" verification method
2. Copy the verification code from the meta tag
3. Edit `hugo.toml` and uncomment/update the line:
   ```toml
   [params.seo]
     googleSiteVerification = "YOUR_CODE_HERE"
   ```
4. Build and deploy site
5. Return to Search Console and click "Verify"

## Step 3: Submit Sitemap

1. In Search Console, go to "Sitemaps"
2. Enter: `sitemap.xml`
3. Click "Submit"
4. Wait for processing (may take a few hours)

## Step 4: Monitor

Check weekly for first month, then monthly:
- Coverage report for indexing errors
- Performance report for search queries
- Core Web Vitals report
- Mobile Usability report

## Troubleshooting

**Verification fails:**
- Ensure hugo.toml has correct code
- Rebuild site: `hugo --minify`
- Deploy updated site
- Wait 5 minutes, try verification again

**Sitemap not found:**
- Verify sitemap accessible at: https://sgurwinderr.com/sitemap.xml
- Check robots.txt has correct sitemap URL
- Resubmit sitemap after 24 hours

**Pages not indexed:**
- Check robots.txt isn't blocking
- Verify pages in sitemap
- Request indexing via URL Inspection tool
- Wait 1-2 weeks for Google to crawl
```

- [ ] **Step 4: Build and verify**

```bash
hugo server -D --bind 0.0.0.0 --port 1313
```

View source, verify google-site-verification meta tag is NOT present (because no code set yet)

Expected: Tag only appears when googleSiteVerification is set in config

- [ ] **Step 5: Commit**

```bash
git add hugo.toml layouts/partials/_shared/head.html docs/google-search-console-setup.md
git commit -m "feat(seo): add Google Search Console verification support

- Add [params.seo] section to hugo.toml
- Add conditional google-site-verification meta tag
- Create setup documentation
- Ready for user to add verification code after GSC setup

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Self-Review Checklist

**Spec coverage check:**
- ✅ Enhanced meta tags (description, keywords, article tags) - Task 1
- ✅ Resource hints (preconnect, dns-prefetch) - Task 1
- ✅ TechArticle schema - Task 3
- ✅ BreadcrumbList schema - Task 4
- ✅ Organization/Person schema - Task 2
- ✅ Visual breadcrumb navigation - Task 5
- ✅ Breadcrumb CSS styling - Task 5
- ✅ Featured image alt text - Task 6
- ✅ GA4 custom events (outbound, scroll, course, deep read) - Task 7
- ✅ Search Console setup support - Task 10
- ✅ CLAUDE.md SEO guidelines - Task 8
- ✅ CLAUDE.md monitoring checklist - Task 8
- ✅ Post frontmatter template update - Task 8
- ✅ Comprehensive testing - Task 9

**Placeholder scan:**
- No TBD or TODO markers
- All code blocks complete
- All file paths exact
- All commands with expected output
- Twitter handle intentionally left as optional (not in use)

**Type consistency:**
- Schema fields: headline, description, image, datePublished, author, publisher, mainEntityOfPage, proficiencyLevel, articleSection, keywords, wordCount, timeRequired
- Template variables: .IsPage, .Title, .Description, .Content, .Params.image, .Params.imageAlt, .Params.categories, .Params.tags
- GA4 events: click, course_visit, scroll, deep_read with consistent event_category and event_label
- CSS classes: breadcrumb-nav, breadcrumb, breadcrumb-item, featured-image

All consistent throughout plan.

---

## Execution Notes

**Order matters:**
- Task 1 must complete before Task 2 (head.html includes organization schema)
- Task 4 modifies single.html (adds breadcrumb schema)
- Task 5 modifies single.html (adds breadcrumb nav)
- Task 6 modifies single.html (updates featured image)
- Task 7 modifies single.html (adds analytics events)
- Tasks 4-7 all modify single.html - ensure changes don't conflict

**Testing requirements:**
- Hugo server must be running for browser testing
- GA4 events require network inspection to verify
- Schema validation requires online validators
- All tests can be done locally before deployment

**No external dependencies:**
- All changes are template and CSS
- No new Hugo modules or npm packages
- No changes to base theme files
- All changes in `layouts/` directory override theme

**Backward compatibility:**
- All frontmatter fields optional (imageAlt, description)
- Intelligent fallbacks for missing data
- Existing posts work without changes
- Can add new fields gradually to posts
