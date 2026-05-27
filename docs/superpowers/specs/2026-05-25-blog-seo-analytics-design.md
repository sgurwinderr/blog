# Blog SEO and Analytics Enhancement Design

**Date:** 2026-05-25  
**Author:** Gurwinder (with Claude)  
**Status:** Draft for Review

---

## Context

This design addresses making Gurwinder's technical blog (https://sgurwinderr.com/) SEO-friendly and enhancing analytics capabilities. The blog currently has 26+ deep technical posts covering AI, GPU programming, graphics, and game development, targeting GPU developers, ML engineers, and systems programmers.

**Why this change is needed:**
- Publication-quality content deserves maximum visibility in search results
- Current SEO foundation is basic (Open Graph tags, simple schema) but leaves opportunities on table
- Target mixed audience: both developers searching for solutions ("PyTorch autograd internals") and learners exploring topics ("how does GPU programming work")
- Google Analytics is configured but only tracks basic page views
- Want both search engine optimization and social media sharing optimization

**Current state:**
- Hugo static site with custom Medium theme
- GA4 configured (`G-KD3KXHFEQM`)
- Basic Open Graph and Twitter Card meta tags
- Simple Article schema (JSON-LD)
- Sitemap generation enabled
- robots.txt in place
- No systematic image alt text
- No custom analytics events
- No Google Search Console integration

**Intended outcome:**
- Rich search results with breadcrumbs, author info, article metadata
- Better social sharing previews on LinkedIn/Twitter
- Track meaningful engagement (outbound clicks, scroll depth, course visits)
- Improved search rankings through technical SEO
- Accessible images with proper alt text
- Monitoring capability via Search Console

---

## Architecture Overview

The enhancement follows Hugo's template hierarchy, keeping customizations in `layouts/` directory (not modifying theme directly). Components are isolated and can be updated independently:

1. **Meta tags layer** (`layouts/partials/_shared/head.html`)
   - Pulls data from post frontmatter and site config
   - Generates SEO meta tags, Open Graph, Twitter Cards
   - Adds performance resource hints

2. **Structured data layer** (`layouts/partials/_shared/schema.html`, new breadcrumb partial)
   - JSON-LD schema generation
   - TechArticle vs Article conditional logic
   - Organization and BreadcrumbList schemas

3. **Analytics layer** (new `layouts/partials/_shared/analytics-events.html`)
   - Custom GA4 event tracking
   - Non-blocking JavaScript
   - Privacy-respectful tracking

4. **Content layer** (post frontmatter updates)
   - Add optional `description`, `imageAlt` fields
   - Backward compatible (generates defaults if missing)

5. **Navigation layer** (breadcrumb component)
   - Visual breadcrumbs in UI
   - Matching JSON-LD schema for search engines

**Data flow:**
```
Post Frontmatter → Hugo Template Variables → Partial Templates → HTML Output
     ↓                                              ↓
  Defaults if missing                    Conditional logic based on page type
```

---

## Component Design

### 1. Meta Tags and SEO Headers

**File:** `layouts/partials/_shared/head.html`

**Changes:**

Add article-specific meta tags:
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

Enhanced Open Graph tags for articles:
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

Performance resource hints (add near top of `<head>`):
```html
<!-- Preconnect to external domains for performance -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="preconnect" href="https://cdn.jsdelivr.net">
<link rel="dns-prefetch" href="https://www.googletagmanager.com">
<link rel="dns-prefetch" href="https://www.google-analytics.com">
```

Add Twitter creator handle (if Twitter account exists):
```html
<meta name="twitter:creator" content="@yourhandle" />
```

**Fallback logic:**
- Description: Use `.Description` from frontmatter, else first 160 chars of content
- Keywords: Combine categories and tags
- OG image: Use `.Params.image`, else site logo as fallback

**Testing:**
- Check meta tags with browser dev tools
- Validate Open Graph with https://www.opengraph.xyz/
- Validate Twitter Cards with https://cards-dev.twitter.com/validator

---

### 2. Enhanced Structured Data (JSON-LD)

**File:** `layouts/partials/_shared/schema.html`

**Changes:**

Replace generic Article schema with TechArticle:
```html
{{- if .IsPage -}}
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": {{ .Title | jsonify }},
  "description": {{ (with .Description){{ . }}{{ else }}{{ .Summary | plainify }}{{ end }} | jsonify }},
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
  "articleSection": {{ (index .Params.categories 0) | jsonify }},
  {{ with .Params.tags }}
  "keywords": {{ delimit . ", " | jsonify }},
  {{ end }}
  "wordCount": {{ .WordCount }},
  "timeRequired": "PT{{ .ReadingTime }}M"
}
</script>
{{- end -}}
```

**New file:** `layouts/partials/_shared/breadcrumb-schema.html`

Add BreadcrumbList schema:
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

**New file:** `layouts/partials/_shared/organization-schema.html`

Add Organization schema (include once, site-wide):
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

**Include in templates:**
- Call `breadcrumb-schema.html` in `layouts/_default/single.html` (post pages)
- Call `organization-schema.html` in `layouts/partials/_shared/head.html` or footer (site-wide, once)

**Testing:**
- Validate with https://validator.schema.org/
- Test in Google Rich Results Test: https://search.google.com/test/rich-results

---

### 3. Image Optimization and Alt Text

**File:** `layouts/_default/single.html`

**Changes:**

Update featured image to use alt text from frontmatter:
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

**Frontmatter additions:**

Add to post template in CLAUDE.md:
```yaml
---
image: assets/images/post-cover.jpg
imageAlt: "Descriptive alt text for featured image"
description: "SEO-friendly description (160 chars max)"
---
```

**Alt text guidelines (document in CLAUDE.md):**

For technical diagrams:
- ✅ "Architecture diagram showing PyTorch Triton kernel integration flow"
- ✅ "Graph comparing online vs standard softmax memory usage across batch sizes"
- ❌ "Diagram" (too vague)
- ❌ "Image showing technical stuff" (not descriptive)

For code screenshots:
- ✅ "Code snippet demonstrating PyTorch autograd hook registration"
- ✅ "Terminal output showing successful Triton kernel compilation"

For illustrations:
- ✅ "Conceptual illustration of GPU memory hierarchy with L1/L2 caches"

**Image format recommendations (guidance, not automation):**
- SVG for diagrams and architecture illustrations
- WebP with JPEG fallback for screenshots
- Keep under 200KB per image
- Minimum 1200x630px for featured images (Open Graph)

**One-time task:**
- Audit existing posts and add `imageAlt` to frontmatter where missing
- Can be done gradually, not blocking

---

### 4. Internal Linking and Site Structure

**File:** `layouts/partials/single-partials/suggested-posts.html`

**Changes:**

Enhance related posts logic to prioritize same category:
```html
{{- $related := where .Site.RegularPages "Section" .Section -}}
{{- $related := where $related ".Permalink" "!=" .Permalink -}}

{{- /* Filter by primary category if exists */ -}}
{{- if .Params.categories -}}
  {{- $primaryCategory := index .Params.categories 0 -}}
  {{- $categoryMatches := where $related ".Params.categories" "intersect" (slice $primaryCategory) -}}
  {{- if gt (len $categoryMatches) 0 -}}
    {{- $related = $categoryMatches -}}
  {{- end -}}
{{- end -}}

{{- $related := $related.ByDate.Reverse -}}
{{- $related := first 3 $related -}}
```

**New file:** `layouts/partials/_shared/breadcrumb-nav.html`

Visual breadcrumb navigation:
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

**CSS for breadcrumbs (add to `static/css/custom.css`):**
```css
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

**Include in template:**

Add to `layouts/_default/single.html` after opening `<div class="container">`:
```html
{{- partial "_shared/breadcrumb-nav.html" . -}}
```

**Internal linking guidelines (add to CLAUDE.md):**

When writing posts:
- Link to related posts for context: "For background on PyTorch's autograd system, see [Understanding PyTorch Autograd](/pytorch-autograd-internals)"
- Link to prerequisite knowledge: "This post assumes familiarity with [GPU memory hierarchies](/gpu-memory-basics)"
- Create "series" posts that reference each other
- Link from new posts to older foundational posts

**Sitemap priority (handled by Hugo automatically, verify in output):**
- Homepage: 1.0
- Posts: 0.8 (0.9 if `featured: true`)
- Static pages (About, Courses): 0.6
- Taxonomy pages (categories/tags): 0.4

---

### 5. GA4 Custom Events and Enhanced Tracking

**New file:** `layouts/partials/_shared/analytics-events.html`

Custom event tracking JavaScript:
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

**Include in template:**

Add to `layouts/_default/single.html` footer (after GA4 loads):
```html
{{ define "footer"}}
    {{- partial "_shared/footer.html" . -}}
    
    <!-- MathJax (existing) -->
    <script>...</script>
    
    <!-- Analytics custom events -->
    {{- partial "_shared/analytics-events.html" . -}}
    
    <!-- Other existing scripts -->
{{ end }}
```

**Events tracked:**
1. **Outbound clicks**: LinkedIn, GitHub, arXiv papers, other external links
2. **Course visits**: Clicks to /learn-ai page
3. **Scroll depth**: 25%, 50%, 75%, 100% of article
4. **Deep reading**: After 5 minutes on page

**Privacy considerations:**
- Respects Do Not Track header
- No personal data collected
- No keystroke or mouse tracking
- Anonymous event data only

**Viewing events in GA4:**
- Go to GA4 dashboard → Reports → Engagement → Events
- Custom events appear as: `click`, `course_visit`, `scroll`, `deep_read`
- Filter by event_category and event_label for details

---

### 6. Performance Optimizations

**File:** `layouts/partials/_shared/head.html`

**Resource hints (already covered in Section 1):**
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="preconnect" href="https://cdn.jsdelivr.net">
<link rel="dns-prefetch" href="https://www.googletagmanager.com">
<link rel="dns-prefetch" href="https://www.google-analytics.com">
```

**Verification checklist:**
- ✅ MathJax loads with `async` attribute
- ✅ Post JavaScript loads with `defer` attribute
- ✅ CSS minified and fingerprinted via Hugo Pipes
- ✅ Images use `loading="lazy"` and `decoding="async"`
- ✅ Cache headers configured for static assets
- ✅ Hugo build uses `--minify` flag in production

**Hugo build command for production:**
```bash
hugo --minify --cleanDestinationDir
```

**Performance testing tools:**
- Google PageSpeed Insights: https://pagespeed.web.dev/
- WebPageTest: https://www.webpagetest.org/
- Lighthouse (Chrome DevTools)

**Target metrics:**
- First Contentful Paint (FCP): < 1.5s
- Largest Contentful Paint (LCP): < 2.5s
- Cumulative Layout Shift (CLS): < 0.1
- Time to Interactive (TTI): < 3.5s

**No changes needed if metrics already good.** Current setup is already performance-optimized.

---

### 7. Google Search Console Integration

**Setup process (manual, one-time):**

1. **Add site to Search Console:**
   - Visit https://search.google.com/search-console
   - Click "Add property"
   - Enter: `https://sgurwinderr.com`
   - Choose "URL prefix" method

2. **Verify ownership via meta tag:**
   - Search Console provides a meta tag like:
     ```html
     <meta name="google-site-verification" content="YOUR_VERIFICATION_CODE" />
     ```
   - Add to Hugo config as a parameter:
     ```toml
     [params.seo]
       googleSiteVerification = "YOUR_VERIFICATION_CODE"
     ```

3. **Update head.html to include verification tag:**
   ```html
   {{ with .Site.Params.seo.googleSiteVerification }}
     <meta name="google-site-verification" content="{{ . }}" />
   {{ end }}
   ```

4. **Submit sitemap:**
   - In Search Console, go to "Sitemaps"
   - Submit: `https://sgurwinderr.com/sitemap.xml`
   - Verify it's successfully processed

**Monitoring checklist (add to CLAUDE.md):**

Monthly SEO health check:
- [ ] Check Search Console for indexing errors
- [ ] Review Coverage report (indexed vs excluded pages)
- [ ] Check Mobile Usability issues
- [ ] Review Core Web Vitals report
- [ ] Analyze Search Performance (queries, impressions, clicks)
- [ ] Check for manual actions or security issues
- [ ] Review top-performing pages and queries

**robots.txt verification:**

Current file at `static/robots.txt`:
```
User-agent: *
Allow: /
Disallow: /categories/
Disallow: /tags/
Disallow: /search

Sitemap: https://sgurwinderr.com/sitemap.xml
```

**Consideration:** You may want to allow `/categories/` and `/tags/` for indexing since they aggregate related content. Current setup blocks them, which is fine if you prefer posts to be the only indexed content.

**No change needed unless** you want category/tag pages indexed. If so, remove those Disallow lines.

---

## Implementation Priority

**Phase 1: Core SEO (highest impact, lowest effort)**
1. Enhanced meta tags in head.html
2. TechArticle schema update
3. BreadcrumbList schema
4. Organization schema
5. Resource hints

**Phase 2: Analytics Enhancement**
1. Custom GA4 events script
2. Search Console setup and verification

**Phase 3: Content Improvements**
1. Add imageAlt field to post template
2. Update existing posts' frontmatter (gradual)
3. Add breadcrumb navigation component
4. Enhance suggested posts logic

**Phase 4: Monitoring and Optimization**
1. Test all meta tags and schema
2. Verify GA4 events firing
3. Monitor Search Console
4. Performance testing

---

## Testing and Verification

**SEO validation:**
1. **Meta tags:** Browser DevTools → Elements → `<head>` section
2. **Open Graph:** https://www.opengraph.xyz/
3. **Twitter Cards:** https://cards-dev.twitter.com/validator
4. **Structured data:** https://validator.schema.org/
5. **Rich results:** https://search.google.com/test/rich-results
6. **Mobile-friendly:** https://search.google.com/test/mobile-friendly

**Analytics validation:**
1. **GA4 real-time report:** Check events firing live
2. **GA4 DebugView:** Enable debug mode to see event parameters
3. **Browser console:** Check for gtag errors
4. **Test events manually:** Click outbound links, scroll, wait 5 minutes

**Performance validation:**
1. **PageSpeed Insights:** https://pagespeed.web.dev/
2. **Lighthouse:** Chrome DevTools → Lighthouse tab
3. **WebPageTest:** https://www.webpagetest.org/

**Accessibility validation:**
1. **WAVE:** https://wave.webaim.org/
2. **axe DevTools:** Browser extension
3. **Screen reader test:** VoiceOver (Mac) or NVDA (Windows)

---

## Critical Files to Modify

1. **layouts/partials/_shared/head.html**
   - Add enhanced meta tags
   - Add resource hints
   - Add Search Console verification tag

2. **layouts/partials/_shared/schema.html**
   - Replace Article with TechArticle
   - Add extended metadata

3. **layouts/partials/_shared/breadcrumb-schema.html** (new)
   - Add BreadcrumbList JSON-LD

4. **layouts/partials/_shared/organization-schema.html** (new)
   - Add Person/Organization JSON-LD

5. **layouts/partials/_shared/analytics-events.html** (new)
   - Custom GA4 event tracking

6. **layouts/partials/_shared/breadcrumb-nav.html** (new)
   - Visual breadcrumb navigation

7. **layouts/_default/single.html**
   - Update featured image with alt text
   - Include breadcrumb navigation
   - Include analytics events script

8. **layouts/partials/single-partials/suggested-posts.html**
   - Enhance related posts logic

9. **static/css/custom.css**
   - Add breadcrumb styling

10. **hugo.toml**
    - Add `[params.seo]` section for Search Console verification
    - Verify sitemap configuration

11. **CLAUDE.md**
    - Add image alt text guidelines
    - Add internal linking guidelines
    - Add monthly SEO checklist

---

## Post Frontmatter Template (Updated)

Add to CLAUDE.md:

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

**New fields:**
- `imageAlt`: Descriptive alt text for featured image
- `description`: SEO meta description (max 160 chars)

**Both optional** - system generates intelligent defaults if missing.

---

## Backward Compatibility

All enhancements are backward compatible:
- Existing posts without `imageAlt` get auto-generated alt text from title
- Existing posts without `description` get first 160 chars of content
- No breaking changes to existing templates
- Hugo builds will succeed with or without new frontmatter fields

**No migration required** - enhancements work immediately, optional fields improve quality when added.

---

## Success Metrics

**SEO metrics (track in Search Console, 30-90 days):**
- Increase in indexed pages
- Improvement in average search position
- Increase in impressions for target keywords
- Higher click-through rate (CTR) from search results
- More rich result appearances

**Analytics metrics (track in GA4):**
- Outbound link clicks (measure engagement)
- Course page visits (conversion funnel)
- Scroll depth distribution (content engagement)
- Deep read rate (quality engagement)

**Performance metrics (track in PageSpeed Insights):**
- Maintain or improve Core Web Vitals scores
- Keep performance budget under control
- No increase in page load time despite added features

**Target keywords to monitor:**
- "PyTorch profiler internals"
- "Triton kernel optimization"
- "online softmax algorithm"
- "GPU programming tutorial"
- "CUDA memory optimization"
- "DirectX ray tracing"

---

## Maintenance

**Weekly:**
- Check GA4 for unusual traffic patterns
- Monitor real-time events for errors

**Monthly:**
- Review Search Console for issues
- Analyze top-performing content
- Check Core Web Vitals
- Review search queries and impressions

**Quarterly:**
- Audit posts for missing imageAlt/description fields
- Update internal links in newer posts
- Review and update target keywords
- Performance testing and optimization

**Yearly:**
- Comprehensive SEO audit
- Review and update structured data
- Check for schema.org updates
- Evaluate new SEO opportunities

---

## Notes and Considerations

**What NOT to do:**
- Don't keyword-stuff meta tags or content
- Don't create duplicate content
- Don't use misleading descriptions
- Don't track personal user data
- Don't sacrifice performance for features

**Trade-offs accepted:**
- Some manual work (adding imageAlt to existing posts) for better accessibility
- Additional JavaScript (analytics events) for better insights, but kept minimal
- More meta tags = slightly larger HTML, but gzipped size impact is negligible

**Future enhancements (not in this design):**
- Automated internal link suggestions
- A/B testing for post titles
- Advanced analytics (heatmaps, session recordings) - requires paid tools
- Internationalization (i18n) if targeting non-English audiences
- AMP pages (Accelerated Mobile Pages) - declining relevance, Hugo support limited

---

## References

- Hugo SEO best practices: https://gohugo.io/templates/internal/#open-graph
- Schema.org TechArticle: https://schema.org/TechArticle
- Google Search Central: https://developers.google.com/search
- GA4 event tracking: https://developers.google.com/analytics/devguides/collection/ga4/events
- Web Vitals: https://web.dev/vitals/

---

**End of Design Document**
