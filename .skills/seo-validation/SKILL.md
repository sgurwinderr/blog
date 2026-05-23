---
name: seo-validation
description: Use after publishing blog posts to validate SEO requirements locally before pushing
---

# SEO Validation

Run SEO checks locally after writing/publishing blog posts, before pushing to GitHub.

## When to Use

After creating or updating blog posts, before committing/pushing.

## Quick Validation

### 1. Frontmatter Check
```bash
# Check required SEO fields
post="content/post/YYYY-MM-DD-slug.md"

grep -q "^title:" "$post" || echo "❌ Missing title"
grep -q "^date:" "$post" || echo "❌ Missing date"
grep -q "^slug:" "$post" || echo "❌ Missing slug"
grep -q "^categories:" "$post" || echo "❌ Missing categories"
grep -q "^image:" "$post" || echo "❌ Missing image"

# Check title length (50-60 optimal)
title=$(grep "^title:" "$post" | cut -d'"' -f2)
echo "Title length: ${#title} chars (optimal: 50-60)"
```

### 2. Image Alt Text
```bash
# Check markdown images
grep -n '!\[]\|!\[\s*]' content/post/*.md && echo "❌ Images missing alt text"
```

### 3. Build & Preview
```bash
hugo --minify
ls -lh public/sitemap.xml  # Check sitemap generated
```

### 4. Local Link Check
```bash
# After hugo server running
curl -s http://localhost:1313/ | grep -i "title\|description\|og:"
```

## GitHub Actions (Automated)

4 workflows run on push:

**seo-audit.yml** - Lighthouse performance + SEO audit
**link-checker.yml** - Find broken links (weekly + on push)
**image-alt-checker.yml** - Verify all images have alt text
**sitemap-validator.yml** - Validate sitemap.xml structure

View results: GitHub repo → Actions tab

## Common Issues

**Missing OG image**: Add `image: assets/images/cover.jpg` to frontmatter
**Title too long**: Keep under 60 characters for Google SERP
**No alt text**: Use `![description](path)` not `![](path)`
**Broken internal links**: Use `/slug/` format, test locally first

## Checklist Before Push

- [ ] All frontmatter fields present
- [ ] Title 50-60 characters
- [ ] Image has alt text
- [ ] Internal links work locally
- [ ] `hugo --minify` succeeds
- [ ] Sitemap includes new post
