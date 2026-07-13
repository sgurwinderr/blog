---
name: seo-validation
description: Validate SEO, accessibility, sitemap, metadata, and rendered Hugo output for this blog. Use after adding or editing posts, layouts, catalog pages, images, schema, analytics, or before a publish/deploy check.
---

# SEO Validation

Run local checks before a push or release. Prefer verifying rendered `public/` output, because many SEO issues are introduced by templates rather than Markdown alone.

## Required Checks

1. Build the site:

```bash
hugo --minify --cleanDestinationDir
```

2. List drafts and confirm whether they are expected:

```bash
hugo list drafts
```

3. For published posts, verify frontmatter includes:

- `title`
- `date`
- `slug`
- `categories`
- `description` under 160 characters when practical
- `image` when the post has a featured image
- `imageAlt` when `image` is present

4. Verify featured image files exist. `image: assets/images/foo.png` must map to `static/assets/images/foo.png`.

5. Check Markdown images for empty alt text:

```bash
rg -n '!\[\s*\]\(' content
```

6. Parse the generated sitemap and confirm important pages are present:

```bash
hugo --minify --cleanDestinationDir
```

Then inspect `public/sitemap.xml` for the new slug and catalog pages.

7. Inspect rendered HTML for missing or semantically wrong alt text. A passing `alt=` presence check is not enough; related/suggested cards should describe the image or target post, not the current page.

## Template-Specific Checks

- Post cards: use each post's `.Params.imageAlt`, falling back to that post's `.Title`.
- Suggested posts: ensure `range` scoping does not accidentally use the current page title for suggested post images.
- Homepage featured posts: ensure image alt text comes from the ranged post, not the home page context.
- Open Graph and Twitter tags: confirm `og:title`, `og:description`, `og:image`, and `og:image:alt` render for posts with images.
- Canonical URLs: confirm `.Permalink` points at the slug URL, not an old alias.

## GitHub Workflows

Relevant checks live in `.github/workflows/`:

- `hugo.yml`: production Pages build and deploy.
- `image-alt-checker.yml`: empty/missing image alt checks.
- `link-checker.yml`: scheduled and content-push link checks.
- `seo-audit.yml`: Lighthouse audit against selected local URLs.
- `sitemap-validator.yml`: generated sitemap XML check.

If local Hugo is newer than CI Hugo, treat local deprecation warnings as maintenance items but verify any template/config update remains compatible with the workflow Hugo version.

## Report Format

Report:

- Build result and warnings.
- Drafts found.
- Missing or weak SEO fields.
- Missing image files or bad alt text.
- Sitemap status.
- Any checks skipped and why.
