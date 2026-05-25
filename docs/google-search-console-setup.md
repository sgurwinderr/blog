# Google Search Console Setup Guide

This guide walks you through setting up Google Search Console (GSC) verification for your technical blog.

## What is Google Search Console?

Google Search Console is a free service that helps you monitor, maintain, and troubleshoot your site's presence in Google Search results. It provides:

- **Search performance metrics** (impressions, clicks, CTR, average position)
- **Index coverage reports** (which pages are indexed, any errors)
- **Mobile usability issues** (responsive design problems)
- **Core Web Vitals** (performance metrics)
- **URL inspection tool** (test individual pages)
- **Sitemap submission** (help Google discover content)
- **Security issues** (malware, hacking alerts)

## Prerequisites

- Google account
- Your blog domain properly configured (DNS, hosting)
- Hugo build tools (already set up)

## Step 1: Create/Access Google Search Console

1. Go to **https://search.google.com/search-console**
2. Click **Start Now** or sign in with your Google account
3. Select **Web** as your property type
4. Enter your domain:
   - For the full site: **https://sgurwinderr.com**
   - Or use a URL-prefix property for more control

## Step 2: Verify Domain Ownership

Google provides multiple verification methods. The **meta tag method** is easiest for Hugo users:

### Via HTML Meta Tag (Recommended)

1. In Google Search Console, go to **Settings** → **Verification**
2. Select **HTML tag** as the verification method
3. Copy the verification code (looks like):
   ```html
   <meta name="google-site-verification" content="ABC123DEF456GHI789JKL" />
   ```

## Step 3: Add Verification Code to Your Blog

1. Open `hugo.toml` in your blog root
2. Find the `[params.seo]` section (added after `[params.social]`)
3. Uncomment and add your verification code:
   ```toml
   [params.seo]
     googleSiteVerification = "ABC123DEF456GHI789JKL"
   ```
   (Replace with your actual code from Google)

4. Save the file

## Step 4: Rebuild and Deploy

### Local Testing

```bash
# Build your site
hugo

# Verify meta tag is present
grep -r "google-site-verification" public/

# Should output something like:
# public/index.html:    <meta name="google-site-verification" content="ABC123DEF456GHI789JKL" />
```

### Deploy to Production

Push your changes to GitHub (or your hosting provider):

```bash
git add hugo.toml
git commit -m "feat(seo): add Google Search Console verification code"
git push origin master
```

Your hosting provider should automatically deploy the updated site.

## Step 5: Verify with Google

1. Back in Google Search Console
2. Click **Verify** button
3. Google will check your domain for the meta tag

If verification fails:
- Confirm the meta tag is in your HTML: `curl https://sgurwinderr.com | grep google-site-verification`
- Check you copied the entire code (spaces, dashes, underscores matter)
- Wait a few minutes for DNS/cache propagation
- Try the **DNS TXT record** method as backup (more permanent)

## Step 6: Submit Sitemap

Once verified:

1. In Google Search Console, go to **Sitemaps**
2. Add your sitemap:
   ```
   https://sgurwinderr.com/sitemap.xml
   ```
3. Hugo auto-generates this; Google will crawl it to discover your posts

## Step 7: Monitor Performance

After a few days to weeks:

1. **Performance tab**: See search impressions, clicks, CTR
2. **Coverage tab**: Verify all posts are indexed
3. **Core Web Vitals**: Check page speed metrics
4. **Mobile Usability**: Ensure responsive design works
5. **Security Issues**: Monitor for hacking/malware

## Troubleshooting

### Meta tag not found in HTML

- Run `hugo` to rebuild (not just `hugo server`)
- Verify code is actually in `hugo.toml` under `[params.seo]`
- Check spelling: must be exactly `googleSiteVerification`
- Hard refresh your browser cache (Ctrl+Shift+R)

### Verification keeps failing

- Verify code from Google again (copy-paste carefully)
- Check for typos in meta tag
- Wait 24-48 hours (GSC caches sometimes)
- Try **DNS TXT record** verification instead (more reliable)
- Contact Google Search Console help for further support

### Pages not indexing

- Submit sitemap in GSC
- Use **URL Inspection** tool to test individual posts
- Check robots.txt isn't blocking crawlers
- Verify posts have `draft: false` in frontmatter
- Wait for Google's crawl queue (can take weeks for new sites)

## DNS TXT Record Method (Backup)

If meta tag verification doesn't work:

1. In Google Search Console, select **DNS TXT record**
2. Copy the TXT record value
3. Add to your domain's DNS settings (typically your registrar's dashboard)
4. Wait for DNS propagation (up to 48 hours)
5. Return to GSC and click **Verify**

This method is more permanent and survives cache clears.

## Tips for Best Results

1. **Keep your sitemap updated**: Hugo auto-generates; just keep posts published (`draft: false`)
2. **Monitor impressions**: Posts with high impressions but low CTR need better title/snippet optimization
3. **Fix crawl errors early**: CSC will alert you; fix 404s and broken links quickly
4. **Optimize Core Web Vitals**: Fast pages rank better (monitor the metrics tab)
5. **Mobile first**: Ensure all content looks good on mobile devices
6. **Post regularly**: New content helps Google crawl your site more frequently

## Common GSC Insights for Technical Blogs

- **Average position 11-30**: Your content appears on page 2-3 of Google; improve title/description to increase CTR
- **Low clicks, high impressions**: Your title/snippet needs to be more compelling
- **High CTR, low impressions**: Your content is good; use SEO to rank for more keywords
- **Index coverage errors**: Check for 404s, redirect loops, or blocked by robots.txt
- **Core Web Vitals "Needs Improvement"**: Optimize image sizes, CSS, JavaScript loading

## Next Steps

After verification:

1. Monitor **Performance** tab weekly
2. Check **Coverage** after publishing new posts
3. Respond to any **Security Issues** alerts immediately
4. Use insights to optimize post titles and descriptions
5. Track which topics drive most traffic to guide future content

## Additional Resources

- **Google Search Central**: https://developers.google.com/search
- **Core Web Vitals Guide**: https://web.dev/vitals
- **Crawl Stats Help**: https://support.google.com/webmasters/answer/7347206
- **Search Appearance Settings**: https://support.google.com/webmasters/answer/80092

---

**Questions?**
- Check Google Search Console Help: https://support.google.com/webmasters
- Review your blog's CLAUDE.md for Hugo configuration details
- For immediate issues, use the "Help" button in Google Search Console
