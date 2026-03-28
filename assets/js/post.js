// TOC active-link scroll spy + hide when past article
(function () {
    var tocEl   = document.getElementById('toc-sidebar');
    var tocLinks = document.querySelectorAll('.toc-nav a');
    if (!tocLinks.length) return;

    // Sentinel: bottom of the article post div
    var articlePost = document.querySelector('.article-post');

    var headings = [];
    tocLinks.forEach(function (a) {
        var id = decodeURIComponent(a.getAttribute('href').replace('#', ''));
        var el = document.getElementById(id);
        if (el) headings.push({ el: el, a: a });
    });

    function onScroll() {
        var scrollY = window.scrollY + 100;

        // Hide TOC once the user scrolls past the article content
        if (tocEl && articlePost) {
            var articleBottom = articlePost.getBoundingClientRect().bottom + window.scrollY;
            if (scrollY > articleBottom) {
                tocEl.style.opacity = '0';
                tocEl.style.pointerEvents = 'none';
            } else {
                tocEl.style.opacity = '1';
                tocEl.style.pointerEvents = 'auto';
            }
        }

        // Active link highlight
        var active = null;
        for (var i = 0; i < headings.length; i++) {
            if (headings[i].el.offsetTop <= scrollY) active = headings[i];
            else break;
        }
        tocLinks.forEach(function (a) { a.classList.remove('active'); });
        if (active) active.a.classList.add('active');
    }
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
})();

// Copy button on syntax-highlighted code blocks
document.querySelectorAll('.highlight').forEach(function (block) {
    var btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    block.appendChild(btn);
    btn.addEventListener('click', function () {
        navigator.clipboard.writeText(block.querySelector('pre').innerText).then(function () {
            btn.textContent = 'Copied!';
            btn.classList.add('copied');
            setTimeout(function () {
                btn.textContent = 'Copy';
                btn.classList.remove('copied');
            }, 2000);
        });
    });
});
