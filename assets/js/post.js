// TOC active-link scroll spy
(function () {
    var tocLinks = document.querySelectorAll('.toc-nav a');
    if (!tocLinks.length) return;
    var headings = [];
    tocLinks.forEach(function (a) {
        var id = decodeURIComponent(a.getAttribute('href').replace('#', ''));
        var el = document.getElementById(id);
        if (el) headings.push({ el: el, a: a });
    });
    function onScroll() {
        var scrollY = window.scrollY + 100;
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
