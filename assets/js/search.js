(function () {
    var overlay  = document.getElementById('search-overlay');
    var input    = document.getElementById('search-input');
    var results  = document.getElementById('search-results');
    var footer   = document.getElementById('search-footer');
    var countEl  = document.getElementById('search-count');
    var openBtn  = document.getElementById('search-open');
    var closeBtn = document.getElementById('search-close');

    if (!overlay || !input) return;

    var indexData = null;
    var indexUrl = (document.querySelector('head[data-baseurl]') || {}).dataset
        ? (document.querySelector('head').dataset.baseurl || '').replace(/\/$/, '') + '/index.json'
        : '/index.json';

    function loadIndex(cb) {
        if (indexData) { cb(); return; }
        fetch(indexUrl)
            .then(function (r) { return r.json(); })
            .then(function (data) {
                indexData = data || [];
                cb();
            });
    }

    function rank(query, post) {
        var q = query.toLowerCase();
        var title = String(post.title || '').toLowerCase();
        var summary = String(post.summary || '').toLowerCase();
        var categories = (post.categories || []).join(' ').toLowerCase();
        var score = 0;

        if (title.indexOf(q) !== -1) score += title.startsWith(q) ? 7 : 5;
        if (summary.indexOf(q) !== -1) score += 2;
        if (categories.indexOf(q) !== -1) score += 1;

        // Fuzzy-like token match fallback for multi-word queries.
        q.split(/\s+/).filter(Boolean).forEach(function (tok) {
            if (tok.length < 2) return;
            if (title.indexOf(tok) !== -1) score += 2;
            if (summary.indexOf(tok) !== -1) score += 1;
        });

        return score;
    }

    function openSearch() {
        overlay.classList.add('active');
        document.body.classList.add('search-open');
        loadIndex(function () { input.focus(); });
    }

    function closeSearch() {
        overlay.classList.remove('active');
        document.body.classList.remove('search-open');
        input.value = '';
        results.innerHTML = '';
        footer.style.display = 'none';
    }

    function renderResults(query) {
        if (!indexData || query.length < 2) {
            results.innerHTML = '';
            footer.style.display = 'none';
            return;
        }

        var hits = indexData
            .map(function (p) {
                return { item: p, rank: rank(query, p) };
            })
            .filter(function (h) { return h.rank > 0; })
            .sort(function (a, b) { return b.rank - a.rank; })
            .slice(0, 8);

        if (!hits.length) {
            results.innerHTML = '<p class="search-no-results">No results for <strong>' + escHtml(query) + '</strong></p>';
            footer.style.display = 'none';
            return;
        }
        results.innerHTML = hits.map(function (h) {
            var p = h.item;
            var cat = (p.categories && p.categories.length) ? p.categories[0] : '';
            var imgHtml = p.image
                ? '<img src="' + escHtml(p.image) + '" class="sr-thumb" loading="lazy" alt="">'
                : '<div class="sr-thumb sr-thumb-placeholder"></div>';
            return '<a href="' + escHtml(p.url) + '" class="sr-item">'
                + imgHtml
                + '<div class="sr-body">'
                + (cat ? '<span class="sr-cat">' + escHtml(cat) + '</span>' : '')
                + '<span class="sr-title">' + escHtml(p.title) + '</span>'
                + '<span class="sr-meta">' + escHtml(p.date) + ' · ' + p.readingTime + ' min read</span>'
                + '</div>'
                + '</a>';
        }).join('');
        countEl.textContent = hits.length + ' result' + (hits.length !== 1 ? 's' : '');
        footer.style.display = 'block';
    }

    function escHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    openBtn.addEventListener('click', openSearch);
    closeBtn.addEventListener('click', closeSearch);

    overlay.addEventListener('click', function (e) {
        if (e.target === overlay) closeSearch();
    });

    document.addEventListener('keydown', function (e) {
        if ((e.key === 'k' && (e.metaKey || e.ctrlKey)) || e.key === '/') {
            if (document.activeElement.tagName === 'INPUT' ||
                document.activeElement.tagName === 'TEXTAREA') return;
            e.preventDefault();
            openSearch();
        }
        if (e.key === 'Escape') closeSearch();
    });

    var debounceTimer;
    input.addEventListener('input', function () {
        clearTimeout(debounceTimer);
        var q = input.value.trim();
        debounceTimer = setTimeout(function () { renderResults(q); }, 150);
    });
})();
