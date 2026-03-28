(function () {
    var STORAGE_KEY = 'color-scheme';
    var root = document.documentElement;
    var btn  = document.getElementById('dark-mode-toggle');

    // Apply saved preference; default to light if nothing saved
    function getPreferred() {
        var saved = localStorage.getItem(STORAGE_KEY);
        if (saved) return saved;
        return 'light';
    }

    function applyTheme(theme) {
        if (theme === 'dark') {
            root.setAttribute('data-theme', 'dark');
        } else {
            root.removeAttribute('data-theme');
        }
        localStorage.setItem(STORAGE_KEY, theme);
        if (btn) btn.setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
    }

    // Apply on load
    applyTheme(getPreferred());

    // Wire toggle button
    if (btn) {
        btn.addEventListener('click', function () {
            var current = root.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
            applyTheme(current === 'dark' ? 'light' : 'dark');
        });
    }
})();
