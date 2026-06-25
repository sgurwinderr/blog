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

// Scroll-triggered reveal for post visualization blocks
(function () {
    var stageLabels = [
        'Ready',
        'Stage 1: Projection',
        'Stage 2: Flow',
        'Stage 3: Tile Pop'
    ];

    function restartClassAnimation(el, cls) {
        el.classList.remove(cls);
        // Force style recalc so CSS animation can replay.
        void el.getBoundingClientRect();
        el.classList.add(cls);
    }

    function triggerGroup(group, cls, baseDelay, stride) {
        group.forEach(function (el, index) {
            el.style.setProperty('--diagram-delay', String(baseDelay + index * stride) + 'ms');
            restartClassAnimation(el, cls);
        });
    }

    function stopPlayback(state) {
        state.isPlaying = false;
        if (state.playTimer) {
            clearInterval(state.playTimer);
            state.playTimer = null;
        }
        if (state.playBtn) state.playBtn.textContent = 'Play';
    }

    function setStage(state, stage) {
        state.stage = Math.max(0, Math.min(3, stage));
        if (state.stageLabel) state.stageLabel.textContent = stageLabels[state.stage];

        if (state.stage === 1) triggerGroup(state.accents, 'diagram-pulse', 0, 120);
        if (state.stage === 2) triggerGroup(state.flows, 'diagram-flow', 60, 90);
        if (state.stage === 3) triggerGroup(state.pops, 'diagram-pop', 30, 24);
    }

    function startPlayback(state) {
        if (state.isPlaying) {
            stopPlayback(state);
            return;
        }
        state.isPlaying = true;
        if (state.playBtn) state.playBtn.textContent = 'Pause';

        if (state.stage === 0) setStage(state, 1);

        state.playTimer = setInterval(function () {
            if (state.stage >= 3) {
                stopPlayback(state);
                return;
            }
            setStage(state, state.stage + 1);
        }, 1200);
    }

    function createDiagramControls(diagramEl, state) {
        if (diagramEl.querySelector('.post-diagram-controls')) return;

        var controls = document.createElement('div');
        controls.className = 'post-diagram-controls';

        var prevBtn = document.createElement('button');
        prevBtn.type = 'button';
        prevBtn.className = 'post-diagram-btn';
        prevBtn.textContent = 'Prev';

        var playBtn = document.createElement('button');
        playBtn.type = 'button';
        playBtn.className = 'post-diagram-btn post-diagram-btn-primary';
        playBtn.textContent = 'Play';

        var nextBtn = document.createElement('button');
        nextBtn.type = 'button';
        nextBtn.className = 'post-diagram-btn';
        nextBtn.textContent = 'Next';

        var resetBtn = document.createElement('button');
        resetBtn.type = 'button';
        resetBtn.className = 'post-diagram-btn';
        resetBtn.textContent = 'Reset';

        var stageLabel = document.createElement('span');
        stageLabel.className = 'post-diagram-stage';
        stageLabel.textContent = stageLabels[0];

        controls.appendChild(prevBtn);
        controls.appendChild(playBtn);
        controls.appendChild(nextBtn);
        controls.appendChild(resetBtn);
        controls.appendChild(stageLabel);

        var caption = diagramEl.querySelector('.post-diagram-caption');
        if (caption) diagramEl.insertBefore(controls, caption);
        else diagramEl.appendChild(controls);

        state.stageLabel = stageLabel;
        state.playBtn = playBtn;

        prevBtn.addEventListener('click', function () {
            stopPlayback(state);
            setStage(state, state.stage - 1);
        });

        nextBtn.addEventListener('click', function () {
            stopPlayback(state);
            setStage(state, state.stage + 1);
        });

        resetBtn.addEventListener('click', function () {
            stopPlayback(state);
            setStage(state, 0);
        });

        playBtn.addEventListener('click', function () {
            startPlayback(state);
        });
    }

    function initDiagram(diagramEl) {
        if (diagramEl._diagramState) return diagramEl._diagramState;

        var svg = diagramEl.querySelector('svg');
        if (!svg) return null;

        var state = {
            stage: 0,
            isPlaying: false,
            playTimer: null,
            stageLabel: null,
            playBtn: null,
            accents: Array.from(svg.querySelectorAll('[opacity="0.5"], [opacity="0.55"], [opacity="0.6"]')),
            flows: Array.from(svg.querySelectorAll('path[marker-end], line[marker-end], path[stroke-dasharray], line[stroke-dasharray]')),
            pops: Array.from(svg.querySelectorAll('rect[rx], rect[stroke-width]'))
        };

        svg.classList.add('diagram-active');
        createDiagramControls(diagramEl, state);
        diagramEl._diagramState = state;
        return state;
    }

    function animateDiagram(diagramEl) {
        if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
            return;
        }
        var state = initDiagram(diagramEl);
        if (!state) return;
        if (diagramEl.dataset.diagramAnimated === '1') return;

        setStage(state, 1);
        diagramEl.dataset.diagramAnimated = '1';
    }

    var revealTargets = document.querySelectorAll(
        '.article-post .post-diagram, .article-post .post-step-card, .article-post .post-callout, .article-post .post-bench'
    );
    if (!revealTargets.length) return;

    revealTargets.forEach(function (el) {
        el.classList.add('post-animate-in');
    });

    if (!('IntersectionObserver' in window)) {
        revealTargets.forEach(function (el) {
            el.classList.add('visible');
        });
        return;
    }

    var revealObserver = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                if (entry.target.classList.contains('post-diagram')) {
                    animateDiagram(entry.target);
                }
                revealObserver.unobserve(entry.target);
            }
        });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });

    revealTargets.forEach(function (el, index) {
        el.style.setProperty('--post-stagger-index', String(index % 6));
        revealObserver.observe(el);
    });
})();
