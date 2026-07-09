/* =====================================================================
   slider.js — catalog carousel for the homepage Learn AI / PR Walkthroughs
   sections. Vanilla JS, no dependencies (Bootstrap JS is not loaded).

   Markup (see layouts/index.html):
     <div class="catalog-slider" data-visible="3">
       <button class="slider-btn slider-prev">…</button>
       <div class="slider-viewport">
         <div class="slider-track">
           <div class="slider-slide"> … </div> × N
         </div>
       </div>
       <button class="slider-btn slider-next">…</button>
     </div>

   Behaviour: show `visible` cards at a time (responsive: 3 → 2 → 1),
   prev/next step by one card, loop at the ends, smooth translateX.
   Hides the controls entirely when there is nothing to scroll. Honours
   prefers-reduced-motion (jumps instead of animating).
   ===================================================================== */
(function () {
  'use strict';

  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function visibleFor(base) {
    var w = window.innerWidth;
    if (w < 640) return 1;
    if (w < 992) return Math.min(2, base);
    return base;
  }

  function initSlider(root) {
    var track = root.querySelector('.slider-track');
    var slides = track ? Array.prototype.slice.call(track.querySelectorAll('.slider-slide')) : [];
    var prev = root.querySelector('.slider-prev');
    var next = root.querySelector('.slider-next');
    if (!track || slides.length === 0) return;

    var base = parseInt(root.getAttribute('data-visible'), 10) || 3;
    var index = 0;      // index of the left-most visible slide
    var visible = visibleFor(base);

    function maxIndex() { return Math.max(0, slides.length - visible); }

    function layout() {
      visible = visibleFor(base);
      // if everything fits, disable the slider entirely
      var scrollable = slides.length > visible;
      root.classList.toggle('is-static', !scrollable);
      if (prev) prev.hidden = !scrollable;
      if (next) next.hidden = !scrollable;
      if (index > maxIndex()) index = maxIndex();
      apply(true);
    }

    function apply(instant) {
      // slide width is measured from the first slide including the flex gap
      var slideW = slides[0].getBoundingClientRect().width;
      var styles = window.getComputedStyle(track);
      var gap = parseFloat(styles.columnGap || styles.gap || '0') || 0;
      var offset = index * (slideW + gap);
      if (instant || reduce) {
        var prevTrans = track.style.transition;
        track.style.transition = 'none';
        track.style.transform = 'translateX(' + (-offset) + 'px)';
        // force reflow then restore transition
        void track.offsetHeight;
        track.style.transition = prevTrans;
      } else {
        track.style.transform = 'translateX(' + (-offset) + 'px)';
      }
    }

    function go(dir) {
      var last = maxIndex();
      if (last === 0) return;
      index += dir;
      if (index < 0) index = last;        // loop to end
      else if (index > last) index = 0;   // loop to start
      apply(false);
    }

    if (prev) prev.addEventListener('click', function () { go(-1); });
    if (next) next.addEventListener('click', function () { go(1); });

    // keyboard support when the slider has focus
    root.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowLeft') { go(-1); }
      else if (e.key === 'ArrowRight') { go(1); }
    });

    var rAF = null;
    window.addEventListener('resize', function () {
      if (rAF) cancelAnimationFrame(rAF);
      rAF = requestAnimationFrame(layout);
    });

    layout();
    // mark ready so CSS can reveal the track without a first-paint jump
    root.classList.add('slider-ready');
  }

  function init() {
    var sliders = document.querySelectorAll('.catalog-slider');
    Array.prototype.forEach.call(sliders, initSlider);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
