(function () {
  var nav = document.querySelector('nav');
  var backToTop = document.querySelector('.back-to-top');
  var alertbar = document.querySelector('.alertbar');
  var toggler = document.querySelector('.navbar-toggler');
  var navbarCollapse = document.querySelector('#navbarMediumish');
  var lastScrollTop = 0;
  var delta = 5;
  var ticking = false;

  function onScroll() {
    var y = window.scrollY || window.pageYOffset;

    if (backToTop) {
      var show = y > 1250;
      backToTop.style.opacity = show ? '1' : '0';
      backToTop.style.pointerEvents = show ? 'auto' : 'none';
    }

    if (alertbar) {
      var maxScroll = document.documentElement.scrollHeight - window.innerHeight;
      var showAlert = y > 350 || y + 100 > maxScroll;
      alertbar.style.display = showAlert ? 'block' : 'none';
    }

    if (nav) {
      if (Math.abs(lastScrollTop - y) > delta) {
        var navHeight = nav.offsetHeight;
        if (y > lastScrollTop && y > navHeight) {
          nav.classList.remove('nav-down');
          nav.classList.add('nav-up');
          nav.style.top = -navHeight + 'px';
        } else if (y + window.innerHeight < document.documentElement.scrollHeight) {
          nav.classList.remove('nav-up');
          nav.classList.add('nav-down');
          nav.style.top = '0px';
        }
        lastScrollTop = y;
      }
    }
  }

  function smoothToHash(hash) {
    if (!hash || hash === '#') return;
    var id = hash.slice(1);
    var target = document.getElementById(id) || document.querySelector('[name="' + id + '"]');
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  if (backToTop) {
    backToTop.addEventListener('click', function (event) {
      event.preventDefault();
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    backToTop.style.opacity = '0';
    backToTop.style.pointerEvents = 'none';
  }

  document.addEventListener('click', function (event) {
    var link = event.target.closest('a[href*="#"]');
    if (!link) return;
    var url = new URL(link.href, window.location.origin);
    if (url.pathname === window.location.pathname && url.hash) {
      event.preventDefault();
      smoothToHash(url.hash);
    }
  });

  if (window.location.hash) {
    setTimeout(function () {
      window.scrollTo(0, 0);
      smoothToHash(window.location.hash);
    }, 1);
  }

  if (toggler && navbarCollapse) {
    toggler.addEventListener('click', function () {
      var expanded = toggler.getAttribute('aria-expanded') === 'true';
      toggler.setAttribute('aria-expanded', expanded ? 'false' : 'true');
      navbarCollapse.classList.toggle('show', !expanded);
    });

    navbarCollapse.querySelectorAll('a').forEach(function (a) {
      a.addEventListener('click', function () {
        toggler.setAttribute('aria-expanded', 'false');
        navbarCollapse.classList.remove('show');
      });
    });
  }

  window.addEventListener('scroll', function () {
    if (!ticking) {
      window.requestAnimationFrame(function () {
        onScroll();
        ticking = false;
      });
      ticking = true;
    }
  }, { passive: true });

  onScroll();
})();
