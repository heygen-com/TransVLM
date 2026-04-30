/* -- Toggle helpers -- */
function togglePrompt(btn) {
  const content = btn.nextElementSibling;
  const isOpen = content.classList.toggle('open');
  btn.classList.toggle('active', isOpen);
  const textNode = btn.childNodes[btn.childNodes.length - 1];
  if (textNode) textNode.textContent = isOpen ? ' Hide Text Prompt' : ' Show Text Prompt';
}

function toggleCollapse(header) {
  const body = header.nextElementSibling;
  const isOpen = body.classList.toggle('open');
  header.classList.toggle('open', isOpen);
  /* pause all videos inside when collapsing */
  if (!isOpen) {
    body.querySelectorAll('video').forEach(v => v.pause());
  }
}

function playAllInCard(btn) {
  const card = btn.closest('.result-card');
  if (!card) return;
  /* ensure the card is expanded first */
  const body = card.querySelector('.collapsible-body');
  const header = card.querySelector('.collapsible-header');
  if (body && !body.classList.contains('open')) {
    body.classList.add('open');
    if (header) header.classList.add('open');
  }
  /* play all videos inside this card from the start */
  card.querySelectorAll('video').forEach(v => {
    v.currentTime = 0;
    v.play().catch(() => {});
  });
}

document.addEventListener('DOMContentLoaded', () => {

  /* -- Highlights: inject blurred background-fill video to avoid black pillarbox -- */
  document.querySelectorAll('.highlights-item .video-showcase').forEach(showcase => {
    if (showcase.querySelector('.bg-fill')) return;
    const main = showcase.querySelector('.main-video');
    if (!main) return;
    const src = main.querySelector('source');
    if (!src) return;
    const bg = document.createElement('video');
    bg.className = 'bg-fill';
    bg.muted = true;
    bg.loop = true;
    bg.defaultMuted = true;
    bg.playsInline = true;
    bg.setAttribute('muted', '');
    bg.setAttribute('playsinline', '');
    bg.setAttribute('aria-hidden', 'true');
    bg.preload = 'metadata';
    const newSrc = document.createElement('source');
    newSrc.src = src.src;
    newSrc.type = src.type;
    bg.appendChild(newSrc);
    showcase.insertBefore(bg, showcase.firstChild);
    /* sync bg with main playback state (only active slide plays) */
    main.addEventListener('play', () => bg.play().catch(() => {}));
    main.addEventListener('pause', () => bg.pause());
    main.addEventListener('seeking', () => { try { bg.currentTime = main.currentTime; } catch (e) {} });
  });

  /* -- Highlights scroll dots -- */
  const hlScroll = document.getElementById('highlights-scroll');
  const hlDotsContainer = document.getElementById('highlights-dots');
  if (hlScroll && hlDotsContainer) {
    const items = hlScroll.querySelectorAll('.highlights-item');
    items.forEach((_, i) => {
      const dot = document.createElement('button');
      dot.className = 'highlights-dot' + (i === 0 ? ' active' : '');
      dot.onclick = () => {
        items[i].scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
      };
      hlDotsContainer.appendChild(dot);
    });
    const dots = hlDotsContainer.querySelectorAll('.highlights-dot');
    let hlActiveIdx = 0;
    hlScroll.addEventListener('scroll', () => {
      const scrollLeft = hlScroll.scrollLeft;
      const width = hlScroll.offsetWidth;
      const idx = Math.round(scrollLeft / width);
      dots.forEach((d, i) => d.classList.toggle('active', i === idx));
      /* auto-play current, pause others */
      if (idx !== hlActiveIdx) {
        items.forEach((item, i) => {
          const vid = item.querySelector('.main-video');
          if (!vid) return;
          if (i === idx) {
            vid.currentTime = 0;
            vid.play().catch(() => {});
          } else {
            vid.pause();
          }
        });
        hlActiveIdx = idx;
      }
    }, { passive: true });
    /* auto-play first one when it scrolls into the page viewport */
    const hlSectionObs = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        const vid = items[hlActiveIdx]?.querySelector('.main-video');
        if (!vid) return;
        if (e.isIntersecting) {
          vid.play().catch(() => {});
        } else {
          vid.pause();
        }
      });
    }, { threshold: 0.3 });
    hlSectionObs.observe(hlScroll);

    /* arrow buttons with seamless wrap (clone-edge technique) */
    const hlPrev = document.getElementById('hl-prev');
    const hlNext = document.getElementById('hl-next');
    const EPS = 4;
    let wrapping = false;

    function waitScrollEnd(cb) {
      let done = false;
      const finish = () => { if (done) return; done = true; cb(); };
      if ('onscrollend' in window) {
        hlScroll.addEventListener('scrollend', finish, { once: true });
        setTimeout(finish, 900);
      } else {
        setTimeout(finish, 650);
      }
    }

    function getRealItems() {
      return hlScroll.querySelectorAll('.highlights-item:not(.hl-clone)');
    }

    function wrapForward() {
      if (wrapping) return;
      wrapping = true;
      const real = getRealItems();
      const w = hlScroll.offsetWidth;
      const clone = real[0].cloneNode(true);
      clone.classList.add('hl-clone');
      /* pause any video inside clone (no autoplay flicker) */
      clone.querySelectorAll('video').forEach(v => { v.muted = true; try { v.pause(); } catch (e) {} });
      hlScroll.classList.add('no-snap');
      hlScroll.appendChild(clone);
      requestAnimationFrame(() => {
        hlScroll.scrollBy({ left: w, behavior: 'smooth' });
        waitScrollEnd(() => {
          hlScroll.scrollTo({ left: 0, behavior: 'auto' });
          clone.remove();
          hlScroll.classList.remove('no-snap');
          wrapping = false;
        });
      });
    }

    function wrapBackward() {
      if (wrapping) return;
      wrapping = true;
      const real = getRealItems();
      const w = hlScroll.offsetWidth;
      const last = real[real.length - 1];
      const clone = last.cloneNode(true);
      clone.classList.add('hl-clone');
      clone.querySelectorAll('video').forEach(v => { v.muted = true; try { v.pause(); } catch (e) {} });
      hlScroll.classList.add('no-snap');
      hlScroll.insertBefore(clone, hlScroll.firstChild);
      /* keep view on original first slide (now at position w due to prepend) */
      hlScroll.scrollTo({ left: w, behavior: 'auto' });
      requestAnimationFrame(() => {
        hlScroll.scrollBy({ left: -w, behavior: 'smooth' });
        waitScrollEnd(() => {
          hlScroll.scrollTo({ left: (real.length - 1) * w, behavior: 'auto' });
          clone.remove();
          hlScroll.classList.remove('no-snap');
          wrapping = false;
        });
      });
    }

    if (hlPrev) hlPrev.onclick = () => {
      if (wrapping) return;
      const w = hlScroll.offsetWidth;
      if (hlScroll.scrollLeft <= EPS) wrapBackward();
      else hlScroll.scrollBy({ left: -w, behavior: 'smooth' });
    };
    if (hlNext) hlNext.onclick = () => {
      if (wrapping) return;
      const w = hlScroll.offsetWidth;
      const max = hlScroll.scrollWidth - w;
      if (hlScroll.scrollLeft >= max - EPS) wrapForward();
      else hlScroll.scrollBy({ left: w, behavior: 'smooth' });
    };
  }

  /* -- Intro video tab switcher -- */
  const introTabs = document.querySelectorAll('.intro-tab');
  const introPanels = document.querySelectorAll('.intro-video-panel');

  introTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      /* deactivate all */
      introTabs.forEach(t => t.classList.remove('active'));
      introPanels.forEach(p => {
        p.classList.remove('active');
        p.querySelectorAll('video').forEach(v => v.pause());
      });
      /* activate selected */
      tab.classList.add('active');
      const panel = document.getElementById(tab.dataset.introTarget);
      if (panel) {
        panel.classList.add('active');
        const mainVid = panel.querySelector('.intro-main-video');
        if (mainVid) { mainVid.currentTime = 0; mainVid.play().catch(() => {}); }
      }
    });
  });

  /* auto-play first intro video */
  const firstIntroVid = document.querySelector('#intro-panel-1 .intro-main-video');
  if (firstIntroVid) {
    const introObs = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) e.target.play().catch(() => {});
        else e.target.pause();
      });
    }, { threshold: 0.3 });
    document.querySelectorAll('.intro-main-video').forEach(v => introObs.observe(v));
  }

  document.querySelectorAll('.ref-pip').forEach(pip => {
    pip.addEventListener('mouseenter', () => pip.play().catch(() => {}));
    pip.addEventListener('mouseleave', () => { pip.pause(); pip.currentTime = 0; });
  });

  /* -- Auto-wrap showcase cards as collapsible -- */
  document.querySelectorAll('#showcases .result-card').forEach(card => {
    const label = card.querySelector('.card-label');
    if (!label || card.querySelector('.collapsible-header')) return;

    /* create header */
    const header = document.createElement('div');
    header.className = 'collapsible-header';
    header.onclick = function() { toggleCollapse(this); };
    header.innerHTML = '<svg class="collapse-icon" viewBox="0 0 24 24"><path d="M7 10l5 5 5-5z"/></svg>';
    header.appendChild(label);

    /* wrap remaining children */
    const body = document.createElement('div');
    body.className = 'collapsible-body';
    while (card.firstChild && card.firstChild !== header) {
      if (card.firstChild === label) { card.removeChild(label); continue; }
      body.appendChild(card.firstChild);
    }
    /* move anything after header marker too */
    while (card.childNodes.length > 0) {
      body.appendChild(card.firstChild);
    }

    card.appendChild(header);
    card.appendChild(body);
  });

  /* -- Synchronized comparison playback -- */
  const activeSyncIntervals = new Map();

  function syncPlayCard(card) {
    const allVideos = card.querySelectorAll('.video-compare video, .ref-inputs video');
    if (!allVideos.length) return;

    allVideos.forEach(v => { v.currentTime = 0; });
    requestAnimationFrame(() => {
      allVideos.forEach(v => v.play().catch(() => {}));
    });

    if (activeSyncIntervals.has(card)) clearInterval(activeSyncIntervals.get(card));
    const master = allVideos[allVideos.length - 1];
    const interval = setInterval(() => {
      const t = master.currentTime;
      allVideos.forEach(v => {
        if (v !== master && Math.abs(v.currentTime - t) > 0.15) {
          v.currentTime = t;
        }
      });
    }, 500);
    activeSyncIntervals.set(card, interval);
  }

  function syncPauseCard(card) {
    card.querySelectorAll('.video-compare video, .ref-inputs video')
        .forEach(v => v.pause());
    if (activeSyncIntervals.has(card)) {
      clearInterval(activeSyncIntervals.get(card));
      activeSyncIntervals.delete(card);
    }
  }

  const compObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      /* only auto-play if the collapsible-body is open (or there is none) */
      const body = entry.target.querySelector('.collapsible-body');
      const isExpanded = !body || body.classList.contains('open');
      if (entry.isIntersecting && isExpanded) {
        syncPlayCard(entry.target);
      } else {
        syncPauseCard(entry.target);
      }
    });
  }, { threshold: 0.2 });

  document.querySelectorAll('#comparison .result-card, #showcases .result-card').forEach(card => {
    compObserver.observe(card);

    const oursVideo = card.querySelector('.video-label.ours')
                          ?.closest('.video-item')
                          ?.querySelector('video');
    if (oursVideo) {
      oursVideo.muted = false;
      oursVideo.volume = 0.8;
    }
  });

  /* -- Sidebar TOC: scroll spy -- */
  const tocLinks = document.querySelectorAll('#toc a');
  const tocItems = document.querySelectorAll('#toc li');
  const sectionIds = Array.from(tocLinks).map(a => a.getAttribute('href').slice(1));

  function updateActiveToc() {
    /* Active = last section whose top has passed viewport center, so the section
       currently occupying the middle of the screen is highlighted. */
    const threshold = window.innerHeight / 2;
    let current = sectionIds[0];
    for (const id of sectionIds) {
      const el = document.getElementById(id);
      if (el && el.getBoundingClientRect().top <= threshold) {
        current = id;
      }
    }
    tocItems.forEach(li => {
      const href = li.querySelector('a')?.getAttribute('href')?.slice(1);
      li.classList.toggle('active', href === current);
    });
  }

  window.addEventListener('scroll', updateActiveToc, { passive: true });
  updateActiveToc();

  tocLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const target = document.getElementById(link.getAttribute('href').slice(1));
      if (!target) return;
      const r = target.getBoundingClientRect();
      const scrollY = window.scrollY || window.pageYOffset;
      const targetCenter = scrollY + r.top + r.height / 2;
      const dest = Math.max(0, targetCenter - window.innerHeight / 2);
      window.scrollTo({ top: dest, behavior: 'smooth' });
    });
  });
});

/* =====================================================================
   STD-specific extensions: interactive Quant + Ablation widgets
   ===================================================================== */
(function () {
  function fillSelect(sel, items, defaultId) {
    if (!sel) return;
    sel.innerHTML = '';
    items.forEach(function (it) {
      var o = document.createElement('option');
      o.value = it.id;
      o.textContent = it.label;
      if (defaultId && it.id === defaultId) o.selected = true;
      sel.appendChild(o);
    });
  }

  function setupWidget(opts) {
    /* opts: { rootId, manifest, assetBase } */
    var manifest = opts.manifest;
    var base = opts.assetBase;
    var root = document.getElementById(opts.rootId);
    if (!root || !manifest) return;

    var tDs = root.querySelector('[data-role="t-ds"]');
    var tTt = root.querySelector('[data-role="t-tt"]');
    var pDs = root.querySelector('[data-role="p-ds"]');
    var pMt = root.querySelector('[data-role="p-mt"]');
    var tFr = root.querySelector('[data-role="t-frame"]');
    var pImg = root.querySelector('[data-role="p-img"]');

    fillSelect(tDs, manifest.datasets, 'd00');
    fillSelect(tTt, manifest.ttOptions, 'mean');
    fillSelect(pDs, manifest.datasets, 'd00');
    fillSelect(pMt, manifest.metrics, 'segment_f1');

    function updTable() {
      var key = tDs.value + '|' + tTt.value;
      var p = manifest.tables && manifest.tables[key];
      if (p) {
        tFr.src = base + p;
      } else {
        tFr.removeAttribute('src');
      }
    }
    function updPlot() {
      var d = manifest.plotDirs && manifest.plotDirs[pDs.value];
      if (d) {
        pImg.src = base + d + pMt.value + '.svg';
      } else {
        pImg.removeAttribute('src');
      }
    }

    [tDs, tTt].forEach(function (s) { s.addEventListener('change', updTable); });
    [pDs, pMt].forEach(function (s) { s.addEventListener('change', updPlot); });

    updTable();
    updPlot();
  }

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      s.src = src;
      s.onload = resolve;
      s.onerror = reject;
      document.head.appendChild(s);
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    var tasks = [];
    if (document.getElementById('quant-widget')) {
      tasks.push(loadScript('./Packages/STD/data/quantitative/manifest.js').then(function () {
        setupWidget({
          rootId: 'quant-widget',
          manifest: window.QUANT_MANIFEST,
          assetBase: './Packages/STD/data/quantitative/'
        });
      }));
    }
    if (document.getElementById('abla-widget')) {
      tasks.push(loadScript('./Packages/STD/data/ablation/manifest.js').then(function () {
        setupWidget({
          rootId: 'abla-widget',
          manifest: window.ABLA_MANIFEST,
          assetBase: './Packages/STD/data/ablation/'
        });
      }));
    }
  });
})();
