/* TransVLM project page behaviour.
 *
 * Four components over one shared piece. The comparison player and the
 * failure-case section draw the same thing -- four method tracks against a
 * video's timeline -- from two different sources, so the strip itself lives in
 * tracks.js and this file only wires it up.
 *
 * The per-tau panels these used to mount -- 168 pre-generated tables and 98
 * curves that already ship under Packages/STD/data/ -- are not on this page. They
 * belong to the leaderboard, which is not built yet; assets/js/widget.js still
 * mounts them there. See handoff.md.
 */
(function () {
  'use strict';

  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var el = function (tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };

  function getJSON(path) {
    return fetch(path, { cache: 'no-cache' }).then(function (r) {
      if (!r.ok) throw new Error(path + ': HTTP ' + r.status);
      return r.json();
    });
  }

  /* How long a card must have been in front before finishing a pass is allowed
     to move the deck on. Both decks that advance on their own content need it:
     a card is left playing wherever it was when it arrived, so it can reach the
     end of its clip a moment later, and advancing on that would flick past a
     card nobody has had time to look at. */
  var FRONT_MIN = 3000;

  /* Playback speed for the comparison clips, and the default is half.
     A cut is two frames: at 1x it is over before the eye has left the card and
     reached the rails, which is the one thing the section is asking the reader
     to do. Half speed is what makes the boundary something you can watch
     happen; 0.25 is for looking at a dissolve frame by frame, 2 for skimming.
     FRONT_MIN stays in wall time, so a slower clip gets more looks before the
     deck moves on -- which is the intent, not a side effect. */
  var SPEEDS = [0.25, 0.5, 1, 2];
  var DEFAULT_SPEED = 0.5;

  /* THE PLAYHEAD MOVES EVERY FRAME, NOT FOUR TIMES A SECOND. `timeupdate` is
     what both decks repainted their strip on, and it fires at about 4 Hz, so
     the head stepped along the rails in quarter-second hops. While the front
     card is playing the strip is repainted from a rAF loop instead; the loop
     starts on the card's `playing`, stops itself the moment that card is paused
     or ended, and is kicked once when a card that is already playing arrives at
     the front, because arriving fires no `playing`. One loop per deck, none
     while nothing plays. `timeupdate` stays bound in the callers: it is what
     repaints a stopped element after a seek. */
  function smoothPaint(host, activeOf, paint) {
    var raf = 0;
    function loop() {
      var v = activeOf();
      if (!v || v.paused || v.ended) { raf = 0; return; }
      paint();
      raf = requestAnimationFrame(loop);
    }
    function kick() { if (!raf) raf = requestAnimationFrame(loop); }
    // Neither event bubbles, so both capture on the deck, like timeupdate.
    host.addEventListener('playing', function (ev) {
      if (ev.target === activeOf()) kick();
    }, true);
    return kick;
  }

  /* mulberry32: 32 bits of state, uniform enough for dealing a deck and nowhere
     near anything that needs to be unguessable. Copied out of collage.js rather
     than shared -- the site loads plain scripts and has no module system, and a
     global for six lines buys nothing. */
  function mulberry32(a) {
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* ------------------------------------------------------------ theme */

  function initTheme() {
    var btn = $('.theme');
    if (!btn) return;
    btn.addEventListener('click', function () {
      var root = document.documentElement;
      var dark = root.getAttribute('data-theme') === 'dark';
      var next = dark ? 'light' : 'dark';
      /* Swap with transitions off -- see the note on [data-theme-swap]. The
         reflow is what makes the suppression land before the tokens move
         rather than in the same style pass; two frames of waiting is what
         makes it outlast the swap's own recalc. */
      root.setAttribute('data-theme-swap', '');
      void root.offsetWidth;
      root.setAttribute('data-theme', next);
      requestAnimationFrame(function () {
        requestAnimationFrame(function () { root.removeAttribute('data-theme-swap'); });
      });
      btn.setAttribute('aria-label', 'Switch to ' + (dark ? 'dark' : 'light') + ' theme');
      try { localStorage.setItem('transvlm-theme', next); } catch (e) { /* private mode */ }
    });
  }

  /* ----------------------------------------------------------- rail */

  /* The rail is a readout, not a menu. RAIL_N ticks stand in for the whole
     document; the read position is a gaussian over them, so width, opacity and
     colour all come off one falloff and the thing reads as a single object
     rather than three effects. A section start is a tick that never shrinks
     below RAIL_ANCHOR, placed where the section actually falls in the scroll
     range -- the gaps between those ticks are what says a section is long. */
  var RAIL_N = 45;          // ticks; the frame's height is this times the pitch
  var RAIL_PITCH = 10;      // px per tick, and .rail .tick in the stylesheet
  var RAIL_MIN = 2;         // px, a tick nowhere near the reader
  var RAIL_MAX = 18;        // px, the tick under the reader, and the rail's width
  var RAIL_ANCHOR = 6;      // px, the floor for a section start
  var RAIL_SIGMA = 2.26;    // ticks; the falloff dies out by about four of them
  var RAIL_DIM = 0.28;      // opacity of a tick at rest
  var RAIL_LABEL_GAP = 18;  // px, just over the 16px a name's box is tall

  function initRail() {
    var rail = $('.rail');
    if (!rail) return;

    var labels = [].slice.call(rail.querySelectorAll('.lbl'));
    var sections = labels.map(function (l) {
      var id = l.getAttribute('href').slice(1);
      var s = document.getElementById(id);
      // A name pointing at a section that is not on the page means the rail is
      // out of step with the document, and a rail that lies about where things
      // are is worse than no rail.
      if (!s) throw new Error('rail: no section #' + id);
      return s;
    });

    // The ticks carry no information a heading does not, and 45 of them in the
    // tab order would be 45 things to skip.
    var track = el('span', 'ticks');
    track.setAttribute('aria-hidden', 'true');
    var ticks = [];
    for (var i = 0; i < RAIL_N; i++) {
      var row = el('span', 'tick');
      var bar = el('i');
      row.appendChild(bar);
      track.appendChild(row);
      ticks.push(bar);
    }
    rail.appendChild(track);

    /* Open on reaching the strip, and stay open for as long as the pointer is
       anywhere in the column the names occupy -- including the gaps between
       them, which is the whole point: two of those gaps are ~90px tall, and a
       reveal that ends when the pointer is over neither name closes halfway to
       the one being reached for.

       A region test rather than a grace period, because the failing case is
       reading the names slowly. Any timer short enough not to linger on the way
       out is too short for a deliberate 97px traverse. And a region test rather
       than a wider hover pad, because a pad is a hit target and the figures run
       underneath this one.

       The right edge is measured once on opening: the names do not change width
       while they are up. The vertical bounds come off the strip each move --
       one rect, and it follows the rail if the page scrolls under the pointer. */
    var railOpen = false, railRight = 0;
    function railOut(ev) {
      var r = rail.getBoundingClientRect();
      if (ev.clientX >= r.left - 20 && ev.clientX <= railRight &&
          ev.clientY >= r.top - 16 && ev.clientY <= r.bottom + 16) return;
      railOpen = false;
      rail.classList.remove('is-open');
      document.removeEventListener('pointermove', railOut);
    }
    rail.addEventListener('pointerenter', function () {
      if (railOpen) return;
      railOpen = true;
      railRight = labels.reduce(function (x, l) {
        return Math.max(x, l.getBoundingClientRect().right);
      }, rail.getBoundingClientRect().right) + 12;
      rail.classList.add('is-open');
      document.addEventListener('pointermove', railOut);
    });

    var anchored = [];   // tick index -> is a section start
    var current = null;

    function progress() {
      var range = document.documentElement.scrollHeight - window.innerHeight;
      if (range <= 0) return 0;
      return Math.min(1, Math.max(0, window.scrollY / range));
    }

    /* Where each section falls in the scroll range, which is not where it falls
       in the document: the last screenful of the page is one scroll position, so
       measuring against scrollHeight alone would push the closing sections off
       the end of the rail. */
    function layout() {
      var range = document.documentElement.scrollHeight - window.innerHeight;
      anchored = [];
      var floor = -Infinity;
      sections.forEach(function (s, i) {
        var top = s.getBoundingClientRect().top + window.scrollY;
        var at = range > 0 ? Math.min(1, Math.max(0, top / range)) : 0;
        var tick = at * (RAIL_N - 1);
        anchored[Math.round(tick)] = true;
        // Two sections can sit close enough that their names would overlap once
        // revealed. Only the text moves; the tick keeps the true position.
        var y = Math.max(tick * RAIL_PITCH + RAIL_PITCH / 2, floor + RAIL_LABEL_GAP);
        labels[i].style.top = y + 'px';
        floor = y;
      });
    }

    function paint() {
      var head = progress() * (RAIL_N - 1);
      for (var i = 0; i < RAIL_N; i++) {
        var d = (i - head) / RAIL_SIGMA;
        var g = Math.exp(-d * d);
        var w = RAIL_MIN + (RAIL_MAX - RAIL_MIN) * g;
        var o = RAIL_DIM + (1 - RAIL_DIM) * g;
        if (anchored[i]) { w = Math.max(w, RAIL_ANCHOR); o = Math.max(o, 0.45); }
        var bar = ticks[i];
        bar.style.width = w.toFixed(2) + 'px';
        bar.style.opacity = o.toFixed(3);
        bar.style.background =
          'color-mix(in oklab, var(--fg) ' + (g * 100).toFixed(1) + '%, var(--fg-3))';
      }

      // The name that stays lit is the section you are reading: the last one
      // whose top has crossed a third of the viewport.
      var line = window.innerHeight / 3;
      var at = 0;
      sections.forEach(function (s, i) {
        if (s.getBoundingClientRect().top <= line) at = i;
      });
      if (at !== current) {
        if (current !== null) labels[current].removeAttribute('data-current');
        labels[at].setAttribute('data-current', '');
        current = at;
      }
    }

    var queued = false;
    function frame() {
      if (queued) return;
      queued = true;
      requestAnimationFrame(function () { queued = false; paint(); });
    }

    addEventListener('scroll', frame, { passive: true });
    addEventListener('resize', function () { layout(); frame(); }, { passive: true });
    // Videos, iframes and the figures all get their boxes after this runs, and
    // every one of them moves the sections the rail is drawn from.
    new ResizeObserver(function () { layout(); frame(); }).observe(document.body);
    layout();
    paint();
  }

  /* A hash in the URL scrolls before the figures have their boxes. Re-aim once
     layout settles, but stop the moment the reader takes over. */
  function initHash() {
    var id = decodeURIComponent(location.hash.slice(1));
    var target = id && document.getElementById(id);
    if (!target) return;
    // Whether the reader has taken over is answered by the scroll position
    // itself: if it is no longer where we last put it, stop moving it. Watching
    // input events instead disarms on the first stray pointer event.
    var placed = null;
    function aim() {
      if (placed !== null && Math.abs(window.scrollY - placed) > 4) return;
      var top = target.getBoundingClientRect().top + window.scrollY - 24;
      window.scrollTo({ top: Math.max(0, top), behavior: 'instant' });
      placed = Math.round(window.scrollY);
    }
    aim();
    addEventListener('load', aim);
    setTimeout(aim, 250);
    setTimeout(aim, 900);
  }

  /* --------------------------------------------------- shared: tracks */

  /* The four-method strip and the readout under it live in tracks.js: the player, the failure cases and the deck all draw the same
     thing, and a second copy is a second thing to keep in step. */
  var trackStrip = TVTracks.strip;
  var readout = TVTracks.readout;

  /* ------------------------------------------- 02 comparison deck */

  /* One card per extracted sample, and the sample *is* the transition: a
     1.6-13.5 s cut around a ground-truth boundary TransVLM overlaps while both
     SBD baselines are silent. So every card plays its whole file, loops it, and
     the four rails underneath run over that file's own seconds. Nothing here is
     a window into a longer video -- extract_samples.py already made the cut, in
     a tool that can show its work (lab/samples.html), and this section shows
     what it kept.

     The deck advances on the clip rather than on a clock, so a 1.6 s cut is not
     held for as long as a 13.5 s dissolve and neither is cut off part-way. */

  /* The 84 cards are dealt in a fixed pseudo-random order.
   *
   * Source order is twelve videos' worth of samples end to end, which opens the
   * deck on thirty-two consecutive RAI clips: the reader would have to sit
   * through a third of the deck to learn that it covers more than one dataset.
   * Shuffling mixes them. Seeding the shuffle is what makes the order a property
   * of the page rather than of the load -- two readers see the same deck, a
   * reload does not re-deal, and "the third card" is a thing two people can talk
   * about. Changing DECK_SEED is how to re-deal. */
  var DECK_SEED = 0x7B5C21;

  function initPlayer(items, legend) {
    var host = $('#clip-deck');
    if (!host) return;
    var under = $('#clip-under');
    var dots = $('#clip-dots');
    var trackHost = $('.tracks', under);
    var readHost = $('.readout', under);
    var speedHost = $('#clip-speed', under);

    // Fisher-Yates, over the seeded PRNG. See DECK_SEED above.
    var clips = items.slice();
    var rnd = mulberry32(DECK_SEED);
    for (var i = clips.length - 1; i > 0; i--) {
      var j = (rnd() * (i + 1)) | 0;
      var tmp = clips[i]; clips[i] = clips[j]; clips[j] = tmp;
    }

    // TVDeck.mount() calls onIndex synchronously, so everything that binding
    // touches is declared before the mount. Initialising it afterwards would
    // silently undo the first bind and leave the playhead stuck at zero until
    // the reader happened to change cards.
    var deck = null, strip = null, active = null, current = null, arrivedAt = 0;
    var rate = DEFAULT_SPEED;   // the section's speed, not the front card's

    // How long a card has to have been at the front before finishing a pass is
    // allowed to move the deck on. See tick().
    // FRONT_MIN is shared with section 04; see its declaration above.

    function render(it) {
      var card = el('figure', 'card');
      var media = el('div', 'card-media');
      var v = document.createElement('video');
      v.muted = true;
      v.playsInline = true;
      v.preload = 'none';
      // The poster is a frame of the transition itself, so a card that has not
      // been fetched yet still shows the thing it is about rather than a hole.
      v.poster = it.poster;
      v.setAttribute('aria-label', it.dataset + ' clip, ' + it.domain);
      v.dataset.src = it.src;
      // Click pauses the section, not just this clip. Five cards playing is one
      // exhibit, and stopping only the front one leaves four still moving in the
      // corner of your eye. Only the active card gets the click at all -- the
      // deck claims clicks on the others to mean "bring me here".
      v.addEventListener('click', function () { run(!running); });
      media.appendChild(v);
      card.appendChild(media);
      // The clip's filename identifies a file, not the thing the reader is
      // looking at. It stays in the markup and off the page.
      card.dataset.clip = it.id;
      var foot = el('figcaption', 'card-foot');
      // Dataset and domain for every card; the synthetic ones also name the
      // ffmpeg effect each covered transition was rendered with, which is the
      // one case where the page can say exactly what the transition is.
      var label = it.dataset + ' · ' + it.domain;
      if (it.effects.length) label += ' · ' + it.effects.join(', ');
      foot.appendChild(el('span', 'grp', label));
      card.appendChild(foot);
      return card;
    }

    // Only the cards you can see are fetched, and only they play. The 84 cuts are
    // 26 MB between them; pulling all of them in to fill a deck of five is not
    // defensible on a page most people scroll past.
    function wake(v) {
      if (!v.src) {
        // play() issued before the first frame is decodable is aborted by the
        // pending load(), so arm it on loadeddata rather than fire and hope.
        v.addEventListener('loadeddata', function () {
          if (v.dataset.want === '1') v.play().catch(function () {});
        });
        // Some browsers reset playbackRate when a new source finishes loading,
        // so it is set again once the element has its metadata rather than only
        // here.
        v.addEventListener('loadedmetadata', function () { v.playbackRate = rate; });
        v.src = v.dataset.src;
        v.playbackRate = rate;
        v.load();
      }
      v.dataset.want = '1';
      // Swallowed deliberately: a rejected play() here means the browser declined
      // an autoplay it is entitled to decline, or the element was torn down
      // mid-call. Neither is actionable, and neither should reach the console.
      if (v.readyState >= 2) v.play().catch(function () {});
    }
    function sleep(v) {
      v.dataset.want = '0';
      if (!v.paused) v.pause();
    }

    /* The loop is written here and the element's own `loop` attribute is
     * deliberately not used: with `loop` set the wrap happens inside the element
     * and no `ended` fires, and `ended` is the one event the deck's pacing hangs
     * off. Rewinding by hand costs one assignment and gives the section the end
     * of a pass to act on.
     *
     * A scrub can leave currentTime anywhere and none of that ends a pass, so
     * `ended` is the whole of the test -- there is nothing else to check. */
    function tick(v) {
      v.currentTime = 0;
      if (v.paused && v.dataset.want === '1') v.play().catch(function () {});
      // One pass has just finished. Only the front card moves the deck on; the
      // neighbours loop quietly. advance() is a request, not a command -- it
      // declines mid-drag, or with the section scrolled away.
      //
      // The front card is left wherever it was playing when it arrived, so it
      // can reach its end a moment later. Advancing on that wrap would flick
      // straight past a card nobody has had time to look at, so a wrap inside
      // the first FRONT_MIN of a card's turn is spent looping and the next one
      // moves the deck on.
      if (v !== active || !deck) return;
      if (performance.now() - arrivedAt < FRONT_MIN) return;
      deck.advance();
    }

    // Five clips decoding at once is the price of the stack, and it is only worth
    // paying while the section is on screen. Scrolled past, the deck keeps its
    // state and stops its videos.
    var shown = [], onScreen = true, running = true, mounted = [];
    function sync() {
      mounted.forEach(function (card, i) {
        var v = card.querySelector('video');
        if (running && onScreen && shown.indexOf(i) >= 0) wake(v); else sleep(v);
      });
    }
    function run(on) {
      running = on;
      host.classList.toggle('is-paused', !on);
      sync();
    }

    function paint() {
      if (!strip || !active || !current) return;
      var t = active.currentTime || 0;
      // Frames here, and only here, and frames only: these clips are the
      // extracted cuts, so the page knows the source frame rate exactly and a
      // two-frame cut is worth counting in frames.
      readout(readHost, t, current.duration, strip.paint(t), { fps: current.fps, ours: legend.ours.label });
    }

    /* The speed buttons, one per SPEEDS entry. Built here rather than written
       into the markup so the set and the default cannot disagree with the
       constant the videos are actually started at. */
    var rateBtns = SPEEDS.map(function (r) {
      var b = el('button', 'btn' + (r === rate ? ' solid' : ''), r + '×');
      b.type = 'button';
      b.setAttribute('aria-pressed', r === rate ? 'true' : 'false');
      b.addEventListener('click', function () { setRate(r); });
      speedHost.appendChild(b);
      return b;
    });

    function setRate(r) {
      rate = r;
      // Every mounted card, not just the front one: the neighbours are playing
      // too, and a stack running at two speeds reads as a broken deck.
      mounted.forEach(function (card) { card.querySelector('video').playbackRate = r; });
      rateBtns.forEach(function (b, i) {
        var on = SPEEDS[i] === r;
        b.setAttribute('aria-pressed', on ? 'true' : 'false');
        b.classList.toggle('solid', on);
      });
    }

    function bind(it, card) {
      current = it;
      active = card.querySelector('video');
      arrivedAt = performance.now();
      /* A card arriving at the front keeps playing exactly where it was. It used
         to be rewound, on the argument that "after each clip" should mean a whole
         clip -- but a card that has been looping in the wings is one nobody has
         asked anything of, and restarting it for having been looked at is a jump
         backwards the reader did not ask for. The pacing floor in tick() covers
         the case the rewind was there to prevent. */
      strip = trackStrip(trackHost, {
        tracks: it.tracks, duration: it.duration, legend: legend,
        keys: false,              // left and right belong to the deck, not the rail
        now: function () { return active.currentTime || 0; },
        onSeek: function (t) {
          active.currentTime = t;
          paint();
        }
      });
      paint();
      smooth();
    }

    // Before the mount, which binds the first card synchronously and calls it.
    var smooth = smoothPaint(host, function () { return active; }, paint);

    deck = TVDeck.mount(host, {
      items: clips,
      ariaLabel: 'Comparison clips',
      render: render,
      publishTo: under,           // the tracks line up with the card, not the page
      onVisible: function (vis, items, cards) {
        shown = vis;
        mounted = cards;
        sync();
      },
      dots: dots,
      dotLabel: function (it) { return it.dataset + ' clip, ' + it.domain; },
      onIndex: function (i, it, card) { bind(it, card); }
    });

    // On the deck, not on the document: neither event bubbles, so these have to
    // capture, and capturing on document would also catch section 04's videos,
    // whose pacing is its own handler's business and whose `active` is a
    // different element from this one's.
    host.addEventListener('timeupdate', function (ev) {
      if (ev.target === active) paint();
    }, true);
    host.addEventListener('ended', function (ev) {
      if (ev.target.tagName === 'VIDEO') tick(ev.target);
    }, true);

    new IntersectionObserver(function (entries) {
      onScreen = entries[0].isIntersecting;
      sync();
    }, { rootMargin: '200px 0px' }).observe(host);

    // A page loaded in a background tab has its autoplay refused, and the refusal
    // is not retried by anything else here -- sync() only runs when the deck
    // scrolls or changes card. Without this the clips are simply still when the
    // reader finally switches to the tab.
    document.addEventListener('visibilitychange', sync);
  }

  /* ------------------------------------------------- 03 the effect belts */

  /* Three belts over the 59, 20 / 20 / 19. The component owns the motion, the
     hold and the keyboard; this owns the items -- where the files are, what
     order they are in, and which of them is `cut`.

     The split is written out rather than derived. 59 is prime, so no even
     division exists, and a computed "roughly a third each" would silently
     re-partition the whole section the day a 60th effect lands in ffmpeg --
     which is exactly the kind of drift this directory keeps asserting against
     elsewhere. Three numbers that add to 59, and the component throws if they
     do not. */
  var BELTS = [20, 20, 19];

  function initStrips(effects) {
    var host = $('#strips');
    if (!host) return;
    var dir = 'assets/media/effects/';
    // cut first: it is the degenerate case the other 58 are measured against.
    var names = Object.keys(effects).sort(function (a, b) {
      if (a === 'cut') return -1;
      if (b === 'cut') return 1;
      return a.localeCompare(b);
    });
    TVStrips.mount(host, {
      rows: BELTS,
      // The belts drift and nothing plays until the reader points at a tile.
      // A card popping itself open on a section with no copy reads as the page
      // deciding what is interesting; the drift already brings all 59 past.
      tour: false,
      items: names.map(function (name) {
        return {
          id: name,
          poster: dir + name + '.jpg',
          media: dir + name + '.webm',
          title: name,
          body: effects[name],
          label: name + ' transition'
        };
      })
    });
  }

  /* ------------------------------------------------- 04 failure cases */

  /* The same deck as section 02, with two things switched off. It does not
     advance on its own: two cases are a claim to read, not a reel to watch, and
     a section that moves you off a failure you are still reading is worse than
     one that waits. And it does not loop -- with two cards, wrapping past the
     end lands you back where you started with no way to tell that it happened,
     so the ends stay ends.
   
     Its four tracks come straight from each method's prediction file rather than
     from a status grid burnt into the frames, so they are exact rather than
     decoded, and they are already in the excerpt's own timebase. */
  function initFailures(data, legend) {
    var host = $('#fail-deck');
    if (!host) return;
    var under = $('#fail-under');
    var dots = $('#fail-dots');
    var trackHost = $('.tracks', under);
    var readHost = $('.readout', under);
    var why = $('.why', under);

    // Declared ahead of the mount: TVDeck calls onIndex synchronously.
    var strip = null, active = null, current = null, deck = null, arrivedAt = 0;
    var shown = [], onScreen = true, running = true, mounted = [];

    function render(c) {
      var card = el('figure', 'card');
      var media = el('div', 'card-media' + (c.lowRes ? ' low' : ''));
      var v = document.createElement('video');
      v.muted = true;
      v.playsInline = true;
      v.preload = 'none';
      v.setAttribute('aria-label', c.modeLabel + ' failure case');
      if (c.poster) v.poster = c.poster;
      v.dataset.src = c.clip;
      v.addEventListener('click', function () { run(!running); });
      media.appendChild(v);
      card.appendChild(media);
      var foot = el('figcaption', 'card-foot');
      foot.appendChild(el('span', 'grp', c.id + ' · ' + c.modeLabel));
      card.appendChild(foot);
      return card;
    }

    function wake(v) {
      if (!v.src) {
        v.addEventListener('loadeddata', function () {
          if (v.dataset.want === '1') v.play().catch(function () {});
        });
        v.src = v.dataset.src;
        v.load();
      }
      v.dataset.want = '1';
      if (v.readyState >= 2) v.play().catch(function () {});
    }
    function sleep(v) {
      v.dataset.want = '0';
      if (!v.paused) v.pause();
    }
    function sync() {
      mounted.forEach(function (card, i) {
        var v = card.querySelector('video');
        if (running && onScreen && shown.indexOf(i) >= 0) wake(v); else sleep(v);
      });
    }
    function run(on) {
      running = on;
      host.classList.toggle('is-paused', !on);
      sync();
    }

    // The excerpt is the whole clip, so the wrap is just the end of it -- and
    // that is what moves the deck on, as in section 02. On a clock instead, one
    // interval would have to serve a 15.9 s excerpt and an 18 s one: short
    // enough for C1 and it cuts C2 off before the false positive it exists to
    // show, long enough for C2 and it holds C1 well past its end. The clip
    // knows when it is done. Only the front card counts; the neighbour loops
    // quietly. There are two cards and the deck does not wrap, so advance()
    // turns round at each end -- see the note on it in deck.js.
    function tick(v, ended) {
      if (v.dataset.free === '1') return;
      if (!ended && v.currentTime > 0.05) return;
      v.currentTime = 0;
      if (v.paused && v.dataset.want === '1') v.play().catch(function () {});
      if (v !== active || !deck || !running) return;
      if (performance.now() - arrivedAt < FRONT_MIN) return;
      deck.advance();
    }

    function paint() {
      if (!strip || !active || !current) return;
      var t = active.currentTime || 0;
      // Frames here too, on the user's call: the excerpt's frame rate comes with
      // its case, so a boundary reads in the unit its tracks are drawn in.
      readout(readHost, t, current.duration, strip.paint(t), { fps: current.fps, ours: legend.ours.label });
    }

    function bind(c, card) {
      current = c;
      arrivedAt = performance.now();
      active = card.querySelector('video');
      // Left where it was playing, like the clips in section 02: a card that has
      // been looping in the wings is not restarted for having been looked at.
      delete active.dataset.free;

      // Only the sentence that says what broke. The per-case F1/ABE/FP table and
      // the provenance line that used to sit here were the whole video's
      // figures, not the excerpt's, which took two more lines of caveat to
      // explain than the numbers were worth in a section whose point is the
      // shape of the tracks.
      why.innerHTML = '';
      why.appendChild(el('p', null, c.why));

      strip = trackStrip(trackHost, {
        tracks: c.tracks, duration: c.duration, legend: legend,
        keys: false,              // left and right belong to the deck, not the rail
        now: function () { return active.currentTime || 0; },
        onSeek: function (t) {
          active.dataset.free = '1';
          active.currentTime = t;
          paint();
        }
      });
      paint();
      smooth();
    }

    // Before the mount, for the same reason as in section 02.
    var smooth = smoothPaint(host, function () { return active; }, paint);

    deck = TVDeck.mount(host, {
      items: data.cases,
      ariaLabel: 'Failure cases',
      render: render,
      publishTo: under,
      loop: false,
      depth: 1,
      onVisible: function (vis, items, cards) {
        shown = vis;
        mounted = cards;
        sync();
      },
      dots: dots,
      dotLabel: function (c) { return c.id + ' \u2014 ' + c.modeLabel; },
      onIndex: function (i, c, card) { bind(c, card); }
    });

    host.addEventListener('timeupdate', function (ev) {
      if (ev.target.tagName !== 'VIDEO') return;
      if (ev.target === active) paint();
    }, true);
    host.addEventListener('ended', function (ev) {
      if (ev.target.tagName === 'VIDEO') tick(ev.target, true);
    }, true);

    new IntersectionObserver(function (entries) {
      onScreen = entries[0].isIntersecting;
      sync();
    }, { rootMargin: '200px 0px' }).observe(host);
    document.addEventListener('visibilitychange', sync);
  }

  /* ------------------------------------------- 06 the missed transitions */

  /* Fourteen examples the public labels do not mark, on eight cards in one deck.

     THE CARDS ARE NOT THE SUBFIGURES. A single strip of five wide frames is 9:1;
     the strips that share a source video are stacked into one card instead, so
     MovieShots2's four are a 5x4 grid at 2.25:1 rather than four slivers. Which
     strips read as one exhibit is a judgement about the frames, so it lives in
     the extractor's CARDS table, not in a rule here.

     ONE BOX, SIX SHAPES. A deck is one card shape and the cards run 2.25:1 to
     8.94:1, so the card is MISSED_BOX and each grid is centred inside it at its
     own ratio. 3.5:1 sits between the two clusters -- the 5x2 cards at ~4.45 and
     the 5x1 and 5x3 cards at ~2.9 -- so both land inside 80% of the box. The
     price is ClipShots 1, the one strip left on its own at 8.94:1, which keeps
     a third of the card's height; merging it into ClipShots 2-3 would fix that
     and is a one-line change to CARDS.

     NOT A BELT. Section 03's belts were the other candidate and do not fit: the
     unit here is a whole strip, which at belt height is wider than the belt, and
     a belt of individual frames would drift three identical tiles past the
     reader every time a shot holds still -- these strips repeat frames because
     the shot before the cut does not move.

     NO FILENAMES, NO TIMESTAMPS on the page. `-96lap9jYibdZbCgpB8pBw__.mp4`
     identifies a file, not the thing the reader is looking at, and "11.44-11.48
     s" is a coordinate in a video they cannot see. Both go in the markup, as
     section 02 does with its clip ids, and the card says "ClipShots 2-3".

     IT LOOPS AND ADVANCES, which section 04 deliberately does not. A failure
     case is a claim you may still be reading when the deck moves off it; these
     are instances of one fact, and an advance costs the reader nothing. */
  var MISSED_MS = 5200;
  var MISSED_BOX = 3.5;              // the card's aspect; see above

  function initMissed(data) {
    var host = $('#missed');
    if (!host) return;
    var dots = $('#missed-dots');
    /* The card's ratio is written from here rather than from the stylesheet,
       because render() below picks which axis of the grid to pin by comparing
       against it. Two copies of the number is one stylesheet edit away from
       grids that overflow their card. */
    host.style.setProperty('--card-ar', String(MISSED_BOX));

    function render(c) {
      var card = el('figure', 'card');
      // Provenance stays in the markup: enough to find the source frames again
      // without putting a filename on the page.
      card.dataset.sources = c.sources.map(function (s) {
        return s.video + '@' + s.start.toFixed(2) + '-' + s.end.toFixed(2);
      }).join(' ');

      var media = el('div', 'card-media');
      var wide = c.ar[0] / c.ar[1] >= MISSED_BOX;
      var grid = el('div', 'ms-strip ' + (wide ? 'is-wide' : 'is-tall'));
      grid.style.setProperty('--ms-cols', String(c.cols));
      grid.style.setProperty('--ms-ar', c.ar[0] + ' / ' + c.ar[1]);
      c.frames.forEach(function (src, i) {
        var img = el('img');
        img.src = src;
        img.loading = 'lazy';
        img.decoding = 'async';
        // One alt for the card, on its first frame. Twenty alts describing
        // twenty frames of the same four cuts is twenty times the noise and no
        // more information; the rest are presentational.
        img.alt = i ? '' : c.name + ': frames across ' +
          (c.sources.length > 1 ? c.sources.length + ' transitions' : 'a transition') +
          ' the published labels do not mark';
        grid.appendChild(img);
      });
      media.appendChild(grid);
      card.appendChild(media);

      var foot = el('figcaption', 'card-foot');
      foot.appendChild(el('span', 'name', c.name));
      foot.appendChild(el('span', 'grp', c.kind));
      card.appendChild(foot);
      return card;
    }

    TVDeck.mount(host, {
      items: data.cards,
      ariaLabel: 'Transitions missing from the published labels',
      render: render,
      autoplay: { ms: MISSED_MS },
      /* Wider than a deck's default card, because a grid is five frames across
         and the default 0.54 leaves each one 112 px. The slot stays well under
         the card so the neighbours keep stacking behind it. */
      metrics: function (w) {
        if (w < 480) return { slot: 0.40, card: 0.94, depth: 1 };
        if (w < 900) return { slot: 0.34, card: 0.88, depth: 1 };
        return { slot: 0.30, card: 0.78, depth: 1 };
      },
      dots: dots,
      dotLabel: function (c) { return c.name; }
    });
  }

  /* --------------------------------------------------- 05 method figures */

  /* The two figures on one deck, built from the <template> in the section. Same
     stack as section 04 and for the same reason: they are two views of one
     claim, and stacking them puts the second one where the reader can see it
     exists instead of below the fold.

     THE CARD IS SMALLER THAN THE FIGURE WAS. A full-width plate was ~1040 px;
     the active card is 0.80 of an inset deck, ~740. These are dense schematics
     with type set at print size, so the image is a link to the asset itself --
     which is what the old .plate zoom did, kept here rather than dropped.

     THE CAPTION IS IN THE CARD, under its picture, on the user's call. It sat
     in a strip under the deck before, swapping with the index, on the argument
     that prose inside a 2.7:1 card halves the picture and reads as noise on the
     dimmed neighbour. Against that: a caption that changes below a deck is a
     caption the reader has to re-find, and the two here differ by a line, so
     the page below moved on every card change. In the card, each figure
     carries its own words and nothing outside the deck moves. */
  // Long enough to read a schematic with three labelled panels in it. Section
  // 06's montage sits at 5.2 s, but that card is one glance at a grid of frames.
  var METHOD_MS = 7600;

  function initMethod() {
    var host = $('#method-deck');
    if (!host) return;
    var dots = $('#method-dots');

    var figs = [].slice.call($('#method-figs').content.querySelectorAll('figure'))
      .map(function (fg) {
        // The picture is a row of cropped panels, not one <img>: see .fig-split
        // in site.css. data-src is the whole figure, which is what the zoom
        // link opens -- a panel is a view of it, not a file of its own.
        return {
          split: document.importNode(fg.querySelector('.fig-split'), true),
          src: fg.dataset.src,
          title: fg.dataset.title,
          tag: fg.dataset.tag,
          caption: fg.querySelector('figcaption').innerHTML
        };
      });

    function render(f) {
      var card = el('figure', 'card');
      var media = el('div', 'card-media pad split');
      var a = el('a', 'zoom');
      a.href = f.src;
      a.target = '_blank';
      a.rel = 'noopener';
      a.title = 'Open at full size';
      a.appendChild(f.split);
      media.appendChild(a);
      card.appendChild(media);

      // The figure's own caption, from the <template>, under its picture.
      var body = el('div', 'card-body fig-cap');
      body.innerHTML = f.caption;
      card.appendChild(body);

      var foot = el('figcaption', 'card-foot');
      foot.appendChild(el('span', 'name', f.title));
      foot.appendChild(el('span', 'grp', f.tag));
      card.appendChild(foot);
      return card;
    }

    TVDeck.mount(host, {
      items: figs,
      ariaLabel: 'Method figures',
      render: render,
      // Two cards, as in section 04: wrapping past the end lands you back where
      // you started with nothing to say that it happened, so the ends stay ends.
      // It still rotates -- advance() turns round at an end rather than wrapping
      // past it. On a clock rather than on the content, because a still figure
      // has no rhythm of its own to advance on; long enough to read three
      // labelled panels, and the deck holds it while the pointer is over it.
      loop: false,
      autoplay: { ms: METHOD_MS },
      depth: 1,
      /* Far wider than a deck's default card. A 2.7:1 schematic at the default
         0.54 is 500 px across for a figure with three labelled panels in it. The
         price is that the neighbour is a sliver rather than a whole card edge:
         card/2 + slot runs past the viewport's half-width, so the mask cuts it.
         That is the same trade the montage deck in section 06 makes. */
      metrics: function (w) {
        if (w < 480) return { slot: 0.30, card: 0.94, depth: 1 };
        if (w < 900) return { slot: 0.29, card: 0.88, depth: 1 };
        return { slot: 0.28, card: 0.80, depth: 1 };
      },
      dots: dots,
      dotLabel: function (f) { return f.title; }
    });
  }

  /* ------------------------------------------------------- 00 the hero */

  /* A DECODER PER TILE, SO THE BUDGET SCALES WITH AREA. Tiles are drawn at a
     fraction of the host's width, so a bigger viewport does not make them
     bigger relative to it -- it makes room for more of them, and an area-linear
     count is what keeps the stage looking equally busy at every size instead of
     sparse on a laptop and crowded on a phone. 150000 px^2 per tile puts it at
     9 on a 1440x900 laptop, 12 at 1920x1080 (the cap), and 4 on a 390x844
     phone, where the clamp floor also matches the governor's own minCount. The ceiling is
     not a performance guess: the governor MEASURES frame time and permanently
     retires tiles when the machine cannot keep up, so this only has to be a
     sane opening bid, not a safe one. */
  function heroCount() {
    return Math.max(4, Math.min(12, Math.round(innerWidth * innerHeight / 150000)));
  }

  function initHero() {
    /* No `if (!bg) return` guard, unlike the sections below: index.html is the
       only page that loads this file, and .hero-bg is in its markup. A guard
       here would be a branch that can only ever fire on a page that has broken
       its own hero -- which should throw, not silently render nothing. */
    var bg = $('.hero-bg');
    getJSON('assets/data/samples.json').then(function (data) {
      /* The file nests samples under their source video and states the media
         root once. Flattened, because the collage has no notion of a source
         video and should not be handed one. */
      var items = [];
      data.videos.forEach(function (v) {
        v.samples.forEach(function (s) {
          items.push({
            id: s.id,
            src: data.media + '/' + s.hero,
            poster: data.media + '/' + s.poster,
            width: s.heroOut.width,
            height: s.heroOut.height,
            duration: s.duration
          });
        });
      });
      /* Not remounted on resize: the component re-measures the stage itself on
         every resize, and a remount would throw away every live decoder to
         arrive at the same tiles. */
      /* Tighter than the component's defaults on the user's call: tiles between
         a fifth and a third of the stage rather than a sixth and nearly two
         fifths, so no tile dominates and none reads as a thumbnail, and a
         faster turnover -- one or two loops a life and at most 0.9 s of dark
         slot -- so a new transition arrives somewhere every few seconds. */
      window.TVCollage.mount(bg, {
        items: items, count: heroCount(), minCount: 4,
        size: [0.2, 0.32], life: [1, 2], gap: [0, 900]
      });
    }).catch(function (e) { console.error('[TransVLM]', e.message); });
  }

  /* ------------------------------------------------ 07 the results figure */

  // One figure, four controls, over the numbers the published project page
  // carries. tools/build_std_results.py writes the data file and asserts it
  // back against that page's markup.
  function initResults() {
    var host = $('#explore');
    if (!host) return;
    getJSON('assets/data/std-results.json').then(function (data) {
      window.TVExplore.mount(host, data);
    }).catch(function (e) { console.error('[TransVLM]', e.message); });
  }

  /* ------------------------------------------------------------ misc */

  function legacyCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.cssText = 'position:fixed;top:-9999px;opacity:0';
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    document.body.removeChild(ta);
    return ok;
  }

  function initCopy() {
    document.querySelectorAll('.js-copy').forEach(function (btn) {
      var label = btn.textContent;
      function flash(msg) {
        btn.textContent = msg;
        setTimeout(function () { btn.textContent = label; }, 1800);
      }
      btn.addEventListener('click', function () {
        var pre = document.getElementById(btn.dataset.copy);
        if (!pre) return;
        var text = pre.innerText.trim();
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text)
            .then(function () { flash('Copied'); })
            .catch(function () { flash(legacyCopy(text) ? 'Copied' : 'Select and copy'); });
        } else {
          flash(legacyCopy(text) ? 'Copied' : 'Select and copy');
        }
      });
    });
  }

  /* ------------------------------------------------------------ boot */

  document.addEventListener('DOMContentLoaded', function () {
    initTheme();
    initRail();
    initHash();
    initCopy();
    initMethod();
    initResults();
    initHero();

    getJSON('assets/data/samples.json').then(function (d) {
      /* The file nests samples under their source video and states the media
         root once. Flattened: a card is a sample, and the deck has no notion of
         a source video -- dataset and domain are the only two facts about the
         source it needs, and they are the two the card foot prints.

         `effects` is the ffmpeg effect name of each covered transition, and it
         is empty for everything but the synthetic clips: a transition in real
         footage was not rendered by anything, so `effect` is null there. */
      var items = [];
      d.videos.forEach(function (v) {
        v.samples.forEach(function (s) {
          items.push({
            id: s.id,
            src: d.media + '/' + s.src,
            poster: d.media + '/' + s.poster,
            duration: s.duration,
            // The source video's frame rate, so the readout can count frames:
            // a sample is a cut out of that file and keeps its timebase.
            fps: v.fps,
            tracks: s.tracks,
            dataset: v.dataset,
            domain: v.domain,
            // Deduplicated: a window covering two slides of the same kind names
            // the effect once, not "hlwind, hlwind".
            effects: s.transitions.map(function (i) { return v.transitions[i].effect; })
              .filter(function (name, k, all) { return name && all.indexOf(name) === k; })
          });
        });
      });
      initPlayer(items, d.legend);
      return getJSON('assets/data/failures.json').then(function (f) {
        initFailures(f, d.legend);
      });
    }).catch(function (e) { console.error('[TransVLM]', e.message); });

    getJSON('assets/data/effects.json')
      .then(initStrips)
      .catch(function (e) { console.error('[TransVLM]', e.message); });

    getJSON('assets/data/missed.json')
      .then(initMissed)
      .catch(function (e) { console.error('[TransVLM]', e.message); });
  });
})();
