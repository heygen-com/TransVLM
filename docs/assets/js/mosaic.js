/* TVMosaic — all fifty-nine at once, one of them awake.
 *
 * THE SHAPE IS DATA. hexmap's best structural idea is that a hexagon is one
 * swappable function rather than a mesh, so the renderer draws a different
 * shape by evaluating a different signed distance (gpumon/hexmap/src/shaders.ts,
 * `sdHex`). That idea is worth more here than its code: TILINGS below is the
 * same extension point in DOM terms -- a lattice, a step, and a polygon -- so
 * hex, rect and triangle are table entries, not code paths. Every polygon is
 * written with SIX vertices, including the rectangle each one opens into,
 * because clip-path only interpolates between polygons of equal length. That
 * constraint is the whole reason the table looks the way it does.
 *
 * THE TWO-OFFSET MORPH, in DOM. hexmap gives every tile a `from` and a `to`
 * centre and mixes them with one uniform, so re-laying out 100k tiles costs
 * one scalar per frame. Here there is one travelling element, so the same idea
 * is two springs and a lerp -- but the reason to want it is identical: the
 * thing that moves carries where it came from, so a redirect mid-flight is a
 * curve rather than a restart.
 *
 * WHAT IS NOT BORROWED, and why. hexmap exists because a DOM cannot hold
 * 200,000 tiles. Fifty-nine is not that problem. It also cannot show video --
 * its fill is `texture2D(uPalette, value)`, a colour ramp indexed by a number,
 * and video tiles would need a per-instance atlas rect, a new sampler, and a
 * texSubImage2D per visible clip per frame on top of the decode we already
 * pay. And the hover state here is an enlarged frame with prose under it,
 * which leaves the canvas for the DOM regardless (hexmap ships underlay.ts for
 * exactly that). Fifty-nine real <button>s are keyboard-reachable and
 * screen-reader-legible for free, which is the cost hexmap has to buy back
 * with an a11y mirror.
 *
 * ONE CLIP PLAYS AT A TIME. The other tiles are their own poster frames. That
 * is what makes a still honeycomb worth looking at, and it means a panel of any
 * size costs one video decode rather than one per tile.
 *
 * ---------------------------------------------------------------------------
 * API
 *
 *   TVMosaic.mount(host, {
 *     items: [{ id, poster, media?, title?, body?, label? }],  // required
 *     shape:  'hex' | 'hexPointy' | 'rect' | 'square' | 'tri', // 'hex'
 *     ratio:  16 / 9,        // the open card's aspect
 *     cols:   null,          // null fits the container; a number is honoured
 *     target: 118,           // wanted cell width in px, when fitting
 *     count:  items.length,  // show fewer than all of them
 *     fill:   0.965,         // tile size inside its lattice slot
 *     zoom:   2.7,           // open card width, in cell widths
 *     dwell:  { loops: 2 },  // or { ms: 3200 } where the items are stills
 *     resume: 1400,          // ms after the pointer leaves before the tour resumes
 *     tour:   true,
 *     onOpen: function (item, index) {}
 *   })
 *   -> { show, sleep, set, get, shapes, destroy }
 *
 * `items` is the whole contract. It carries no convention about where the files
 * live and no opinion about their order -- a component that sorted its own
 * items would be imposing an argument it does not have. `poster` is required on
 * every item because the panel paints before any media can load; `media` is
 * optional, and an item without it is a still.
 *
 * It needs assets/css/mosaic.css and assets/js/spring.js, and it reads the host
 * page's tokens (--bg-elev, --line, --line-2, --fg, --fg-2, --fg-3, --bg-sunk,
 * --accent, --mono) where they exist. Every one has a fallback, so it renders
 * standalone.
 */
window.TVMosaic = (function () {
  'use strict';

  var S = window.TVSpring;
  if (!S) throw new Error('TVMosaic: assets/js/spring.js must load first');

  var SQ3 = Math.sqrt(3);

  /* Each tiling gives: the cell box from a base width, how many base widths
     wide and tall the whole lattice is, where cell (i, j) sits, and the
     polygon -- six vertices, in the same order as the rectangle it opens into.
     `clip` may be a function of (i, j) for lattices whose cells alternate. */
  var TILINGS = {
    hex: {
      label: 'Hexagon (flat top)',
      cell: function (w) { return { w: w, h: w * SQ3 / 2 }; },
      spanW: function (cols) { return 0.75 * cols + 0.25; },
      spanH: function (rows, c) { return c.h * (rows + 0.5); },
      place: function (i, j, c) {
        return { x: i * c.w * 0.75, y: j * c.h + (i % 2 ? c.h / 2 : 0) };
      },
      clip: [[25, 0], [75, 0], [100, 50], [75, 100], [25, 100], [0, 50]],
      open: [[0, 0], [100, 0], [100, 50], [100, 100], [0, 100], [0, 50]]
    },
    hexPointy: {
      label: 'Hexagon (pointy top)',
      cell: function (w) { return { w: w, h: w * 2 / SQ3 }; },
      spanW: function (cols) { return cols + 0.5; },
      spanH: function (rows, c) { return c.h * (0.75 * rows + 0.25); },
      place: function (i, j, c) {
        return { x: i * c.w + (j % 2 ? c.w / 2 : 0), y: j * c.h * 0.75 };
      },
      clip: [[50, 0], [100, 25], [100, 75], [50, 100], [0, 75], [0, 25]],
      open: [[50, 0], [100, 0], [100, 100], [50, 100], [0, 100], [0, 0]]
    },
    rect: {
      label: 'Rectangle (16:9)',
      cell: function (w) { return { w: w, h: w * 9 / 16 }; },
      spanW: function (cols) { return cols; },
      spanH: function (rows, c) { return c.h * rows; },
      place: function (i, j, c) { return { x: i * c.w, y: j * c.h }; },
      clip: [[0, 0], [50, 0], [100, 0], [100, 100], [50, 100], [0, 100]],
      open: [[0, 0], [50, 0], [100, 0], [100, 100], [50, 100], [0, 100]]
    },
    square: {
      label: 'Square',
      cell: function (w) { return { w: w, h: w }; },
      spanW: function (cols) { return cols; },
      spanH: function (rows, c) { return c.h * rows; },
      place: function (i, j, c) { return { x: i * c.w, y: j * c.h }; },
      clip: [[0, 0], [50, 0], [100, 0], [100, 100], [50, 100], [0, 100]],
      open: [[0, 0], [50, 0], [100, 0], [100, 100], [50, 100], [0, 100]]
    },
    tri: {
      label: 'Triangle',
      cell: function (w) { return { w: w, h: w * SQ3 / 2 }; },
      spanW: function (cols) { return 0.5 * cols + 0.5; },
      spanH: function (rows, c) { return c.h * rows; },
      place: function (i, j, c) { return { x: i * c.w * 0.5, y: j * c.h }; },
      // Up and down triangles interlock along a row. Both are written with the
      // two edge midpoints included, so each has six vertices and its own
      // rectangle to open into.
      clip: function (i, j) {
        return (i + j) % 2
          ? [[0, 0], [50, 0], [100, 0], [75, 50], [50, 100], [25, 50]]
          : [[50, 0], [75, 50], [100, 100], [50, 100], [0, 100], [25, 50]];
      },
      open: function (i, j) {
        return (i + j) % 2
          ? [[0, 0], [50, 0], [100, 0], [100, 100], [50, 100], [0, 100]]
          : [[50, 0], [100, 0], [100, 100], [50, 100], [0, 100], [0, 0]];
      }
    }
  };

  /* play() rejects for two reasons that are both expected and both nothing to
     do about here: the browser's autoplay policy declining a muted play in a
     tab that has had no interaction, and a play() cut short by the next load()
     when the tour moves on mid-buffer. Neither is recoverable and neither is a
     fault -- the poster is already painted underneath, so the card is correct
     either way. Anything else surfaces as a media `error` event, which is not
     swallowed. One copy, so that reasoning is stated once rather than five
     times or, as it was, not at all. */
  function roll(v) {
    var p = v.play();
    if (p && p.catch) p.catch(function () {});
  }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function poly(pts) {
    var out = [];
    for (var i = 0; i < pts.length; i++) out.push(pts[i][0] + '% ' + pts[i][1] + '%');
    return 'polygon(' + out.join(',') + ')';
  }

  function lerpPoly(a, b, t) {
    var out = [];
    for (var i = 0; i < a.length; i++) {
      out.push((a[i][0] + (b[i][0] - a[i][0]) * t).toFixed(2) + '% ' +
               (a[i][1] + (b[i][1] - a[i][1]) * t).toFixed(2) + '%');
    }
    return 'polygon(' + out.join(',') + ')';
  }

  function shuffled(n) {
    var a = [];
    for (var i = 0; i < n; i++) a.push(i);
    for (var j = n - 1; j > 0; j--) {
      var k = Math.floor(Math.random() * (j + 1));
      var t = a[j]; a[j] = a[k]; a[k] = t;
    }
    return a;
  }

  function mount(host, opts) {
    var items = opts.items;
    if (!Array.isArray(items) || !items.length) {
      throw new Error('TVMosaic: opts.items must be a non-empty array');
    }
    items.forEach(function (it, i) {
      if (!it || !it.poster) {
        throw new Error('TVMosaic: items[' + i + '] has no poster — every tile ' +
          'needs an image, because the panel paints before any media can load');
      }
    });
    var reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

    /* Ordering is the caller's. A component that sorted its own items would be
       imposing an argument it does not have. */
    var onOpen = opts.onOpen || function () {};
    var dwell = opts.dwell || { loops: 2 };
    if (dwell.loops && !items.some(function (it) { return it.media; })) {
      throw new Error('TVMosaic: dwell.loops counts a clip wrapping, and no item ' +
        'has media — pass dwell: { ms: N } for a mosaic of stills');
    }

    var st = {
      shape: opts.shape || 'hex',
      // null = fit the container. An explicit column count is honoured, and
      // the row count follows from it: every effect is always on screen, so
      // the grid is sized to the library rather than sampled from it.
      cols: opts.cols || null,
      target: opts.target == null ? 118 : opts.target,   // wanted cell width in px
      count: Math.min(opts.count == null ? items.length : opts.count, items.length),
      // hexmap's LayerOptions.fill, same meaning: the tile is drawn slightly
      // smaller than its lattice slot. A margin would break the interlock;
      // a uniform shrink leaves a hairline and keeps the polygon exact.
      fill: opts.fill == null ? 0.965 : opts.fill,
      zoom: opts.zoom == null ? 2.7 : opts.zoom,         // open card width, in cell widths
      ratio: opts.ratio == null ? 16 / 9 : opts.ratio,   // the open card's aspect
      resume: opts.resume == null ? 1400 : opts.resume,  // ms before the tour resumes
      tour: opts.tour !== false && !reduced,
      active: -1,
      pointer: false,
      cols_: 0, rows_: 0, cell: { w: 0, h: 0 }
    };

    host.classList.add('mosaic');
    host.innerHTML = '';

    var grid = el('div', 'mo-grid');
    var spot = el('div', 'mo-spot');
    // The panel behind the frame and its caption, faded and inflated with the
    // open. It is a separate element because the media above it is clipped to
    // a hexagon for most of the animation and a clip cuts a border off.
    var plate = el('div', 'mo-plate');
    var spotShape = el('div', 'mo-spot-shape');
    var spotImg = el('img', 'mo-spot-poster');
    spotImg.alt = '';
    var spotVideo = el('video');
    spotVideo.muted = true; spotVideo.playsInline = true; spotVideo.loop = true;
    spotVideo.preload = 'auto';
    spotVideo.setAttribute('aria-hidden', 'true');
    spotShape.appendChild(spotImg);
    spotShape.appendChild(spotVideo);
    var cap = el('div', 'mo-cap');
    var capName = el('p', 'mo-cap-n');
    var capText = el('p', 'mo-cap-d');
    cap.appendChild(capName);
    cap.appendChild(capText);
    spot.appendChild(plate);
    spot.appendChild(spotShape);
    spot.appendChild(cap);
    grid.appendChild(spot);
    host.appendChild(grid);

    // The section has to say what it is showing without the pointer being on
    // anything. aria-live, so a screen reader hears the tour move too.
    var live = el('p', 'mo-live');
    live.setAttribute('aria-live', 'polite');
    host.appendChild(live);

    /* ------------------------------------------------------------- cells */

    var cells = [];   // { i, j, btn, name, x, y }
    var warmed = false;

    function build() {
      cells.forEach(function (c) { c.btn.remove(); });
      cells = [];
      for (var k = 0; k < st.count; k++) {
        var btn = el('button', 'mo-cell');
        btn.type = 'button';
        btn.tabIndex = k === 0 ? 0 : -1;
        var it = items[k];
        btn.setAttribute('aria-label', it.label || it.title || it.id);
        /* No src yet, and deliberately not loading="lazy" either.
         *
         * The whole set is 476 kB and it is only ever looked at as one thing,
         * so the two states worth having are "not fetched" and "all of it" --
         * and lazy gives neither. Its heuristic decides per image against the
         * viewport, and these are absolutely positioned with no size until
         * layout() runs, so a panel arriving on screen paints a scatter of
         * empty hexes that fill in one by one. Whatever cell the reader points
         * at first is the one most likely to still be blank.
         *
         * So the srcs are filled in from one observer, once, a screen early.
         * See `warm` below. */
        var img = el('img');
        img.dataset.src = it.poster;
        img.alt = '';
        img.decoding = 'async';
        img.width = 320; img.height = 180;
        btn.appendChild(img);
        btn.appendChild(el('span', 'mo-cell-n', it.title || it.id));
        var c = { k: k, item: it, btn: btn, x: 0, y: 0, i: 0, j: 0 };
        btn.__c = c;
        cells.push(c);
        grid.appendChild(btn);
      }
      bindCells();
      // A rebuild -- a shape or column change -- makes 59 new <img>s, and the
      // warm-up observer has long since disconnected. Whatever was already
      // fetched is in the HTTP cache, so this is free.
      if (warmed) cells.forEach(function (c) {
        var img = c.btn.querySelector('img');
        img.src = img.dataset.src;
      });
      layout();
    }

    /* ------------------------------------------------------------ layout */

    function layout() {
      var T = TILINGS[st.shape];
      var W = grid.clientWidth;
      if (!W) return;

      var cols = st.cols || fitCols(W, T);
      var rows = Math.ceil(st.count / cols);
      /* 59 is prime, so no column count divides it and the last row is always
         short. Left-aligned that reads as a stub hanging off the bottom-left
         with holes up and to its right; shifted by whole columns it reads as
         the bottom of a honeycomb. Whole columns, because a fractional shift
         would take the cells off the lattice and they would stop interlocking. */
      var stub = st.count - (rows - 1) * cols;
      var shift = stub === cols ? 0 : Math.floor((cols - stub) / 2);
      var baseW = W / T.spanW(cols);
      var cell = T.cell(baseW);

      st.cols_ = cols; st.rows_ = rows; st.cell = cell;
      var fw = cell.w * st.fill, fh = cell.h * st.fill;
      st.fw = fw; st.fh = fh;
      grid.style.height = T.spanH(rows, cell) + 'px';

      cells.forEach(function (c, k) {
        var j = Math.floor(k / cols);
        var i = k % cols + (j === rows - 1 ? shift : 0);
        var p = T.place(i, j, cell);
        c.i = i; c.j = j; c.x = p.x; c.y = p.y;
        c.btn.style.width = fw + 'px';
        c.btn.style.height = fh + 'px';
        c.btn.style.transform =
          'translate3d(' + (p.x + (cell.w - fw) / 2) + 'px,' +
                           (p.y + (cell.h - fh) / 2) + 'px,0)';
        var cl = typeof T.clip === 'function' ? T.clip(i, j) : T.clip;
        c.btn.style.clipPath = poly(cl);
      });

      // The caption is as wide as the open frame and no wider, so the panel is
      // one rectangle and the prose cannot reflow part-way through an opening.
      cap.style.width = openW() + 'px';
      if (st.active >= 0) { anchor(st.active); }
      if (shown >= 0) capH = cap.offsetHeight;
      wrote.w = wrote.h = -1;
      paint();
    }

    /* ---------------------------------------------------- the spotlight */

    /* THE CARD NEVER TRAVELS. It belongs to one cell, grows out of that cell's
       own slot, and goes back into it -- so the anchor is a plain number, set
       only while the card is invisible, rather than a spring. Making it a
       number is the point: a card that could be interpolated between two cells
       is a card that can be caught between them with no home to return to, and
       the whole reason the open state reads as the CELL enlarging is that it
       has never been anywhere else.

       One spring, retuned. Opening and closing are the same object reversing,
       which is what lets a pointer that comes back mid-retraction re-expand
       from where it got to instead of from the beginning. */
    var ax = 0, ay = 0;
    /* True while the card is on its way home with nothing to open after it --
       the difference between a hop (a queued target) and a sleep (none). */
    var homing = false;
    var so = S.Spring(0, S.PRESET.open, 0.0006);
    /* Below this the card is visually home; the remaining tail of the spring
       is under a pixel and there is nothing to wait for. */
    var CLOSED = 0.02;
    var PAD = 10;   // the plate's margin around the frame, px at full open

    /* Choosing the column count is not just "what gets closest to the wanted
       cell size". At 1040 px and a 118 px target, 11 columns give a 122 px cell
       -- almost exactly right -- and 11x6 = 66 slots for 59 cells, so SEVEN of
       them are empty and all seven are in the last row. 12 columns give a 113 px
       cell, 4% smaller, and 12x5 = 60 slots: one hole, in a corner. The second
       is obviously the better grid, and a solver that only minimises size error
       cannot see it. So both are scored, with a whole empty slot weighted
       against a sixth of the target size. */
    function fitCols(W, T) {
      var slope = T.spanW(1) - T.spanW(0);
      var ideal = (W / st.target - T.spanW(0)) / slope;
      var lo = Math.max(3, Math.floor(ideal) - 3);
      // Reaching well past the ideal, because the height term below can want
      // many more columns than the cell size does: at 375 px the wanted cell
      // gives four columns and fifteen rows of honeycomb, 1417 px of it.
      var hi = Math.min(st.count, Math.ceil(ideal) + 8);
      /* Nine tenths of the viewport, capped. A panel taller than the screen
         stops being one thing you can look at, and on a phone the wanted cell
         size will happily produce two screens of it. */
      var maxH = Math.min(innerHeight * 0.9, 760);
      var best = null;
      for (var c = lo; c <= hi; c++) {
        var cell = T.cell(W / T.spanW(c));
        var waste = c * Math.ceil(st.count / c) - st.count;
        var err = Math.abs(cell.w - st.target) / st.target;
        var over = Math.max(0, T.spanH(Math.ceil(st.count / c), cell) / maxH - 1);
        var score = waste + err * 6 + over * 8;
        if (!best || score < best.score) best = { c: c, score: score };
      }
      // lo > hi is impossible: lo is at least 3 and hi is at least lo, but a
      // container narrower than three cells would make `ideal` less than 3 and
      // the loop would still run from 3.
      return best ? best.c : lo;
    }

    /* One definition of the open size, because the layout sets the caption's
       width from it and paint() sets the frame's -- and a panel whose two
       halves disagree by a pixel is a panel with a seam. */
    function openW() {
      return Math.min(st.fw * st.zoom,
                      grid.clientWidth * 0.62,
                      grid.clientWidth - PAD * 2);
    }

    function anchor(k) {
      var c = cells[k];
      if (!c) return;
      ax = c.x + st.cell.w / 2;
      ay = c.y + st.cell.h / 2;
    }

    /* Geometry is derived every frame from three springs rather than stored,
       so an interrupted hop has no stale state to reconcile. `wrote` is not
       state, it is a write cache: the only property that changes on every frame
       of a hop at full size is the transform, and the rest would re-lay the
       card out for nothing. */
    var wrote = { w: -1, h: -1, vis: '', open: -1, cell: null };
    /* Measured once per content change, never per frame. The caption's width is
       fixed by the layout so its height cannot change mid-animation, and a
       guessed reserve is a guess that clips the longest definition. */
    var capH = 96;

    function paint() {
      var T = TILINGS[st.shape];
      var c = cells[st.active];
      var open = so.x;
      var cw = st.fw, ch = st.fh;
      var ow = openW();
      var oh = ow / st.ratio;

      var w = cw + (ow - cw) * open;
      var h = ch + (oh - ch) * open;

      // The closed box sits exactly on its cell; the open box is centred and
      // then pushed back inside the grid. Clamping only the open box, and
      // interpolating, is what keeps an edge cell's growth continuous instead
      // of snapping as soon as it starts.
      var cx = ax, cy = ay;
      var shutX = cx - cw / 2, shutY = cy - ch / 2;
      // The whole panel has to fit, not just the frame: the plate's margin and
      // the caption below it are part of the footprint being clamped.
      var lo = PAD, hiX = grid.clientWidth - ow - PAD;
      var hiY = grid.clientHeight - oh - capH - PAD;
      var ox = hiX < lo ? lo : Math.max(lo, Math.min(hiX, cx - ow / 2));
      var oy = hiY < lo ? lo : Math.max(lo, Math.min(hiY, cy - oh / 2));
      var x = shutX + (ox - shutX) * open;
      var y = shutY + (oy - shutY) * open;

      spot.style.transform = 'translate3d(' + x + 'px,' + y + 'px,0)';
      if (w !== wrote.w || h !== wrote.h) {
        spot.style.width = w + 'px';
        spot.style.height = h + 'px';
        wrote.w = w; wrote.h = h;
      }
      var vis = open <= 0.001 ? '0' : '1';
      if (vis !== wrote.vis) { spot.style.opacity = vis; wrote.vis = vis; }

      // The panel inflates from nothing: at open 0 it is exactly the frame's
      // box with no margin, no radius and no opacity, so the cell is all there
      // is; at open 1 it is a solid card wrapping the frame and the prose.
      var pad = PAD * open;
      plate.style.left = -pad + 'px';
      plate.style.top = -pad + 'px';
      plate.style.width = (w + pad * 2) + 'px';
      plate.style.height = (h + capH * open + pad * 2) + 'px';
      plate.style.opacity = open;

      if (c && (open !== wrote.open || c !== wrote.cell)) {
        var cl = typeof T.clip === 'function' ? T.clip(c.i, c.j) : T.clip;
        var op = typeof T.open === 'function' ? T.open(c.i, c.j) : T.open;
        spotShape.style.clipPath = lerpPoly(cl, op, open);
        wrote.open = open; wrote.cell = c;
      }
      // The caption arrives in the last part of the open, so it does not read
      // as text being stretched.
      var ct = Math.max(0, (open - 0.45) / 0.55);
      cap.style.opacity = ct;
      cap.style.transform = 'translateY(' + ((1 - ct) * -8) + 'px)';
    }

    var raf = 0, last = 0;
    function loop(now) {
      var dt = last ? (now - last) / 1000 : 1 / 60;
      last = now;
      var done = so.step(dt);
      paint();
      // Step two of a hop, and the end of a sleep, both happen the moment the
      // card is home -- not on a timer that would have to be kept in step with
      // a spring it cannot see.
      if (so.x <= CLOSED) {
        if (queued >= 0) { flip(); raf = requestAnimationFrame(loop); return; }
        if (homing && done) { homing = false; st.active = -1; shown = -1; }
      }
      if (done) { raf = 0; last = 0; return; }
      raf = requestAnimationFrame(loop);
    }
    function kick() {
      if (raf) return;
      last = 0;
      raf = requestAnimationFrame(loop);
    }

    /* ------------------------------------------------------- the content */

    var wantTimer = 0, want = -1, queued = -1;

    /* A HOP IS TWO STEPS, in this order and never overlapped: the cell that is
       open shrinks back into its own slot, and only once it is home does the
       next one grow out of its own. That is the whole difference between a card
       that lives somewhere and a card that flies -- and it is why there is a
       `queued` target rather than a second animation.
     *
     * The 80 ms is not a debounce on the CONTENT, it is one on the INTENTION.
     * A pointer sweeping across the honeycomb crosses six cells on its way to
     * the seventh; without this each crossing starts a retraction that the next
     * one interrupts, and the card never opens at all. Focus and the public
     * API skip it, because both are already a decision. */
    function show(k, immediate) {
      if (k === want && !immediate) return;
      want = k;
      clearTimeout(wantTimer);
      if (immediate) commit();
      else wantTimer = setTimeout(commit, 80);
    }

    function commit() {
      var k = want;
      if (k < 0) return;
      if (k === st.active) {
        // Back to the one that is already open, or on its way home. Reversing
        // the same spring keeps the velocity it had, so coming back mid-close
        // is a curve rather than a restart from nothing.
        homing = false;
        if (so.target !== 1) {
          so.tune(S.PRESET.open).to(1);
          roll(spotVideo);
          kick();
        }
        return;
      }
      queued = k;
      if (so.x <= CLOSED) flip();
      else { so.tune(S.PRESET.shut).to(0); kick(); }
    }

    /* Step two. Only ever called with the card home, which is what makes
       re-anchoring instant rather than a move: there is nothing on screen to
       move. */
    function flip() {
      var k = queued;
      queued = -1;
      st.active = k;
      cells.forEach(function (c, i) { c.btn.classList.toggle('on', i === k); });
      host.classList.add('awake');
      homing = false;
      anchor(k);
      content(k);
      // Unconditional: a sleep paused it, and re-opening the same cell takes
      // content()'s early return, so this is the only place that can resume.
      roll(spotVideo);
      so.tune(S.PRESET.open);
      if (reduced) so.set(1); else so.to(1);
      kick();
    }

    var shown = -1;

    function content(k) {
      if (k === shown) return;
      shown = k;
      var it = cells[k].item;
      var title = it.title || it.id;
      spot.classList.remove('rolling');
      spotImg.src = it.poster;
      // An item with no media is a still, and `rolling` is what reveals the
      // video layer -- so it simply never gets added and the poster is the
      // whole card. The src is cleared rather than left: a stale clip playing
      // under a still it does not belong to is worse than a blank element.
      if (it.media) spotVideo.src = it.media;
      else { spotVideo.removeAttribute('src'); spotVideo.load(); }
      capName.textContent = title;
      capText.textContent = it.body || '';
      live.textContent = it.body ? title + ' — ' + it.body : title;
      onOpen(it, k);
      armDwell();
      // Measured here, with the caption at its final width and its final text,
      // so the plate reserves exactly the prose it is about to show.
      capH = cap.offsetHeight;
      wrote.h = -1;
      loopsSeen = 0;
      // Reset WITH loopsSeen, not just it: lastT still holds the outgoing
      // clip's playhead, and the incoming clip's first frame at t ~= 0 is
      // indistinguishable from a wrap. Left behind, every cell counted a loop
      // it had never played and the dwell was half what it says it is.
      lastT = 0;
      roll(spotVideo);
    }

    // The poster holds the frame until the clip can actually paint, so a
    // cell-to-cell hop never shows a black rectangle. Both are the same two
    // shots, so the handoff is invisible.
    spotVideo.addEventListener('loadeddata', function () {
      spot.classList.add('rolling');
      roll(spotVideo);
    });

    function sleep() {
      want = -1; queued = -1;
      clearTimeout(wantTimer);
      clearTimeout(msTimer);
      cells.forEach(function (c) { c.btn.classList.remove('on'); });
      host.classList.remove('awake');
      so.tune(S.PRESET.shut).to(0);
      kick();
      spotVideo.pause();
      live.textContent = '';
      // st.active is deliberately NOT cleared until the card is home: paint()
      // needs the cell it is retracting into for as long as it is retracting.
      homing = true;
    }

    /* ---------------------------------------------------------- the tour */

    var deck = [], loopsSeen = 0, lastT = 0, resumeTimer = 0, msTimer = 0;

    /* Two ways to decide the tour has stayed long enough, and only one of them
       is a stopwatch. Where there is a clip, the clip is asked -- a wall-clock
       dwell cuts a 0.6 s transition in half about as often as not. Where there
       is not, a timer is all there is. */
    function armDwell() {
      clearTimeout(msTimer);
      if (!dwell.ms || !st.tour || st.pointer) return;
      msTimer = setTimeout(function () { show(draw()); }, dwell.ms);
    }

    /* A shuffled deck, not an independent random draw each time: over one pass
       the tour shows every one of the 59 before it repeats any. A section whose
       claim is the number cannot leave a third of them statistically unvisited. */
    function draw() {
      if (!deck.length) deck = shuffled(st.count);
      var k = deck.pop();
      if (k === st.active && deck.length) {
        // Send it to the bottom rather than discarding it: dropping a card
        // would leave one effect unvisited for the whole pass, which is the
        // single thing a deck is here to rule out.
        deck.unshift(k);
        k = deck.pop();
      }
      return k;
    }

    // Handoff on a loop boundary, asked of the thing that knows: a wall-clock
    // dwell cuts the transition in half about as often as not.
    spotVideo.addEventListener('timeupdate', function () {
      var t = spotVideo.currentTime;
      if (t < lastT) loopsSeen++;
      lastT = t;
      if (!st.tour || st.pointer || !dwell.loops) return;
      if (loopsSeen >= dwell.loops) { loopsSeen = 0; show(draw()); }
    });

    /* ------------------------------------------------------ interaction */

    function bindCells() {
      cells.forEach(function (c, k) {
        c.btn.addEventListener('pointerenter', function () { show(k); });
        c.btn.addEventListener('focus', function () { rove(k); show(k, true); });
      });
    }

    function rove(k) {
      cells.forEach(function (c, i) { c.btn.tabIndex = i === k ? 0 : -1; });
    }

    grid.addEventListener('pointerenter', function () {
      st.pointer = true;
      host.classList.add('hot');
      clearTimeout(resumeTimer);
      clearTimeout(msTimer);
    });
    grid.addEventListener('pointerleave', function () {
      st.pointer = false;
      host.classList.remove('hot');
      // A pointer crossing the section on its way somewhere else should not
      // end the tour; one that has actually left should not have it resume
      // under the cursor either.
      clearTimeout(resumeTimer);
      resumeTimer = setTimeout(function () {
        if (!st.pointer && st.tour) { loopsSeen = 0; show(draw()); }
      }, st.resume);
    });

    grid.addEventListener('keydown', function (e) {
      var k = st.active < 0 ? 0 : st.active;
      var d = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: st.cols_, ArrowUp: -st.cols_ }[e.key];
      if (d == null) {
        if (e.key === 'Escape') { sleep(); e.preventDefault(); }
        return;
      }
      var n = Math.max(0, Math.min(st.count - 1, k + d));
      rove(n);
      cells[n].btn.focus();
      e.preventDefault();
    });

    // Observe the width only. layout() writes grid.style.height, so an
    // observer that reacted to height would report its own write back and
    // re-enter -- convergent, but a loop the browser is entitled to warn about.
    var lastW = 0;
    var ro = new ResizeObserver(function () {
      if (grid.clientWidth === lastW) return;
      lastW = grid.clientWidth;
      layout();
    });
    ro.observe(grid);

    /* One shot, a screen early, then gone: by the time the panel is actually
       in view every poster is either painted or in flight, and there is no
       per-image heuristic left to get it wrong. */
    var warm = new IntersectionObserver(function (en) {
      if (!en[0].isIntersecting) return;
      warm.disconnect();
      warmed = true;
      cells.forEach(function (c) {
        var img = c.btn.querySelector('img');
        if (img && !img.src) img.src = img.dataset.src;
      });
    }, { rootMargin: '600px 0px' });
    warm.observe(host);

    // Off screen, nothing runs: no tour, no decode, no frames.
    var io = new IntersectionObserver(function (en) {
      if (en[0].isIntersecting) {
        if (st.tour && st.active < 0) show(draw());
        else if (st.active >= 0) roll(spotVideo);
        // A hop interrupted by scrolling away is still mid-flight: its springs
        // have a target and a velocity and no frames. Without this the card
        // comes back frozen between two cells and stays there, because kick()
        // is only ever called by something changing.
        kick();
      } else {
        spotVideo.pause();
        if (raf) { cancelAnimationFrame(raf); raf = 0; }
      }
    }, { threshold: 0 });
    io.observe(host);

    build();

    return {
      set: function (k, v) {
        st[k] = v;
        if (k === 'shape' || k === 'cols' || k === 'target' || k === 'count') {
          if (k === 'count') st.count = Math.min(v, items.length);
          build();
        } else layout();
      },
      get: function (k) { return st[k]; },
      show: function (k) { show(k, true); },
      sleep: sleep,
      shapes: Object.keys(TILINGS).map(function (id) {
        return { id: id, label: TILINGS[id].label };
      }),
      destroy: function () {
        if (raf) cancelAnimationFrame(raf);
        clearTimeout(wantTimer); clearTimeout(resumeTimer); clearTimeout(msTimer);
        ro.disconnect(); io.disconnect(); warm.disconnect();
        host.innerHTML = '';
      }
    };
  }

  return { mount: mount, TILINGS: TILINGS };
})();
