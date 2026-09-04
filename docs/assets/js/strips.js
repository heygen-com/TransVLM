/* TVStrips — three belts of live tiles, one row held at a time.
 *
 * Same `items` contract as TVMosaic, and the same spring. What differs is what
 * the motion is FOR: a mosaic moves one thing to a destination, so it springs a
 * position; a belt has no destination, so it holds a velocity and what springs
 * is the RATE. Each row has its own, because only the row under the pointer
 * stops.
 *
 * A RING, NOT TWO COPIES. The obvious way to loop a belt is to put the row's
 * items in twice and translate by one copy's width. That costs one transform
 * per row and it is what this was, until every tile had to carry a playing
 * clip: two copies of 59 items is 118 video elements, and half of them would be
 * decoding a second copy of a frame already on screen. So each item gets
 * exactly one tile, positioned individually, and a tile that slides off the left
 * edge is placed back on the right. 59 transform writes a frame instead of 3 is
 * nothing next to 59 decoders saved.
 *
 * ONE DECODER. Every tile playing its own clip was tried and measured: about 27
 * concurrent VP9 streams once the off-screen ones were paused, which is inside
 * the pixel budget on paper -- 59 tiles at 320x180x20fps is one 1080p30 stream's
 * worth of decoding -- and janks badly in practice, because the cost is
 * per-element, not per-pixel. So the tiles are stills, and the only clip on the
 * page is the one in the popped card. 59 <img> and one <video>.
 *
 * WHICH MAKES THE TOUR THE POINT. With one player, something has to decide what
 * it plays. Left alone the strip picks a tile and pops it, waits two loops of
 * the clip, and moves on; the pointer overrules it instantly and hands it back
 * a beat after leaving. The tour only ever picks a tile that is comfortably
 * on screen -- a card opening past the edge is a card nobody sees -- and the
 * belts drifting is what brings the rest of the 59 into range, so coverage is
 * the motion's job rather than a shuffled deck's.
 *
 * ---------------------------------------------------------------------------
 * API
 *
 *   TVStrips.mount(host, {
 *     items: [{ id, poster, media?, title?, body?, label? }],  // required
 *     rows:  [20, 20, 19],   // how many items in each row; must total items.length
 *     speed: 30,             // px/s, the middle row
 *     spread: 0.14,          // how much the rows differ, fraction of speed
 *     dirs:  [1, -1, 1],     // per row: 1 drifts left, -1 drifts right
 *     cellH: 96,             // row height in px; also what caps the card
 *     ratio: 16 / 9,
 *     gap:   6,
 *     zoom:  2.4,            // MAXIMUM card width, in tile widths; the belt may cap it
 *     tour:  true,           // pop a tile at random when nothing is pointed at
 *     loops: 2,              // clip loops before the tour moves on
 *     rest:  2000,           // ms of nothing open between one card and the next
 *     resume: 1400,          // ms after the pointer leaves before the tour resumes
 *     onOpen: function (item, index) {}
 *   })
 *   -> { hold, release, set, get, rows, destroy }
 *
 * Needs assets/css/strips.css and assets/js/spring.js.
 * ---------------------------------------------------------------------------
 */
window.TVStrips = (function () {
  'use strict';

  var S = window.TVSpring;
  if (!S) throw new Error('TVStrips: assets/js/spring.js must load first');

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  /* play() rejects when the autoplay policy declines a muted play in a tab with
     no interaction yet, and when a play() is cut short by the next load(). The
     poster is painted underneath either way, so there is nothing to recover;
     anything else surfaces as a media `error` event, which is not swallowed. */
  function roll(v) {
    var p = v.play();
    if (p && p.catch) p.catch(function () {});
  }

  function mod(a, n) { return ((a % n) + n) % n; }

  function mount(host, opts) {
    var items = opts.items;
    if (!Array.isArray(items) || !items.length) {
      throw new Error('TVStrips: opts.items must be a non-empty array');
    }
    var counts = opts.rows || [20, 20, 19];
    /* Exact, not "at most". A row split that came up short would quietly drop
       effects off the end of a section whose whole claim is the count, and it
       would do it the day a 60th xfade transition lands in ffmpeg. A caller
       that wants a subset slices `items` itself. */
    var total = counts.reduce(function (a, b) { return a + b; }, 0);
    if (total !== items.length) {
      throw new Error('TVStrips: rows [' + counts.join(', ') + '] account for ' +
        total + ' items but ' + items.length + ' were given — every item has to ' +
        'land in a row, so slice `items` if you mean to show fewer');
    }
    items.forEach(function (it, i) {
      if (!it || !it.poster) {
        throw new Error('TVStrips: items[' + i + '] has no poster — a belt paints ' +
          'before any media can load');
      }
    });

    var reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    var onOpen = opts.onOpen || function () {};

    /* The card is sized against the LONGEST definition in the set, not against
       whichever one is open. Sizing per item makes the card a different size for
       every effect -- a four-line definition eats the frame that a one-line
       definition leaves alone -- and a popup that changes size depending on what
       you point at reads as a layout bug. Character count is a fine proxy for
       wrapped height at a fixed width. */
    var longest = items.reduce(function (a, b) {
      var la = ((a.title || a.id) + ' ' + (a.body || '')).length;
      var lb = ((b.title || b.id) + ' ' + (b.body || '')).length;
      return lb > la ? b : a;
    }, items[0]);

    var st = {
      speed: opts.speed == null ? 30 : opts.speed,
      spread: opts.spread == null ? 0.14 : opts.spread,
      // The middle row against the grain. Three belts all one way read as a
      // single sheet sliding behind a window.
      dirs: opts.dirs || [1, -1, 1],
      cellH: opts.cellH == null ? 96 : opts.cellH,
      ratio: opts.ratio == null ? 16 / 9 : opts.ratio,
      gap: opts.gap == null ? 6 : opts.gap,
      zoom: opts.zoom == null ? 2.4 : opts.zoom,
      loops: opts.loops == null ? 2 : opts.loops,
      // The quiet beat between one card closing and the next opening. Without
      // it a card is always open somewhere and the eye never rests on the
      // belts themselves.
      rest: opts.rest == null ? 2000 : opts.rest,
      resume: opts.resume == null ? 1400 : opts.resume,
      tour: opts.tour !== false && !reduced,
      held: null
    };

    host.classList.add('strips');
    host.innerHTML = '';

    /* The mask that fades the tails belongs to the rows, not to the stage: the
       popped card is a sibling of the masked element, or it would fade out
       along with the tiles whenever it opened near an edge. */
    var stage = el('div', 'st-stage');
    var belt = el('div', 'st-rows');
    stage.appendChild(belt);

    var card = el('div', 'st-card');
    var plate = el('div', 'st-plate');
    var cardFrame = el('div', 'st-frame');
    var cardImg = el('img', 'st-card-poster');
    cardImg.alt = '';
    var cardVid = el('video');
    cardVid.muted = true; cardVid.playsInline = true; cardVid.loop = true;
    cardVid.preload = 'auto';
    cardVid.setAttribute('aria-hidden', 'true');
    cardFrame.appendChild(cardImg);
    cardFrame.appendChild(cardVid);
    var cap = el('div', 'st-cap-box');
    var capName = el('p', 'st-name');
    var capBody = el('p', 'st-body');
    cap.appendChild(capName);
    cap.appendChild(capBody);
    card.appendChild(plate);
    card.appendChild(cardFrame);
    card.appendChild(cap);
    stage.appendChild(card);
    host.appendChild(stage);

    var live = el('p', 'st-live');
    live.setAttribute('aria-live', 'polite');
    host.appendChild(live);

    /* --------------------------------------------------------- the tiles */

    var rows = [];   // { items, tiles, row, pitch, span, dir, v, rate, off }
    var tiles = [];  // flat, all of them
    var warmed = false;

    function tile(it, r, i) {
      var btn = el('button', 'st-tile');
      btn.type = 'button';
      btn.tabIndex = -1;
      btn.setAttribute('aria-label', it.label || it.title || it.id);
      var img = el('img');
      img.dataset.src = it.poster;
      img.alt = '';
      img.decoding = 'async';
      img.width = 320; img.height = 180;
      btn.appendChild(img);
      // Always on, bottom centre. The belt is not a credit roll -- a tile you
      // cannot name is a tile you cannot look up.
      btn.appendChild(el('span', 'st-tag', it.title || it.id));
      var t = { it: it, btn: btn, img: img, r: r, i: i, x: 0 };
      btn.__t = t;
      return t;
    }

    function build() {
      belt.innerHTML = '';
      rows = [];
      tiles = [];
      var cursor = 0;
      counts.forEach(function (n, r) {
        var slice = items.slice(cursor, cursor + n);
        cursor += n;
        var rowEl = el('div', 'st-row');
        var ts = slice.map(function (it, i) {
          var t = tile(it, r, i);
          rowEl.appendChild(t.btn);
          tiles.push(t);
          return t;
        });
        belt.appendChild(rowEl);
        rows.push({
          items: slice, tiles: ts, row: rowEl, off: 0,
          rate: S.Spring(reduced ? 0 : 1, S.PRESET.open, 0.002)
        });
      });
      // One tab stop into the strip; the rest is arrow keys.
      if (rows[0] && rows[0].tiles[0]) rows[0].tiles[0].btn.tabIndex = 0;
      if (warmed) fill();
      layout();
    }

    function fill() {
      warmed = true;
      tiles.forEach(function (t) {
        if (!t.img.getAttribute('src')) t.img.src = t.img.dataset.src;
      });
    }

    /* -------------------------------------------------------- the layout */

    var tileW = 0, beltW = 0;

    function layout() {
      beltW = belt.clientWidth;
      if (!beltW) return;
      tileW = Math.round(st.cellH * st.ratio);
      host.style.setProperty('--st-h', st.cellH + 'px');
      host.style.setProperty('--st-w', tileW + 'px');
      host.style.setProperty('--st-gap', st.gap + 'px');

      /* Speeds spread AROUND the middle row, so `speed` keeps meaning the same
         thing if the row count changes. */
      var mid = (rows.length - 1) / 2;
      rows.forEach(function (row, r) {
        var n = row.items.length;
        /* A ring only covers the belt if the items are wider, laid end to end,
           than the belt plus a tile at each edge. Twenty 149 px tiles span
           3100 px against a belt of at most ~1400, so this never bites in the
           shipped configuration -- but a very wide window with very small tiles
           would leave a hole scrolling through. Widening the pitch spreads the
           same items further apart instead, which is a thinner belt rather than
           a broken one. */
        row.pitch = Math.max(tileW + st.gap, Math.ceil((beltW + 2 * tileW) / n));
        row.span = row.pitch * n;
        row.dir = st.dirs[r] == null ? 1 : st.dirs[r];
        row.v = st.speed * (1 + (r - mid) * st.spread);
        row.off = mod(row.off, row.span);
        row.tiles.forEach(function (t) {
          t.btn.style.width = tileW + 'px';
          t.btn.style.height = st.cellH + 'px';
        });
        place(row);
      });
      sizeCard();
    }

    function place(row) {
      var edge = tileW + st.gap;
      row.tiles.forEach(function (t) {
        // Shifted by one tile before the modulo, so a tile leaving the left
        // edge slides fully out before it reappears on the right instead of
        // popping across.
        t.x = mod(t.i * row.pitch - row.off + edge, row.span) - edge;
        t.btn.style.transform = 'translate3d(' + t.x.toFixed(2) + 'px,0,0)';
      });
    }

    /* -------------------------------------------------------- the motion */

    var raf = 0, last = 0, onScreen = false;

    /* Named `tick`, not `frame`: the card's media wrapper is also a "frame",
       and a `var frame` in the same scope silently overwrites a hoisted
       `function frame` -- which handed requestAnimationFrame a DIV and stopped
       the belts dead before they ever moved. */
    function tick(now) {
      var dt = last ? Math.min((now - last) / 1000, S.DT_MAX) : 1 / 60;
      last = now;
      var moving = false;
      rows.forEach(function (row) {
        row.rate.step(dt);
        if (row.rate.x <= 0.0005) return;
        moving = true;
        row.off = mod(row.off + row.v * row.dir * row.rate.x * dt, row.span);
        place(row);
      });
      var cardStill = openS.step(dt);
      paintCard();
      /* Step two of a hop, and the end of a retract, both happen the moment the
         card is home -- not on a timer that would have to be kept in step with
         a spring it cannot see. */
      if (openS.x <= CLOSED) {
        if (queued) { flip(); raf = requestAnimationFrame(tick); return; }
        if (homing && cardStill) { homing = false; st.held = null; shown = null; }
      }
      // The loop runs while anything is moving. A rate at rest at 1 is a belt
      // at full speed, which is not a reason to stop.
      if (!moving && cardStill) { raf = 0; last = 0; return; }
      raf = requestAnimationFrame(tick);
    }

    function kick() {
      if (raf || !onScreen) return;
      last = 0;
      raf = requestAnimationFrame(tick);
    }

    /* ---------------------------------------------------------- the card */

    var openS = S.Spring(0, S.PRESET.open, 0.0006);
    var cardW = 0, cardH = 0, capH = 96;
    var PAD = 10;
    var wrote = { w: -1, h: -1, vis: '' };

    function beltHeight() {
      return rows.length * st.cellH + (rows.length - 1) * st.gap;
    }

    /* Half a row of slack above and below the belts, which the section reserves
       (see --st-reserve). Fitting the card strictly inside the three rows was
       tried and is worse: it leaves exactly ONE legal vertical position, so
       every card lands in the same place regardless of which tile was held, and
       the bottom row's card flies 200 px up while it grows instead of growing
       where it is. A bounded reserve buys back enough freedom for a card to sit
       on its own row, and the section owns the space so nothing can intrude on
       the next one. */
    function reserve() { return Math.round(st.cellH * 0.5); }

    /* THE BELT CAPS THE CARD. `zoom` is a maximum, not a size: a card taller
       than the rows plus their reserve has nowhere to be clamped to and hangs
       into whatever section comes next. The lever for a bigger card is `cellH`
       -- taller rows, taller belt, bigger reserve, bigger card, everything in
       proportion.

       The loop is there because the two constraints are circular: a narrower
       card wraps the definition onto more lines, a taller caption leaves less
       room for the frame. It converges in one or two passes, since the caption
       only ever changes by whole lines. */
    function sizeCard() {
      var beltH = beltHeight() + reserve() * 2;
      var maxW = Math.min(tileW * st.zoom, beltW * 0.6, beltW - PAD * 2);
      /* A pop has to be a pop. Below this the belt is too short to hold any
         card worth opening -- a very small cellH in the harness -- and a card
         that overflows a little is better than one nobody can see the
         difference of. */
      var floor = tileW * 1.5;
      // Measured on the longest definition, with whatever is on screen put back
      // before anything can paint, so the reserve is the same for every card.
      var keepN = capName.textContent, keepB = capBody.textContent;
      capName.textContent = longest.title || longest.id;
      capBody.textContent = longest.body || '';
      var w = maxW;
      for (var pass = 0; pass < 3; pass++) {
        cap.style.minHeight = '0px';
        cap.style.width = w + 'px';
        capH = cap.offsetHeight || capH;
        var fits = (beltH - capH - PAD * 2) * st.ratio;
        var next = Math.max(floor, Math.min(maxW, fits));
        if (Math.abs(next - w) < 1) break;
        w = next;
      }
      cardW = w;
      cardH = w / st.ratio;
      cap.style.width = cardW + 'px';
      cap.style.minHeight = '0px';
      capH = cap.offsetHeight || capH;
      capName.textContent = keepN;
      capBody.textContent = keepB;
      // So a one-line definition occupies the same box as the longest one and
      // the plate is the same rectangle whatever is open.
      cap.style.minHeight = capH + 'px';
      host.style.setProperty('--st-reserve', reserve() + 'px');
      wrote.w = wrote.h = -1;
    }

    /* Geometry derived every frame from one spring rather than stored, so an
       interrupted open has no stale state to reconcile. */
    function paintCard() {
      var t = st.held;
      var open = openS.x;
      if (!t && open <= 0.001) {
        if (wrote.vis !== '0') { card.style.opacity = '0'; wrote.vis = '0'; }
        return;
      }
      var anchor = t || wrote.t;
      if (!anchor) return;
      wrote.t = anchor;

      var w = tileW + (cardW - tileW) * open;
      var h = st.cellH + (cardH - st.cellH) * open;
      var cx = anchor.x + tileW / 2;
      var cy = anchor.r * (st.cellH + st.gap) + st.cellH / 2;
      var shutX = cx - tileW / 2, shutY = cy - st.cellH / 2;
      var res = reserve();
      var loX = PAD, hiX = beltW - cardW - PAD;
      /* The clamp box is the belts plus their reserve, so the middle row's card
         sits exactly on its row and the outer two are nudged in rather than
         hanging out of the section. sizeCard() keeps the card short enough that
         hiY >= loY; the unclamped branch is only reachable at the size floor. */
      var loY = -res + PAD;
      var hiY = beltHeight() + res - cardH - capH - PAD;
      var ox = hiX < loX ? loX : Math.max(loX, Math.min(hiX, cx - cardW / 2));
      var oy = hiY < loY ? (cy - cardH / 2) : Math.max(loY, Math.min(hiY, cy - cardH / 2));
      var x = shutX + (ox - shutX) * open;
      var y = shutY + (oy - shutY) * open;

      card.style.transform = 'translate3d(' + x + 'px,' + y + 'px,0)';
      if (w !== wrote.w || h !== wrote.h) {
        card.style.width = w + 'px';
        card.style.height = h + 'px';
        wrote.w = w; wrote.h = h;
      }
      var vis = open <= 0.001 ? '0' : '1';
      if (vis !== wrote.vis) { card.style.opacity = vis; wrote.vis = vis; }

      var p = PAD * open;
      plate.style.left = -p + 'px';
      plate.style.top = -p + 'px';
      plate.style.width = (w + p * 2) + 'px';
      plate.style.height = (h + capH * open + p * 2) + 'px';
      plate.style.opacity = open;

      var ct = Math.max(0, (open - 0.45) / 0.55);
      cap.style.opacity = ct;
      cap.style.transform = 'translateY(' + ((1 - ct) * -8) + 'px)';
    }

    /* ----------------------------------------------------------- holding */

    /* A HOP IS TWO STEPS, in this order and never overlapped: the card that is
       open shrinks back into its own tile, and only once it is home does the
       next one grow out of its own. That is the difference between a card that
       lives somewhere and a card that teleports across the strip -- and it is
       why there is a `queued` target rather than a second animation.

       `CLOSED` is where the card is visually home; the remaining tail of the
       spring is under a pixel and there is nothing to wait for. */
    var CLOSED = 0.02;
    var wantT = null, wantTimer = 0, queued = null, homing = false, shown = null;

    /* The 80 ms is a debounce on the INTENTION, not on the content. A pointer
       crossing the belt passes over six tiles on its way to the seventh, and
       without this each crossing starts a retraction the next one interrupts,
       so the card never opens at all. The tour and the keyboard skip it,
       because both are already a decision. */
    function ask(t, immediate) {
      if (t === wantT && !immediate) return;
      wantT = t;
      clearTimeout(wantTimer);
      if (immediate) commit();
      else wantTimer = setTimeout(commit, 80);
    }

    function commit() {
      var t = wantT;
      if (t === st.held) {
        // Back to the one already open, or on its way home. Reversing the same
        // spring keeps the velocity it had, so coming back mid-close is a curve
        // rather than a restart from nothing.
        homing = false;
        if (openS.target !== 1) {
          openS.tune(S.PRESET.open).to(1);
          stopRow(t.r);
          roll(cardVid);
          kick();
        }
        return;
      }
      queued = t;
      if (openS.x <= CLOSED) flip();
      else { openS.tune(S.PRESET.shut).to(0); kick(); }
    }

    /* Step two. Only ever called with the card home, which is what makes
       re-anchoring instant rather than a move: there is nothing on screen. */
    function flip() {
      var t = queued;
      queued = null;
      homing = false;
      if (st.held) {
        st.held.btn.classList.remove('held');
        if (!t || t.r !== st.held.r) resumeRow(st.held.r);
      }
      st.held = t;
      if (!t) { host.classList.remove('holding'); return; }
      t.btn.classList.add('held');
      host.classList.add('holding');
      stopRow(t.r);
      content(t);
      openS.tune(S.PRESET.open);
      if (reduced) openS.set(1); else openS.to(1);
      kick();
    }

    function stopRow(r) { rows[r].rate.tune(S.PRESET.open).to(0); }
    function resumeRow(r) {
      if (!reduced) rows[r].rate.tune(S.PRESET.open).to(1);
    }

    function content(t) {
      if (t.it === shown) { roll(cardVid); return; }
      shown = t.it;
      var it = t.it;
      var title = it.title || it.id;
      capName.textContent = title;
      capBody.textContent = it.body || '';
      live.textContent = it.body ? title + ' — ' + it.body : title;
      card.classList.remove('rolling');
      if (cardImg.getAttribute('src') !== it.poster) cardImg.src = it.poster;
      if (it.media) {
        cardVid.src = it.media;
      } else {
        cardVid.removeAttribute('src');
        cardVid.load();
      }
      loopsSeen = 0;
      lastT = 0;
      onOpen(it, items.indexOf(it));
    }

    /* The card goes home and the belt it was holding restarts, but `st.held`
       stays until it has arrived -- paintCard needs the tile it is retracting
       into for as long as it is retracting. */
    function letGo() {
      wantT = null; queued = null;
      clearTimeout(wantTimer);
      if (!st.held) return;
      resumeRow(st.held.r);
      st.held.btn.classList.remove('held');
      host.classList.remove('holding');
      card.classList.remove('rolling');
      cardVid.pause();
      openS.tune(S.PRESET.shut).to(0);
      homing = true;
      kick();
    }

    // The poster holds the frame until the clip can paint, so a card never
    // opens onto a black rectangle.
    cardVid.addEventListener('loadeddata', function () {
      if (!st.held) return;
      card.classList.add('rolling');
      roll(cardVid);
    });

    /* ------------------------------------------------------------ the tour */

    var deck = [], loopsSeen = 0, lastT = 0, restTimer = 0, resumeTimer = 0;
    var pointerIn = false;

    function shuffledTiles() {
      var a = tiles.slice();
      for (var j = a.length - 1; j > 0; j--) {
        var k = Math.floor(Math.random() * (j + 1));
        var tmp = a[j]; a[j] = a[k]; a[k] = tmp;
      }
      return a;
    }

    /* Comfortably on screen, not merely on screen: a tile at the very edge
       gets its card clamped against the belt every time, which reads as the
       card refusing to centre rather than as the tile being where it is. */
    function reachable(t) {
      return t.x > tileW * 0.25 && t.x < beltW - tileW * 1.25;
    }

    /* A deck for the order and the belt for the coverage. The deck keeps the
       tour from picking the same handful twice in a row; the drift is what
       brings the rest of the 59 into range, so a tile that is off screen when
       its turn comes is left in the deck rather than skipped. */
    function draw() {
      if (!deck.length) deck = shuffledTiles();
      for (var k = 0; k < deck.length; k++) {
        if (deck[k] !== st.held && reachable(deck[k])) return deck.splice(k, 1)[0];
      }
      return null;
    }

    /* Two loops, then home, then a beat of nothing before the next one. The
       beat is the whole reason this reads as calm rather than as a slideshow:
       without it the card is always open somewhere and the eye never gets to
       rest on the belts themselves. */
    function tourNext() {
      letGo();
      clearTimeout(restTimer);
      restTimer = setTimeout(pop, st.rest);
    }

    function pop() {
      clearTimeout(restTimer);
      if (pointerIn || !st.tour || !onScreen) return;
      var t = draw();
      if (t) ask(t, true);
      // Nothing comfortably on screen is possible for a frame or two after a
      // resize. Ask the belts again shortly rather than ending the tour.
      else restTimer = setTimeout(pop, 400);
    }

    // Handoff on a loop boundary, asked of the thing that knows: a wall-clock
    // dwell cuts a 0.6 s transition in half about as often as not.
    cardVid.addEventListener('timeupdate', function () {
      var t = cardVid.currentTime;
      if (t < lastT) loopsSeen++;
      lastT = t;
      if (!st.tour || pointerIn || !st.held) return;
      if (loopsSeen >= st.loops) { loopsSeen = 0; tourNext(); }
    });

    /* ------------------------------------------------------ interaction */

    /* Touch has no hover to lean on, and the sequence below defeats itself on a
       phone: a tap fires pointerover, which opens a card, and then pointerleave
       at the end of that same tap, which closes it -- and iOS does not focus a
       <button> on tap, so the focus path does not stand in either. Where there
       is no hover, a tap latches a tile open and the next tap releases it:
       on the tile, on another tile, or on the belt around them.

       NOT VERIFIED ON A DEVICE. This is reasoned from the event sequence, not
       measured on a phone -- see HAX-62. The code fact it rests on is that
       there was no click or pointerup handler here at all. */
    var HOVERS = window.matchMedia('(any-hover: hover)').matches;

    belt.addEventListener('pointerenter', function () {
      pointerIn = true;
      clearTimeout(restTimer);
      clearTimeout(resumeTimer);
    });
    belt.addEventListener('pointerover', function (e) {
      var btn = e.target.closest ? e.target.closest('.st-tile') : null;
      if (btn && btn.__t) ask(btn.__t);
    });
    /* On the rows, not the tile. A tile-scoped release is a loop: the belt
       restarts, the tile slides out from under the cursor, the neighbour takes
       the hover, that stops it again -- about thirty times a second. */
    belt.addEventListener('pointerleave', function () {
      pointerIn = false;
      // On touch this fires at the end of the tap that just opened a card.
      if (HOVERS) letGo();
      // A pointer crossing the section on its way somewhere else should not end
      // the tour; one that has actually left should not have a card open under
      // it either.
      clearTimeout(resumeTimer);
      if (st.tour) resumeTimer = setTimeout(pop, st.resume);
    });

    /* Clicking a tile focuses it -- a <button> takes focus on mousedown in
       Chrome -- and centre() below then hauls the belt across so that tile
       lands in the middle. Nobody asked for that: the tile the pointer just
       pressed is by definition already on screen, under the cursor. centre()
       is there for the keyboard, where the tile that just took focus may be
       anywhere along a ring wider than the window. So the belt moves only for
       a focus the pointer did not cause, and a click on an open card does
       nothing at all -- hover already opened it.

       A flag rather than :focus-visible, because one answer is wanted in every
       engine and the engines disagree on the premise: Chrome focuses a button
       on click, Firefox on macOS does not. The flag is true in exactly the case
       to exclude either way. Focus is set during mousedown's default action, so
       focusin always lands before the timeout clears it. */
    var pointerFocus = false;
    belt.addEventListener('pointerdown', function () {
      pointerFocus = true;
      setTimeout(function () { pointerFocus = false; }, 0);
    });

    belt.addEventListener('focusin', function (e) {
      var btn = e.target.closest ? e.target.closest('.st-tile') : null;
      if (!btn || !btn.__t) return;
      clearTimeout(restTimer);
      rove(btn);
      if (!pointerFocus) centre(btn.__t);
      ask(btn.__t, true);
    });
    belt.addEventListener('focusout', function (e) {
      if (belt.contains(e.relatedTarget)) return;
      letGo();
      if (st.tour) resumeTimer = setTimeout(pop, st.resume);
    });

    if (!HOVERS) {
      var tapped = null;
      belt.addEventListener('click', function (e) {
        var btn = e.target.closest ? e.target.closest('.st-tile') : null;
        var t = btn && btn.__t;
        if (!t || t === tapped) { tapped = null; letGo(); return; }
        tapped = t;
        centre(t);
        ask(t, true);
      });
    }

    function rove(btn) {
      tiles.forEach(function (t) { t.btn.tabIndex = -1; });
      btn.tabIndex = 0;
    }

    /* A focused tile may be anywhere along a ring wider than the window, so
       keyboard focus moves the belt rather than the page. Without this, tabbing
       in lands on something nobody can see. Keyboard only -- see the flag at
       the focusin handler for why a click must not reach here. */
    function centre(t) {
      var row = rows[t.r];
      row.off = mod(t.i * row.pitch + tileW / 2 - beltW / 2, row.span);
      place(row);
    }

    belt.addEventListener('keydown', function (e) {
      var btn = document.activeElement;
      if (!btn || !btn.__t) return;
      var t = btn.__t, next = null;
      var n = rows[t.r].items.length;
      if (e.key === 'ArrowRight') next = rows[t.r].tiles[(t.i + 1) % n];
      else if (e.key === 'ArrowLeft') next = rows[t.r].tiles[(t.i - 1 + n) % n];
      else if (e.key === 'ArrowDown' && t.r + 1 < rows.length) {
        next = rows[t.r + 1].tiles[Math.min(t.i, rows[t.r + 1].items.length - 1)];
      } else if (e.key === 'ArrowUp' && t.r > 0) {
        next = rows[t.r - 1].tiles[Math.min(t.i, rows[t.r - 1].items.length - 1)];
      } else if (e.key === 'Escape') { btn.blur(); letGo(); e.preventDefault(); return; }
      else return;
      e.preventDefault();
      if (next) next.btn.focus();
    });

    var lastW = 0;
    var ro = new ResizeObserver(function () {
      if (belt.clientWidth === lastW) return;
      lastW = belt.clientWidth;
      layout();
    });
    ro.observe(belt);

    // A screen early: the posters are looked at as one thing, so the two states
    // worth having are "not fetched" and "all of it".
    var warm = new IntersectionObserver(function (en) {
      if (!en[0].isIntersecting) return;
      warm.disconnect();
      fill();
    }, { rootMargin: '600px 0px' });
    warm.observe(host);

    // Off screen, nothing runs: no motion, no decoders, no frames.
    var io = new IntersectionObserver(function (en) {
      onScreen = en[0].isIntersecting;
      if (onScreen) {
        kick();
        if (st.tour && !st.held && !pointerIn) restTimer = setTimeout(pop, st.rest);
      } else {
        if (raf) { cancelAnimationFrame(raf); raf = 0; last = 0; }
        clearTimeout(restTimer); clearTimeout(resumeTimer);
        cardVid.pause();
      }
    }, { threshold: 0 });
    io.observe(host);

    build();

    return {
      hold: function (r, i) { if (rows[r]) ask(rows[r].tiles[i], true); },
      release: letGo,
      get: function (k) { return st[k]; },
      set: function (k, v) {
        st[k] = v;
        if (k === 'cellH' || k === 'ratio' || k === 'gap' || k === 'speed' ||
            k === 'spread' || k === 'dirs' || k === 'zoom') layout();
        else if (k === 'tour' && !v) { clearTimeout(restTimer); clearTimeout(resumeTimer); }
      },
      rows: function () { return rows.map(function (r) { return r.items.length; }); },
      // One clip on the page, and this says whether it is the one running.
      playing: function () { return cardVid.paused ? 0 : 1; },
      held: function () { return st.held ? st.held.it.id : null; },
      destroy: function () {
        if (raf) cancelAnimationFrame(raf);
        clearTimeout(wantTimer); clearTimeout(restTimer); clearTimeout(resumeTimer);
        ro.disconnect(); io.disconnect(); warm.disconnect();
        host.innerHTML = '';
      }
    };
  }

  return { mount: mount };
})();
