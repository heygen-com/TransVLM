/* TVDeck — an infinite stacked deck of cards.
 *
 * Content-agnostic on purpose. The deck owns geometry, the gesture, index state
 * and keyboard; it knows nothing about what a card contains. Callers pass
 * render(item, i) and get onVisible/onIndex back, which is how the video deck
 * knows to play the cards you can see and pause the ones you cannot, and how an
 * image or table deck can do nothing at all.
 *
 * The carriage is Embla (vendored, MIT, 18 KB, no dependencies), not hand-rolled
 * inertia and not CSS scroll-snap. Two reasons it has to be a library here:
 * looping wants slides teleported around the ends mid-drag, and momentum wants a
 * real velocity integrator -- both are the kind of thing that works on a
 * trackpad and then falls apart under a fast thumb.
 *
 * The stack itself is ours. Embla lays the slots out and owns each slot's own
 * transform (that is how it loops them); we transform the card *inside* the
 * slot, so the two never fight. Because transforms do not affect layout, cards
 * can overlap as much as we like without the carriage noticing.
 *
 * Not handled: RTL (Embla's positions flip sign), and vertical decks.
 */
window.TVDeck = (function () {
  'use strict';

  var REDUCED = window.matchMedia
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* Decks that answer the arrow keys without being focused first. Several can be
     on screen at once, and the keys belong to whichever one the reader is
     actually looking at -- so one listener arbitrates, rather than each deck
     grabbing the event and every visible deck moving together. */
  var GLOBAL = [];

  function globalKey(ev) {
    if (ev.metaKey || ev.ctrlKey || ev.altKey || ev.shiftKey) return;
    if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
    var t = ev.target;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;

    var mid = window.innerHeight / 2, best = null, bestD = Infinity;
    GLOBAL.forEach(function (d) {
      var r = d.host.getBoundingClientRect();
      if (r.bottom < mid * 0.15 || r.top > window.innerHeight - mid * 0.15) return;
      var dist = Math.abs((r.top + r.bottom) / 2 - mid);
      if (dist < bestD) { bestD = dist; best = d; }
    });
    if (!best) return;
    if (ev.key === 'ArrowLeft') best.api.scrollPrev(); else best.api.scrollNext();
    ev.preventDefault();
  }

  function register(entry) {
    if (!GLOBAL.length) document.addEventListener('keydown', globalKey);
    GLOBAL.push(entry);
  }
  function unregister(entry) {
    var i = GLOBAL.indexOf(entry);
    if (i >= 0) GLOBAL.splice(i, 1);
    if (!GLOBAL.length) document.removeEventListener('keydown', globalKey);
  }

  var DEFAULTS = {
    index: 0,
    // Fractions of the deck's own width, re-resolved on every resize.
    slot: 0.21,       // the carriage step: how far one card-change travels
    card: 0.54,       // the visible card
    depth: 2,         // neighbours painted on each side
    scale: 0.87,      // multiplied once per step away from centre
    fade: 0.55,       // ditto, for the depth scrim
    falloff: 0.62,    // past the first neighbour, steps compress by this much
    crossfade: 0.3,   // of a step: how far either side of the swap it dissolves
    duration: 28,     // Embla's ease length; lower is snappier
    autoplay: false,  // { ms: 6000 } to advance on a timer, add start:false to
                      // configure the interval but leave it switched off
    keys: 'global',   // 'global' | 'focus' | false
    dots: null,       // an element to fill with the counter; see buildDots
    dotLabel: null,   // function (item, i) -> the button's aria-label
    dotStretch: 0.55  // how much the travelling bar lengthens between two ticks
  };

  // Under ~900px there is no room for a card plus two peeking neighbours, so the
  // card takes most of the width and only one neighbour survives.
  function defaultMetrics(w) {
    if (w < 480) return { slot: 0.34, card: 0.82, depth: 1 };
    if (w < 900) return { slot: 0.26, card: 0.66, depth: 1 };
    return {};
  }

  function el(tag, cls) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    return n;
  }

  function mod(a, n) { return ((a % n) + n) % n; }

  function mount(host, opts) {
    if (!window.EmblaCarousel) throw new Error('TVDeck: embla-carousel is not loaded');
    var o = {}, k;
    for (k in DEFAULTS) o[k] = DEFAULTS[k];
    for (k in opts) o[k] = opts[k];
    var items = o.items || [];
    if (!items.length) return null;

    host.classList.add('deck');
    host.innerHTML = '';
    var view = el('div', 'deck-view');
    var track = el('div', 'deck-track');
    view.appendChild(track);
    view.tabIndex = 0;                        // Safari will not focus a bare div
    view.setAttribute('role', 'group');
    view.setAttribute('aria-label', o.ariaLabel || 'Card deck');
    view.setAttribute('aria-roledescription', 'carousel');
    host.appendChild(view);

    var cards = items.map(function (item, i) {
      var slot = el('div', 'deck-slot');
      var card = el('div', 'deck-card');
      card.setAttribute('data-i', String(i));
      card.appendChild(o.render(item, i));
      slot.appendChild(card);
      track.appendChild(slot);
      // A click on a card that is not the active one means "bring me here".
      // Capture, so it never reaches whatever the card put inside itself.
      card.addEventListener('click', function (ev) {
        if (i === idx) return;
        ev.preventDefault();
        ev.stopPropagation();
        embla.scrollTo(i);
      }, true);
      return card;
    });

    // Embla can only loop when the slides outrun the viewport; a short deck of
    // narrow slots does not, and it degrades to a non-looping track plus a
    // console warning. measure() knows the real numbers, so it decides there.
    var wantLoop = o.loop !== false;
    var looping = wantLoop;
    var idx = Math.max(0, Math.min(items.length - 1, o.index));
    var depth = o.depth, shown = '', unit = 0;

    var embla = window.EmblaCarousel(view, {
      loop: looping,
      align: 'center',
      containScroll: false,
      duration: REDUCED ? 0 : o.duration,
      startIndex: idx,
      dragFree: false,
      skipSnaps: false
    });

    function measure() {
      var w = host.clientWidth;
      if (!w) return;
      var m = o.metrics ? o.metrics(w) : defaultMetrics(w);
      var slotF = m.slot != null ? m.slot : o.slot;
      var cardF = m.card != null ? m.card : o.card;
      depth = m.depth != null ? m.depth : o.depth;
      var step = Math.round(slotF * w);
      host.style.setProperty('--deck-slot', step + 'px');
      host.style.setProperty('--deck-card', Math.round(cardF * w) + 'px');
      // Anything the page hangs under the deck -- a timeline, a caption -- has to
      // line up with the active card, and it lives outside the deck's own subtree.
      if (o.publishTo) o.publishTo.style.setProperty('--deck-card', Math.round(cardF * w) + 'px');

      // Embla needs the slides to outrun the viewport before it will loop. Under
      // that, asking for a loop yields a warning and a track that stops at both
      // ends; a plain non-looping deck is the honest fallback, and it comes back
      // on its own if a resize gives the slides the room.
      var fits = items.length * step >= w * 1.5;
      if (wantLoop && looping !== fits) {
        looping = fits;
        embla.reInit({ loop: looping });
        return;                       // reInit fires its own event; do not double up
      }
      embla.reInit();
    }

    // Embla reports positions as a 0..1 progress over the whole track. One
    // card-step is the gap between two adjacent snaps, so dividing by it turns
    // progress back into the units the stack is described in.
    function stepUnit() {
      var snaps = embla.scrollSnapList();
      if (snaps.length < 2) return 1;
      return Math.abs(snaps[1] - snaps[0]) || 1;
    }

    function paint() {
      var engine = embla.internalEngine();
      var progress = embla.scrollProgress();
      var snaps = embla.scrollSnapList();
      var loopPoints = engine.options.loop ? engine.slideLooper.loopPoints : [];
      var vis = [];
      var nearA = Infinity, nearF = 0;

      snaps.forEach(function (snap, snapIndex) {
        engine.slideRegistry[snapIndex].forEach(function (i) {
          var diff = snap - progress;
          // A slide Embla has teleported to the far end is, in progress terms,
          // a whole track away; measure it against the wrapped position instead
          // or it flies off and pops back at every wrap.
          loopPoints.forEach(function (lp) {
            var target = lp.target();
            if (i !== lp.index || target === 0) return;
            if (Math.sign(target) === -1) diff = snap - (1 + progress);
            else diff = snap + (1 - progress);
          });

          var d = diff / unit, a = Math.abs(d), card = cards[i];
          // Before the cull below: the counter needs the nearest card, and the
          // nearest card is never one of the culled ones.
          if (a < nearA) { nearA = a; nearF = i - d; }
          if (a > depth + 0.55) {
            if (card.style.visibility !== 'hidden') {
              card.style.visibility = 'hidden';
              card.setAttribute('aria-hidden', 'true');
              card.classList.remove('is-active');
              card.parentNode.style.zIndex = '0';
            }
            return;
          }
          // 1:1 with the finger for the first step out, compressed after it, so
          // the far cards read as a stack rather than as a queue.
          var travel = a <= 1 ? a : 1 + (a - 1) * o.falloff;
          var x = (Math.sign(d) * travel - d) * slotPx();
          card.style.visibility = '';
          card.removeAttribute('aria-hidden');
          card.style.transform = 'translate3d(' + x.toFixed(2) + 'px,0,0) scale('
            + Math.pow(o.scale, a).toFixed(4) + ')';
          card.style.setProperty('--dim', (1 - Math.pow(o.fade, a)).toFixed(4));
          /* Position is continuous; the stacking order is not. Halfway between
             two snaps the front card and the one replacing it are the same
             distance from centre, and in that one frame they trade places --
             which, with 60% of each card behind the other, re-sources some 44%
             of the picture at once (measured by replaying this function over a
             step and differencing the composite). It was the last hard cut in a
             move that is otherwise smooth, and it is what the swap looked like:
             a cut, not a change of order.

             So the two of them dissolve through it. Both dip to half opacity at
             the crossing and are opaque again `crossfade` of a step either side,
             which leaves a bit over a quarter of the old cut instead of all of it.

             BOTH, not just the one coming to the front. Fading only the card on
             top is the tempting version and it is worth nothing: its opacity has
             to jump from 0.5 to 1 the instant it drops behind the other, and the
             same 44% moves -- measurably no better than no fade at all. Dipping
             both keeps every card's own opacity continuous, which is what makes
             the swap invisible.

             ONLY the front pair, a < 1. The pairs further out swap in the same
             frame, but they are 78% covered by the depth scrim, so their share
             of the picture is small -- and fading them too doubles how much of
             the page washes through the stack mid-move for no measurable gain.

             This is not the translucency the scrim exists to avoid: a card is
             see-through only while it is moving, what shows through it is mostly
             the card it is trading places with, and at rest every card in the
             deck is opaque again. */
          var fr = a - Math.floor(a), lift = 1;
          if (o.crossfade > 0 && a < 1) {
            lift = 1 - 0.5 * Math.max(0, 1 - Math.abs(fr - 0.5) / o.crossfade);
          }
          card.style.opacity = lift < 1 ? lift.toFixed(3) : '';
          // On the slot, not the card. Embla transforms a slot to loop it, which
          // makes that slot a stacking context -- a z-index on the card inside
          // is then scoped to it, and cards order by DOM position instead, so
          // the card to the right of centre paints over the active one.
          //
          // Rounded fine, not to a tenth of a step. At a tenth, a pair spends
          // the middle 0.1 of every step on the same z-index, ordered by DOM
          // position -- so the swap actually landed wherever that band opened,
          // up to 0.05 of a step early and on the same side whichever way the
          // deck was moving, and the dissolve above would have played out
          // underneath an already-opaque card. The ceiling stays 1000: the page
          // has elements at 20, 30 and 100, and the deck sits over them.
          card.parentNode.style.zIndex = String(1000 - Math.round(a * 300));
          card.classList.toggle('is-active', a < 0.5);
          if (a <= depth + 0.5) vis.push(i);
        });
      });

      // Wrapped, because `i - d` is continuous across a loop teleport but not
      // bounded by it: coming round from the last card it reads as -0.5 where
      // the counter wants n - 0.5, the same place on the ring.
      if (dotBar && nearA < Infinity) paintDots(mod(nearF, items.length));

      var key = vis.sort(function (p, q) { return p - q; }).join(',');
      if (key !== shown) {
        shown = key;
        if (o.onVisible) o.onVisible(vis, items, cards);
      }
    }

    /* ------------------------------------------------------------ the counter
     * One accent bar that TRAVELS between short grey ticks, rather than a tick
     * that turns green in place.
     *
     * IT IS DRIVEN BY THE SAME NUMBER THE CARDS ARE. paint() already resolves
     * Embla's scroll position into a signed distance in card-steps, loop
     * teleports corrected for; the nearest card's `i - d` is a continuous
     * fractional index over the ring, and that is the bar's position. So the bar
     * follows a drag under the finger, follows Embla's own easing on a jump, and
     * cannot disagree with which card is in front -- all for free. A separate
     * animator here, spring or transition, would be a second opinion about where
     * the deck is, and the two would drift apart mid-drag.
     *
     * WHICH IS WHY THERE IS NO TRANSITION ON IT. spring.js's rule is that things
     * which move a distance get a spring and things which merely change get a
     * curve; this moves a distance, but the distance is already being animated
     * one level down. Tracking it is neither of the two.
     *
     * THE STRETCH IS THE POINT. A bar that only translates reads as a sliding
     * block. Lengthening it with how far it is from a tick and letting it settle
     * back on arrival is what makes it read as one mark reaching for the next
     * position, and it costs one scaleX on a transform that is already written.
     *
     * On a looping deck the trip from the last tick to the first sweeps back
     * across the whole row: the ring is drawn as a line, so there is no short
     * way round, and a bar that vanished at one end and reappeared at the other
     * would read as two bars.
     *
     * ITS WIDTH COMES FROM THE CARD, not from the element the caller gave it.
     * The counter is a mark on a scale and the scale belongs to the card the
     * deck is showing; the host is often a full-width panel (990 px against a
     * 683 px card on the page), and a row spread across all of it reads as two
     * counters with nothing in between -- on a two-card deck, as one tick in
     * each corner. So measureDots caps the row at --dd-share of --deck-card and
     * lets flex spend the rest as the gap. See the CSS for the tokens.
     *
     * IT IS ALSO THE SCRUBBER. See "the scrubber" below: the counter takes the
     * drag, and the bar answers it only through Embla, like everything else.
     */
    var dotHost = o.dots || null;
    var dotBtns = [], dotBar = null, dotMid = [], dotBarW = 0;
    var stretch = REDUCED ? 0 : o.dotStretch;

    function buildDots() {
      dotHost.classList.add('deck-dots');
      dotHost.innerHTML = '';
      // The host is the caller's element, so it may already say what it is; only
      // name it when it does not.
      if (!dotHost.getAttribute('role')) dotHost.setAttribute('role', 'group');
      if (!dotHost.getAttribute('aria-label')) dotHost.setAttribute('aria-label', 'Cards');
      // First child, which the CSS's :first-of-type ticks rely on. Paint order
      // is a z-index, not DOM order: the bar goes OVER the ticks -- see the CSS.
      dotBar = el('span', 'dd-bar');
      dotBar.setAttribute('aria-hidden', 'true');
      dotHost.appendChild(dotBar);
      dotBtns = items.map(function (item, i) {
        var b = el('button');
        b.type = 'button';
        b.setAttribute('aria-label', o.dotLabel ? o.dotLabel(item, i) : 'Card ' + (i + 1));
        // Pointers are the scrubber's, below; this click is the one Enter and
        // Space fire on a focused tick, and whatever an AT sends instead.
        b.addEventListener('click', function () { embla.scrollTo(i); });
        dotHost.appendChild(b);
        return b;
      });
      dotHost.addEventListener('pointerdown', dotDown);
      dotHost.addEventListener('pointermove', dotMove);
      dotHost.addEventListener('pointerup', dotUp);
      dotHost.addEventListener('pointercancel', dotUp);
    }

    /* THE ROW IS AS WIDE AS ITS TICKS WANT, up to two thirds of the card. Under
       that cap the gap is whatever the width leaves -- flex spends it with
       space-between -- so it starts at --dd-gap-max and floats down as the card
       narrows or the count grows, and two cards get two ticks a gap apart in
       the middle rather than one in each corner.

       DENSE WHEN THE GAP RUNS OUT. A row of separate ticks is legible while the
       gap can still be seen; below --dd-gap-min it cannot, and eighty-four
       8 px ticks a pixel apart read as a broken dotted line rather than as a
       scale. So there the counter becomes one continuous line -- every tick
       stretches to its share of the row, the gaps close -- and the bar, still
       the only accent, still says where the deck is. Decided from the width and
       the tokens rather than from what flex did, so the switch is a step and
       not a shrink; re-decided on every reInit, which is where a resize
       arrives. */
    function measureDots() {
      if (!dotBar) return;
      var n = dotBtns.length;
      var cs = getComputedStyle(dotHost);
      var tick = parseFloat(cs.getPropertyValue('--dd-tick'));
      var gapMax = parseFloat(cs.getPropertyValue('--dd-gap-max'));
      var gapMin = parseFloat(cs.getPropertyValue('--dd-gap-min'));
      var share = parseFloat(cs.getPropertyValue('--dd-share'));
      if (!(tick > 0 && gapMax > 0 && gapMin > 0 && share > 0)) {
        throw new Error('TVDeck: the counter has no tokens -- expected --dd-tick, '
          + '--dd-gap-max, --dd-gap-min and --dd-share on .deck-dots. Is deck.css loaded?');
      }
      // The card the deck has just measured, which is also what it publishes to
      // the strip under it. Zero only before the deck has been laid out at all
      // -- mounted in a display:none subtree -- and the resize that gives it a
      // width comes back through here.
      var card = parseFloat(getComputedStyle(host).getPropertyValue('--deck-card'));
      if (!card) return;

      // A cap, not a width: the row is still a block in the caller's column, so
      // a column narrower than the cap gives the row less and tightens the gap
      // rather than overflowing.
      dotHost.style.maxWidth = Math.min(share * card, n * tick + (n - 1) * gapMax) + 'px';
      // Which is why the width is read back rather than assumed.
      var w = dotHost.clientWidth;
      dotHost.classList.toggle('is-dense', n > 1 && (w - n * tick) / (n - 1) < gapMin);
      dotBarW = dotBar.offsetWidth;
      dotMid = dotBtns.map(function (b) { return b.offsetLeft + b.offsetWidth / 2; });
    }

    function paintDots(f) {
      var n = dotMid.length;
      if (!dotBar || !n) return;
      var i = Math.floor(f), t = f - i;
      var a = dotMid[mod(i, n)], b = dotMid[mod(i + 1, n)];
      /* How far the bar is from a place it could rest, in ticks, capped at
         half a step -- the stretch reaches its most at that distance and the
         bar is at rest at 0.

         SEPARATE TICKS: the nearest tick. The bar lengthens as it leaves one and
         settles as it reaches the next, one reach per step, which is the mark
         reaching for the next position.

         ONE LINE: the destination card instead. With ticks a few pixels apart a
         scrub crosses one every few pixels, and a reach per tick is a pulse --
         the bar twitched all the way along the line and bounced to a stop. So
         on a line it stretches by its distance to where the deck is going,
         stays long while it travels, and settles once, on arrival. Embla's
         selected snap is that destination from the moment a scroll is asked
         for, and during a drag it is the card the deck will settle on. */
      var off;
      if (dotHost.classList.contains('is-dense')) {
        var dist = Math.abs(f - embla.selectedScrollSnap());
        if (dist > n / 2) dist = n - dist;    // the short way round a loop
        off = Math.min(dist, 0.5);
      } else {
        off = Math.min(t, 1 - t);            // 0 at a tick, 0.5 between two
      }
      // The whole per-frame cost of the counter: one transform, on one element.
      // Nothing has to be hidden under the bar -- it paints over the ticks.
      dotBar.style.transform = 'translate3d(' +
        (a + (b - a) * t - dotBarW / 2).toFixed(2) + 'px,0,0) scaleX(' +
        (1 + stretch * off * 2).toFixed(3) + ')';
    }

    /* ------------------------------------------------------------ the scrubber
     * The counter is a control as well as a readout: press anywhere on its band
     * and drag, and every move maps the pointer's x to the nearest tick and asks
     * Embla for that card. The bar is not moved from here. It goes on tracking
     * Embla's scroll position, which is what makes a scrub read as dragging the
     * deck -- the bar follows the finger because the deck does -- rather than as
     * dragging a slider the deck copies.
     *
     * ONLY WHEN THE INDEX CHANGES. scrollTo restarts Embla's ease, so calling it
     * on every move would keep restarting a 28-frame animation from wherever it
     * had got to, and the deck would crawl a long way behind the finger.
     *
     * CAPTURED ON PRESS, because a scrub leaves the row constantly -- the band
     * is 44 px tall and a thumb travelling sideways does not stay in it -- and
     * an uncaptured pointer simply stops reporting. The cost is that the click
     * ending a press is retargeted to the host, which is why a press that never
     * moved is treated as a click on the tick nearest it, here, rather than
     * being left to the button's own listener (that one still serves the
     * keyboard, where no pointer is captured).
     *
     * Nothing here reaches the deck's own drag: Embla listens on .deck-view,
     * which is inside the deck host, and the counter is not in that subtree.
     */
    var SLOP = 3;            // px of travel before a press is a drag and not a tap
    var dragId = -1, dragX = 0, dragMoved = false, dragSent = -1;

    function tickAt(clientX) {
      var x = clientX - dotHost.getBoundingClientRect().left;
      var best = 0, bestD = Infinity;
      dotMid.forEach(function (m, i) {
        var d = Math.abs(m - x);
        if (d < bestD) { bestD = d; best = i; }
      });
      return best;
    }

    function dotDown(ev) {
      if (ev.pointerType === 'mouse' && ev.button !== 0) return;
      if (!dotMid.length) return;
      dragId = ev.pointerId;
      dragX = ev.clientX;
      dragMoved = false;
      dragSent = -1;
      dotHost.setPointerCapture(dragId);
    }
    function dotMove(ev) {
      if (ev.pointerId !== dragId) return;
      // A tap wobbles. Until it has travelled far enough to be a drag, a move is
      // not a scrub -- otherwise every tap scrolls to its own tick twice, and a
      // tap 2 px away from a tick edge scrolls to the wrong one.
      if (!dragMoved && Math.abs(ev.clientX - dragX) < SLOP) return;
      dragMoved = true;
      var i = tickAt(ev.clientX);
      if (i !== dragSent) { dragSent = i; embla.scrollTo(i); }
      ev.preventDefault();
    }
    function dotUp(ev) {
      if (ev.pointerId !== dragId) return;
      dragId = -1;
      if (!dragMoved) embla.scrollTo(tickAt(ev.clientX));
    }

    // Here, not up beside the embla call: `var dotHost` hoists but its
    // assignment does not, so calling this any earlier reads undefined and
    // silently builds no counter at all.
    if (dotHost) buildDots();

    var slotCache = 0;
    function slotPx() {
      if (!slotCache) {
        slotCache = parseFloat(getComputedStyle(host).getPropertyValue('--deck-slot')) || 1;
      }
      return slotCache;
    }

    function onSelect() {
      idx = embla.selectedScrollSnap();
      // The bar says where the deck is; aria-current says which card is
      // selected, which is a different question and only has a discrete answer.
      dotBtns.forEach(function (b, j) {
        b.setAttribute('aria-current', String(j === idx));
      });
      if (o.onIndex) o.onIndex(idx, items[idx], cards[idx]);
    }
    function onReInit() { slotCache = 0; unit = stepUnit(); measureDots(); paint(); }

    embla.on('scroll', paint);
    embla.on('select', onSelect);
    embla.on('reInit', onReInit);

    /* ---------------------------------------------------------- auto-advance
     * Suppressed while the reader is plainly busy with the deck (pointer over
     * it, mid-drag) or cannot see it (scrolled past, tab in the background). An
     * advance that fires under the cursor reads as the page fighting you.
     *
     * Time-driven advance lives here because the deck owns the clock. Advance
     * driven by the content -- "when this clip has played once" -- cannot: only
     * the caller knows when its own media is done, so it calls advance(), which
     * still honours every suppression above and reports whether it moved.
     */
    var autoMs = (o.autoplay && o.autoplay.ms) || 0;
    var autoOn = !!o.autoplay && o.autoplay.start !== false && !REDUCED;
    var timer = null, hovering = false, dragging = false;

    function offstage() {
      if (document.hidden) return true;
      var r = host.getBoundingClientRect();
      return r.bottom < 0 || r.top > window.innerHeight;
    }
    /* A caller advancing on its own content -- "this clip has now played once" --
     * is held back only by what makes the move pointless or hostile: the deck is
     * off screen, the tab is in the background, or a drag is in progress.
     *
     * Hovering deliberately does not count here. Watching a deck with the cursor
     * resting on it is the normal case, and refusing to advance then is how a
     * deck ends up looping one card forever while the reader waits for it to
     * move. The timer is the one that yields to the pointer, below. */
    /* A deck that cannot loop still has to get past its own last card. With
       loop off, scrollNext() there does nothing, so anything driving advance --
       a timer, or a caller whose clip has just played out -- would sit against
       the wall forever. It turns round and walks back instead. On a two-card
       deck that is the entire behaviour: left, right, left. Looping decks are
       untouched, wrapping being their own way past the end. */
    var back = false;
    function advance() {
      // A scrub on the counter is a drag too: an advance under the finger would
      // move the deck away from the card the reader is reaching for.
      if (dragging || dragId !== -1 || offstage()) return false;
      if (looping) { embla.scrollNext(); return true; }
      if (back && !embla.canScrollPrev()) back = false;
      else if (!back && !embla.canScrollNext()) back = true;
      if (back) embla.scrollPrev(); else embla.scrollNext();
      return true;
    }
    function arm() {
      clearTimeout(timer);
      if (!autoMs || !autoOn) return;
      timer = setTimeout(function () {
        // Nothing about the content is asking to move on, so a pointer resting
        // on the deck means "wait" -- the reader is reading.
        if (!hovering) advance();
        arm();
      }, autoMs);
    }
    view.addEventListener('pointerenter', function () { hovering = true; });
    view.addEventListener('pointerleave', function () { hovering = false; });
    embla.on('pointerDown', function () { dragging = true; });
    embla.on('pointerUp', function () { dragging = false; });
    embla.on('select', arm);          // the interval measures from the last change
    document.addEventListener('visibilitychange', arm);
    arm();

    function focusKey(ev) {
      if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
      if (ev.key === 'ArrowLeft') { embla.scrollPrev(); ev.preventDefault(); }
      else if (ev.key === 'ArrowRight') { embla.scrollNext(); ev.preventDefault(); }
    }
    var entry = null;
    if (o.keys === 'global') {
      entry = { host: host, api: embla };
      register(entry);
    } else if (o.keys === 'focus') {
      view.addEventListener('keydown', focusKey);
    }

    var ro = new ResizeObserver(measure);
    ro.observe(host);
    /* The counter is not the deck: it sits under it, in a column the caller
       sizes, and both the gap and is-dense follow the width it actually gets.
       Same observer, second target.

       Observing the element measureDots writes to is how a ResizeObserver
       loops, so what it writes is a max-width and not a width: capping a
       990 px row at 455 px reports once more and then agrees with itself. */
    if (dotHost) ro.observe(dotHost);
    measure();
    onReInit();
    onSelect();

    return {
      embla: embla,
      go: function (i) { embla.scrollTo(i); },
      next: function () { embla.scrollNext(); },
      prev: function () { embla.scrollPrev(); },
      cards: cards,
      at: function () { return idx; },
      advance: advance,
      // auto(true|false) toggles; auto(ms) also sets the interval, so a page can
      // offer a switch without the deck having been given one at mount.
      auto: function (on) {
        if (typeof on === 'number') { autoMs = on; autoOn = on > 0; }
        else autoOn = !!on;
        if (REDUCED) autoOn = false;
        arm();
        return autoOn;
      },
      autoOn: function () { return autoOn; },
      refresh: measure,
      destroy: function () {
        ro.disconnect();
        clearTimeout(timer);
        document.removeEventListener('visibilitychange', arm);
        if (entry) unregister(entry);
        embla.destroy();
        host.innerHTML = '';
        // The host element is the caller's and outlives the deck -- the lab page
        // re-mounts onto the same one on every knob move. Its buttons go with
        // its innerHTML, but the scrubber's listeners are on the host itself and
        // would be left holding a destroyed carousel.
        if (dotHost) {
          dotHost.removeEventListener('pointerdown', dotDown);
          dotHost.removeEventListener('pointermove', dotMove);
          dotHost.removeEventListener('pointerup', dotUp);
          dotHost.removeEventListener('pointercancel', dotUp);
          dotHost.classList.remove('is-dense');
          dotHost.style.maxWidth = '';
          dotHost.innerHTML = '';
        }
      }
    };
  }

  return { mount: mount };
})();
