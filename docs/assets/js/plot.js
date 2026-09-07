/* TVPlot -- the page's third shared component, after TVDeck and TVTracks.
 *
 * One figure over the paper's comparison table's own axes: which half of the
 * benchmark, segment or frame level, precision recall or F1, for 21 methods and
 * 7 ablated components. The reader turns it; the slice it opens on is the page's
 * headline, public x segment x F1.
 *
 * Two render modes, because "which is best here" and "is it good on both halves
 * of the benchmark" are different questions and one shape cannot ask both:
 *
 *   ranked   one metric, every row, sorted. The default.
 *   pair     one metric on both halves, against each other.
 *
 * Both take the same rows, share one control bar, and read one data file. Both
 * carry the same furniture: linear axes, reveal on scroll-in with replay on
 * click, hover-to-focus with click-to-pin, and a floating panel that reads a
 * mark out on the figure while dodging the other marks.
 *
 * Hand-written SVG on purpose. No chart library, no new dependency. Every
 * rendered string is ASCII-escaped -- see section 5 of the design note.
 */
window.TVPlot = (function () {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';

  /* Shared by the stylesheet's entrance animation and the timer that cleans it
   * up. If these two disagree, either the entrance is cut short or its delays
   * outlive it and slow the highlight down. */
  var REVEAL_MS = 340, STAGGER_MS = 22;
  var REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)');

  function n(tag, attrs, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function scale(min, max, lo, hi) {
    if (max === min) return function () { return lo; };
    return function (v) { return lo + (v - min) / (max - min) * (hi - lo); };
  }

  function axisTicks(min, max, step) {
    var out = [], v = min;
    /* Accumulate from min rather than multiplying, so a 0.1 step does not
     * drift into 0.30000000000000004 in the tick label. */
    for (var i = 0; v <= max + 1e-9; i++, v = min + i * step) {
      out.push(Math.round(v * 1e6) / 1e6);
    }
    return out;
  }

  /* A round step that leaves roughly six ticks on the axis. Boundary error runs
   * to tens of seconds while an F1 never leaves 0-1, so the step cannot be a
   * constant. */
  function niceStep(span) {
    var raw = span / 6;
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    /* The epsilon is not decoration: 1.2 / 6 is 0.19999999999999998, whose
     * mantissa is a hair under 2 and rounded down to a step of 0.1 -- eleven
     * ticks where six were asked for. */
    var mult = raw / mag + 1e-9;
    return (mult >= 5 ? 5 : mult >= 2 ? 2 : 1) * mag;
  }

  /* The next step up the 1-2-5 ladder: .1 -> .2 -> .5 -> 1 -> 2 ... */
  function coarser(step) {
    var mag = Math.pow(10, Math.floor(Math.log10(step) + 1e-9));
    var m = Math.round(step / mag);
    return m === 1 ? 2 * mag : m === 2 ? 5 * mag : 10 * mag;
  }

  /* A CLIPPED CEILING FOR A METRIC WITH NO CEILING. Boundary error is a
   * distance, and one method's 60 s of it against everyone else's few seconds
   * decides the whole axis if the axis has to hold it: seventeen marks in a
   * band by the origin so that one can be at the far end. So the axis stops
   * at the largest value that is not an outlier -- Tukey's upper fence, the
   * third quartile plus one and a half interquartile ranges, the usual
   * definition and not a number tuned to this data -- snapped outward to the
   * tick step, and whatever lies past it is drawn AT the edge, marked as
   * clipped, with its true value written beside it. Nothing is dropped and
   * nothing is drawn where it is not; the reader is told the axis stops. With
   * no outlier the fence is above the maximum and this is the maximum, padded.
   * Bounded metrics never come here: their ceiling is their bound. */
  function clipCeiling(values) {
    var v = values.slice().sort(function (a, b) { return a - b; });
    var q = function (f) {
      var i = (v.length - 1) * f, lo = Math.floor(i), hi = Math.ceil(i);
      return v[lo] + (v[hi] - v[lo]) * (i - lo);
    };
    var fence = q(0.75) + 1.5 * (q(0.75) - q(0.25));
    var top = 0;
    v.forEach(function (x) { if (x <= fence) top = Math.max(top, x); });
    if (!top) top = v[v.length - 1] || 1;
    var step = niceStep(top * 1.08);
    return Math.ceil(top * 1.08 / step) * step;
  }

  /* ----------------------------------------------- family -> mark colour
   *
   * The published page has five method classes. Only two get a hue, and both
   * are hues the page already assigns: SBD is the baseline the abstract argues
   * against directly, Open VLM is the family the headline compares against.
   * Ours is the page's accent. The other two stay neutral on purpose -- a hue
   * on them would claim a significance the data does not have.
   *
   * Every value here is a token whose contract is "figure colour". Heuristic
   * and Ablation used to read --fg-3, a *text* token, which made them the only
   * marks on the figure that changed colour with the theme. Defined in
   * site.css; see HAX-36. */
  var FAMILY = {
    'Ours':                 'var(--ours)',
    'SBD':                  'var(--tn)',
    'Open VLM (zero-shot)': 'var(--as)',
    'API VLM':              'var(--gt)',
    'Heuristic':            'var(--heur)',
    'TAL':                  'var(--tal)',
    'Ablation':             'var(--abla)'
  };

  /* A family with no colour throws rather than falling back to grey: a new
   * family in the data needs a deliberate colour decision (HAX-36), and
   * greying it would put two families on one colour and read as data. */
  function colour(row) {
    if (row.ours) return FAMILY.Ours;
    var f = FAMILY[row.family];
    if (f == null) {
      throw new Error('TVPlot: no colour for family "' + row.family + '" (' +
                      row.method + '). Assign one in FAMILY.');
    }
    return f;
  }

  /* -------------------------------------------------- reveal and panel */

  /* Reveal when the figure scrolls in, replay on click. Under reduced motion
   * the figure is simply already in -- no observer, no timers. */
  /* Reveal when the figure scrolls in, replay on click. Under reduced motion
   * the figure is simply already in -- no observer, no timers.
   *
   * The entrance is a keyframe animation, and both classes come off once it has
   * played. Nothing about the entrance may outlive it: while `will-reveal` is
   * on, every mark carries that row's stagger delay, and a delay on a mark is
   * a delay on the hover highlight too. */
  function reveal(fig, rows) {
    if (REDUCED.matches) return;

    /* Added here rather than in the stylesheet, so a page whose script never
     * runs shows a plain, complete figure instead of an empty box. */
    fig.classList.add('will-reveal');

    var done = false;
    function go() {
      if (done) return;
      done = true;
      fig.classList.add('is-in');
      setTimeout(function () {
        fig.classList.remove('will-reveal', 'is-in');
      }, REVEAL_MS + rows * STAGGER_MS + 80);
    }

    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting) { go(); io.unobserve(e.target); }
      });
    }, { threshold: 0.15 });
    io.observe(fig);

    /* Fallback: an IntersectionObserver reports nothing while the tab is
     * backgrounded or the pane is occluded, and a figure stuck at opacity 0 is
     * strictly worse than one that arrives without its entrance. Reveal
     * unconditionally after 1.2 s -- a reader who is actually looking at the
     * figure has already had the observer fire long before this. */
    setTimeout(go, 1200);
  }

  function replay(fig, rows) {
    if (REDUCED.matches) return;
    fig.classList.remove('will-reveal', 'is-in');
    void fig.offsetWidth;                       // force a reflow so it restarts
    reveal(fig, rows);
    fig.classList.add('is-in');
  }

  /* What the panel should try not to cover: the marks, and every word the
   * figure needs to be read -- the row names, the number at a bar's end, the
   * tick labels, the axis titles, the note on the parity line. Grid lines are
   * furniture and do not count; a scale the panel has covered is a figure
   * without units. */
  var OBSTACLES = '.bar, .pair-dot, .plot-label, .bar-num,' +
                  '.plot-tick, .plot-axis, .plot-annot';

  /* The panel that reads a mark out. It floats over the figure beside the mark
   * rather than sitting in a line underneath it: a strip of digits below the
   * plot is too far from the mark it describes to be read at all.
   *
   * HTML, not foreignObject -- the panel is typography, and it should take the
   * page's type scale rather than the SVG's viewBox units. aria-live carries it
   * to a screen reader as the keyboard cursor moves. */
  function makePanel(fig) {
    var box = el('div', 'plot-panel');
    box.setAttribute('role', 'tooltip');
    box.setAttribute('aria-live', 'polite');
    box.hidden = true;
    fig.appendChild(box);

    return function (fields, anchor) {
      if (!fields) { box.hidden = true; return; }

      box.textContent = '';
      box.appendChild(el('p', 'panel-name', fields.name));
      if (fields.family) {
        var chip = el('p', 'panel-family', fields.family);
        chip.style.setProperty('--c', fields.colour || 'var(--fg-3)');
        box.appendChild(chip);
      }
      var dl = el('dl', 'panel-rows');
      fields.rows.forEach(function (r) {
        dl.appendChild(el('dt', r[2] ? 'is-' + r[2] : null, r[0]));
        dl.appendChild(el('dd', r[2] ? 'is-' + r[2] : null, r[1]));
      });
      box.appendChild(dl);
      if (fields.note) box.appendChild(el('p', 'panel-note', fields.note));

      /* Everything below is measured from client rects, not viewBox units: the
       * SVG scales to the column width, so a viewBox coordinate is not a pixel.
       * Every rect is taken once, up front, and the search that follows is pure
       * arithmetic -- measuring inside the loop would thrash layout. */
      box.hidden = false;
      var fb = fig.getBoundingClientRect();
      var pw = box.offsetWidth, ph = box.offsetHeight;
      var ab = anchor.getBoundingClientRect();
      var ax = ab.left - fb.left, ay = ab.top - fb.top;
      var acx = ax + ab.width / 2, acy = ay + ab.height / 2;

      /* The marks and text the panel should try not to cover, in figure
       * coordinates. Nothing in the anchor's own group counts: the panel is
       * allowed to sit against the thing it describes, and it repeats that
       * mark's name and numbers in its own type. Counted, a mark's own label
       * made the free slot beside its own dot look as expensive as one over a
       * neighbour's. Marks are shapes and text is text, which is the whole
       * distinction the cost function needs. */
      var obs = [];
      var kin = anchor.parentNode;
      [].forEach.call(fig.querySelectorAll(OBSTACLES), function (o) {
        if (o === anchor || kin.contains(o)) return;
        var b = o.getBoundingClientRect();
        if (!b.width || !b.height) return;
        obs.push({
          l: b.left - fb.left, t: b.top - fb.top,
          r: b.right - fb.left, b: b.bottom - fb.top,
          name: o.tagName === 'text'
        });
      });

      /* What a panel at this spot would hide. Covering a mark at all is a
       * failure rather than something to trade off by area -- a mark half
       * hidden is a mark you cannot read -- so the count is what the cost
       * function is built on.
       *
       * A word is the same: past a fifth of its box it stops being readable,
       * and it counts as a mark from there. Only a clipped edge -- a
       * descender, the last digit of a number the panel has grazed -- is
       * traded off by area. Measured by area alone, twenty-one row labels at a
       * quarter of a point per pixel were cheaper than one bar, and the panel
       * parked itself over the whole name column, including the name of the
       * row it was describing. */
      var HARD_NAME = 0.2;

      function hides(left, top) {
        var marks = 0, names = 0;
        for (var i = 0; i < obs.length; i++) {
          var o = obs[i];
          var w = Math.min(left + pw, o.r) - Math.max(left, o.l);
          if (w <= 0) continue;
          var h = Math.min(top + ph, o.b) - Math.max(top, o.t);
          if (h <= 0) continue;
          if (!o.name) { marks++; continue; }
          if (w * h > (o.r - o.l) * (o.b - o.t) * HARD_NAME) marks++;
          else names += w * h;
        }
        return { marks: marks, names: names };
      }

      var maxL = Math.max(4, fb.width - pw - 4);
      var maxT = Math.max(4, fb.height - ph - 4);

      /* MARK_COST against AWAY sets the exchange rate: the panel will travel
       * about 200 px across the figure rather than cover one mark, and focus
       * dims everything else so a panel that has travelled still reads as
       * belonging to the mark that is lit. */
      var MARK_COST = 900, NAME_COST = 0.25, AWAY = 4, SLID = 4;

      function pick(list, surcharge) {
        var win = null;
        for (var i = 0; i < list.length; i++) {
          var left = Math.max(4, Math.min(list[i][0], maxL));
          var top = Math.max(4, Math.min(list[i][1], maxT));
          var hidden = hides(left, top);
          var slid = Math.abs(left - list[i][0]) + Math.abs(top - list[i][1]);
          var away = Math.abs(left + pw / 2 - acx) + Math.abs(top + ph / 2 - acy);
          var cost = hidden.marks * MARK_COST + hidden.names * NAME_COST +
                     surcharge + slid * SLID + away * AWAY + i * 0.5;
          if (!win || cost < win.cost) {
            win = { left: left, top: top, cost: cost, marks: hidden.marks };
          }
        }
        return win;
      }

      /* Eight slots around the mark first: a panel touching what it describes
       * reads best. Sides lead, because that needs no eye movement at all. */
      var GAP = 14;
      var best = pick([
        [ax + ab.width + GAP, acy - ph / 2],
        [ax - pw - GAP, acy - ph / 2],
        [ax + ab.width + GAP, ay - ph - GAP],
        [ax - pw - GAP, ay - ph - GAP],
        [ax + ab.width + GAP, ay + ab.height + GAP],
        [ax - pw - GAP, ay + ab.height + GAP],
        [acx - pw / 2, ay - ph - GAP],
        [acx - pw / 2, ay + ab.height + GAP]
      ], 0);

      /* If the least bad of those still hides a mark, search the whole figure
       * on a coarse grid. The grid pays an entry fee, so a slot beside the mark
       * wins any near-tie. */
      if (best.marks > 0) {
        var grid = [], STEP = 18;
        for (var gx = 4; gx <= maxL; gx += STEP) {
          for (var gy = 4; gy <= maxT; gy += STEP) grid.push([gx, gy]);
        }
        var far = pick(grid, 120);
        if (far.cost < best.cost) best = far;
      }

      box.style.left = best.left + 'px';
      box.style.top = best.top + 'px';
    };
  }

  /* Hover focuses, click pins, click the background releases. Arrow keys move a
   * cursor through the marks in the order they were drawn, which both modes
   * draw already sorted.
   *
   * Every mode gives each mark an `__anchor`: the shape the panel should sit
   * beside. That is not the mark itself wherever a mark spans a whole row. */
  function wire(fig, svg, marks, show, describe) {
    var pinned = null, at = -1;

    function paint(i) {
      marks.forEach(function (m, j) { m.classList.toggle('is-on', j === i); });
      fig.classList.toggle('has-focus', i >= 0);
      show(i >= 0 ? describe(i) : null, i >= 0 ? marks[i].__anchor : null);
    }
    function focus(i) { at = i; paint(pinned != null ? pinned : i); }

    marks.forEach(function (m, i) {
      m.addEventListener('mouseenter', function () { if (pinned == null) focus(i); });
      m.addEventListener('click', function (ev) {
        ev.stopPropagation();
        pinned = pinned === i ? null : i;
        at = i;
        paint(pinned != null ? pinned : i);
      });
    });

    svg.addEventListener('mouseleave', function () {
      if (pinned == null) { at = -1; paint(-1); }
    });
    fig.addEventListener('click', function () {
      pinned = null; paint(-1); replay(fig, marks.length);
    });

    svg.setAttribute('tabindex', '0');
    svg.addEventListener('keydown', function (ev) {
      var d = ev.key === 'ArrowRight' || ev.key === 'ArrowDown' ? 1
            : ev.key === 'ArrowLeft' || ev.key === 'ArrowUp' ? -1 : 0;
      if (d) {
        ev.preventDefault();
        pinned = null;
        focus((at + d + marks.length) % marks.length);
      } else if (ev.key === 'Escape') {
        pinned = null; at = -1; paint(-1);
      }
    });
    svg.addEventListener('blur', function () { if (pinned == null) { at = -1; paint(-1); } });
  }

  /* A figure is its marks and nothing else: no title, no caption, no source
   * line. The controls above it name every axis, and a figure that explains
   * itself in prose is telling the reader it does not. What it cannot do
   * without is the alt text, which is the figure for anyone not looking at
   * it. */
  function frame(host, opts, w, h) {
    if (!opts.alt) throw new Error('TVPlot: opts.alt is required');

    host.textContent = '';
    var fig = el('figure', 'plot');

    var svg = n('svg', {
      viewBox: '0 0 ' + w + ' ' + h,
      class: 'plot-svg',
      role: 'img',
      'aria-label': opts.alt
    });
    fig.appendChild(svg);
    host.appendChild(fig);
    return { fig: fig, svg: svg };
  }

  function finish(fig, rows) {
    var show = makePanel(fig);
    reveal(fig, rows);
    return show;
  }

  /* ============================================================= ranked
   *
   * One metric, every row, sorted. Bars start at zero and length is the value,
   * always -- for boundary error a long bar is a bad one, which is why the
   * metric carries its own direction and the axis says which way is better
   * rather than the bars quietly changing meaning.
   *
   * "How does it do on the other half?" is not answered by decorating this
   * shape with a second marker -- it is the pair mode, which the Data control
   * reaches directly. */
  /* The figure is drawn in CSS pixels: the viewBox width is the width the SVG
   * is actually rendered at, so one unit is one pixel and the type in plot.css
   * comes out at the size it asks for.
   *
   * It used to be a flat 1120 scaled to fit, which multiplied every label by
   * the column's own fraction of that -- 3.4px at 375, 6.9px at 768, a bar
   * chart with grey smudges where the method names should be. The rule at the
   * foot of plot.css that meant to fix it set font-size in viewBox units, so it
   * was scaled by the very factor it was correcting.
   *
   * One pixel per unit makes the label column a real column, so it has to
   * reflow like one rather than shrink: 236px where there is room for it, then
   * narrower, and under 520 there is no room for a column at all and the name
   * goes above its own bar -- the same trade the clip timeline makes. */
  function rankedLayout(w) {
    if (w >= 760) return { RH: 30, LAB: 236, NUM: 74, TOP: 62, stack: false };
    if (w >= 520) return { RH: 30, LAB: 148, NUM: 62, TOP: 62, stack: false };
    return { RH: 48, LAB: 0, NUM: 0, TOP: 56, stack: true };
  }

  function ranked(host, opts) {
    // Taking the host over from a scatter: its observer must not redraw it back.
    if (host.__pairRO) { host.__pairRO.disconnect(); host.__pairRO = null; }
    function order(o) {
      var lower = o.metric.dir === 'asc';
      var items = [];
      o.rows.forEach(function (row, i) {
        if (o.values[i] == null) return;
        items.push({ row: row, v: o.values[i] });
      });
      /* Array sort is stable, so rows that tie keep the order they came in
       * rather than shuffling on every redraw. */
      items.sort(function (a, b) { return lower ? a.v - b.v : b.v - a.v; });
      return items;
    }

    function domain(items, metric) {
      if (metric.max != null) return metric.max;
      return clipCeiling(items.map(function (it) { return it.v; }));
    }

    var items = order(opts);
    /* The identity of the row *set*, so it is stable under a re-sort -- keyed
     * on the sorted order it would change on exactly the changes it is meant
     * to absorb. */
    var keys = items.map(function (it) { return it.row.method; })
                    .slice().sort().join('|');

    /* Same rows, so this is a filter change rather than a different figure:
     * slide them instead of building a new one. Rebuilding is what made every
     * control change flush the whole chart in from the top. */
    var live = host.__ranked;
    if (live && live.mode === 'ranked' && live.keys === keys) {
      live.update(opts, items);
      return;
    }

    /* Before the host has been laid out there is nothing to measure; the old
     * fixed width stands in and the observer at the bottom corrects it. */
    function hostWidth() {
      return Math.round(host.getBoundingClientRect().width) || 1120;
    }

    var W = hostWidth(), L = rankedLayout(W);
    var RH = L.RH, LAB = L.LAB, NUM = L.NUM, TOP = L.TOP, stack = L.stack;
    var H = items.length * RH + 100;
    var f = frame(host, opts, W, H), svg = f.svg;

    var grid = n('g', { class: 'plot-grid' });
    var axis = n('text', { 'text-anchor': 'middle', class: 'plot-axis' });
    svg.appendChild(grid);
    svg.appendChild(axis);

    var byMethod = {};
    var layer = n('g', { class: 'plot-marks' }), marks = [];
    items.forEach(function (it, ri) {
      /* Every row is drawn about its own origin and placed with a transform, so
       * re-sorting is one property per row and the browser can animate it. */
      var m = n('g', {
        class: 'mark' + (it.row.ours ? ' is-ours' : ''),
        style: '--c:' + colour(it.row) + ';--i:' + ri, tabindex: '-1'
      });
      /* Geometry is left to place(): it is the half of this figure that depends
       * on the width, and it has to be re-decided rather than re-scaled. */
      var hit = n('rect', { x: 0, class: 'mark-hit' });
      var lbl = n('text', { class: 'plot-label' }, it.row.method);
      var bar = n('rect', { height: 14, rx: 2, class: 'bar' });
      var num = n('text', { class: 'bar-num' });
      m.appendChild(hit);
      m.appendChild(lbl);
      m.appendChild(bar);
      m.appendChild(num);

      layer.appendChild(m);
      m.__row = it.row;
      m.__anchor = bar;
      marks.push(m);
      byMethod[it.row.method] = { g: m, hit: hit, lbl: lbl, bar: bar, num: num };
    });

    var current = opts, list = items;

    /* Everything that depends on the width. Attribute writes only -- the row
     * elements themselves never change, so a resize is not a rebuild and does
     * not flush the reveal in from the top. */
    function place() {
      svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
      axis.setAttribute('x', stack ? W / 2 : (LAB + W - NUM) / 2);
      axis.setAttribute('y', H - 6);
      items.forEach(function (it) {
        var r = byMethod[it.row.method];
        r.hit.setAttribute('y', -RH / 2);
        r.hit.setAttribute('width', W);
        r.hit.setAttribute('height', RH);
        if (stack) {
          /* The name on its own line, the value at the end of it, the bar the
           * whole width underneath. */
          r.lbl.setAttribute('x', 0);
          r.lbl.setAttribute('y', -13);
          r.lbl.setAttribute('text-anchor', 'start');
          r.num.setAttribute('x', W);
          r.num.setAttribute('y', -13);
          r.num.setAttribute('text-anchor', 'end');
          r.bar.setAttribute('x', 0);
          r.bar.setAttribute('y', 3);
        } else {
          r.lbl.setAttribute('x', LAB - 16);
          r.lbl.setAttribute('y', 4);
          r.lbl.setAttribute('text-anchor', 'end');
          r.num.setAttribute('x', W - NUM + 12);
          r.num.setAttribute('y', 4);
          r.num.setAttribute('text-anchor', 'start');
          r.bar.setAttribute('x', LAB);
          r.bar.setAttribute('y', -7);
        }
      });
    }

    /* The values, the axis and the row positions, for a fresh build and for
     * every update after it. */
    function paint(o, rows) {
      current = o;
      list = rows;
      var hi = domain(rows, o.metric);
      var x0 = stack ? 0 : LAB, x1 = stack ? W : W - NUM;
      var x = scale(0, hi, x0, x1);

      grid.textContent = '';
      /* The tick step follows the axis's width in pixels, not only its span in
       * units: niceStep aims at six ticks, which is right across 780px and puts
       * eleven 36px labels shoulder to shoulder across the 323px a phone has.
       * One label per ~64px is the most that reads as separate numbers, so the
       * step is asked for as many ticks as that allows and no more. */
      var want = Math.max(3, Math.floor((x1 - x0) / 64));
      var step = niceStep(hi);
      while (hi / step > want) step = coarser(step);
      var ticks = axisTicks(0, hi, step);
      ticks.forEach(function (t, i) {
        grid.appendChild(n('line', { x1: x(t), x2: x(t), y1: 42, y2: H - 46 }));
        /* Stacked, the bars run the full width of the SVG, so a label centred
         * on the first or last tick has half of itself outside it and the SVG
         * clips it. The two end labels hang inward instead. In the row layout
         * the label and number columns leave room, and centred stays. */
        var anchor = !stack ? 'middle' : i === 0 ? 'start'
                   : i === ticks.length - 1 ? 'end' : 'middle';
        grid.appendChild(n('text', { x: x(t), y: H - 28, 'text-anchor': anchor,
                                     class: 'plot-tick' }, t.toFixed(o.metric.dp)));
      });
      axis.textContent = o.metric.label +
        (o.metric.dir === 'asc' ? ' \u00b7 lower is better'
                                : ' \u00b7 higher is better');

      rows.forEach(function (it, ri) {
        var row = byMethod[it.row.method];
        // Past the ceiling the bar runs to the axis end and says so: hatched,
        // with an arrow after the true number. See clipCeiling.
        var clipped = it.v > hi;
        var w = Math.max(x(Math.min(it.v, hi)) - x0, 1.5);
        row.g.style.transform = 'translate(0px,' + (TOP + ri * RH) + 'px)';
        /* Both: the attribute keeps the bar right where the CSS width geometry
         * property is unsupported, and the style is what transitions where it
         * is. */
        row.bar.setAttribute('width', w);
        row.bar.style.width = w + 'px';
        row.bar.classList.toggle('is-clipped', clipped);
        row.num.textContent = it.v.toFixed(o.metric.dp) + (clipped ? ' \u2192' : '');
      });
    }

    /* The first paint happens while `layer` is still detached, so the rows are
     * inserted already at their positions. Painting after insertion would
     * transition every row in from y = 0 -- the entrance flush this change
     * exists to get rid of. */
    place();
    paint(opts, items);
    svg.appendChild(layer);

    var show = finish(f.fig, items.length);
    wire(f.fig, svg, marks, show, function (i) {
      return current.describe(marks[i].__row);
    });

    /* A width change re-decides the layout, so it cannot be left to the SVG's
     * own scaling -- that is the bug this whole function was rewritten for.
     * Guarded on the width actually having moved: setting the viewBox changes
     * the rendered height, which is a resize of the host, which would call this
     * again. Coalesced to a frame so a drag does not relayout per pixel. */
    if (window.ResizeObserver) {
      var seen = W, queued = false;
      new ResizeObserver(function () {
        if (queued || hostWidth() === seen) return;
        queued = true;
        requestAnimationFrame(function () {
          queued = false;
          seen = W = hostWidth();
          var L2 = rankedLayout(W);
          RH = L2.RH; LAB = L2.LAB; NUM = L2.NUM; TOP = L2.TOP; stack = L2.stack;
          H = items.length * RH + 100;
          place();
          paint(current, list);
        });
      }).observe(host);
    }

    host.__ranked = {
      mode: 'ranked',
      keys: keys,
      update: function (o, rows) {
        f.svg.setAttribute('aria-label', o.alt);
        paint(o, rows);
      }
    };
  }

  /* =============================================================== pair
   *
   * The same metric on two datasets, one dot per row, both axes on the same
   * scale so the dashed diagonal means "equally good on both" and the distance
   * from it is what the figure is about.
   *
   * Every row is named on the plot, beside its own dot -- see placeLabels. */
  function pair(host, opts) {
    host.__ranked = null;
    /* CSS PIXELS, like ranked(). The scatter used to be a fixed 1180x720
     * viewBox scaled to fit, so its names shrank with the column -- 9.7px at
     * a 1000px column, threes on a phone -- and the page hid the whole view
     * below 1000px rather than offer that. One unit is one pixel now: the type
     * is the page's type at every width, and the gate is gone. What still
     * scales is the geometry -- height and side room follow the width, within
     * bounds -- and a width change redraws the figure (below). */
    var W = Math.round(host.getBoundingClientRect().width) || 1120;
    var H = Math.round(Math.max(380, Math.min(720, W * 0.61)));
    /* The axes sit well inside the frame: a name beside a dot at either end
     * or on the top row needs somewhere to go, and the label placer will not
     * put one outside this box. The same room on both sides, so the axes box
     * is centred in the figure -- it used to be 84 against 210, which put the
     * whole distribution 63 units left of centre on a page that centres
     * everything else, and now that the domain follows the data a dot can
     * sit at the left edge as easily as the right. 150px at the full width,
     * shrinking with it to a floor a name can still hang in. */
    var side = Math.round(Math.max(56, Math.min(150, W * 0.127)));
    var P = { l: side, r: side, t: 54, b: 78 };
    var f = frame(host, opts, W, H), svg = f.svg;
    var metric = opts.metric;
    /* THE AXES FIT THE DATA, on both axes alike. A fixed 0-1 put a set that
     * runs 0.3-0.9 into the top corner of an empty square, and boundary error
     * has no ceiling at all, so a fixed 1 left an error of 1.4 s outside the
     * frame. So the domain is the data's own extent, padded by a few percent
     * and snapped outward to the tick step, and it is ONE domain for both
     * axes: the dashed diagonal means "equally good on both" only while a
     * unit is the same distance in x as in y. It never goes below zero, and
     * a bounded metric never gets an axis past its bound. */
    var vals = [];
    opts.rows.forEach(function (row, i) {
      if (opts.xValues[i] != null && opts.yValues[i] != null) {
        vals.push(opts.xValues[i], opts.yValues[i]);
      }
    });
    var lo = 0, hi = metric.max != null ? metric.max : 1;   // the empty figure
    if (vals.length && metric.max == null) {
      // Unbounded: the axis stops short of the outliers, which are pinned to
      // its edge below. See clipCeiling.
      hi = clipCeiling(vals);
    } else if (vals.length) {
      var vmin = Math.min.apply(null, vals), vmax = Math.max.apply(null, vals);
      var span = Math.max(vmax - vmin, metric.max * 0.05);
      var pad = span * 0.06;
      var step = niceStep(span + 2 * pad);
      lo = Math.max(0, Math.floor((vmin - pad) / step) * step);
      hi = Math.min(Math.ceil((vmax + pad) / step) * step, metric.max);
      if (hi <= lo) hi = lo + step;
    }
    // As many ticks as the axis's width allows at ~64px apiece, as in ranked().
    var want = Math.max(3, Math.floor((W - P.l - P.r) / 64));
    var tick = niceStep(hi - lo);
    while ((hi - lo) / tick > want) tick = coarser(tick);
    var x = scale(lo, hi, P.l, W - P.r);
    var y = scale(lo, hi, H - P.b, P.t);

    var g = n('g', { class: 'plot-grid' });
    axisTicks(lo, hi, tick).forEach(function (t) {
      g.appendChild(n('line', { x1: x(t), x2: x(t), y1: P.t, y2: H - P.b }));
      g.appendChild(n('line', { x1: P.l, x2: W - P.r, y1: y(t), y2: y(t) }));
      g.appendChild(n('text', { x: x(t), y: H - P.b + 20, 'text-anchor': 'middle',
                                class: 'plot-tick' }, t.toFixed(metric.dp)));
      g.appendChild(n('text', { x: P.l - 12, y: y(t) + 4, 'text-anchor': 'end',
                                class: 'plot-tick' }, t.toFixed(metric.dp)));
    });
    svg.appendChild(g);

    svg.appendChild(n('line', {
      x1: x(lo), y1: y(lo), x2: x(hi), y2: y(hi), class: 'plot-parity'
    }));
    var at = lo + (hi - lo) * 0.29;
    svg.appendChild(n('text', {
      x: x(at), y: y(at) - 9, class: 'plot-annot',
      transform: 'rotate(-37 ' + x(at) + ' ' + (y(at) - 9) + ')'
    }, 'equally good on both'));

    svg.appendChild(n('text', { x: (P.l + W - P.r) / 2, y: H - 22,
                                'text-anchor': 'middle', class: 'plot-axis' },
                      opts.x.label));
    svg.appendChild(n('text', { x: 18, y: (P.t + H - P.b) / 2, class: 'plot-axis',
                                transform: 'rotate(-90 18 ' + (P.t + H - P.b) / 2 + ')',
                                'text-anchor': 'middle' }, opts.y.label));

    var leaders = n('g', { class: 'plot-leaders' });
    var layer = n('g', { class: 'plot-marks' });
    svg.appendChild(leaders);
    svg.appendChild(layer);

    var marks = [], placed = [];
    opts.rows.forEach(function (row, i) {
      var vx = opts.xValues[i], vy = opts.yValues[i];
      if (vx == null || vy == null) return;
      var m = n('g', {
        class: 'mark' + (row.ours ? ' is-ours' : ''),
        style: '--c:' + colour(row) + ';--i:' + i, tabindex: '-1'
      });
      var r = row.ours ? 9 : 6.5;
      /* Past the ceiling on either axis the dot sits on that edge, hollow, and
       * its name carries the true value, so the figure says where it really is
       * while the axis stays readable for everyone else. See clipCeiling. */
      var clipped = vx > hi || vy > hi;
      var dot = n('circle', { cx: x(Math.min(vx, hi)), cy: y(Math.min(vy, hi)), r: r,
                              class: 'pair-dot' + (clipped ? ' is-clipped' : '') });
      m.appendChild(dot);
      layer.appendChild(m);
      m.__row = row;
      m.__anchor = dot;
      marks.push(m);
      var name = row.method;
      if (clipped) {
        name += ' \u00b7 ' + vx.toFixed(metric.dp) + ' / ' + vy.toFixed(metric.dp) + (metric.unit || '');
      }
      placed.push({ mark: m, row: row, name: name,
                    cx: x(Math.min(vx, hi)), cy: y(Math.min(vy, hi)), r: r });
    });

    var text = measure(svg, placed.map(function (it) { return it.name; }));
    var area = { x: 6, y: P.t - 26, w: W - 12, h: H - P.b - P.t + 48 };

    placeLabels(placed, area, text).forEach(function (it) {
      /* The label belongs to the mark, so pointing at the name is the same
       * gesture as pointing at the dot and both light up together. */
      it.mark.appendChild(n('text', {
        x: it.lx, y: it.ly, 'text-anchor': it.anchor,
        class: 'plot-label is-on-plot' + (it.row.ours ? ' is-ours' : '') +
               (it.crowded ? ' is-crowded' : '')
      }, it.name));

      /* Only the ones that had to move away from their dot get a leader. */
      if (!it.leader) return;
      var toward = it.anchor === 'end' ? -1 : 1;
      leaders.appendChild(n('polyline', {
        points: [it.cx + toward * (it.r + 2), it.cy,
                 it.lx - toward * 4, it.ly - 4].join(' '),
        class: 'plot-leader', style: '--c:' + colour(it.row)
      }));
    });

    var show = finish(f.fig, marks.length);
    wire(f.fig, svg, marks, show, function (i) {
      return opts.describe(marks[i].__row);
    });

    /* A width change is a new figure: the geometry above was decided from the
     * width, and scaling the old one is the bug this rewrite removes. One
     * observer per host -- the last draw's is disconnected first, so a filter
     * change does not leave an older observer redrawing older data -- and it
     * fires only when the width has actually moved, because a redraw changes
     * the height, which is itself a resize of the host. */
    if (host.__pairRO) host.__pairRO.disconnect();
    if (window.ResizeObserver) {
      var seen = W;
      host.__pairRO = new ResizeObserver(function () {
        var w = Math.round(host.getBoundingClientRect().width);
        if (!w || w === seen) return;
        pair(host, opts);
      });
      host.__pairRO.observe(host);
    }
  }

  /* --------------------------------------------------- label placement */

  /* Text width in viewBox units, measured by rendering probes once.
   *
   * Fallback: getComputedTextLength returns 0 when the SVG has not been laid
   * out yet -- a hidden pane, or a display:none ancestor. A per-character
   * estimate only ever sizes a label box, so being a few units out costs a
   * little whitespace or one extra leader, never a clipped or overlapped
   * name. */
  function measure(svg, texts) {
    var widths = {}, height = 0;
    var probes = texts.map(function (t) {
      var e = n('text', { class: 'plot-label is-on-plot', x: -9999, y: -9999 }, t);
      svg.appendChild(e);
      return e;
    });
    texts.forEach(function (t, i) {
      widths[t] = probes[i].getComputedTextLength() || t.length * 6.6;
      /* The line's real height, not its font size: the placer reserves boxes
       * with it, and a box shorter than what gets drawn lets two labels in
       * adjacent lanes touch. */
      try {
        height = Math.max(height, probes[i].getBBox().height);
      } catch (e) { /* getBBox throws on an unrendered SVG; the guard below
                     * covers it. */ }
      probes[i].remove();
    });
    return { widths: widths, height: height || 19 };
  }

  function hits(a, b) {
    return a.x < b.x + b.w && b.x < a.x + a.w &&
           a.y < b.y + b.h && b.y < a.y + a.h;
  }

  /* Put every name next to its own mark.
   *
   * Eight candidate slots around the dot, tried in order of how well they read;
   * the first that clears every dot and every name already placed wins. A point
   * in a tight cluster runs out of slots, so it then searches upward and
   * downward for the nearest free lane and takes a hairline leader back to its
   * dot. A gutter of names down the side was built first and thrown away: a
   * name you have to trace across the whole figure is barely a label.
   *
   * Order matters: the marks that carry the argument are placed first, so they
   * get the slot that reads best and the crowd works around them. */
  function placeLabels(items, box, text) {
    var widths = text.widths, H = text.height, PAD = 3;

    var taken = items.map(function (it) {
      return { x: it.cx - it.r, y: it.cy - it.r, w: it.r * 2, h: it.r * 2 };
    });

    var order = items.slice().sort(function (a, b) {
      return (b.row.ours ? 1 : 0) - (a.row.ours ? 1 : 0) || b.r - a.r;
    });

    order.forEach(function (it) {
      var w = widths[it.name] + PAD * 2;
      var g = it.r + 7;

      var slots = [
        [g, 4, 'start'], [-g, 4, 'end'],
        [g, -g, 'start'], [-g, -g, 'end'],
        [g, g + 6, 'start'], [-g, g + 6, 'end'],
        [0, -it.r - 8, 'middle'], [0, it.r + 16, 'middle']
      ];

      for (var i = 0; i < slots.length; i++) {
        var s = slots[i];
        var lx = s[2] === 'middle' ? it.cx : it.cx + s[0];
        var bx = s[2] === 'start' ? lx : s[2] === 'end' ? lx - w : lx - w / 2;
        var by = it.cy + s[1] - H + 2;
        var b = { x: bx, y: by, w: w, h: H };
        if (bx < box.x || bx + w > box.x + box.w) continue;
        if (by < box.y || by + H > box.y + box.h) continue;
        if (taken.some(function (t) { return hits(b, t); })) continue;
        it.lx = lx;
        it.ly = it.cy + s[1];
        it.anchor = s[2];
        taken.push(b);
        return;
      }

      /* No slot beside the dot. Step away vertically, alternating up and down,
       * until a lane is free -- and mark it for a leader, because the name is
       * no longer touching what it names. */
      for (var away = 1; away <= 14; away++) {
        for (var dir = 0; dir < 2; dir++) {
          var dy = (dir ? -1 : 1) * away * (H + 3);
          var sides = [[it.cx + g, 'start'], [it.cx - g, 'end']];
          for (var k = 0; k < sides.length; k++) {
            var sx = sides[k][1] === 'start' ? sides[k][0] : sides[k][0] - w;
            var sy = it.cy + dy - H + 2;
            var sb = { x: sx, y: sy, w: w, h: H };
            if (sx < box.x || sx + w > box.x + box.w) continue;
            if (sy < box.y || sy + H > box.y + box.h) continue;
            if (taken.some(function (t) { return hits(sb, t); })) continue;
            it.lx = sides[k][0];
            it.ly = it.cy + dy;
            it.anchor = sides[k][1];
            it.leader = true;
            taken.push(sb);
            return;
          }
        }
      }

      /* Eighteen names in one plot and this one still has nowhere to go. It
       * gets drawn on top of its dot rather than dropped -- an unreadable name
       * is a layout bug to fix, a missing one is a lie about the data. */
      it.lx = it.cx + g;
      it.ly = it.cy + 4;
      it.anchor = 'start';
      it.crowded = true;
    });

    return items;
  }

  return {
    ranked: ranked,
    pair: pair,
    families: FAMILY
  };
}());
