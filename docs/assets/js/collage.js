/* TVCollage — the hero's background: many short clips looping at once.
 *
 * MANY DECODERS, ON PURPOSE — WHICH IS WHY THERE IS A GOVERNOR. TVStrips ships
 * one <video> and 59 <img> because 27 concurrent VP9 streams janked badly, and
 * its header records the reason: the cost is per-element, not per-pixel. That
 * finding is about a number of elements and a codec, not a law. Here the tiles
 * are nine-ish H.264 clips with a 360 px short side, so the same argument does
 * not settle it either way — and a background whose whole idea is "several
 * things are alive at once" cannot be built on one player. So the budget is
 * MEASURED IN THE PAGE instead of assumed: an EMA of frame time, and a
 * governor that permanently retires tiles until the frame comes back. A
 * component that plays this much has to be able to give some of it up.
 *
 * NO SPRING, DELIBERATELY. spring.js is what everything else here animates on,
 * because everything else MOVES A DISTANCE and should arrive under its own
 * weight. Nothing in a collage arrives. What is integrated here is a heading
 * and a rate, and a spring on a heading is a wobble rather than an arrival: it
 * overshoots the target angle and comes back, which reads as a tile changing
 * its mind. So the heading eases exponentially toward a re-drawn target with a
 * hard rad/s cap on top -- the cap is the calm, and it is `turn`.
 *
 * ONE WRITE PER TILE PER FRAME. Every per-frame quantity -- position, tilt,
 * breathe -- lands in a single `transform`, plus `opacity`. Size, z-order and
 * the media swap happen at birth and nowhere else, so the frame loop never
 * reads geometry and never invalidates layout. The stage is measured once per
 * resize, at the top of a frame, ahead of any write.
 *
 * A POOL, NOT CHURN. `count` tiles are built at mount and reused forever: a
 * death re-randomises a tile in place and swaps its `src`/`poster`. Creating
 * and dropping <video> elements to the rhythm of the lifecycle would hand the
 * media stack a fresh decoder every few seconds, which is the one cost this
 * component cannot afford to pay twice.
 *
 * STEER, NEVER BOUNCE. A tile near an edge has its TARGET heading pulled toward
 * the stage centre and turns at its own capped rate; nothing reflects and
 * nothing teleports. Because `turn` is slow enough that a U-turn outlasts a
 * narrow stage, the boundary also projects the velocity: at the edge the
 * outward component is dropped, so the tile slides along the frame while its
 * heading comes round instead of leaving and never returning.
 *
 * ---------------------------------------------------------------------------
 * API
 *
 *   TVCollage.mount(host, {
 *     items: [{ id, src, poster, width, height, duration }],  // required
 *     count: 9,            // tiles alive at once, before the budget governor trims it
 *     minCount: 4,         // the governor never goes below this
 *     life: [2, 3],        // whole loops of its clip a tile lives, integer drawn per birth
 *     size: [0.16, 0.38],  // tile width as a fraction of the host width, drawn per birth
 *     maxHeight: 0.42,     // a tile is never taller than this fraction of the host height
 *     avoid: 0.5,          // radius of the central zone births tend to skip and drift steers out of,
 *                          // as a fraction of the half-stage; 0 turns it off
 *     breathe: 0.12,       // +/- fraction of that width, one slow sine over the tile's life
 *     drift: 14,           // px/s mean speed; each tile draws +/-40% of it
 *     turn: 0.25,          // rad/s cap on how fast a heading wanders
 *     tilt: 5,             // degrees, max rotation either way, wandering slowly
 *     fade: 900,           // ms, fade in and fade out
 *     gap: [0, 1500],      // ms of empty slot between one tile's death and the next birth there
 *     alpha: 0.92,         // resting opacity
 *     seed: null,          // number for a reproducible run, else Math.random
 *     onBudget: function (count, frameMs) {}   // called whenever the governor trims
 *   })
 *   -> { pause, resume, destroy, count }
 *
 * `items` is the whole contract: no convention about where the files live, no
 * opinion about their order. `width`/`height` are the file's own pixels, not a
 * display size -- they cap the upscale and set the tile's aspect. `duration` is
 * the life clock's fallback for a clip that never gets to play.
 *
 * The host is positioned by the CALLER. The component appends one absolutely
 * inset stage to it and nothing else, and takes no pointer events: it is a
 * background, and a background that could be clicked would eat the hero's own
 * links.
 *
 * Needs assets/css/collage.css. Reads the page's tokens (--bg, --shadow) where
 * they exist; every one has a fallback, so it renders standalone.
 * ---------------------------------------------------------------------------
 */
window.TVCollage = (function () {
  'use strict';

  var TAU = Math.PI * 2;

  var DEFAULTS = {
    count: 9,
    minCount: 4,
    life: [2, 3],
    size: [0.16, 0.38],
    maxHeight: 0.42,
    avoid: 0.5,
    breathe: 0.12,
    drift: 14,
    turn: 0.25,
    tilt: 5,
    fade: 900,
    gap: [0, 1500],
    alpha: 0.92,
    seed: null,
    onBudget: null
  };

  /* How fast a heading closes on its target BEFORE `turn` clips it, as an
     exponential rate. A plain lerp toward the target is frame-rate dependent;
     1 - exp(-k dt) is the same curve at any dt, which matters here because dt
     is exactly what the governor is watching move around. Tilt gets half the
     rate: a tile that re-levels as briskly as it changes course reads as
     twitchy at these amplitudes. */
  var EASE_HEAD = 1.1;
  var EASE_TILT = 0.5;

  /* The breathe frequency band, rad/s. 0.15 is a 42 s period and 0.35 an 18 s
     one; both are slower than a tile's typical life, which is the point -- a
     full cycle would read as a pulse. */
  var OMEGA = [0.15, 0.35];

  /* A tile never renders larger than twice the pixels its file actually has.
     Same rule and same reason as .card-media.low in site.css: this set has
     40 clips at 128x128, and fitting one to a 400 px tile would upscale it
     three times over and invite the reader to read detail that is not in the
     file. */
  var MAX_UPSCALE = 2;

  /* A 60 Hz frame is 16.7 ms. 22 ms sustained is therefore not a slow frame, it
     is a MISSED one roughly every third frame -- the compositor has already
     dropped to 45 fps and is heading for 30. Below that the EMA still sees
     normal variance, so a lower gate would trim tiles off a page that is
     actually fine. Three continuous seconds because one bad second is a GC
     pause, a texture upload, or a clip's first keyframe, and none of those is
     a reason to make the page permanently smaller. */
  var HOT_MS = 22;
  var HOT_FOR = 3;

  function el(tag, cls) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    return n;
  }

  /* mulberry32: 32 bits of state, uniform enough for placement and nowhere near
     anything that needs to be unguessable. Its whole job is that `seed` makes a
     run reproducible, so a scatter worth keeping can be looked at twice. */
  function mulberry32(a) {
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function shuffle(a, rnd) {
    for (var i = a.length - 1; i > 0; i--) {
      var j = (rnd() * (i + 1)) | 0;
      var v = a[i]; a[i] = a[j]; a[j] = v;
    }
  }

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  /* The shortest way round. Without it a target at +179 deg from a heading at
     -179 deg is a 358 deg journey, and the tile spins the long way. */
  function wrapPi(a) {
    a = (a + Math.PI) % TAU;
    if (a < 0) a += TAU;
    return a - Math.PI;
  }

  /* Ease-in-out on the fades. A linear opacity ramp has a visible corner at
     both ends -- the tile appears to start and stop appearing. */
  function ease(t) {
    t = clamp(t, 0, 1);
    return t < 0.5 ? 2 * t * t : 1 - 2 * (1 - t) * (1 - t);
  }

  /* Swallowed deliberately: a rejected play() here means the browser declined
     an autoplay it is entitled to decline, or the element was torn down
     mid-call. Neither is actionable, and neither should reach the console. */
  function roll(v) {
    var p = v.play();
    if (p && p.catch) p.catch(function () {});
  }

  function mount(host, opts) {
    if (!host || !host.appendChild) {
      throw new Error('TVCollage.mount: host must be an element');
    }
    var o = {}, k;
    for (k in DEFAULTS) o[k] = DEFAULTS[k];
    for (k in (opts || {})) if (opts[k] !== undefined) o[k] = opts[k];

    var items = o.items;
    if (!items || !items.length) {
      throw new Error('TVCollage.mount: items is required and must not be empty');
    }
    for (var i = 0; i < items.length; i++) {
      var it = items[i];
      ['id', 'src', 'poster'].forEach(function (f) {
        if (!it[f]) throw new Error('TVCollage.mount: items[' + i + '] has no ' + f);
      });
      ['width', 'height', 'duration'].forEach(function (f) {
        if (!(it[f] > 0)) {
          throw new Error('TVCollage.mount: items[' + i + '].' + f +
            ' must be a positive number, got ' + it[f]);
        }
      });
    }
    /* Two tiles showing the same clip is the one composition mistake this
       cannot make -- side by side they read as a mistake rather than as a
       repeat. The bag guarantees it by never dealing a live item, which is only
       possible while there are more items than slots. */
    if (items.length <= o.count) {
      throw new Error('TVCollage.mount: count ' + o.count + ' needs more than ' +
        o.count + ' items to avoid repeats, got ' + items.length);
    }

    var minCount = Math.min(o.minCount, o.count);
    var reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    var rnd = o.seed == null ? Math.random : mulberry32(o.seed >>> 0);
    var sizeSpan = o.size[1] - o.size[0];

    var stage = el('div', 'cl-stage');
    // Decoration. The hero's text is the accessible content of this region.
    stage.setAttribute('aria-hidden', 'true');
    host.appendChild(stage);

    var W = 0, H = 0, dirty = true;
    function measure() { W = stage.clientWidth; H = stage.clientHeight; dirty = false; }
    measure();

    /* ------------------------------------------------------------ the bag */
    // live[i] is true while items[i] is on the stage.
    var live = new Array(items.length);
    var bag = [];

    function refill() {
      bag.length = 0;
      for (var j = 0; j < items.length; j++) bag.push(j);
      shuffle(bag, rnd);
    }
    refill();

    /* Deal the first item from the end of the bag that is not already on the
       stage; the ones passed over stay in the bag and come up later, so the
       "everything before any repeat" property survives a skip. */
    function deal() {
      for (var pass = 0; pass < 2; pass++) {
        for (var j = bag.length - 1; j >= 0; j--) {
          if (!live[bag[j]]) return bag.splice(j, 1)[0];
        }
        refill();
      }
      // Unreachable while items.length > count, which mount() enforces.
      throw new Error('TVCollage: every item is on stage; nothing left to deal');
    }

    /* THE MIDDLE IS WHERE THE TEXT IS. A tile born under the title spends its
       whole life behind the frosted panel, seen by nobody, and a tile drifting
       through the middle disappears for as long as the crossing takes. So the
       centre is avoided -- probabilistically, not forbidden: a hard hole reads
       as a hole, and the panel's edges are soft. Distance is measured on the
       stage's own ellipse, 0 at the centre and 1 at the middle of an edge, so
       the zone has the stage's proportions rather than being a circle on a
       wide rectangle. */
    function centreDist(x, y) {
      var nx = (x - W / 2) / (W / 2), ny = (y - H / 2) / (H / 2);
      return Math.sqrt(nx * nx + ny * ny);
    }

    /* A birth position: drawn inside the middle 80%, and re-drawn while it
       falls in the zone, accepted there only with a probability that grows
       with the square of its distance out. Eight draws, then whatever came
       last -- on a very small stage every point is in the zone, and a birth
       still has to happen. */
    function spawn() {
      var x, y;
      for (var k = 0; k < 8; k++) {
        x = 0.1 * W + rnd() * 0.8 * W;
        y = 0.1 * H + rnd() * 0.8 * H;
        if (!o.avoid) break;
        var d = centreDist(x, y);
        if (d >= o.avoid) break;
        var p = d / o.avoid;
        if (rnd() < p * p) break;
      }
      return [x, y];
    }

    /* ---------------------------------------------------------- the tiles */
    var tiles = [];

    function build(slot) {
      var t = {
        slot: slot, item: null, state: 'idle', retired: false, gone: false,
        idleT: slot * 0.6 * o.fade,   // the staggered first birth
        node: el('div', 'cl-tile'), media: null, want: false,
        w: 0, h: 0, x: 0, y: 0, dur: 0,
        head: 0, headT: 0, headNext: 0, spd: 0,
        rot: 0, rotT: 0, rotNext: 0,
        ph: 0, om: 0,
        fadeT: 0, wall: 0, loops: 0, lifeLoops: 0, lastT: 0, everPlayed: false
      };
      if (reduced) {
        t.media = el('img');
        t.media.alt = '';
        t.media.decoding = 'async';
      } else {
        var v = el('video');
        v.muted = true;
        v.playsInline = true;
        v.loop = true;
        v.preload = 'auto';
        // The properties above do not all survive as attributes, and iOS reads
        // the attributes: without them a muted inline clip goes fullscreen.
        v.setAttribute('muted', '');
        v.setAttribute('playsinline', '');
        v.setAttribute('loop', '');
        v.setAttribute('preload', 'auto');
        // play() issued before the first frame is decodable is aborted by the
        // pending load(), so arm it on loadeddata rather than fire and hope.
        v.addEventListener('loadeddata', function () { if (t.want && running) roll(v); });
        t.media = v;
      }
      t.node.appendChild(t.media);
      stage.appendChild(t.node);
      return t;
    }

    function swap(t, it) {
      var m = t.media;
      if (reduced) { m.src = it.poster; return; }
      m.poster = it.poster;
      t.want = true;
      m.src = it.src;
      m.load();
      if (running) roll(m);
    }

    function birth(t) {
      var idx = deal();
      var it = items[idx];
      t.item = idx;
      live[idx] = true;

      var frac = o.size[0] + rnd() * sizeSpan;
      var w = Math.min(frac * W, MAX_UPSCALE * it.width);
      var h = w * it.height / it.width;
      /* The width draw is what sizes a tile, and for a landscape clip it is
         enough; a portrait clip at the same width is nearly twice as tall, and
         at a third of the stage's width it stood taller than the stage's
         middle. So height is capped too, as a fraction of the stage, and the
         width follows it down -- the aspect is the clip's and is never bent. */
      if (h > o.maxHeight * H) {
        h = o.maxHeight * H;
        w = h * it.width / it.height;
      }
      t.w = w;
      t.h = h;
      t.node.style.width = t.w.toFixed(1) + 'px';
      t.node.style.height = t.h.toFixed(1) + 'px';
      /* Bigger is nearer, so it sits on top. Off the RENDERED width, not off
         the drawn fraction: 40 of these clips are 128x128 and cap at 256 px, so
         a small file that drew a large fraction comes out as one of the
         smallest boxes on the stage -- and ranking it by what it asked for
         rather than by what it got would put a 256 px tile on top of a 433 px
         one, which is the one thing depth here is supposed to say. Written once,
         at birth: a z-order that changed while a tile breathed would re-sort
         the stack mid-fade. */
      var lo = o.size[0] * W, hi = o.size[1] * W;
      t.node.style.zIndex = String(1 + Math.round(98 *
        (hi > lo ? clamp((w - lo) / (hi - lo), 0, 1) : 1)));

      // Inside the middle 80%, so a tile is comfortably on screen before it has
      // drifted anywhere, and away from the centre by preference -- see spawn().
      var at = spawn();
      t.x = at[0];
      t.y = at[1];
      t.head = rnd() * TAU;
      t.headT = t.head;
      t.headNext = 2 + rnd() * 2;
      t.spd = o.drift * (1 + (rnd() * 2 - 1) * 0.4);
      t.rot = (rnd() * 2 - 1) * o.tilt;
      t.rotT = (rnd() * 2 - 1) * o.tilt;
      t.rotNext = 2 + rnd() * 2;
      t.ph = rnd() * TAU;
      t.om = OMEGA[0] + rnd() * (OMEGA[1] - OMEGA[0]);

      t.dur = it.duration;
      t.lifeLoops = o.life[0] + Math.floor(rnd() * (o.life[1] - o.life[0] + 1));
      t.loops = 0;
      t.wall = 0;
      t.lastT = 0;
      t.everPlayed = false;
      t.fadeT = 0;
      t.state = 'in';
      swap(t, it);
    }

    // The end of a fade-out: the slot goes dark and either waits out its gap or,
    // if the governor took it, leaves for good.
    function land(t) {
      if (t.item != null) { live[t.item] = false; t.item = null; }
      t.want = false;
      if (!reduced && !t.media.paused) t.media.pause();
      t.node.style.opacity = '0';
      if (t.retired) {
        t.gone = true;
        t.state = 'gone';
        /* The point of retiring is to give the decoder back, so the element goes
           with it. This is the one permanent removal in the component; nothing
           ever puts a tile back, so it is not churn. */
        if (!reduced) { t.media.removeAttribute('src'); t.media.load(); }
        if (t.node.parentNode) t.node.parentNode.removeChild(t.node);
        return;
      }
      t.state = 'idle';
      t.idleT = o.gap[0] + rnd() * (o.gap[1] - o.gap[0]);
    }

    /* --------------------------------------------------------- the budget */
    var ema = 16.7, hot = 0, warm = 0;

    /* The first births are the one moment this component is expensive by
       design: `count` clips fetching and decoding their first keyframes inside
       a few seconds, staggered by 0.6 fade apiece. Measured, that spike alone
       was enough to trim a page from nine tiles to six that then ran at 17 ms
       for the rest of its life -- a permanent verdict on a transient. So the
       governor only starts counting once the last staggered birth has had its
       whole fade to arrive. */
    var WARM_S = ((o.count - 1) * 0.6 + 1) * o.fade / 1000;

    function budget(ms, dt) {
      /* Anything past 200 ms is not a frame rate, it is a stall -- a tab
         throttle, a GC pause, a dragged window. Folding it into the EMA would
         poison several seconds of it, and the governor would trim a page that
         was never slow. */
      if (ms > 200) { hot = 0; return; }
      ema += (ms - ema) * 0.1;
      warm += dt;
      if (warm < WARM_S) { hot = 0; return; }
      if (ema > HOT_MS) hot += dt; else hot = 0;
      if (hot < HOT_FOR) return;
      hot = 0;
      if (alive() <= minCount) return;
      var victim = pickVictim();
      if (!victim) return;
      victim.retired = true;
      // An idle slot is already dark; retiring it costs nothing to look at.
      if (victim.state === 'idle') land(victim);
      if (typeof o.onBudget === 'function') o.onBudget(alive(), ema);
    }

    /* Retire the least disruptive slot: one that is already dark, else the tile
       furthest through its own life, which was about to fade out anyway. NOT the
       biggest -- the cost of a tile is per-element, not per-pixel (see the
       TVStrips header), so taking the largest one would only make the collage
       look smaller without buying back more frame. */
    function pickVictim() {
      var best = null, bestAge = -1;
      for (var j = 0; j < tiles.length; j++) {
        var t = tiles[j];
        if (t.gone || t.retired) continue;
        if (t.state === 'idle') return t;
        var age = t.everPlayed ? t.loops / t.lifeLoops : t.wall / (t.lifeLoops * t.dur);
        if (age > bestAge) { bestAge = age; best = t; }
      }
      return best;
    }

    /* Two counts, because a retired tile is doomed before it is dark. `alive` is
       what the governor budgets against and what onBudget reports -- the slot
       will never rebirth, so it is spent the moment it is marked. `count()` is
       what is in the DOM, which is one more than that for as long as the last
       retirement takes to fade. */
    function alive() {
      var n = 0;
      for (var j = 0; j < tiles.length; j++) if (!tiles[j].gone && !tiles[j].retired) n++;
      return n;
    }

    function count() {
      var n = 0;
      for (var j = 0; j < tiles.length; j++) if (!tiles[j].gone) n++;
      return n;
    }

    /* ----------------------------------------------------------- the loop */
    var raf = 0, last = 0, running = false;
    var userPaused = false, onScreen = true, shown = !document.hidden;

    function step(t, dt) {
      if (t.gone) return;

      if (t.state === 'idle') {
        t.idleT -= dt * 1000;
        if (t.idleT <= 0) birth(t);
        return;
      }

      /* The life clock. A loop is a wrap: `loop` on the element means
         currentTime falls back toward zero rather than passing duration, so the
         only observable boundary is the drop. The wall-time arm is not a
         belt-and-braces duplicate of it -- a clip whose play() the autoplay
         policy refused never wraps at all, and without it that slot would live
         forever and the collage would freeze one tile at a time. */
      if (!reduced) {
        var ct = t.media.currentTime;
        if (ct > 0.02) t.everPlayed = true;
        if (ct + 0.02 < t.lastT) t.loops++;
        t.lastT = ct;
      }
      t.wall += dt;
      var spent = t.everPlayed ? t.loops >= t.lifeLoops
                               : t.wall >= t.lifeLoops * t.dur;

      // ---- heading: a re-drawn target, eased, then clipped by `turn`
      t.headNext -= dt;
      if (t.headNext <= 0) {
        t.headT = rnd() * TAU;
        t.headNext = 2 + rnd() * 2;
      }
      // Inside the central zone the target heading points radially out, so a
      // tile that has drifted under the text turns to leave rather than sit
      // there. Re-taken every frame like the edge rule below, and the edge rule
      // wins where both apply, since leaving the stage is the worse outcome.
      if (o.avoid && centreDist(t.x, t.y) < o.avoid) {
        t.headT = Math.atan2(t.y - H / 2, t.x - W / 2);
      }
      // Within 15% of an edge -- or already past one, after a resize -- the
      // target becomes the stage centre. Re-taken every frame while it is in
      // the band, so the pull does not expire mid-approach.
      if (Math.min(t.x, W - t.x) < 0.15 * W || Math.min(t.y, H - t.y) < 0.15 * H) {
        t.headT = Math.atan2(H / 2 - t.y, W / 2 - t.x);
      }
      var d = wrapPi(t.headT - t.head);
      var cap = o.turn * dt;
      t.head += clamp(d * (1 - Math.exp(-EASE_HEAD * dt)), -cap, cap);

      var vx = Math.cos(t.head) * t.spd;
      var vy = Math.sin(t.head) * t.spd;
      /* At the frame, the outward component is dropped rather than reflected.
         `turn` is slow by design -- a U-turn at 0.25 rad/s takes 12.6 s, which
         on a phone-width stage is further than the band is wide -- so steering
         alone cannot promise a tile comes back. This makes it slide along the
         edge while its heading comes round, which is a stall, not a bounce. */
      if ((t.x <= 0 && vx < 0) || (t.x >= W && vx > 0)) vx = 0;
      if ((t.y <= 0 && vy < 0) || (t.y >= H && vy > 0)) vy = 0;
      t.x += vx * dt;
      t.y += vy * dt;

      // ---- tilt: the same wander, slower, and bounded by +/- tilt
      t.rotNext -= dt;
      if (t.rotNext <= 0) {
        t.rotT = (rnd() * 2 - 1) * o.tilt;
        t.rotNext = 2 + rnd() * 2;
      }
      t.rot += (t.rotT - t.rot) * (1 - Math.exp(-EASE_TILT * dt));

      // ---- breathe: one slow sine over the life, about the drawn size
      var s = 1 + o.breathe * Math.sin(t.ph + t.om * t.wall);

      // ---- opacity and the state it implies
      var op;
      if (t.state === 'in') {
        t.fadeT += dt * 1000;
        op = o.alpha * ease(t.fadeT / o.fade);
        if (t.fadeT >= o.fade) t.state = 'live';
      } else if (t.state === 'live') {
        op = o.alpha;
        /* A retired tile leaves the same way a spent one does. If the governor
           caught it mid-fade-in it finishes arriving first: cutting to a
           fade-out from a partial opacity would jump, and one extra `fade` of
           a tile that is on its way out is not worth a visible seam. */
        if (spent || t.retired) { t.state = 'out'; t.fadeT = 0; }
      } else {
        t.fadeT += dt * 1000;
        op = o.alpha * (1 - ease(t.fadeT / o.fade));
        if (t.fadeT >= o.fade) { land(t); return; }
      }

      // The only two writes in the loop.
      t.node.style.transform =
        'translate3d(' + (t.x - t.w / 2).toFixed(1) + 'px,' +
        (t.y - t.h / 2).toFixed(1) + 'px,0) rotate(' +
        t.rot.toFixed(2) + 'deg) scale(' + s.toFixed(3) + ')';
      t.node.style.opacity = op.toFixed(3);
    }

    function tick(now) {
      raf = requestAnimationFrame(tick);
      var ms = last ? now - last : 16.7;
      last = now;
      var dt = Math.min(ms, 50) / 1000;
      budget(ms, dt);
      // Read before any write, so a resize cannot cost a second layout pass.
      if (dirty) measure();
      for (var j = 0; j < tiles.length; j++) step(tiles[j], dt);
    }

    function sync() {
      // Reduced motion has no loop and no videos to hold, so there is nothing
      // for pause/resume to do -- and an <img> has no pause().
      if (reduced) return;
      var want = !userPaused && onScreen && shown;
      if (want === running) return;
      running = want;
      var j;
      if (want) {
        // Zeroed, so the first dt after a pause is one frame and not the length
        // of the pause -- which would otherwise arrive as a stall in the EMA.
        last = 0;
        raf = requestAnimationFrame(tick);
        for (j = 0; j < tiles.length; j++) {
          if (tiles[j].want && tiles[j].media.paused) roll(tiles[j].media);
        }
      } else {
        cancelAnimationFrame(raf);
        raf = 0;
        // `want` is untouched, so resume knows which slots were playing and
        // each clip picks up at the currentTime it was paused on.
        for (j = 0; j < tiles.length; j++) {
          if (!tiles[j].media.paused) tiles[j].media.pause();
        }
      }
    }

    /* --------------------------------------------------------- reduced motion
       No drift, no breathe, no tilt, no fades and no video: a still scatter of
       posters. Keeping the lifecycle and dropping only the fades was the other
       reading, and it is worse -- tiles appearing and vanishing instantly is
       more motion to ignore, not less. So nothing here runs a frame loop. */
    function scatter() {
      for (var j = 0; j < o.count; j++) {
        var t = build(j);
        tiles.push(t);
        birth(t);
        t.state = 'live';
        t.node.style.transform =
          'translate3d(' + (t.x - t.w / 2).toFixed(1) + 'px,' +
          (t.y - t.h / 2).toFixed(1) + 'px,0)';
        t.node.style.opacity = String(o.alpha);
      }
    }

    /* ---------------------------------------------------------- observers */
    var io = null, ro = null;

    function onVis() { shown = !document.hidden; sync(); }

    if (!reduced) {
      for (var n = 0; n < o.count; n++) tiles.push(build(n));

      // Off screen, none of this is worth a decoder or a frame.
      if (window.IntersectionObserver) {
        io = new IntersectionObserver(function (es) {
          onScreen = es[es.length - 1].isIntersecting;
          sync();
        }, { threshold: 0 });
        io.observe(host);
      }
      document.addEventListener('visibilitychange', onVis);
      if (window.ResizeObserver) {
        /* Only the bounds are re-read. A live tile keeps the size it was born
           at: re-deriving widths from the new stage would resize every tile at
           once on a window drag, which is the one motion here that is not slow.
           The next birth uses the new width. */
        ro = new ResizeObserver(function () { dirty = true; });
        ro.observe(stage);
      }
      sync();
    } else {
      scatter();
    }

    return {
      pause: function () { userPaused = true; sync(); },
      resume: function () { userPaused = false; sync(); },
      count: count,
      destroy: function () {
        userPaused = true;
        sync();
        if (io) io.disconnect();
        if (ro) ro.disconnect();
        document.removeEventListener('visibilitychange', onVis);
        for (var j = 0; j < tiles.length; j++) {
          var m = tiles[j].media;
          if (!reduced) { m.removeAttribute('src'); m.load(); }
          tiles[j].gone = true;
        }
        tiles.length = 0;
        if (stage.parentNode) stage.parentNode.removeChild(stage);
      }
    };
  }

  return { mount: mount };
})();
