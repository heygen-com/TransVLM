/* TVSpring — a critically-damped-ish spring, integrated by hand.
 *
 * WHY NOT A CSS TRANSITION. Everything this page animates is interruptible: the
 * spotlight is chasing a pointer that changes its mind mid-flight, and the
 * auto-tour can be overruled at any frame. A CSS transition restarts from zero
 * velocity every time its target changes, which is what makes a followed
 * element feel like it is being dragged in steps. A spring carries its
 * velocity, so a redirect mid-flight curves instead of stuttering. That is the
 * whole argument for owning the integrator.
 *
 * WHY NOT `motion`. This page has no framework and one vendored dependency; a
 * ~30 kB animation runtime for two springs is not a trade worth making. The
 * PARAMETERISATION here is framer-motion's, on purpose -- a = (-k(x - target)
 * - c·v)/m with stiffness/damping/mass -- so the constants tuned in gpumon
 * transfer verbatim rather than being re-guessed by eye.
 *
 * The presets come from gpumon's own tuning, cited per entry. Its DESIGN.md §4
 * splits the vocabulary the way this module does: springs carry things that
 * MOVE A DISTANCE and should arrive under their own weight; the three CSS
 * curves carry things that merely CHANGE -- hover colour, a fade, a swap.
 */
window.TVSpring = (function () {
  'use strict';

  /* Fixed substep. A stiff spring integrated over one long frame explodes:
     at k=520 a single 50 ms Euler step overshoots by more than the travel. */
  var SUB = 1 / 240;
  /* A backgrounded tab, a GC pause, or a dragged window hands the next frame a
     dt of seconds. Physics integrated over that gap is not slow motion, it is
     a different trajectory -- so the gap is clamped and the spring simply
     arrives later than it would have. */
  var DT_MAX = 0.05;

  /* Both lifted from gpumon rather than tuned again here. Its numbers are
     internally consistent -- stiffness falls and mass rises as the thing being
     moved gets bigger -- and a second person guessing at 0.8 damping ratios by
     eye would only produce a near-miss of them.

     zeta = c / (2*sqrt(k*m)) is what these actually say:
       travel  zeta 0.79   ~1.0% overshoot, at rest in ~0.45 s over 400 px
       open    zeta 1.00   no overshoot at all
     Which is the right asymmetry. An arrival is worth watching and gets a
     little weight; a card that overshoots WHILE GROWING escapes the cursor
     and drops its own hover -- hexmap's MarkStyle doc makes the same point
     about its highlight scale. */
  var PRESET = {
    /* gpumon's PILL, web/src/components/surface.tsx:380, verbatim. */
    open: { k: 520, c: 38, m: 0.70 },
    /* NOT lifted: gpumon leaves things with a curve, not a spring, so there is
       no constant of theirs to take. This is their asymmetry rule written as
       one -- "something leaving accelerates away and is shorter; doors swing
       shut faster than they swing open" (DESIGN.md 4). Overdamped on purpose:
       a retraction that overshoots reads as a bounce, and the thing it would
       bounce against is the cell it is going home to. */
    shut: { k: 900, c: 52, m: 0.60 }
  };

  /* `eps` is in the unit being animated: pixels want a coarser rest test than
     a 0..1 progress, and a spring that never rests keeps a rAF loop alive. */
  function Spring(value, cfg, eps) {
    this.x = value;
    this.target = value;
    this.v = 0;
    this.k = cfg.k; this.c = cfg.c; this.m = cfg.m;
    this.eps = eps == null ? 0.05 : eps;
    this.rest = true;
  }

  /** Teleport: no motion, no velocity. For a first layout or a resize. */
  Spring.prototype.set = function (value) {
    this.x = this.target = value;
    this.v = 0;
    this.rest = true;
    return this;
  };

  /**
   * Swap the constants without touching position or velocity.
   *
   * Opening and closing are the same spring at different stiffnesses, not two
   * springs handing off: a target reversed mid-flight has to carry the
   * velocity it already had, and two springs cannot without copying state
   * between them on exactly the right frame.
   */
  Spring.prototype.tune = function (cfg) {
    this.k = cfg.k; this.c = cfg.c; this.m = cfg.m;
    return this;
  };

  /** Redirect. Velocity is deliberately kept -- that is the point. */
  Spring.prototype.to = function (target) {
    if (target === this.target) return this;
    this.target = target;
    this.rest = false;
    return this;
  };

  /** Advance by dt seconds. Returns true once it has come to rest. */
  Spring.prototype.step = function (dt) {
    if (this.rest) return true;
    var left = Math.min(dt, DT_MAX);
    var n = Math.max(1, Math.ceil(left / SUB));
    var h = left / n;
    for (var i = 0; i < n; i++) {
      var a = (-this.k * (this.x - this.target) - this.c * this.v) / this.m;
      this.v += a * h;
      this.x += this.v * h;
    }
    if (Math.abs(this.x - this.target) < this.eps &&
        Math.abs(this.v) < this.eps * 12) {
      this.x = this.target;
      this.v = 0;
      this.rest = true;
    }
    return this.rest;
  };

  return {
    Spring: function (value, cfg, eps) { return new Spring(value, cfg, eps); },
    PRESET: PRESET,
    DT_MAX: DT_MAX
  };
})();
