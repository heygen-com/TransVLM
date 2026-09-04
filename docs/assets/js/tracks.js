/* TVTracks — the four-method timeline, on its own.
 *
 * Lifted out of site.js, where it was already the one piece the comparison
 * player and the failure cases shared. It moved here so a third caller -- the
 * deck -- gets the real strip, with its playhead and its scrubbing, instead of a
 * read-only copy that drifts from this one.
 *
 * strip() renders four rails against a duration and returns { paint(t) }, which
 * moves the playheads and returns the labels lit at t. The caller owns the
 * video: it supplies now() and receives onSeek(t).
 *
 * readout() prints that position and those labels. Its opts are what a caller
 * knows and this file does not: the clip's frame rate, so the position can be
 * given in frames instead of seconds, and which of the four labels is ours, so
 * it can be set apart from the methods it is being compared with.
 */
window.TVTracks = (function () {
  'use strict';

  var ORDER = ['gt', 'tn', 'as', 'ours'];

  var el = function (tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };

  function fmt(t) { return t.toFixed(2) + 's'; }

  function strip(host, opts) {
    var legend = opts.legend, dur = opts.duration, tracks = opts.tracks;
    var keys = opts.keys !== false;
    host.innerHTML = '';
    var heads = {}, labels = {};

    /* The part of the clip that actually plays, drawn around the rails.
     *
     * A caller may loop a window rather than the whole file -- the comparison
     * clips run to 454 s and the transition worth showing is 17 s of that -- and
     * without this the rails say the clip is seven minutes long while the
     * playhead crawls across 4% of them and turns round. The frame is what makes
     * that read as a choice instead of a stuck player.
     *
     * One frame around all four rails, not one per rail: four boxes with the
     * row gaps showing through read as a ladder. It is positioned against the
     * same two custom properties the row grid uses for its label column, so the
     * frame cannot drift from the rails it is framing. */
    if (opts.window) {
      var w0 = opts.window[0] / dur, w1 = opts.window[1] / dur;
      if (w0 > 0.001 || w1 < 0.999) {
        var win = el('div', 'win');
        win.style.setProperty('--w0', w0.toFixed(5));
        win.style.setProperty('--wd', (w1 - w0).toFixed(5));
        host.appendChild(win);
      }
    }

    ORDER.forEach(function (key) {
      var meta = legend[key];
      var row = el('div', 'trk');

      // Our own rail is named in the label's class, not only in its colour: the
      // strip is a comparison, and the row the reader should find first is the
      // same row in all three places the strip is drawn.
      var lbl = el('span', 'lbl' + (key === 'ours' ? ' ours' : ''));
      var sw = el('span', 'swatch');
      sw.style.background = meta.color;
      lbl.appendChild(sw);
      lbl.appendChild(document.createTextNode(meta.label));
      labels[key] = lbl;

      var rail = el('div', 'rail-t');
      // Arrow keys step the playhead only where nothing else has claimed them.
      // In the deck they switch cards, and a rail that also swallowed them would
      // leave the reader unable to tell which thing is about to move.
      if (keys) {
        rail.setAttribute('role', 'slider');
        rail.setAttribute('tabindex', '0');
        rail.setAttribute('aria-label', 'Seek. ' + meta.label + ' track.');
        rail.setAttribute('aria-valuemin', '0');
        rail.setAttribute('aria-valuemax', dur.toFixed(2));
      }

      (tracks[key] || []).forEach(function (seg) {
        var s = el('span', 'seg');
        s.style.left = (seg[0] / dur * 100) + '%';
        // A 0.04 s cut is 1/2500 of a 100 s clip: without a floor it renders
        // sub-pixel and the SBD tracks look empty.
        s.style.width = 'max(2px,' + ((seg[1] - seg[0]) / dur * 100) + '%)';
        s.style.background = meta.color;
        rail.appendChild(s);
      });

      var head = el('span', 'head');
      rail.appendChild(head);
      heads[key] = head;

      function seek(ev) {
        var box = rail.getBoundingClientRect();
        var x = (ev.clientX - box.left) / box.width;
        opts.onSeek(Math.max(0, Math.min(1, x)) * dur);
      }
      rail.addEventListener('pointerdown', function (ev) {
        rail.setPointerCapture(ev.pointerId);
        seek(ev);
        rail.addEventListener('pointermove', seek);
      });
      rail.addEventListener('pointerup', function (ev) {
        rail.releasePointerCapture(ev.pointerId);
        rail.removeEventListener('pointermove', seek);
      });
      if (keys) {
        rail.addEventListener('keydown', function (ev) {
          var step = ev.shiftKey ? 1 : 0.2;
          if (ev.key === 'ArrowRight') { opts.onSeek(Math.min(dur, opts.now() + step)); ev.preventDefault(); }
          if (ev.key === 'ArrowLeft') { opts.onSeek(Math.max(0, opts.now() - step)); ev.preventDefault(); }
        });
      }

      row.appendChild(lbl);
      row.appendChild(rail);
      host.appendChild(row);
    });

    function activeAt(key, t) {
      return (tracks[key] || []).some(function (s) { return t >= s[0] && t < s[1]; });
    }

    return {
      paint: function (t) {
        var pct = (Math.max(0, Math.min(dur, t)) / dur * 100) + '%';
        var on = [];
        ORDER.forEach(function (k) {
          heads[k].style.left = pct;
          var hit = activeAt(k, t);
          labels[k].classList.toggle('on', hit);
          if (hit) on.push(legend[k].label);
        });
        return on;
      }
    };
  }

  function clickToToggle(video) {
    video.addEventListener('click', function () {
      if (video.paused) video.play().catch(function () {});
      else video.pause();
    });
  }

  /* opts = { fps, ours }, both optional.
   *
   * With fps the position is given in frames -- `frame 29 / 57` -- and in
   * frames only: a cut is two frames, so on a clip this short the frame is the
   * unit the tracks are drawn in, and a seconds figure beside it was a second
   * reading of the same position. Without fps the line is seconds, as before.
   *
   * With ours, the lit label matching it is a <b> rather than a <span>, so the
   * one method the page is arguing for is not one name in a list of four. */
  function readout(node, t, dur, on, opts) {
    opts = opts || {};
    node.innerHTML = '';
    if (opts.fps) {
      // 1-based, because it names the frame on screen rather than the number of
      // whole frames behind it. The epsilon covers currentTime landing a float
      // hair under a frame boundary, which would otherwise read a frame early;
      // the clamp covers the last frame, where t == dur puts the count one past
      // the end.
      var total = Math.round(dur * opts.fps);
      var n = Math.min(total, Math.floor(t * opts.fps + 1e-6) + 1);
      node.appendChild(el('span', 't', 'frame ' + n + ' / ' + total));
    } else {
      node.appendChild(el('span', 't', fmt(t) + ' / ' + dur.toFixed(1) + 's'));
    }
    // Nothing is said when nothing is firing. The line this used to print --
    // "all four methods report no transition" -- counted ground truth as a
    // method, and an empty stretch of four empty rails needs no caption anyway.
    //
    // One element per name, rather than one joined string, so opts.ours can be
    // emphasised where it appears.
    if (on.length) {
      var hit = el('span', 'hit');
      hit.appendChild(document.createTextNode('transition: '));
      on.forEach(function (name, i) {
        if (i) hit.appendChild(document.createTextNode(', '));
        var mine = name === opts.ours;
        hit.appendChild(el(mine ? 'b' : 'span', mine ? 'name ours' : 'name', name));
      });
      node.appendChild(hit);
    }
  }

  return {
    ORDER: ORDER,
    strip: strip,
    readout: readout,
    clickToToggle: clickToToggle,
    fmt: fmt
  };
})();
