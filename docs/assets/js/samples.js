/* lab/samples.html — the extraction review page.
 *
 * One section per source video: the original whole, the verdict on every ground
 * truth transition in it, and the clips the extractor cut. Everything on the page
 * comes out of assets/data/samples.json — the thresholds in the header, the four
 * method names and colours, the counts, the tracks. Nothing is typed twice.
 *
 * It reuses TVTracks unchanged. The one thing this page needs that the strip does
 * not offer -- N window frames on one set of rails instead of one -- is done by
 * appending .win elements to the host tracks.js already built, positioned off the
 * same two custom properties it writes for its own single window. See frames().
 */
(function () {
  'use strict';

  var DATA = '../assets/data/samples.json';

  /* The score at which a detector is treated as having fired, for the table's
     tint only. Deliberately not read out of `rules`: ours_min is the acceptance
     floor for TransVLM specifically, and reusing it here would say the table is
     showing the rule when it is showing the three raw confidences. */
  var HOT = 0.5;

  /* Which order the three score columns go in, and where their names come from.
     Ground truth has no score -- it is the thing being scored against. */
  var SCORES = ['ours', 'tn', 'as'];

  /* Names and units for the thresholds. The chips are built by walking the JSON's
     own keys, so a rule this page has never heard of still renders, under its raw
     name -- the page can then be wrong about the wording but never about which
     thresholds the generator actually applied. */
  var RULES = {
    tol: ['match tolerance', '±', 's'],
    cut_max_len: ['a cut is', '≤', 's'],
    ours_cut_max_span: ['TransVLM span on a cut', '≤', 's'],
    ours_min: ['TransVLM score', '≥', ''],
    base_max: ['both baselines', '≤', ''],
    pad_min: ['window pad, least', '', 's'],
    pad_max: ['window pad, most', '', 's']
  };

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  function $(sel) { return document.querySelector(sel); }
  function noop() {}

  /* A missing required field throws, naming the field and the object it was
     missing from. Nothing here substitutes a zero, an empty array or an em dash
     for data the generator failed to write: a page that renders a video with no
     duration as a 0.00s video is a page that lies quietly. Genuinely nullable
     fields -- `effect`, `sample`, `group` -- are read directly instead. */
  function need(obj, key, where) {
    var v = obj == null ? undefined : obj[key];
    if (v === undefined || v === null) {
      throw new Error('samples.json: ' + where + ' has no "' + key + '"');
    }
    return v;
  }

  // ------------------------------------------------------------- formatting

  function s2(x) { return x.toFixed(2); }
  function secs(x) { return x.toFixed(2) + 's'; }
  function span(a, b) { return s2(a) + '–' + s2(b) + 's'; }
  function fps(f) {
    var whole = Math.abs(f - Math.round(f)) < 0.005;
    return (whole ? String(Math.round(f)) : f.toFixed(2)) + ' fps';
  }
  function pad2(n) { return (n < 10 ? '0' : '') + n; }

  /* The legend colours are hex in every file that ships, and the tint is only
     ever decoration. So a colour this cannot parse loses its tint rather than
     taking a guessed one -- the number in the cell is the data either way. */
  function tint(hex, alpha) {
    var m = /^#([0-9a-f]{6})$/i.exec(String(hex).trim());
    if (!m) return null;
    var n = parseInt(m[1], 16);
    return 'rgba(' + [(n >> 16) & 255, (n >> 8) & 255, n & 255, alpha].join(',') + ')';
  }

  // ------------------------------------------------------ playback economy

  /* Every video on the page goes through one observer. Entering the margin sets
     its src for the first time and, unless it is an original, plays it; leaving
     pauses it. The originals are click-driven -- an original can be 50 MB and
     seven minutes long -- but they still take their src here, so a section nobody
     scrolled to costs nothing and one that is on screen is ready to play the
     moment it is clicked. */
  var live = [];

  function wake(v) {
    v.dataset.want = '1';
    if (!v.src) {
      /* play() issued before the first frame is decodable is aborted by the
         load() still in flight, so it is armed on loadeddata rather than fired
         and hoped for. Same reasoning as site.js's own wake(). */
      v.addEventListener('loadeddata', function () {
        if (v.dataset.want === '1' && v.dataset.auto === '1') v.play().catch(noop);
      });
      v.src = v.dataset.src;
      v.load();
      return;
    }
    // Swallowed deliberately: a rejected play() here means the browser declined
    // an autoplay it is entitled to decline -- a background tab, or a reduced
    // data setting -- or the element was torn down mid-call. Neither is
    // actionable and neither should reach the console.
    if (v.dataset.auto === '1' && v.readyState >= 2) v.play().catch(noop);
  }
  function sleep(v) {
    v.dataset.want = '0';
    if (!v.paused) v.pause();
  }

  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      var v = e.target;
      var at = live.indexOf(v);
      if (e.isIntersecting) {
        if (at < 0) live.push(v);
        wake(v);
      } else {
        if (at >= 0) live.splice(at, 1);
        sleep(v);
      }
    });
  }, { rootMargin: '200px 0px' });

  /* A page opened in a background tab has every autoplay refused, and nothing
     else here retries: the observer has already fired and will not fire again
     until something scrolls. Without this the grid is simply still when the
     reader switches to the tab. */
  document.addEventListener('visibilitychange', function () {
    if (!document.hidden) live.forEach(wake);
  });

  // --------------------------------------------------------------- the strip

  /* timeupdate fires about four times a second, which is a playhead that visibly
     steps between the rails' segments. So the smooth painting is a rAF loop, and
     it runs only while this element is actually playing -- one loop per playing
     video, none for the dozens that are paused or off screen. timeupdate and
     seeked stay bound because they are what repaints a *stopped* element: after
     a scrub, after a seek, at the end of a pass. */
  function drive(video, strip, readHost, dur) {
    var raf = 0;
    function paint() {
      var t = video.currentTime || 0;
      TVTracks.readout(readHost, t, dur, strip.paint(t));
    }
    function loop() { paint(); raf = requestAnimationFrame(loop); }
    function start() { if (!raf) raf = requestAnimationFrame(loop); }
    function stop() {
      if (raf) { cancelAnimationFrame(raf); raf = 0; }
      paint();
    }
    video.addEventListener('timeupdate', paint);
    video.addEventListener('seeked', paint);
    video.addEventListener('playing', start);
    video.addEventListener('pause', stop);
    video.addEventListener('ended', stop);
    paint();
    return paint;
  }

  function mountStrip(video, tracks, dur, legend, keys) {
    var trackHost = el('div', 'tracks');
    var readHost = el('div', 'readout');
    // onSeek is handed to strip() before drive() exists, so it reaches the
    // painter through a variable rather than by closing over it directly.
    var repaint = noop;
    var strip = TVTracks.strip(trackHost, {
      tracks: tracks, duration: dur, legend: legend, keys: keys,
      now: function () { return video.currentTime || 0; },
      onSeek: function (t) { video.currentTime = t; repaint(); }
    });
    repaint = drive(video, strip, readHost, dur);
    return { tracks: trackHost, readout: readHost };
  }

  /* Every sample's window, framed on the original's rails.
   *
   * tracks.js draws one frame, for the stretch of a clip that actually plays,
   * and positions it off --w0 / --wd against the row grid's own label column so
   * the frame cannot drift from the rails. This wants N of them, named, on rails
   * that play whole. Rather than growing the component an option with one
   * caller, the same two properties are computed the same way here and the
   * frames are appended to the host it just built: .win's rule in site.css does
   * the rest, and tracks.js is untouched.
   *
   * The tag hangs above the frame rather than inside it. Inside, it sits on the
   * ground-truth rail -- the one rail the frame exists to be read against. */
  function frames(trackHost, samples, dur) {
    if (!samples.length) return;
    trackHost.classList.add('has-wins');
    samples.forEach(function (s, i) {
      var w0 = s.window[0] / dur, w1 = s.window[1] / dur;
      var win = el('div', 'win');
      win.style.setProperty('--w0', w0.toFixed(5));
      win.style.setProperty('--wd', (w1 - w0).toFixed(5));
      win.appendChild(el('span', 'wtag', tagOf(s, i)));
      trackHost.appendChild(win);
    });
  }

  /* The number the tag shows is the number in the sample's own id, so the frame
     on the rails and the card it points at cannot disagree -- an extractor that
     drops s02 must not leave the third sample labelled s03 in the table and s02
     on the rails. Position is the fallback, for an id shaped some other way. */
  function tagOf(s, i) {
    var m = /-(s\d+)$/.exec(String(s.id));
    return m ? m[1] : 's' + pad2(i + 1);
  }

  // ---------------------------------------------------------------- header

  function rulesLine(host, rules) {
    Object.keys(rules).forEach(function (k) {
      var meta = RULES[k] || [k, '', ''];
      var chip = el('span', 'sm-chip');
      chip.appendChild(el('b', null, meta[0]));
      chip.appendChild(document.createTextNode(
        (meta[1] ? meta[1] + ' ' : '') + rules[k] + (meta[2] ? ' ' + meta[2] : '')));
      host.appendChild(chip);
    });
  }

  function indexLine(host, videos) {
    var ul = el('ul');
    videos.forEach(function (v) {
      var li = el('li');
      var a = el('a');
      a.href = '#v-' + v.id;
      a.appendChild(el('i', null, v.id));
      a.appendChild(document.createTextNode(
        ' · ' + v.dataset + ' · ' + v.samples.length +
        (v.samples.length === 1 ? ' sample' : ' samples')));
      li.appendChild(a);
      ul.appendChild(li);
    });
    host.appendChild(ul);
  }

  function sub(text, note) {
    var h = el('h3', 'sm-sub');
    h.appendChild(document.createTextNode(text));
    if (note) h.appendChild(el('em', null, note));
    return h;
  }

  // --------------------------------------------------------------- verdicts

  function scoreCell(val, meta) {
    var td = el('td', 'sc');
    if (val === undefined || val === null) { td.textContent = '—'; return td; }
    td.textContent = s2(val);
    if (val >= HOT) {
      td.classList.add('hot');
      var bg = tint(meta.color, 0.18);
      if (bg) td.style.background = bg;
    }
    return td;
  }

  function verdicts(v, legend) {
    var wrap = el('div', 'sm-scroll');
    var table = el('table', 'tbl vtbl');
    var head = el('tr');
    ['#', 'span', 'length', 'kind', 'effect'].forEach(function (h) {
      head.appendChild(el('th', null, h));
    });
    SCORES.forEach(function (k) {
      head.appendChild(el('th', null, need(legend, k, 'legend').label));
    });
    ['verdict', 'why', 'sample'].forEach(function (h) {
      head.appendChild(el('th', null, h));
    });
    var thead = el('thead'); thead.appendChild(head); table.appendChild(thead);

    var body = el('tbody');
    v.transitions.forEach(function (t) {
      var where = v.id + ' transition ' + t.i;
      var a = need(t, 'span', where)[0], b = t.span[1];
      var verdict = need(t, 'verdict', where);
      var tr = el('tr', verdict === 'accepted' ? null : 'rej');
      tr.appendChild(el('td', 'n', '#' + need(t, 'i', where)));
      tr.appendChild(el('td', 'n', span(a, b)));
      tr.appendChild(el('td', 'n', secs(b - a)));
      tr.appendChild(el('td', 'kind', need(t, 'kind', where)));
      tr.appendChild(el('td', 'eff', t.effect || '—'));
      SCORES.forEach(function (k) {
        tr.appendChild(scoreCell(need(t, 'score', where)[k], legend[k]));
      });
      tr.appendChild(el('td', 'vd', verdict));
      tr.appendChild(el('td', 'why', need(t, 'why', where)));

      var last = el('td', 'smp');
      if (t.sample) {
        var link = el('a', null, t.sample);
        link.href = '#card-' + t.sample;
        last.appendChild(link);
      } else {
        last.textContent = '—';
      }
      tr.appendChild(last);
      body.appendChild(tr);
    });
    table.appendChild(body);
    wrap.appendChild(table);
    return wrap;
  }

  // ------------------------------------------------------------ the media

  /* One media box for both the originals and the samples, and one cap on both.
     --nat-w is the encoded width out of the JSON, which is what lets the CSS
     refuse to upscale any clip past 2x without this code having to decide what
     counts as small. */
  function media(url, poster, out, where, tap) {
    var box = el('div', 'sm-media' + (tap ? ' tap' : ''));
    var w = need(out, 'width', where), h = need(out, 'height', where);
    box.style.setProperty('--nat-w', w);
    // ClipShots distributes some videos at 128x128. Pixelated is honest there
    // and wrong everywhere else; see .sm-media.low in samples.css.
    if (w <= 256 || h <= 256) box.classList.add('low');
    var v = document.createElement('video');
    v.muted = true;
    v.playsInline = true;
    v.preload = 'none';
    // The intrinsic ratio, before a byte of video has arrived: without it the
    // box is zero-height until metadata lands and the whole page reflows.
    v.width = w;
    v.height = h;
    if (poster) v.poster = poster;
    v.dataset.src = url;
    box.appendChild(v);
    return { box: box, video: v };
  }

  // ------------------------------------------------------------- the cards

  function card(v, s, byI, data, i) {
    var where = v.id + ' sample ' + s.id;
    var fig = el('figure', 's-card');
    fig.id = 'card-' + need(s, 'id', where);

    var m = media(mediaUrl(data, need(s, 'src', where)),
                  s.poster ? mediaUrl(data, s.poster) : null,
                  need(s, 'out', where), where, false);
    // A sample is a few seconds long and exists to be watched over and over, so
    // it loops natively -- no window to enforce from outside, the file *is* the
    // window. The observer decides only whether it is playing at all.
    m.video.loop = true;
    m.video.dataset.auto = '1';
    m.video.setAttribute('aria-label',
      'Extracted sample ' + s.id + ', ' + v.dataset + ', ' + v.domain);
    fig.appendChild(m.box);

    var strip = mountStrip(m.video, need(s, 'tracks', where),
                           need(s, 'duration', where), data.legend, false);
    fig.appendChild(strip.tracks);
    fig.appendChild(strip.readout);

    var foot = el('figcaption', 's-foot');
    var win = need(s, 'window', where);
    foot.appendChild(el('span', 'id', tagOf(s, i) + ' · ' + s.id));
    foot.appendChild(el('div', null,
      'window ' + span(win[0], win[1]) + ' of the original · ' + secs(s.duration)));

    var dl = el('div', 'trs');
    need(s, 'transitions', where).forEach(function (k) {
      var t = byI[k];
      if (!t) throw new Error('samples.json: ' + where + ' covers transition #' + k +
                              ', which is not in this video\'s transitions');
      var line = el('div', 'tr');
      var bits = ['#' + t.i, t.kind];
      if (t.effect) bits.push(t.effect);
      line.appendChild(document.createTextNode(bits.join(' · ') + ' · '));
      SCORES.forEach(function (mk, n) {
        var sc = el('span', 'sc',
          (n ? ' · ' : '') + data.legend[mk].label + ' ' + s2(t.score[mk]));
        line.appendChild(sc);
      });
      dl.appendChild(line);
    });
    foot.appendChild(dl);

    var others = need(s, 'others', where);
    if (others.length) {
      foot.appendChild(el('span', 's-warn', 'contains rejected ' +
        others.map(function (k) { return '#' + k; }).join(', ')));
    }
    fig.appendChild(foot);
    io.observe(m.video);
    return fig;
  }

  // ------------------------------------------------------------- a section

  function mediaUrl(data, rel) { return '../' + data.media + '/' + rel; }

  function section(v, n, data) {
    var where = 'video ' + (v.id || '#' + n);
    var sec = el('section');
    sec.id = 'v-' + need(v, 'id', where);

    var head = el('div', 'wrap sm-w');
    var eye = el('p', 'eyebrow');
    eye.appendChild(el('b', null, pad2(n + 1)));
    eye.appendChild(el('span', null, need(v, 'dataset', where)));
    if (v.group) eye.appendChild(el('span', null, v.group));
    head.appendChild(eye);
    head.appendChild(el('h2', null, need(v, 'domain', where)));

    var meta = el('p', 'sm-meta');
    meta.appendChild(el('span', 'id', v.id));
    meta.appendChild(document.createTextNode(' · ' +
      need(v, 'width', where) + '×' + need(v, 'height', where) + ' · ' +
      fps(need(v, 'fps', where)) + ' · ' + secs(need(v, 'duration', where))));
    head.appendChild(meta);

    var counts = el('div', 'sm-counts');
    TVTracks.ORDER.forEach(function (k) {
      var sp = el('span');
      var sw = el('span', 'swatch');
      sw.style.background = need(data.legend, k, 'legend').color;
      sp.appendChild(sw);
      sp.appendChild(el('b', null, need(v, 'counts', where)[k]));
      sp.appendChild(document.createTextNode(data.legend[k].label));
      counts.appendChild(sp);
    });
    head.appendChild(counts);
    sec.appendChild(head);

    var samples = need(v, 'samples', where);
    var byI = {};
    need(v, 'transitions', where).forEach(function (t) { byI[t.i] = t; });

    // ---- the original, whole
    var orig = el('div', 'wrap sm-w');
    orig.appendChild(sub('Original', samples.length
      ? 'the whole file, with every extracted window framed on the rails'
      : 'the whole file'));
    var om = media(mediaUrl(data, need(v, 'src', where)), null,
                   need(v, 'out', where), where, true);
    // Click, not autoplay: these are the full source videos, and one of them is
    // ten minutes of television. Native loop is right here because nothing
    // outside is enforcing a window on it -- the whole file is the window.
    om.video.loop = true;
    om.video.dataset.auto = '0';
    om.video.setAttribute('aria-label', 'Original video, ' + v.dataset + ', ' + v.domain);
    TVTracks.clickToToggle(om.video);
    orig.appendChild(om.box);
    orig.appendChild(el('p', 'sm-hint',
      'Click the picture to play or pause. Drag a rail to scrub.'));
    var ostrip = mountStrip(om.video, v.tracks, v.duration, data.legend, true);
    frames(ostrip.tracks, samples, v.duration);
    orig.appendChild(ostrip.tracks);
    orig.appendChild(ostrip.readout);
    io.observe(om.video);
    sec.appendChild(orig);

    // ---- the verdict on every ground truth transition
    var vd = el('div', 'wrap sm-w');
    vd.appendChild(sub('Verdicts', 'one row per ground-truth transition'));
    vd.appendChild(verdicts(v, data.legend));
    sec.appendChild(vd);

    // ---- the clips that came out
    var ex = el('div', 'wrap sm-w');
    ex.appendChild(sub('Extractions', samples.length + (samples.length === 1 ? ' clip' : ' clips')));
    if (!samples.length) {
      // No empty grid: a grid with nothing in it reads as a page that failed to
      // load rather than as a video the rule passed over.
      ex.appendChild(el('p', 'sm-none',
        'Nothing was extracted from this video — no transition in it met the rule.'));
    } else {
      var grid = el('div', 'sm-grid');
      samples.forEach(function (s, i) { grid.appendChild(card(v, s, byI, data, i)); });
      ex.appendChild(grid);
    }
    sec.appendChild(ex);
    return sec;
  }

  // ------------------------------------------------------------------ build

  function build(data) {
    need(data, 'legend', 'the file');
    need(data, 'media', 'the file');
    need(data, 'rules', 'the file');
    var videos = need(data, 'videos', 'the file');

    rulesLine($('#rules'), data.rules);
    indexLine($('#index'), videos);

    var host = $('#videos');
    var nSamples = 0, nTrans = 0;
    videos.forEach(function (v, i) {
      host.appendChild(section(v, i, data));
      nSamples += v.samples.length;
      nTrans += v.transitions.length;
    });

    $('#status').textContent = videos.length + ' videos · ' + nTrans +
      ' ground-truth transitions · ' + nSamples + ' extracted clips';
  }

  fetch(DATA).then(function (r) {
    // Fail loudly and by name. A 404 that fell through to an empty render would
    // look exactly like a run in which the extractor found nothing.
    if (!r.ok) throw new Error(DATA + ': HTTP ' + r.status);
    return r.json();
  }).then(build).catch(function (err) {
    var st = $('#status');
    st.className = 'sm-status bad';
    st.textContent = String(err.message || err);
    throw err;
  });
})();
