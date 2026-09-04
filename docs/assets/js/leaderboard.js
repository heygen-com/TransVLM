/* STD leaderboard: nine views over one dataset of results.
 *
 * Everything comes from assets/data/leaderboard.json, built by
 * tools/build_data.py out of the paper's own tables and five reviewer-response
 * files. Nothing is typed into this file, which is why the dimension list, the
 * columns, the subset switches and the footnotes are all data-driven.
 *
 * Three shapes of dimension:
 *   table   -- rows in `rows`, or in `rowsBySubset` when a subset switch applies
 *   scatter -- a table plus a Pareto plot of cost against quality
 *   widget  -- mounts the 168 pre-generated per-tau tables instead of its own
 */
(function () {
  'use strict';

  var $ = function (s, r) { return (r || document).querySelector(s); };
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  var DATA = null;
  var state = {
    dim: 'overall',
    subset: null,
    families: null,     // null = all
    query: '',
    sortBy: null,
    sortDir: null,
    expanded: false,
    pareto: false
  };
  var TOP_N = 10;

  /* ------------------------------------------------------------- helpers */

  function dimension(id) {
    return DATA.dimensions.filter(function (d) { return d.id === id; })[0];
  }

  function rowsFor(dim) {
    if (dim.rowsBySubset) {
      var key = state.subset && dim.rowsBySubset[state.subset]
        ? state.subset : dim.subsets[0];
      return dim.rowsBySubset[key] || [];
    }
    if (dim.rowsRef) return DATA.rowSets[dim.rowsRef] || [];
    return dim.rows || [];
  }

  function allFamilies() {
    var seen = [];
    DATA.rowSets.tab3.forEach(function (r) {
      if (seen.indexOf(r.family) < 0) seen.push(r.family);
    });
    return seen;
  }

  function filtered(dim) {
    var rows = rowsFor(dim).slice();
    if (state.families) {
      rows = rows.filter(function (r) { return state.families[r.family]; });
    }
    if (state.query) {
      var q = state.query.toLowerCase();
      rows = rows.filter(function (r) { return r.method.toLowerCase().indexOf(q) >= 0; });
    }
    var sortBy = state.sortBy || primaryColumn(dim);
    var dir = state.sortDir || columnDir(dim, sortBy);
    if (sortBy) rows.sort(window.TVTable.comparator(sortBy, dir));
    return rows;
  }

  function primaryColumn(dim) {
    var cols = dim.columns || [];
    for (var i = 0; i < cols.length; i++) if (cols[i].primary) return cols[i].id;
    return cols.length ? cols[0].id : null;
  }

  function columnDir(dim, id) {
    var cols = dim.columns || [];
    for (var i = 0; i < cols.length; i++) if (cols[i].id === id) return cols[i].dir || 'desc';
    return 'desc';
  }

  /* --------------------------------------------------------------- rail */

  function buildRail() {
    var host = $('#rail-body');
    host.innerHTML = '';

    host.appendChild(el('div', 'rgroup-h', 'View as'));
    var view = el('div', 'seg');
    [['Ranking', false], ['Pareto', true]].forEach(function (v) {
      var b = el('button', null, v[0]);
      b.type = 'button';
      b.setAttribute('aria-pressed', String(state.pareto === v[1]));
      b.addEventListener('click', function () {
        state.pareto = v[1];
        if (v[1] && dimension(state.dim).kind !== 'scatter') pick('efficiency');
        else render();
      });
      view.appendChild(b);
    });
    host.appendChild(view);

    host.appendChild(el('div', 'rgroup-h', 'Dimensions'));
    var list = el('div', 'rlist');
    DATA.dimensions.forEach(function (d) {
      var a = el('button', null);
      a.type = 'button';
      a.appendChild(el('span', 'tick' + (d.id === state.dim ? ' on' : ''),
                       d.id === state.dim ? '✓' : ''));
      a.appendChild(document.createTextNode(d.label));
      if (d.id === state.dim) a.setAttribute('aria-current', 'true');
      a.addEventListener('click', function () { pick(d.id); });
      list.appendChild(a);
    });
    host.appendChild(list);

    var dim = dimension(state.dim);
    if (dim.subsets && dim.subsets.length > 1) {
      host.appendChild(el('div', 'rgroup-h', 'Subset'));
      var subs = el('div', 'rlist');
      dim.subsets.forEach(function (key) {
        var b = el('button', null);
        b.type = 'button';
        var sel = (state.subset || dim.subsets[0]) === key;
        b.appendChild(el('span', 'tick' + (sel ? ' on' : ''), sel ? '✓' : '·'));
        b.appendChild(document.createTextNode(DATA.subsetLabels[key] || key));
        b.addEventListener('click', function () { state.subset = key; render(); });
        subs.appendChild(b);
      });
      host.appendChild(subs);
    }

    if (dim.kind !== 'widget') {
      host.appendChild(el('div', 'rgroup-h', 'Family'));
      var fams = el('div', 'rlist');
      allFamilies().forEach(function (f) {
        var on = !state.families || state.families[f];
        var b = el('button', null);
        b.type = 'button';
        b.appendChild(el('span', 'tick' + (on ? ' on' : ''), on ? '✓' : '·'));
        b.appendChild(document.createTextNode(f));
        b.addEventListener('click', function () {
          if (!state.families) {
            state.families = {};
            allFamilies().forEach(function (x) { state.families[x] = true; });
          }
          state.families[f] = !state.families[f];
          var any = allFamilies().some(function (x) { return state.families[x]; });
          if (!any) state.families = null;   // never show an empty table
          render();
        });
        fams.appendChild(b);
      });
      host.appendChild(fams);

      host.appendChild(el('div', 'rgroup-h', 'Search'));
      var inp = el('input', 'search');
      inp.type = 'search';
      inp.placeholder = 'method name';
      inp.value = state.query;
      inp.addEventListener('input', function () {
        state.query = inp.value.trim();
        renderMain();
      });
      host.appendChild(inp);
    }
  }

  function pick(id) {
    state.dim = id;
    state.subset = null;
    state.sortBy = null;
    state.sortDir = null;
    state.expanded = false;
    var dim = dimension(id);
    if (dim.kind !== 'scatter') state.pareto = false;
    if (location.hash.slice(1) !== id) history.replaceState(null, '', '#' + id);
    render();
  }

  /* --------------------------------------------------------------- main */

  function renderHead(dim) {
    var host = $('#lb-head');
    host.innerHTML = '';
    var h = el('h1');
    h.appendChild(document.createTextNode('STD Benchmark'));
    h.appendChild(el('span', 'badge', dim.label));
    host.appendChild(h);

    var m = DATA.meta;
    var meta = el('p', 'lb-meta');
    [m.videos.toLocaleString('en-US') + ' videos',
     m.hours.toFixed(2) + ' h',
     m.transitions.toLocaleString('en-US') + ' transitions',
     m.methods + ' methods',
     m.families + ' families'].forEach(function (t, i) {
      if (i) meta.appendChild(el('span', 'dot', '·'));
      meta.appendChild(el('span', null, t));
    });
    host.appendChild(meta);

    if (dim.blurb) {
      var p = el('p', 'lb-blurb');
      p.innerHTML = dim.blurb;
      host.appendChild(p);
    }
  }

  function renderMain() {
    var dim = dimension(state.dim);
    var host = $('#lb-body');
    host.innerHTML = '';

    if (dim.kind === 'widget') {
      host.appendChild(buildWidgetPanel());
      window.TVWidget.mount($('.qe', host), dim.widget);
      return;
    }

    var rows = filtered(dim);
    var shown = state.expanded ? rows : rows.slice(0, TOP_N);
    var sortBy = state.sortBy || primaryColumn(dim);
    var sortDir = state.sortDir || columnDir(dim, sortBy);

    if (state.pareto && dim.scatter) {
      host.appendChild(scatter(rows, dim));
    }

    var box = el('div', 'tblbox');
    window.TVTable.render(box, {
      columns: dim.columns,
      rows: shown,
      sortable: true,
      bars: true,
      showFamily: true,
      sortBy: sortBy,
      sortDir: sortDir,
      firstLabel: dim.id === 'ablation' ? 'Ablation' : 'Method',
      onSort: function (c) {
        if (state.sortBy === c.id) {
          state.sortDir = state.sortDir === 'asc' ? 'desc' : 'asc';
        } else {
          state.sortBy = c.id;
          state.sortDir = c.dir || 'desc';
        }
        renderMain();
      }
    });

    // rank column, after sorting, so it reflects the view rather than the source
    var body = box.querySelector('tbody');
    var trs = [].slice.call(body.querySelectorAll('tr'));
    box.querySelector('thead tr').insertBefore(el('th', 'rankh', '#'),
      box.querySelector('thead tr').firstChild);
    trs.forEach(function (tr, i) {
      tr.insertBefore(el('td', 'rank', String(i + 1)), tr.firstChild);
    });
    host.appendChild(box);

    if (rows.length > TOP_N) {
      var more = el('div', 'morewrap');
      var b = el('button', 'btn', state.expanded
        ? 'Show top ' + TOP_N
        : 'View all ' + rows.length);
      b.type = 'button';
      b.addEventListener('click', function () {
        state.expanded = !state.expanded;
        renderMain();
      });
      more.appendChild(b);
      host.appendChild(more);
    }
    if (!rows.length) {
      host.appendChild(el('p', 'note', 'No method matches this filter.'));
    }

  }

  function buildWidgetPanel() {
    var wrap = el('div', 'qe');
    [['Per-τ table', 't-ds', 't-tt', 'Dataset', 'τ', 'iframe'],
     ['Metric vs τ', 'p-ds', 'p-mt', 'Dataset', 'Metric', 'img']].forEach(function (spec) {
      var panel = el('div', 'qe-panel');
      panel.appendChild(el('h3', null, spec[0]));
      var ctl = el('div', 'qe-ctl');
      [[spec[3], spec[1]], [spec[4], spec[2]]].forEach(function (pair) {
        var lab = el('label', null, pair[0] + ' ');
        var sel = el('select');
        sel.setAttribute('data-role', pair[1]);
        lab.appendChild(sel);
        ctl.appendChild(lab);
      });
      panel.appendChild(ctl);
      var box = el('div', 'qe-box ' + (spec[5] === 'iframe' ? 't' : 'p'));
      if (spec[5] === 'iframe') {
        var f = el('iframe');
        f.setAttribute('data-role', 't-frame');
        f.title = 'Per-tau comparison table';
        box.appendChild(f);
      } else {
        var i = el('img');
        i.setAttribute('data-role', 'p-img');
        i.alt = 'Metric plotted against temporal tolerance';
        box.appendChild(i);
      }
      panel.appendChild(box);
      wrap.appendChild(panel);
    });
    return wrap;
  }

  /* ------------------------------------------------------------ scatter */

  var SVGNS = 'http://www.w3.org/2000/svg';
  function svg(tag, attrs) {
    var n = document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    return n;
  }

  var X_TICKS = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4];

  function scatter(rows, dim) {
    var spec = dim.scatter;
    var W = 760, H = 420, L = 62, R = 96, T = 20, B = 46;
    var pts = rows.map(function (r) {
      var x = (r.cells[spec.x] || {}).v;
      var y = (r.cells[spec.y] || {}).v;
      var s = (r.cells[spec.size] || {});
      return { r: r, x: x, y: y, size: s.norm != null ? s.norm : null };
    }).filter(function (p) { return p.x != null && p.y != null && p.x > 0; });
    if (!pts.length) return el('div');

    // Cost spans 0.02 to 3.3 -- two orders of magnitude. On a linear axis two
    // thirds of the methods pile into the leftmost 5% of the plot and the
    // trade-off the view exists to show is unreadable.
    var lo = Math.log10(Math.min.apply(null, pts.map(function (p) { return p.x; })) * 0.8);
    var hi = Math.log10(Math.max.apply(null, pts.map(function (p) { return p.x; })) * 1.25);
    var sx = function (v) { return L + (Math.log10(v) - lo) / (hi - lo) * (W - L - R); };
    var sy = function (v) { return T + (1 - v) * (H - T - B); };
    var smax = Math.max.apply(null, pts.map(function (p) { return p.size || 0; })) || 1;
    var sr = function (v) { return v == null ? 3.5 : 5 + 15 * Math.sqrt(v / smax); };

    var root = svg('svg', {
      viewBox: '0 0 ' + W + ' ' + H, class: 'scat', role: 'img',
      'aria-label': 'Public Seg-F1 against end-to-end real-time factor on a '
        + 'logarithmic axis; bubble area is parameter count'
    });

    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
      root.appendChild(svg('line', { x1: L, x2: W - R, y1: sy(v), y2: sy(v), class: 'grid' }));
      var t = svg('text', { x: L - 8, y: sy(v) + 4, class: 'axl', 'text-anchor': 'end' });
      t.textContent = v.toFixed(2);
      root.appendChild(t);
    });
    X_TICKS.forEach(function (v) {
      if (Math.log10(v) < lo || Math.log10(v) > hi) return;
      root.appendChild(svg('line', {
        x1: sx(v), x2: sx(v), y1: T, y2: H - B, class: 'grid faint'
      }));
      var t = svg('text', { x: sx(v), y: H - B + 18, class: 'axl', 'text-anchor': 'middle' });
      t.textContent = v < 1 ? String(v) : v.toFixed(0) + '×';
      root.appendChild(t);
    });
    var xl = svg('text', { x: (L + W - R) / 2, y: H - 8, class: 'axt', 'text-anchor': 'middle' });
    xl.textContent = 'real-time factor, end to end  (log scale, lower is better)';
    root.appendChild(xl);
    var yl = svg('text', {
      x: 14, y: (T + H - B) / 2, class: 'axt', 'text-anchor': 'middle',
      transform: 'rotate(-90 14 ' + ((T + H - B) / 2) + ')'
    });
    yl.textContent = 'public Seg-F1';
    root.appendChild(yl);

    // Pareto front: nothing is both cheaper and better.
    var front = pts.filter(function (p) {
      return !pts.some(function (q) {
        return q !== p && q.x <= p.x && q.y >= p.y && (q.x < p.x || q.y > p.y);
      });
    }).sort(function (a, b) { return a.x - b.x; });
    if (front.length > 1) {
      root.appendChild(svg('polyline', {
        class: 'front',
        points: front.map(function (p) { return sx(p.x) + ',' + sy(p.y); }).join(' ')
      }));
    }

    pts.forEach(function (p) {
      var g = svg('g', { class: 'pt' + (p.r.ours ? ' mine' : '') });
      g.appendChild(svg('circle', { cx: sx(p.x), cy: sy(p.y), r: sr(p.size) }));
      var title = svg('title');
      title.textContent = p.r.method + ' — Seg-F1 ' + p.y.toFixed(3)
        + ', RTF ' + p.x.toFixed(2)
        + (p.size ? ', ' + (p.size >= 1000 ? (p.size / 1000).toFixed(1) + ' B'
                                           : p.size.toFixed(1) + ' M') : '');
      g.appendChild(title);
      root.appendChild(g);
    });

    // Labels last, and only where they fit: ours, then the frontier in order,
    // skipping any that would land on one already placed.
    var placed = [];
    function fits(x, y) {
      return !placed.some(function (b) {
        return Math.abs(b.y - y) < 12 && x < b.x2 && b.x < x + 130;
      });
    }
    var wanted = pts.filter(function (p) { return p.r.ours; })
      .concat(front.filter(function (p) { return !p.r.ours; }));
    wanted.forEach(function (p) {
      var right = sx(p.x) + sr(p.size) + 6;
      var x = right, anchor = 'start';
      if (right > W - R + 60) { x = sx(p.x) - sr(p.size) - 6; anchor = 'end'; }
      if (!fits(anchor === 'start' ? x : x - 130, sy(p.y))) return;
      placed.push({ x: anchor === 'start' ? x : x - 130, x2: x + 130, y: sy(p.y) });
      var lab = svg('text', {
        x: x, y: sy(p.y) + 4, class: 'ptl' + (p.r.ours ? ' mine' : ''),
        'text-anchor': anchor
      });
      lab.textContent = p.r.method;
      root.appendChild(lab);
    });

    var box = el('div', 'scatbox');
    box.appendChild(root);
    box.appendChild(el('p', 'note',
      'Bubble area is parameter count; small hollow dots are methods with none to '
      + 'report. The dashed line joins the methods nothing beats on both axes at '
      + 'once. Hover any bubble for its numbers.'));
    return box;
  }

  /* --------------------------------------------------------------- boot */

  function render() {
    var dim = dimension(state.dim);
    buildRail();
    renderHead(dim);
    renderMain();
    renderFootnotes(dim);
  }

  function renderFootnotes(dim) {
    var host = $('#lb-foot');
    host.innerHTML = '';
    (dim.footnotes || []).forEach(function (t, i) {
      var p = el('p', 'fn');
      p.innerHTML = '<span class="fnn">' + (i + 1) + '</span>'
                  + '<span class="fnb">' + t + '</span>';
      host.appendChild(p);
    });
    var src = el('p', 'fn src');
    src.innerHTML = '<span class="fnn">src</span><span class="fnb">Generated '
      + DATA.meta.generated + ' from '
      + DATA.meta.sources.map(function (s) { return '<code>' + s + '</code>'; }).join(', ')
      + '.</span>';
    host.appendChild(src);
  }

  document.addEventListener('DOMContentLoaded', function () {
    fetch('assets/data/leaderboard.json', { cache: 'no-cache' })
      .then(function (r) {
        if (!r.ok) throw new Error('leaderboard.json: HTTP ' + r.status);
        return r.json();
      })
      .then(function (d) {
        DATA = d;
        var want = location.hash.slice(1);
        if (want && dimension(want)) state.dim = want;
        if (dimension(state.dim).kind === 'scatter') state.pareto = true;
        render();
      })
      .catch(function (e) {
        console.error('[TransVLM]', e.message);
        $('#lb-body').textContent = 'Could not load the results: ' + e.message;
      });

    addEventListener('hashchange', function () {
      var want = location.hash.slice(1);
      if (DATA && want && dimension(want) && want !== state.dim) pick(want);
    });
  });
})();
