/* One table renderer, used by the homepage's condensed views and by the
 * leaderboard's sortable one.
 *
 * Every cell arrives as {v, mark, raw, unit, norm} from build_data.py: the value
 * and its rank are separate fields, so a table can sort on v while still showing
 * the paper's own best/second-best marking. A cell with no v at all -- a
 * heuristic method has no parameter count -- renders as an em dash and sorts last
 * in either direction.
 */
window.TVTable = (function () {
  'use strict';

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function format(cellv, col) {
    if (!cellv) return '—';
    if (cellv.raw != null && (col.fmt === 'raw' || cellv.v == null)) return cellv.raw;
    if (cellv.v == null) return '—';
    var s;
    switch (col.fmt) {
      case 'int': s = cellv.v.toLocaleString('en-US'); break;
      case '2f': s = cellv.v.toFixed(2); break;
      case '1f': s = cellv.v.toFixed(1); break;
      default: s = cellv.v.toFixed(3);
    }
    return s + (col.unit || '');
  }

  /* Missing values sort to the bottom whichever way the column points. */
  function comparator(colId, dir) {
    return function (a, b) {
      var x = (a.cells[colId] || {}).v;
      var y = (b.cells[colId] || {}).v;
      var xn = (a.cells[colId] || {}).norm;
      var yn = (b.cells[colId] || {}).norm;
      if (xn != null || yn != null) { x = xn; y = yn; }
      if (x == null && y == null) return 0;
      if (x == null) return 1;
      if (y == null) return -1;
      return dir === 'asc' ? x - y : y - x;
    };
  }

  /* Bar length for the sorted column: share of the best value in that column,
   * flipped for columns where lower is better so a short bar always means worse. */
  function scaler(rows, colId, dir) {
    var vals = rows.map(function (r) {
      var c = r.cells[colId] || {};
      return c.norm != null ? c.norm : c.v;
    }).filter(function (v) { return v != null; });
    if (!vals.length) return null;
    var max = Math.max.apply(null, vals);
    var min = Math.min.apply(null, vals);
    if (max === min) return null;
    return function (v) {
      if (v == null) return 0;
      var f = dir === 'asc' ? (max - v) / (max - min) : (v - min) / (max - min);
      return Math.max(0.02, Math.min(1, f));
    };
  }

  function render(host, opts) {
    var columns = opts.columns, rows = opts.rows;
    var sortId = opts.sortBy, sortDir = opts.sortDir;

    host.innerHTML = '';
    var table = el('table', 'tbl');
    var thead = el('thead');
    var htr = el('tr');
    htr.appendChild(el('th', 'mcell', opts.firstLabel || 'Method'));
    columns.forEach(function (c) {
      var th = el('th');
      if (opts.sortable) {
        var b = el('button', 'sorth');
        b.type = 'button';
        b.appendChild(document.createTextNode(c.label));
        b.appendChild(el('span', 'caret', c.id === sortId ? (sortDir === 'asc' ? '↑' : '↓') : ''));
        b.addEventListener('click', function () { opts.onSort(c); });
        th.appendChild(b);
        if (c.id === sortId) th.className = 'sorted';
      } else {
        th.textContent = c.label;
      }
      htr.appendChild(th);
    });
    thead.appendChild(htr);
    table.appendChild(thead);

    var tbody = el('tbody');
    var bar = opts.bars && sortId ? scaler(rows, sortId, sortDir) : null;
    var lastFamily = null;

    rows.forEach(function (r) {
      if (opts.groupByFamily && r.family !== lastFamily) {
        lastFamily = r.family;
        var gtr = el('tr', 'group');
        var gtd = el('td', null, r.family);
        gtd.colSpan = columns.length + 1;
        gtr.appendChild(gtd);
        tbody.appendChild(gtr);
      }
      var tr = el('tr', r.ours ? 'ours' : null);
      // Explicit, because the leaderboard inserts a rank column in front of this
      // one afterwards and :first-child stops matching.
      var name = el('td', 'mcell');
      name.appendChild(el('span', 'mname', r.method));
      if (opts.showFamily && r.family) name.appendChild(el('span', 'mfam', r.family));
      tr.appendChild(name);

      columns.forEach(function (c) {
        var cv = r.cells[c.id];
        var td = el('td', cv && cv.mark ? cv.mark : (cv && cv.v != null ? null : 'nil'));
        if (bar && c.id === sortId) {
          var wrapper = el('span', 'barcell');
          var b = el('span', 'bar');
          var v = cv && (cv.norm != null ? cv.norm : cv.v);
          b.style.width = (bar(v) * 100) + '%';
          if (r.ours) b.classList.add('mine');
          wrapper.appendChild(b);
          wrapper.appendChild(el('span', 'barv', format(cv, c)));
          td.appendChild(wrapper);
        } else {
          td.textContent = format(cv, c);
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    var scroll = el('div', 'scroller');
    scroll.appendChild(table);
    host.appendChild(scroll);

    // A wide table clipped at the container edge looks like a rendering bug
    // rather than something you can scroll. Fade the edge that has more to show.
    function edges() {
      var over = scroll.scrollWidth - scroll.clientWidth;
      scroll.classList.toggle('more-r', over > 1 && scroll.scrollLeft < over - 1);
      scroll.classList.toggle('more-l', scroll.scrollLeft > 1);
    }
    scroll.addEventListener('scroll', edges, { passive: true });
    addEventListener('resize', edges, { passive: true });
    requestAnimationFrame(edges);
    return table;
  }

  return { render: render, format: format, comparator: comparator };
})();
