/* The control bar over TVPlot, and the wiring from the data file to a figure.
 *
 * Four controls, and each one is an axis the published comparison table already
 * has: Rows (methods or ablated components), Data (public, synthetic, or both),
 * Level (segment, frame, or boundary error) and Statistic (precision, recall,
 * F1). The default is the page's headline, public x segment x F1, so the figure
 * makes the argument before anyone touches anything.
 *
 * Data = Both is the scatter. It is the same question -- does it hold on both
 * halves -- so it belongs on the control that names the halves, not on a second
 * control the reader has to connect to the first.
 *
 * Two things deliberately absent from the controls. Boundary error has no
 * precision/recall/F1 split, so choosing it hides the Statistic control rather
 * than leaving an inert one on screen. Real-time factor is a cost rather than a
 * quality and never shares an axis with an F1 -- it is in the panel instead.
 *
 * Nothing here writes a caption. The controls say what the figure is; if the
 * figure needs a sentence explaining how to read it, it is the wrong figure.
 */
window.TVExplore = (function () {
  'use strict';

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function fmt(v, dp, unit) {
    return v == null ? '--' : v.toFixed(dp) + (unit || '');
  }

  function segmented(label, options, value, onChange) {
    var wrap = el('div', 'ctl');
    wrap.appendChild(el('span', 'ctl-k', label));
    var seg = el('div', 'seg');
    options.forEach(function (o) {
      var b = el('button', null, o.label);
      b.type = 'button';
      b.value = o.id;
      b.setAttribute('aria-pressed', String(o.id === value));
      b.addEventListener('click', function () {
        [].forEach.call(seg.children, function (c) {
          c.setAttribute('aria-pressed', String(c === b));
        });
        onChange(o.id);
      });
      seg.appendChild(b);
    });
    wrap.appendChild(seg);
    wrap.__seg = seg;
    return wrap;
  }

  function press(ctl, id) {
    [].forEach.call(ctl.__seg.children, function (c) {
      c.setAttribute('aria-pressed', String(c.value === id));
    });
  }

  /* data is the parsed std-results.json; host is the element to build into. */
  function mount(host, data) {
    var halfById = {}, statById = {};
    data.halves.forEach(function (h) { halfById[h.id] = h; });
    data.stats.forEach(function (s) { statById[s.id] = s; });

    /* `half` is 'public', 'synthetic', or 'both'. 'both' is not a third
     * dataset -- it is the question "does it hold on both?", which is a scatter,
     * so the Data control reaches the pair mode directly rather than through a
     * second control the reader has to connect to the first. */
    var state = {
      group: 'quant',
      half: data.defaults.half,
      level: data.defaults.level,
      stat: data.defaults.stat
    };

    var bar = el('div', 'ctlbar');
    var figure = el('div', 'ctl-figure');
    host.textContent = '';
    host.appendChild(bar);
    host.appendChild(figure);

    function set(key) {
      return function (v) { state[key] = v; draw(); };
    }

    var rowsCtl = segmented('Rows', [
      { id: 'quant', label: 'Methods' },
      { id: 'abla', label: 'Ablations' }
    ], state.group, set('group'));

    var halfCtl = segmented('Data', data.halves.map(function (h) {
      return { id: h.id, label: h.label.replace(' data', '') };
    }).concat([{ id: 'both', label: 'Both' }]), state.half, set('half'));

    var levelCtl = segmented('Level', data.levels.map(function (l) {
      return { id: l.id, label: l.label };
    }).concat([{ id: 'abe', label: 'Boundary error' }]), state.level,
      set('level'));

    var statCtl = segmented('Statistic', data.stats.map(function (s) {
      return { id: s.id, label: s.label };
    }), state.stat, set('stat'));


    [rowsCtl, halfCtl, levelCtl, statCtl]
      .forEach(function (c) { bar.appendChild(c); });

    var reset = el('button', 'ctl-reset', 'Back to the headline');
    reset.type = 'button';
    reset.addEventListener('click', function () {
      /* The headline is the methods ranking, so the row set comes back too --
       * a button labelled "back to the headline" that leaves you on the
       * ablations has not done what it says. */
      state.group = 'quant';
      state.half = data.defaults.half;
      state.level = data.defaults.level;
      state.stat = data.defaults.stat;
      press(rowsCtl, state.group);
      press(halfCtl, state.half);
      press(levelCtl, state.level);
      press(statCtl, state.stat);
      draw();
    });
    bar.appendChild(reset);

    /* --- what the controls currently name */

    function metricId() {
      return state.level === 'abe' ? 'abe' : state.level + '_' + state.stat;
    }

    function metric() {
      if (state.level === 'abe') {
        return { id: 'abe', label: 'Boundary error', dir: 'asc', dp: 2,
                 unit: ' s', max: null };
      }
      var level = state.level === 'seg' ? 'Segment' : 'Frame';
      return {
        id: metricId(), dir: 'desc', dp: 3, unit: '', max: 1,
        label: level + ' ' + statById[state.stat].inline
      };
    }

    /* Every metric of a row, for the panel. The figure shows one number per
     * row; reading a mark should answer the next question too, instead of
     * sending the reader back to the controls. In the both-halves view it
     * carries both columns, because that view is about the difference. */
    function describe() {
      var halves = [].slice.call(arguments);
      var m = metricId();
      return function (row) {
        var rows = [];

        function line(label, get, dp, unit, id) {
          var cells = halves.map(function (h) { return get(row.values[h]); });
          rows.push([label, cells.map(function (c) {
            return fmt(c, dp, unit);
          }).join('  /  '), id === m ? 'lead' : null]);
        }

        data.levels.forEach(function (level) {
          data.stats.forEach(function (stat) {
            var id = level.id + '_' + stat.id;
            line(level.label + ' ' + stat.inline,
                 function (v) { return v[id]; }, 3, '', id);
          });
        });
        line('Boundary error', function (v) { return v.abe; }, 2, ' s', 'abe');
        if (row.rtf != null) {
          rows.push(['Real-time factor', fmt(row.rtf, 2)]);
        }

        return {
          name: row.method,
          family: row.family === 'Ablation' ? 'component removed' : row.family,
          colour: window.TVPlot.families[row.ours ? 'Ours' : row.family],
          rows: rows,
          note: halves.map(function (h) {
            return halfById[h].label.toLowerCase();
          }).join(' / ')
        };
      };
    }

    function values(half) {
      var m = metricId();
      return data.groups[state.group].rows.map(function (r) {
        return r.values[half][m];
      });
    }

    function draw() {
      /* Precision, recall and F1 do not apply to a boundary error in seconds,
       * so the control goes away rather than sitting there inert. */
      statCtl.hidden = state.level === 'abe';

      var g = data.groups[state.group];
      var met = metric();

      if (state.half === 'both') {
        window.TVPlot.pair(figure, {
          rows: g.rows,
          xValues: values('public'),
          yValues: values('synthetic'),
          metric: met,
          x: { label: met.label + ' \u00b7 public' },
          y: { label: met.label + ' \u00b7 synthetic' },
          alt: met.label + ' on public data against synthetic data, ' +
               g.rows.length + ' rows, each named on the plot.',
          describe: describe('public', 'synthetic')
        });
        return;
      }

      window.TVPlot.ranked(figure, {
        rows: g.rows,
        values: values(state.half),
        metric: met,
        alt: met.label + ' on ' + halfById[state.half].label.toLowerCase() +
             ', ' + g.rows.length + ' rows, sorted.',
        describe: describe(state.half)
      });
    }

    draw();
    /* Nothing here watches the width. Both figures do their own width work:
       ranked() since HAX-55, and pair() now that it draws in CSS pixels too --
       which is also what retired the 1000px floor this used to keep the
       scatter's control above. */

    return { state: state, draw: draw };
  }

  return { mount: mount };
}());
