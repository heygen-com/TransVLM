/* Mounts the two pre-generated per-tau widgets.
 *
 * Ported from Packages/STD/js/index.js:369-433, which already ships with the
 * site and drives the quantitative panel on the published page. The ablation
 * half -- another 168 tables and 98 curves, already being served -- has never
 * had an element to mount into, which is the whole reason it was invisible.
 *
 * Shared here because both the homepage and the leaderboard need it.
 */
window.TVWidget = (function () {
  'use strict';

  var GROUPS = {
    quantitative: 'QUANT_MANIFEST',
    ablation: 'ABLA_MANIFEST'
  };
  var loaded = {};

  function fillSelect(sel, items, defaultId) {
    if (!sel) return;
    sel.innerHTML = '';
    items.forEach(function (it) {
      var o = document.createElement('option');
      o.value = it.id;
      o.textContent = it.label;
      if (defaultId && it.id === defaultId) o.selected = true;
      sel.appendChild(o);
    });
  }

  function loadScript(src) {
    if (loaded[src]) return loaded[src];
    loaded[src] = new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      s.src = src;
      s.onload = resolve;
      s.onerror = function () { reject(new Error('failed to load ' + src)); };
      document.head.appendChild(s);
    });
    return loaded[src];
  }

  function wire(root, manifest, base) {
    var q = function (role) { return root.querySelector('[data-role="' + role + '"]'); };
    var tDs = q('t-ds'), tTt = q('t-tt'), pDs = q('p-ds'), pMt = q('p-mt');
    var tFr = q('t-frame'), pImg = q('p-img');

    fillSelect(tDs, manifest.datasets, 'd00');
    fillSelect(tTt, manifest.ttOptions, 'mean');
    fillSelect(pDs, manifest.datasets, 'd00');
    fillSelect(pMt, manifest.metrics, 'segment_f1');

    function updTable() {
      var p = manifest.tables && manifest.tables[tDs.value + '|' + tTt.value];
      if (p) tFr.src = base + p; else tFr.removeAttribute('src');
    }
    function updPlot() {
      var d = manifest.plotDirs && manifest.plotDirs[pDs.value];
      if (d) pImg.src = base + d + pMt.value + '.svg'; else pImg.removeAttribute('src');
    }
    [tDs, tTt].forEach(function (s) { s.addEventListener('change', updTable); });
    [pDs, pMt].forEach(function (s) { s.addEventListener('change', updPlot); });
    updTable();
    updPlot();
  }

  /* group is 'quantitative' or 'ablation'; root is an element or an id. */
  function mount(root, group) {
    var node = typeof root === 'string' ? document.getElementById(root) : root;
    if (!node) return Promise.resolve();
    var base = './Packages/STD/data/' + group + '/';
    return loadScript(base + 'manifest.js').then(function () {
      wire(node, window[GROUPS[group]], base);
    }).catch(function (e) { console.error('[TransVLM]', e.message); });
  }

  return { mount: mount };
})();
