/**
 * stats-panel.js
 *
 * Statistics and comparison panel for the Adinkra chromotopology visualizer.
 * Renders compact D3 charts summarizing the 145 doubly-even binary linear
 * code equivalence classes at N=16.
 *
 * Expects D3 v7 globally available.
 *
 * Usage:
 *   StatsPanel.render(containerEl, codesArray, colorScale);
 *   StatsPanel.update(filteredCodesArray);
 */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ */
  /*  Internal state                                                     */
  /* ------------------------------------------------------------------ */
  let _allData = [];
  let _colorScale = null;
  let _container = null;
  let _comparedCodes = [];

  /* Cached DOM sections */
  let _kChartSvg = null;
  let _propSvg = null;
  let _summaryDiv = null;
  let _heatSvg = null;
  let _compDiv = null;

  /* ------------------------------------------------------------------ */
  /*  Helpers                                                            */
  /* ------------------------------------------------------------------ */

  function groupByK(data) {
    const map = {};
    for (let k = 1; k <= 8; k++) map[k] = [];
    data.forEach(function (c) { (map[c.k] || (map[c.k] = [])).push(c); });
    return map;
  }

  function kValues() { return [1, 2, 3, 4, 5, 6, 7, 8]; }

  function fmt(n) { return n.toLocaleString(); }

  /* ------------------------------------------------------------------ */
  /*  A. k-Distribution bar chart                                       */
  /* ------------------------------------------------------------------ */

  function drawKDistribution(parent, data) {
    const byK = groupByK(data);
    const counts = kValues().map(function (k) { return { k: k, count: byK[k].length }; });
    const maxCount = d3.max(counts, function (d) { return d.count; }) || 1;

    const margin = { top: 14, right: 8, bottom: 36, left: 28 };
    const width = 260 - margin.left - margin.right;
    const height = 160 - margin.top - margin.bottom;

    const wrapper = parent.append('div').attr('class', 'sp-section sp-k-chart');
    wrapper.append('div').attr('class', 'sp-title').text('k-Distribution');

    const svg = wrapper.append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    const x = d3.scaleBand().domain(kValues()).range([0, width]).padding(0.25);
    const y = d3.scaleLinear().domain([0, maxCount * 1.15]).range([height, 0]);

    svg.append('g')
      .attr('transform', 'translate(0,' + height + ')')
      .call(d3.axisBottom(x).tickSize(0))
      .selectAll('text').style('fill', '#aaa').style('font-size', '10px');

    svg.append('g')
      .call(d3.axisLeft(y).ticks(4).tickSize(-width))
      .selectAll('text').style('fill', '#888').style('font-size', '9px');

    svg.selectAll('.tick line').style('stroke', '#222').style('stroke-dasharray', '2,3');
    svg.selectAll('.domain').style('stroke', '#333');

    svg.selectAll('.bar')
      .data(counts)
      .enter().append('rect')
      .attr('class', 'bar')
      .attr('x', function (d) { return x(d.k); })
      .attr('y', function (d) { return y(d.count); })
      .attr('width', x.bandwidth())
      .attr('height', function (d) { return height - y(d.count); })
      .attr('fill', function (d) { return _colorScale ? _colorScale(d.k) : '#4fc3f7'; })
      .attr('rx', 2);

    svg.selectAll('.bar-label')
      .data(counts)
      .enter().append('text')
      .attr('class', 'bar-label')
      .attr('x', function (d) { return x(d.k) + x.bandwidth() / 2; })
      .attr('y', function (d) { return y(d.count) - 3; })
      .attr('text-anchor', 'middle')
      .style('fill', '#ccc')
      .style('font-size', '9px')
      .text(function (d) { return d.count; });

    /* x-axis label */
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height + 26)
      .attr('text-anchor', 'middle')
      .style('fill', '#888')
      .style('font-size', '9px')
      .text('k (code dimension)');

    /* Citation annotation */
    wrapper.append('div')
      .style('font-size', '8px')
      .style('color', '#555')
      .style('margin-top', '2px')
      .style('text-align', 'center')
      .text('arXiv:0806.0050 Table 4');

    _kChartSvg = wrapper;
    return wrapper;
  }

  /* ------------------------------------------------------------------ */
  /*  B. Property breakdown (indecomposable vs decomposable per k)      */
  /* ------------------------------------------------------------------ */

  function drawPropertyBreakdown(parent, data) {
    const byK = groupByK(data);
    const rows = kValues().map(function (k) {
      var codes = byK[k];
      var indec = codes.filter(function (c) { return c.is_indecomposable; }).length;
      return { k: k, total: codes.length, indec: indec, dec: codes.length - indec };
    });
    var maxTotal = d3.max(rows, function (d) { return d.total; }) || 1;

    var margin = { top: 12, right: 8, bottom: 4, left: 28 };
    var width = 220 - margin.left - margin.right;
    var height = 150 - margin.top - margin.bottom;

    var wrapper = parent.append('div').attr('class', 'sp-section sp-prop');
    wrapper.append('div').attr('class', 'sp-title').text('Indecomp. / Decomp. by k');

    var svg = wrapper.append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    var y = d3.scaleBand().domain(kValues()).range([0, height]).padding(0.2);
    var x = d3.scaleLinear().domain([0, maxTotal]).range([0, width]);

    svg.append('g')
      .call(d3.axisLeft(y).tickSize(0))
      .selectAll('text').style('fill', '#aaa').style('font-size', '9px');
    svg.selectAll('.domain').style('stroke', '#333');

    /* Indecomposable (filled) */
    svg.selectAll('.bar-indec')
      .data(rows)
      .enter().append('rect')
      .attr('class', 'bar-indec')
      .attr('x', 0)
      .attr('y', function (d) { return y(d.k); })
      .attr('width', function (d) { return x(d.indec); })
      .attr('height', y.bandwidth())
      .attr('fill', function (d) { return _colorScale ? _colorScale(d.k) : '#4fc3f7'; })
      .attr('rx', 1);

    /* Decomposable (hollow outline) */
    svg.selectAll('.bar-dec')
      .data(rows)
      .enter().append('rect')
      .attr('class', 'bar-dec')
      .attr('x', function (d) { return x(d.indec); })
      .attr('y', function (d) { return y(d.k); })
      .attr('width', function (d) { return x(d.dec); })
      .attr('height', y.bandwidth())
      .attr('fill', 'none')
      .attr('stroke', function (d) { return _colorScale ? _colorScale(d.k) : '#4fc3f7'; })
      .attr('stroke-width', 1.2)
      .attr('stroke-dasharray', '3,2')
      .attr('rx', 1);

    /* Count labels */
    svg.selectAll('.bar-count')
      .data(rows)
      .enter().append('text')
      .attr('x', function (d) { return x(d.total) + 4; })
      .attr('y', function (d) { return y(d.k) + y.bandwidth() / 2 + 3; })
      .style('fill', '#888')
      .style('font-size', '8px')
      .text(function (d) { return d.indec + '/' + d.dec; });

    /* Legend */
    var leg = wrapper.append('div').style('display', 'flex').style('gap', '10px')
      .style('font-size', '8px').style('color', '#777').style('margin-top', '2px');
    leg.append('span').html('<span style="display:inline-block;width:8px;height:8px;background:#4fc3f7;border-radius:1px;margin-right:2px;vertical-align:middle"></span> indecomp.');
    leg.append('span').html('<span style="display:inline-block;width:8px;height:8px;border:1px dashed #4fc3f7;border-radius:1px;margin-right:2px;vertical-align:middle"></span> decomp.');

    _propSvg = wrapper;
    return wrapper;
  }

  /* ------------------------------------------------------------------ */
  /*  C. Summary statistics box                                         */
  /* ------------------------------------------------------------------ */

  function drawSummary(parent, data) {
    var wrapper = parent.append('div').attr('class', 'sp-section sp-summary');
    wrapper.append('div').attr('class', 'sp-title').text('Summary');

    var indec = data.filter(function (c) { return c.is_indecomposable; }).length;
    var dec = data.length - indec;
    var selfDual = data.filter(function (c) { return c.is_self_dual; }).length;
    var autoSizes = data.map(function (c) { return c.automorphism_group_size; });
    var meanAuto = autoSizes.length > 0 ? (autoSizes.reduce(function (a, b) { return a + b; }, 0) / autoSizes.length) : 0;
    var maxAuto = autoSizes.length > 0 ? d3.max(autoSizes) : 0;

    var lines = [
      { label: fmt(data.length) + ' equivalence classes', cls: 'sp-stat-highlight' },
      { label: fmt(indec) + ' indecomposable / ' + fmt(dec) + ' decomposable', cls: '' },
      { label: fmt(selfDual) + ' Type II (self-dual doubly-even) codes', cls: '' },
      { label: 'Validated: Doran-Faux-Gates bijection (arXiv:0806.0050)', cls: 'sp-stat-cite' },
      { label: 'Mean |Aut|: ' + meanAuto.toFixed(1) + '  Max |Aut|: ' + fmt(maxAuto), cls: '' }
    ];

    var ul = wrapper.append('div').attr('class', 'sp-stat-list');
    lines.forEach(function (ln) {
      ul.append('div')
        .attr('class', 'sp-stat-line ' + ln.cls)
        .text(ln.label);
    });

    _summaryDiv = wrapper;
    return wrapper;
  }

  /* ------------------------------------------------------------------ */
  /*  D. Zero-columns heatmap                                           */
  /* ------------------------------------------------------------------ */

  function drawZeroColumnsHeatmap(parent, data) {
    var kVals = kValues();
    var zcVals = d3.range(0, 13); /* 0 through 12 */

    /* Build count matrix */
    var countMap = {};
    kVals.forEach(function (k) {
      zcVals.forEach(function (zc) { countMap[k + ',' + zc] = 0; });
    });
    data.forEach(function (c) {
      var key = c.k + ',' + c.zero_columns;
      if (countMap.hasOwnProperty(key)) countMap[key]++;
    });
    var maxVal = d3.max(Object.values(countMap)) || 1;

    var margin = { top: 14, right: 8, bottom: 28, left: 28 };
    var cellSize = 14;
    var width = zcVals.length * cellSize;
    var height = kVals.length * cellSize;

    var wrapper = parent.append('div').attr('class', 'sp-section sp-heatmap');
    wrapper.append('div').attr('class', 'sp-title').text('Zero-Columns Heatmap');

    var svg = wrapper.append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    var colorInterp = d3.scaleSequential(d3.interpolateViridis).domain([0, maxVal]);

    /* Cells */
    kVals.forEach(function (k, ki) {
      zcVals.forEach(function (zc, zi) {
        var val = countMap[k + ',' + zc];
        svg.append('rect')
          .attr('x', zi * cellSize)
          .attr('y', ki * cellSize)
          .attr('width', cellSize - 1)
          .attr('height', cellSize - 1)
          .attr('rx', 1)
          .attr('fill', val > 0 ? colorInterp(val) : '#111')
          .attr('stroke', '#1a1a1a')
          .attr('stroke-width', 0.5)
          .append('title')
          .text('k=' + k + ', zc=' + zc + ': ' + val + ' codes');

        if (val > 0 && cellSize >= 12) {
          svg.append('text')
            .attr('x', zi * cellSize + cellSize / 2 - 0.5)
            .attr('y', ki * cellSize + cellSize / 2 + 3)
            .attr('text-anchor', 'middle')
            .style('fill', val > maxVal * 0.6 ? '#000' : '#ccc')
            .style('font-size', '7px')
            .style('pointer-events', 'none')
            .text(val);
        }
      });
    });

    /* k-axis (rows) */
    kVals.forEach(function (k, ki) {
      svg.append('text')
        .attr('x', -4)
        .attr('y', ki * cellSize + cellSize / 2 + 3)
        .attr('text-anchor', 'end')
        .style('fill', '#aaa')
        .style('font-size', '8px')
        .text(k);
    });

    /* zero-columns axis (cols) */
    zcVals.forEach(function (zc, zi) {
      svg.append('text')
        .attr('x', zi * cellSize + cellSize / 2 - 0.5)
        .attr('y', height + 10)
        .attr('text-anchor', 'middle')
        .style('fill', '#aaa')
        .style('font-size', '7px')
        .text(zc);
    });

    /* Axis labels */
    svg.append('text')
      .attr('x', -4)
      .attr('y', -4)
      .attr('text-anchor', 'end')
      .style('fill', '#666')
      .style('font-size', '7px')
      .text('k');

    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height + 22)
      .attr('text-anchor', 'middle')
      .style('fill', '#666')
      .style('font-size', '8px')
      .text('zero columns');

    _heatSvg = wrapper;
    return wrapper;
  }

  /* ------------------------------------------------------------------ */
  /*  E. Comparison table                                               */
  /* ------------------------------------------------------------------ */

  function drawComparisonPanel(parent) {
    var wrapper = parent.append('div')
      .attr('class', 'sp-section sp-compare')
      .style('display', 'none');

    wrapper.append('div').attr('class', 'sp-title').text('Code Comparison');
    wrapper.append('div').attr('class', 'sp-compare-body');

    _compDiv = wrapper;
    return wrapper;
  }

  function updateComparison(codes) {
    if (!_compDiv) return;
    var body = _compDiv.select('.sp-compare-body');
    body.html('');

    if (!codes || codes.length < 2) {
      _compDiv.style('display', 'none');
      return;
    }

    _compDiv.style('display', null);
    var a = codes[0];
    var b = codes[1];

    var props = [
      { key: 'index',                label: 'Index' },
      { key: 'k',                    label: 'k (dimension)' },
      { key: 'num_codewords',        label: 'Codewords' },
      { key: 'min_distance',         label: 'Min distance' },
      { key: 'is_self_orthogonal',   label: 'Self-orthogonal' },
      { key: 'is_self_dual',         label: 'Self-dual' },
      { key: 'is_indecomposable',    label: 'Indecomposable' },
      { key: 'automorphism_group_size', label: '|Aut| size' },
      { key: 'zero_columns',         label: 'Zero columns' },
      { key: 'weight_distribution',  label: 'Weight dist.' }
    ];

    var table = body.append('table').attr('class', 'sp-comp-table');
    var thead = table.append('thead').append('tr');
    thead.append('th').text('Property');
    thead.append('th').text('#' + a.index);
    thead.append('th').text('#' + b.index);

    var tbody = table.append('tbody');
    props.forEach(function (p) {
      var tr = tbody.append('tr');
      tr.append('td').text(p.label);

      var va = a[p.key];
      var vb = b[p.key];

      function fmtVal(v) {
        if (v === true) return '✓';
        if (v === false) return '✗';
        if (Array.isArray(v)) {
          /* weight_distribution: array of [weight, count] pairs */
          return v.map(function (pair) {
            return Array.isArray(pair) ? pair[0] + ':' + pair[1] : String(pair);
          }).join('  ');
        }
        return String(v);
      }

      var cellA = tr.append('td').text(fmtVal(va));
      var cellB = tr.append('td').text(fmtVal(vb));

      /* Highlight differences */
      var sameVal = JSON.stringify(va) === JSON.stringify(vb);
      if (!sameVal) {
        cellA.style('color', '#f7c948');
        cellB.style('color', '#f7c948');
      }
    });
  }

  /* ------------------------------------------------------------------ */
  /*  CSS injection                                                     */
  /* ------------------------------------------------------------------ */

  function injectStyles() {
    if (document.getElementById('sp-styles')) return;
    var style = document.createElement('style');
    style.id = 'sp-styles';
    style.textContent = [
      '.sp-root {',
      '  display: grid;',
      '  grid-template-columns: 1fr 1fr;',
      '  grid-template-rows: auto auto auto;',
      '  gap: 10px;',
      '  padding: 10px;',
      '  background: #0a0a0f;',
      '  color: #e0e0e0;',
      '  font-family: "Inter", "Segoe UI", system-ui, sans-serif;',
      '  font-size: 11px;',
      '  overflow-y: auto;',
      '  max-height: 100%;',
      '}',
      '.sp-section {',
      '  background: #12121a;',
      '  border: 1px solid #1e1e2e;',
      '  border-radius: 6px;',
      '  padding: 8px;',
      '}',
      '.sp-title {',
      '  font-size: 10px;',
      '  font-weight: 600;',
      '  text-transform: uppercase;',
      '  letter-spacing: 0.6px;',
      '  color: #888;',
      '  margin-bottom: 6px;',
      '}',
      '.sp-k-chart   { grid-column: 1; grid-row: 1; }',
      '.sp-prop      { grid-column: 2; grid-row: 1; }',
      '.sp-summary   { grid-column: 1; grid-row: 2; }',
      '.sp-heatmap   { grid-column: 2; grid-row: 2; }',
      '.sp-compare   { grid-column: 1 / -1; grid-row: 3; }',
      '',
      '.sp-stat-list { display: flex; flex-direction: column; gap: 4px; }',
      '.sp-stat-line { font-size: 11px; color: #bbb; line-height: 1.4; }',
      '.sp-stat-highlight { font-size: 13px; font-weight: 700; color: #e0e0e0; }',
      '.sp-stat-cite { font-size: 9px; color: #666; font-style: italic; }',
      '',
      '.sp-comp-table {',
      '  width: 100%;',
      '  border-collapse: collapse;',
      '  font-size: 10px;',
      '}',
      '.sp-comp-table th {',
      '  text-align: left;',
      '  padding: 3px 6px;',
      '  border-bottom: 1px solid #2a2a3a;',
      '  color: #888;',
      '  font-weight: 600;',
      '}',
      '.sp-comp-table td {',
      '  padding: 2px 6px;',
      '  border-bottom: 1px solid #1a1a2a;',
      '  color: #ccc;',
      '}',
      '.sp-comp-table tr:hover td { background: #1a1a2a; }',
      '',
      '/* SVG axis styling */',
      '.sp-section svg { display: block; }',
      '.sp-section .domain { stroke: #333; }',
      '.sp-section .tick line { stroke: #222; }'
    ].join('\n');
    document.head.appendChild(style);
  }

  /* ------------------------------------------------------------------ */
  /*  Public API                                                        */
  /* ------------------------------------------------------------------ */

  function render(container, data, colorScale) {
    injectStyles();

    _allData = data;
    _colorScale = colorScale;
    _container = d3.select(container);
    _comparedCodes = [];

    /* Clear previous content */
    _container.selectAll('*').remove();

    var root = _container.append('div').attr('class', 'sp-root');

    drawKDistribution(root, data);
    drawPropertyBreakdown(root, data);
    drawSummary(root, data);
    drawZeroColumnsHeatmap(root, data);
    drawComparisonPanel(root);

    /* Listen for code comparison events from scatter plot */
    window.addEventListener('code-compared', function (e) {
      var detail = e.detail;
      if (detail && Array.isArray(detail.codes)) {
        _comparedCodes = detail.codes;
        updateComparison(_comparedCodes);
      }
    });
  }

  function update(filteredData) {
    if (!_container) return;

    var root = _container.select('.sp-root');
    if (root.empty()) {
      render(_container.node(), filteredData, _colorScale);
      return;
    }

    /* Rebuild all sections with filtered data */
    root.selectAll('*').remove();

    drawKDistribution(root, filteredData);
    drawPropertyBreakdown(root, filteredData);
    drawSummary(root, filteredData);
    drawZeroColumnsHeatmap(root, filteredData);
    drawComparisonPanel(root);

    /* Restore comparison if active */
    if (_comparedCodes.length >= 2) {
      updateComparison(_comparedCodes);
    }
  }

  /* ------------------------------------------------------------------ */
  /*  Export                                                             */
  /* ------------------------------------------------------------------ */

  window.StatsPanel = {
    render: render,
    update: update
  };

})();
