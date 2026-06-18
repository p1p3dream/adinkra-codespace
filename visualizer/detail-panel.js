/**
 * Detail Panel for Adinkra Chromotopology Visualizer
 *
 * Renders an interactive inspection panel for a selected binary linear code.
 * Sections: properties card, generator matrix grid, weight enumerator bar chart,
 * and column weight profile.
 *
 * Expects D3 v7 globally available.
 * Exports: window.DetailPanel = { render, update }
 */
(function () {
  'use strict';

  // ---------------------------------------------------------------------------
  // Style injection (once)
  // ---------------------------------------------------------------------------
  const STYLE_ID = 'detail-panel-styles';

  function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = `
      .dp-root {
        font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', 'Cascadia Code', monospace;
        color: #e0e0e0;
        background: #0a0a0f;
        padding: 16px;
        overflow-y: auto;
        height: 100%;
        box-sizing: border-box;
      }

      /* Placeholder */
      .dp-placeholder {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        font-size: 14px;
        color: #555;
        font-style: italic;
        user-select: none;
      }

      /* Section shared */
      .dp-section {
        margin-bottom: 20px;
        opacity: 1;
        transition: opacity 0.3s ease;
      }
      .dp-section-title {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #666;
        margin-bottom: 8px;
      }

      /* Properties card */
      .dp-props-card {
        background: #12121a;
        border: 1px solid #222;
        border-radius: 8px;
        padding: 16px;
      }
      .dp-code-heading {
        font-size: 28px;
        font-weight: 700;
        margin: 0 0 10px 0;
        color: #fff;
        transition: color 0.4s ease;
      }
      .dp-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 12px;
      }
      .dp-badge {
        display: inline-block;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.5px;
        padding: 3px 8px;
        border-radius: 4px;
        text-transform: uppercase;
      }
      .dp-badge-self-dual {
        background: rgba(255, 193, 7, 0.18);
        color: #ffc107;
        border: 1px solid rgba(255, 193, 7, 0.35);
      }
      .dp-badge-self-orthogonal {
        background: rgba(66, 165, 245, 0.18);
        color: #42a5f5;
        border: 1px solid rgba(66, 165, 245, 0.35);
      }
      .dp-badge-indecomposable {
        background: rgba(76, 175, 80, 0.18);
        color: #4caf50;
        border: 1px solid rgba(76, 175, 80, 0.35);
      }
      .dp-badge-decomposable {
        background: rgba(255, 152, 0, 0.18);
        color: #ff9800;
        border: 1px solid rgba(255, 152, 0, 0.35);
      }
      .dp-stats {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 8px;
      }
      .dp-stat {
        background: #0e0e16;
        border-radius: 4px;
        padding: 8px;
        text-align: center;
      }
      .dp-stat-value {
        font-size: 18px;
        font-weight: 700;
        color: #fff;
      }
      .dp-stat-label {
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #666;
        margin-top: 2px;
      }

      /* Generator matrix grid */
      .dp-gen-matrix {
        display: inline-block;
        background: #0e0e16;
        border-radius: 6px;
        padding: 8px;
        overflow-x: auto;
      }
      .dp-gen-table {
        border-collapse: collapse;
      }
      .dp-gen-table th {
        font-size: 9px;
        font-weight: 400;
        color: #555;
        padding: 2px 0;
        text-align: center;
        width: 22px;
        min-width: 22px;
      }
      .dp-gen-table th.dp-row-header {
        text-align: right;
        padding-right: 6px;
        width: auto;
        min-width: auto;
        color: #555;
      }
      .dp-gen-table td {
        padding: 1px;
      }
      .dp-gen-cell {
        width: 20px;
        height: 20px;
        border-radius: 3px;
        transition: background-color 0.4s ease;
      }

      /* Weight enumerator chart */
      .dp-we-chart {
        background: #0e0e16;
        border-radius: 6px;
        padding: 8px;
      }
      .dp-we-chart svg {
        display: block;
      }
      .dp-we-bar {
        transition: fill 0.4s ease, height 0.4s ease, y 0.4s ease;
      }
      .dp-we-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        fill: #999;
        text-anchor: middle;
        transition: opacity 0.3s ease;
      }
      .dp-we-axis text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        fill: #555;
      }
      .dp-we-axis line,
      .dp-we-axis path {
        stroke: #333;
      }

      /* Column weight profile */
      .dp-cwp-container {
        background: #0e0e16;
        border-radius: 6px;
        padding: 8px;
      }
      .dp-cwp-container svg {
        display: block;
      }
      .dp-cwp-bar {
        transition: fill 0.4s ease, height 0.4s ease, y 0.4s ease, opacity 0.4s ease;
      }
      .dp-cwp-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 8px;
        fill: #555;
        text-anchor: middle;
      }
    `;
    document.head.appendChild(style);
  }

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  let rootEl = null;
  let currentCode = null;

  // ---------------------------------------------------------------------------
  // render(container) -- set up the DOM skeleton
  // ---------------------------------------------------------------------------
  function render(container) {
    injectStyles();

    container.innerHTML = '';
    const root = document.createElement('div');
    root.className = 'dp-root';
    container.appendChild(root);
    rootEl = root;

    // Build skeleton with data-section markers
    root.innerHTML = `
      <div class="dp-placeholder" data-dp="placeholder">Click a code to inspect</div>
      <div data-dp="content" style="display:none;">
        <div class="dp-section" data-dp="props">
          <div class="dp-props-card">
            <h2 class="dp-code-heading" data-dp="heading"></h2>
            <div class="dp-badges" data-dp="badges"></div>
            <div class="dp-stats">
              <div class="dp-stat">
                <div class="dp-stat-value" data-dp="stat-aut"></div>
                <div class="dp-stat-label">Aut Group</div>
              </div>
              <div class="dp-stat">
                <div class="dp-stat-value" data-dp="stat-zero"></div>
                <div class="dp-stat-label">Zero Cols</div>
              </div>
              <div class="dp-stat">
                <div class="dp-stat-value" data-dp="stat-cw"></div>
                <div class="dp-stat-label">Codewords</div>
              </div>
            </div>
          </div>
        </div>
        <div class="dp-section" data-dp="gen-section">
          <div class="dp-section-title">Generator Matrix</div>
          <div class="dp-gen-matrix" data-dp="gen-matrix"></div>
        </div>
        <div class="dp-section" data-dp="we-section">
          <div class="dp-section-title">Weight Enumerator</div>
          <div class="dp-we-chart" data-dp="we-chart"></div>
        </div>
        <div class="dp-section" data-dp="cwp-section">
          <div class="dp-section-title">Column Weight Profile</div>
          <div class="dp-cwp-container" data-dp="cwp-chart"></div>
        </div>
      </div>
    `;
  }

  // ---------------------------------------------------------------------------
  // Helper: query by data-dp attribute
  // ---------------------------------------------------------------------------
  function q(key) {
    return rootEl.querySelector('[data-dp="' + key + '"]');
  }

  // ---------------------------------------------------------------------------
  // update(code, colorScale) -- populate the panel
  // ---------------------------------------------------------------------------
  function update(code, colorScale) {
    if (!rootEl) return;

    const placeholder = q('placeholder');
    const content = q('content');

    if (!code) {
      placeholder.style.display = 'flex';
      content.style.display = 'none';
      currentCode = null;
      return;
    }

    placeholder.style.display = 'none';
    content.style.display = 'block';

    const kColor = colorScale ? colorScale(code.k) : '#42a5f5';
    const isNewCode = !currentCode || currentCode.index !== code.index;
    currentCode = code;

    updateProperties(code, kColor);
    updateGeneratorMatrix(code, kColor, isNewCode);
    updateWeightEnumerator(code, kColor);
    updateColumnWeightProfile(code, kColor);
  }

  // ---------------------------------------------------------------------------
  // A. Properties card
  // ---------------------------------------------------------------------------
  function updateProperties(code, kColor) {
    // Heading: [n, k, d]
    const heading = q('heading');
    heading.textContent = '[' + code.n + ', ' + code.k + ', ' + code.min_distance + ']';
    heading.style.color = kColor;

    // Badges
    const badgesEl = q('badges');
    const badges = [];
    if (code.is_self_dual) {
      badges.push('<span class="dp-badge dp-badge-self-dual">Self-Dual</span>');
    }
    if (code.is_self_orthogonal) {
      badges.push('<span class="dp-badge dp-badge-self-orthogonal">Self-Orthogonal</span>');
    }
    if (code.is_indecomposable) {
      badges.push('<span class="dp-badge dp-badge-indecomposable">Indecomposable</span>');
    } else {
      badges.push('<span class="dp-badge dp-badge-decomposable">Decomposable</span>');
    }
    badgesEl.innerHTML = badges.join('');

    // Stats
    q('stat-aut').textContent = formatNumber(code.automorphism_group_size);
    q('stat-zero').textContent = code.zero_columns;
    q('stat-cw').textContent = code.num_codewords;
  }

  function formatNumber(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e4) return (n / 1e3).toFixed(1) + 'K';
    return n.toLocaleString();
  }

  // ---------------------------------------------------------------------------
  // B. Generator matrix grid
  // ---------------------------------------------------------------------------
  function updateGeneratorMatrix(code, kColor, forceRebuild) {
    const container = q('gen-matrix');
    const k = code.k;
    const n = code.n;
    const generators = code.generators_binary;

    // Always rebuild when k changes (different number of rows)
    const prevK = container.dataset.prevK ? parseInt(container.dataset.prevK, 10) : -1;
    const needsRebuild = forceRebuild || prevK !== k;
    container.dataset.prevK = k;

    if (needsRebuild) {
      // Build the table from scratch
      let html = '<table class="dp-gen-table"><thead><tr><th class="dp-row-header"></th>';
      for (let col = 0; col < n; col++) {
        html += '<th>' + col + '</th>';
      }
      html += '</tr></thead><tbody>';

      for (let row = 0; row < k; row++) {
        html += '<tr><th class="dp-row-header">g' + row + '</th>';
        const bits = generators[row] || '';
        for (let col = 0; col < n; col++) {
          const bit = bits.charAt(col) === '1' ? 1 : 0;
          const bg = bit ? kColor : '#1a1a2e';
          html += '<td><div class="dp-gen-cell" data-row="' + row + '" data-col="' + col +
            '" style="background-color:' + bg + ';"></div></td>';
        }
        html += '</tr>';
      }
      html += '</tbody></table>';
      container.innerHTML = html;
    } else {
      // Just update colors via transitions
      const cells = container.querySelectorAll('.dp-gen-cell');
      cells.forEach(function (cell) {
        const row = parseInt(cell.dataset.row, 10);
        const col = parseInt(cell.dataset.col, 10);
        const bits = generators[row] || '';
        const bit = bits.charAt(col) === '1' ? 1 : 0;
        cell.style.backgroundColor = bit ? kColor : '#1a1a2e';
      });
    }
  }

  // ---------------------------------------------------------------------------
  // C. Weight enumerator bar chart (D3)
  // ---------------------------------------------------------------------------
  const WE_MARGIN = { top: 20, right: 10, bottom: 28, left: 40 };
  const WE_WIDTH = 340;
  const WE_HEIGHT = 160;
  const WE_INNER_W = WE_WIDTH - WE_MARGIN.left - WE_MARGIN.right;
  const WE_INNER_H = WE_HEIGHT - WE_MARGIN.top - WE_MARGIN.bottom;

  let weSvg = null;
  let weXScale = null;
  let weYScale = null;
  let weBarGroup = null;
  let weLabelGroup = null;
  let weYAxisGroup = null;

  function initWeightEnumeratorChart() {
    const container = q('we-chart');
    container.innerHTML = '';

    const svg = d3.select(container)
      .append('svg')
      .attr('width', WE_WIDTH)
      .attr('height', WE_HEIGHT);

    weSvg = svg;

    const g = svg.append('g')
      .attr('transform', 'translate(' + WE_MARGIN.left + ',' + WE_MARGIN.top + ')');

    // X scale: 17 bands (weights 0-16)
    weXScale = d3.scaleBand()
      .domain(d3.range(17))
      .range([0, WE_INNER_W])
      .padding(0.2);

    // Y scale: will be updated per code
    weYScale = d3.scaleLinear()
      .range([WE_INNER_H, 0]);

    // X axis
    g.append('g')
      .attr('class', 'dp-we-axis')
      .attr('transform', 'translate(0,' + WE_INNER_H + ')')
      .call(
        d3.axisBottom(weXScale)
          .tickValues([0, 4, 8, 12, 16])
          .tickSize(0)
      )
      .select('.domain').remove();

    // Y axis group (updated dynamically)
    weYAxisGroup = g.append('g')
      .attr('class', 'dp-we-axis');

    // Bar group
    weBarGroup = g.append('g');

    // Label group
    weLabelGroup = g.append('g');
  }

  function updateWeightEnumerator(code, kColor) {
    const data = code.weight_enumerator_full; // array of 17 integers

    // Initialize chart if needed
    if (!weSvg) {
      initWeightEnumeratorChart();
    }

    const maxVal = d3.max(data) || 1;
    weYScale.domain([0, maxVal]);

    // Update Y axis
    weYAxisGroup
      .transition()
      .duration(400)
      .call(
        d3.axisLeft(weYScale)
          .ticks(4)
          .tickSize(-WE_INNER_W)
          .tickFormat(function (d) {
            if (d >= 1000) return (d / 1000) + 'k';
            return d;
          })
      );
    weYAxisGroup.select('.domain').remove();
    weYAxisGroup.selectAll('.tick line')
      .attr('stroke', '#1a1a2e')
      .attr('stroke-dasharray', '2,2');

    // Bars
    const bars = weBarGroup.selectAll('.dp-we-bar')
      .data(data);

    bars.enter()
      .append('rect')
      .attr('class', 'dp-we-bar')
      .attr('x', function (_d, i) { return weXScale(i); })
      .attr('width', weXScale.bandwidth())
      .attr('y', WE_INNER_H)
      .attr('height', 0)
      .attr('rx', 2)
      .attr('fill', kColor)
      .merge(bars)
      .transition()
      .duration(400)
      .attr('x', function (_d, i) { return weXScale(i); })
      .attr('width', weXScale.bandwidth())
      .attr('y', function (d) { return d > 0 ? weYScale(d) : WE_INNER_H; })
      .attr('height', function (d) { return d > 0 ? WE_INNER_H - weYScale(d) : 0; })
      .attr('fill', function (d) { return d > 0 ? kColor : 'transparent'; });

    bars.exit().remove();

    // Labels above nonzero bars
    const labels = weLabelGroup.selectAll('.dp-we-label')
      .data(data);

    labels.enter()
      .append('text')
      .attr('class', 'dp-we-label')
      .attr('x', function (_d, i) { return weXScale(i) + weXScale.bandwidth() / 2; })
      .attr('y', WE_INNER_H)
      .merge(labels)
      .transition()
      .duration(400)
      .attr('x', function (_d, i) { return weXScale(i) + weXScale.bandwidth() / 2; })
      .attr('y', function (d) { return d > 0 ? weYScale(d) - 4 : WE_INNER_H; })
      .attr('opacity', function (d) { return d > 0 ? 1 : 0; })
      .text(function (d) { return d > 0 ? d : ''; });

    labels.exit().remove();
  }

  // ---------------------------------------------------------------------------
  // D. Column weight profile (16 mini bars)
  // ---------------------------------------------------------------------------
  const CWP_MARGIN = { top: 8, right: 4, bottom: 20, left: 4 };
  const CWP_WIDTH = 340;
  const CWP_HEIGHT = 80;
  const CWP_INNER_W = CWP_WIDTH - CWP_MARGIN.left - CWP_MARGIN.right;
  const CWP_INNER_H = CWP_HEIGHT - CWP_MARGIN.top - CWP_MARGIN.bottom;

  let cwpSvg = null;
  let cwpXScale = null;
  let cwpYScale = null;
  let cwpBarGroup = null;
  let cwpLabelGroup = null;

  function initColumnWeightProfile() {
    const container = q('cwp-chart');
    container.innerHTML = '';

    const svg = d3.select(container)
      .append('svg')
      .attr('width', CWP_WIDTH)
      .attr('height', CWP_HEIGHT);

    cwpSvg = svg;

    const g = svg.append('g')
      .attr('transform', 'translate(' + CWP_MARGIN.left + ',' + CWP_MARGIN.top + ')');

    cwpXScale = d3.scaleBand()
      .domain(d3.range(16))
      .range([0, CWP_INNER_W])
      .padding(0.15);

    cwpYScale = d3.scaleLinear()
      .range([CWP_INNER_H, 0]);

    cwpBarGroup = g.append('g');

    // Column index labels
    cwpLabelGroup = g.append('g');
    for (let i = 0; i < 16; i++) {
      cwpLabelGroup.append('text')
        .attr('class', 'dp-cwp-label')
        .attr('x', cwpXScale(i) + cwpXScale.bandwidth() / 2)
        .attr('y', CWP_INNER_H + 14)
        .text(i);
    }
  }

  function updateColumnWeightProfile(code, kColor) {
    // column_weight_profile is sorted ascending, but we want per-column data.
    // Actually, looking at the data: for index 0 (k=1, gen "1111000000000000"),
    // column_weight_profile is [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1].
    // This is sorted. The actual per-column weights come from analyzing generators,
    // but the profile as given is sorted. We'll display it as-is (sorted profile).
    //
    // For a true per-column display we'd need to compute from generators.
    // The profile is sorted, so column 0 = lightest, column 15 = heaviest.
    // We display position index 0-15 of the sorted profile.
    const profile = code.column_weight_profile; // array of 16 sorted ints

    if (!cwpSvg) {
      initColumnWeightProfile();
    }

    const maxVal = d3.max(profile) || 1;
    cwpYScale.domain([0, maxVal]);

    // Derive a dimmed color for zero-weight columns
    const dimColor = '#1a1a2e';

    const bars = cwpBarGroup.selectAll('.dp-cwp-bar')
      .data(profile);

    bars.enter()
      .append('rect')
      .attr('class', 'dp-cwp-bar')
      .attr('x', function (_d, i) { return cwpXScale(i); })
      .attr('width', cwpXScale.bandwidth())
      .attr('rx', 2)
      .attr('y', CWP_INNER_H)
      .attr('height', 0)
      .attr('fill', kColor)
      .attr('opacity', 1)
      .merge(bars)
      .transition()
      .duration(400)
      .attr('x', function (_d, i) { return cwpXScale(i); })
      .attr('width', cwpXScale.bandwidth())
      .attr('y', function (d) { return d > 0 ? cwpYScale(d) : CWP_INNER_H - 2; })
      .attr('height', function (d) { return d > 0 ? CWP_INNER_H - cwpYScale(d) : 2; })
      .attr('fill', function (d) { return d > 0 ? kColor : dimColor; })
      .attr('opacity', function (d) { return d > 0 ? 1 : 0.3; });

    bars.exit().remove();
  }

  // ---------------------------------------------------------------------------
  // Expose public API
  // ---------------------------------------------------------------------------
  window.DetailPanel = {
    render: render,
    update: update
  };
})();
