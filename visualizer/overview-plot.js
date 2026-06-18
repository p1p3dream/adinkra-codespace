// overview-plot.js -- Scatter plot of 145 doubly-even [16,k,d] code equivalence classes
// Requires D3 v7 globally available

(function () {
  'use strict';

  // ---- constants ----
  const MARGIN = { top: 52, right: 160, bottom: 56, left: 64 };
  const TRANSITION_MS = 750;
  const JITTER_BAND = 0.35; // +/- fraction of one x-unit
  const MIN_RADIUS = 4;
  const MAX_RADIUS = 14;

  const Y_METRICS = [
    { key: 'min_distance',           label: 'Min Distance (d)' },
    { key: 'num_codewords',          label: 'Codewords (2^k)' },
    { key: 'zero_columns',           label: 'Zero Columns' },
    { key: 'automorphism_group_size', label: 'Aut(C) Size' },
  ];

  // ---- module state (singleton) ----
  let svg, gPlot, gXAxis, gYAxis, gGrid, tooltip;
  let xScale, yScale, rScale;
  let currentMetric = Y_METRICS[0].key;
  let currentData = [];
  let currentColorScale = null;
  let width = 0, height = 0;
  let jitterMap = new Map(); // code.index -> deterministic jitter value

  // ---- helpers ----

  function seededRandom(seed) {
    // simple deterministic hash so jitter is stable across renders
    let x = Math.sin(seed * 9301 + 49297) * 49297;
    return x - Math.floor(x);
  }

  function getJitter(code) {
    if (!jitterMap.has(code.index)) {
      jitterMap.set(code.index, (seededRandom(code.index) - 0.5) * 2 * JITTER_BAND);
    }
    return jitterMap.get(code.index);
  }

  function radiusForCode(code) {
    return rScale(code.num_codewords);
  }

  function xPos(code) {
    return xScale(code.k + getJitter(code));
  }

  function yPos(code) {
    return yScale(code[currentMetric]);
  }

  function markerPath(code, r) {
    if (code.is_self_dual) {
      // 4-point diamond
      return `M0,${-r} L${r},0 L0,${r} L${-r},0 Z`;
    }
    // circle (used for both indecomposable filled and decomposable hollow)
    return null;
  }

  // ---- tooltip ----

  function showTooltip(event, code) {
    const props = [];
    if (code.is_self_orthogonal) props.push('self-orthogonal');
    if (code.is_self_dual) props.push('self-dual');
    if (code.is_indecomposable) props.push('indecomposable');
    if (!code.is_indecomposable) props.push('decomposable');

    tooltip
      .style('opacity', 1)
      .html(
        `<strong>Class #${code.index}</strong><br/>` +
        `[${code.n}, ${code.k}, ${code.min_distance}] code<br/>` +
        `Codewords: ${code.num_codewords}<br/>` +
        `Zero columns: ${code.zero_columns}<br/>` +
        `${props.join(', ')}`
      );

    positionTooltip(event);
  }

  function positionTooltip(event) {
    const ttNode = tooltip.node();
    const ttRect = ttNode.getBoundingClientRect();
    let x = event.pageX + 14;
    let y = event.pageY - 14;
    // keep on screen
    if (x + ttRect.width > window.innerWidth - 8) {
      x = event.pageX - ttRect.width - 14;
    }
    if (y + ttRect.height > window.innerHeight - 8) {
      y = event.pageY - ttRect.height - 14;
    }
    tooltip.style('left', x + 'px').style('top', y + 'px');
  }

  function hideTooltip() {
    tooltip.style('opacity', 0);
  }

  // ---- dropdown ----

  function buildDropdown(container) {
    const wrap = d3.select(container)
      .append('div')
      .attr('class', 'overview-controls')
      .style('position', 'absolute')
      .style('top', '8px')
      .style('left', MARGIN.left + 'px')
      .style('z-index', '10')
      .style('display', 'flex')
      .style('align-items', 'center')
      .style('gap', '8px');

    wrap.append('label')
      .style('color', '#888')
      .style('font-size', '12px')
      .style('font-family', 'monospace')
      .text('Y-axis:');

    const sel = wrap.append('select')
      .style('background', '#1a1a24')
      .style('color', '#ccc')
      .style('border', '1px solid #333')
      .style('border-radius', '4px')
      .style('padding', '2px 6px')
      .style('font-size', '12px')
      .style('font-family', 'monospace')
      .style('cursor', 'pointer')
      .on('change', function () {
        currentMetric = this.value;
        rebuildYScale();
        updatePlot(currentData);
      });

    sel.selectAll('option')
      .data(Y_METRICS)
      .join('option')
      .attr('value', d => d.key)
      .text(d => d.label)
      .property('selected', d => d.key === currentMetric);
  }

  // ---- legend ----

  function buildLegend() {
    const lg = svg.append('g')
      .attr('class', 'overview-legend')
      .attr('transform', `translate(${width - MARGIN.right + 16}, ${MARGIN.top + 4})`);

    const items = [
      { label: 'Indecomposable', type: 'filled' },
      { label: 'Decomposable',   type: 'hollow' },
      { label: 'Self-dual',      type: 'diamond' },
    ];

    items.forEach((item, i) => {
      const g = lg.append('g')
        .attr('transform', `translate(0, ${i * 22})`);

      if (item.type === 'diamond') {
        g.append('path')
          .attr('d', 'M0,-6 L6,0 L0,6 L-6,0 Z')
          .attr('transform', 'translate(7, 7)')
          .attr('fill', '#888')
          .attr('stroke', '#aaa')
          .attr('stroke-width', 1);
      } else if (item.type === 'filled') {
        g.append('circle')
          .attr('cx', 7).attr('cy', 7).attr('r', 5)
          .attr('fill', '#888')
          .attr('stroke', 'none');
      } else {
        g.append('circle')
          .attr('cx', 7).attr('cy', 7).attr('r', 5)
          .attr('fill', 'none')
          .attr('stroke', '#888')
          .attr('stroke-width', 1.5);
      }

      g.append('text')
        .attr('x', 18)
        .attr('y', 11)
        .attr('fill', '#888')
        .attr('font-size', '11px')
        .attr('font-family', 'monospace')
        .text(item.label);
    });
  }

  // ---- scales ----

  function buildScales(data) {
    const plotW = width - MARGIN.left - MARGIN.right;
    const plotH = height - MARGIN.top - MARGIN.bottom;

    xScale = d3.scaleLinear()
      .domain([0.5, 8.5])
      .range([0, plotW]);

    rScale = d3.scaleSqrt()
      .domain([2, 256])
      .range([MIN_RADIUS, MAX_RADIUS])
      .clamp(true);

    rebuildYScale();
  }

  function rebuildYScale() {
    const plotH = height - MARGIN.top - MARGIN.bottom;
    const vals = currentData.map(d => d[currentMetric]);
    const lo = d3.min(vals);
    const hi = d3.max(vals);
    const pad = Math.max(1, (hi - lo) * 0.08);

    yScale = d3.scaleLinear()
      .domain([Math.max(0, lo - pad), hi + pad])
      .range([plotH, 0])
      .nice();
  }

  // ---- axes & grid ----

  function drawAxes(transition) {
    const plotH = height - MARGIN.top - MARGIN.bottom;
    const plotW = width - MARGIN.left - MARGIN.right;

    // x-axis
    const xAxis = d3.axisBottom(xScale)
      .tickValues([1, 2, 3, 4, 5, 6, 7, 8])
      .tickFormat(d => 'k=' + d);

    if (transition) {
      gXAxis.transition().duration(TRANSITION_MS).call(xAxis);
    } else {
      gXAxis.call(xAxis);
    }

    // y-axis
    const yAxis = d3.axisLeft(yScale)
      .ticks(8);

    if (transition) {
      gYAxis.transition().duration(TRANSITION_MS).call(yAxis);
    } else {
      gYAxis.call(yAxis);
    }

    // grid lines
    gGrid.selectAll('.grid-line-y').remove();
    const yTicks = yScale.ticks(8);
    gGrid.selectAll('.grid-line-y')
      .data(yTicks)
      .join('line')
      .attr('class', 'grid-line-y')
      .attr('x1', 0)
      .attr('x2', plotW)
      .attr('y1', d => yScale(d))
      .attr('y2', d => yScale(d))
      .attr('stroke', '#1e1e2e')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,4');

    gGrid.selectAll('.grid-line-x').remove();
    gGrid.selectAll('.grid-line-x')
      .data([1, 2, 3, 4, 5, 6, 7, 8])
      .join('line')
      .attr('class', 'grid-line-x')
      .attr('x1', d => xScale(d))
      .attr('x2', d => xScale(d))
      .attr('y1', 0)
      .attr('y2', plotH)
      .attr('stroke', '#1e1e2e')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,4');

    // style ticks for dark theme
    svg.selectAll('.tick text')
      .attr('fill', '#888')
      .attr('font-family', 'monospace')
      .attr('font-size', '11px');

    svg.selectAll('.tick line')
      .attr('stroke', '#444');

    svg.selectAll('.domain')
      .attr('stroke', '#444');

    // y-axis label
    svg.selectAll('.y-axis-label').remove();
    const metricLabel = Y_METRICS.find(m => m.key === currentMetric);
    svg.append('text')
      .attr('class', 'y-axis-label')
      .attr('transform', `translate(${MARGIN.left - 44}, ${MARGIN.top + (plotH / 2)}) rotate(-90)`)
      .attr('text-anchor', 'middle')
      .attr('fill', '#888')
      .attr('font-size', '12px')
      .attr('font-family', 'monospace')
      .text(metricLabel ? metricLabel.label : currentMetric);

    // x-axis label
    svg.selectAll('.x-axis-label').remove();
    svg.append('text')
      .attr('class', 'x-axis-label')
      .attr('transform', `translate(${MARGIN.left + plotW / 2}, ${height - 8})`)
      .attr('text-anchor', 'middle')
      .attr('fill', '#888')
      .attr('font-size', '12px')
      .attr('font-family', 'monospace')
      .text('Dimension k');
  }

  // ---- plot points ----

  function updatePlot(data, skipTransition) {
    const t = skipTransition
      ? d3.transition().duration(0)
      : d3.transition().duration(TRANSITION_MS).ease(d3.easeCubicInOut);

    const colorFn = currentColorScale || (window.APP && window.APP.colorScale) || d3.scaleOrdinal(d3.schemeTableau10);

    // bind data by code index
    const marks = gPlot.selectAll('.code-mark')
      .data(data, d => d.index);

    // EXIT
    marks.exit()
      .transition(t)
      .attr('opacity', 0)
      .attr('transform', d => `translate(${xPos(d)}, ${yPos(d)}) scale(0)`)
      .remove();

    // ENTER
    const enter = marks.enter()
      .append('g')
      .attr('class', 'code-mark')
      .attr('transform', d => `translate(${xPos(d)}, ${yPos(d)})`)
      .attr('opacity', 0)
      .style('cursor', 'pointer')
      .on('mouseenter', function (event, d) { showTooltip(event, d); })
      .on('mousemove', function (event) { positionTooltip(event); })
      .on('mouseleave', function () { hideTooltip(); })
      .on('click', function (event, d) {
        document.dispatchEvent(new CustomEvent('code-selected', { detail: d }));
      });

    // append shapes inside each enter group
    enter.each(function (d) {
      const g = d3.select(this);
      const r = radiusForCode(d);
      const color = colorFn(d.k);

      if (d.is_self_dual) {
        // diamond shape
        g.append('path')
          .attr('d', markerPath(d, r))
          .attr('fill', d.is_indecomposable ? color : 'none')
          .attr('stroke', color)
          .attr('stroke-width', d.is_indecomposable ? 0 : 2);
      } else {
        // circle
        g.append('circle')
          .attr('r', r)
          .attr('fill', d.is_indecomposable ? color : 'none')
          .attr('stroke', color)
          .attr('stroke-width', d.is_indecomposable ? 0 : 2);
      }
    });

    // ENTER transition
    enter.transition(t)
      .attr('opacity', 0.85);

    // UPDATE (merge enter + existing)
    const merged = enter.merge(marks);

    merged.transition(t)
      .attr('transform', d => `translate(${xPos(d)}, ${yPos(d)})`)
      .attr('opacity', 0.85);

    // update shape attributes in case color or metric changed
    merged.each(function (d) {
      const g = d3.select(this);
      const r = radiusForCode(d);
      const color = colorFn(d.k);

      const circle = g.select('circle');
      if (!circle.empty()) {
        circle.transition(t)
          .attr('r', r)
          .attr('fill', d.is_indecomposable ? color : 'none')
          .attr('stroke', color)
          .attr('stroke-width', d.is_indecomposable ? 0 : 2);
      }

      const path = g.select('path');
      if (!path.empty()) {
        path.transition(t)
          .attr('d', markerPath(d, r))
          .attr('fill', d.is_indecomposable ? color : 'none')
          .attr('stroke', color)
          .attr('stroke-width', d.is_indecomposable ? 0 : 2);
      }
    });
  }

  // ---- public API ----

  function render(container, data, colorScale) {
    currentData = data;
    currentColorScale = colorScale || null;
    container.innerHTML = '';

    // measure container
    const rect = container.getBoundingClientRect();
    width = rect.width || 900;
    height = rect.height || 520;

    // ensure container is positioned for absolute children
    const containerSel = d3.select(container);
    if (containerSel.style('position') === 'static') {
      containerSel.style('position', 'relative');
    }

    // create tooltip if it does not exist
    if (!tooltip) {
      tooltip = d3.select('body')
        .append('div')
        .attr('class', 'overview-tooltip')
        .style('position', 'absolute')
        .style('pointer-events', 'none')
        .style('background', '#1a1a28')
        .style('color', '#ddd')
        .style('border', '1px solid #444')
        .style('border-radius', '6px')
        .style('padding', '8px 12px')
        .style('font-size', '12px')
        .style('font-family', 'monospace')
        .style('line-height', '1.5')
        .style('opacity', 0)
        .style('z-index', '1000')
        .style('max-width', '260px')
        .style('box-shadow', '0 4px 16px rgba(0,0,0,0.5)');
    }

    // dropdown
    buildDropdown(container);

    // svg
    svg = containerSel.append('svg')
      .attr('width', width)
      .attr('height', height)
      .style('display', 'block');

    // groups in correct draw order
    gGrid = svg.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    gXAxis = svg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(${MARGIN.left}, ${height - MARGIN.bottom})`);

    gYAxis = svg.append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    gPlot = svg.append('g')
      .attr('class', 'plot-area')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    // clip path so points don't overflow axes
    svg.append('defs')
      .append('clipPath')
      .attr('id', 'overview-clip')
      .append('rect')
      .attr('width', width - MARGIN.left - MARGIN.right)
      .attr('height', height - MARGIN.top - MARGIN.bottom);

    gPlot.attr('clip-path', 'url(#overview-clip)');

    // build scales and draw
    buildScales(data);
    drawAxes(false);
    buildLegend();
    updatePlot(data, true);

    // listen for resize
    const ro = new ResizeObserver(() => {
      const r2 = container.getBoundingClientRect();
      if (r2.width === width && r2.height === height) return;
      width = r2.width || 900;
      height = r2.height || 520;

      svg.attr('width', width).attr('height', height);

      const plotW = width - MARGIN.left - MARGIN.right;
      const plotH = height - MARGIN.top - MARGIN.bottom;

      xScale.range([0, plotW]);
      yScale.range([plotH, 0]);

      gXAxis.attr('transform', `translate(${MARGIN.left}, ${height - MARGIN.bottom})`);

      svg.select('#overview-clip rect')
        .attr('width', plotW)
        .attr('height', plotH);

      drawAxes(false);
      updatePlot(currentData, true);

      // reposition legend
      svg.select('.overview-legend')
        .attr('transform', `translate(${width - MARGIN.right + 16}, ${MARGIN.top + 4})`);
    });
    ro.observe(container);
  }

  function update(filteredData) {
    currentData = filteredData;
    rebuildYScale();
    drawAxes(true);
    updatePlot(filteredData, false);
  }

  // expose on window
  window.OverviewPlot = { render: render, update: update };
})();
