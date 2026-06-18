// graph-panel.js — Bipartite graph visualization for Adinkra chromotopology codes
// Renders column vertices (bit positions 0-15) connected to nonzero codeword vertices.
// Codeword vertices colored by Hamming weight; layout adapts based on code dimension k.

(function () {
  'use strict';

  // ── Utility functions ──────────────────────────────────────────────

  function parseHex(s) {
    return parseInt(s, 16);
  }

  function weight(v) {
    let w = 0;
    while (v) { w += v & 1; v >>>= 1; }
    return w;
  }

  function hasBit(v, j) {
    return (v >>> j) & 1;
  }

  function binaryStr(v, n) {
    let s = v.toString(2);
    while (s.length < n) s = '0' + s;
    return s;
  }

  function hexStr(v) {
    return '0x' + v.toString(16);
  }

  // ── Color scales ───────────────────────────────────────────────────

  // Weight color: warm sequential scale for weights 4, 8, 12, 16
  function weightColor(w) {
    // Map weight to [0, 1] domain: weight 4 -> 0.2, 8 -> 0.45, 12 -> 0.7, 16 -> 0.95
    var t = Math.max(0.15, Math.min(0.95, (w - 2) / 16));
    return d3.interpolateYlOrRd(t);
  }

  // ── Constants ──────────────────────────────────────────────────────

  var COL_COLOR = '#00bcd4';
  var COL_COLOR_DIM = '#333';
  var COL_RADIUS = 10;
  var N = 16;
  var TRANSITION_MS = 400;

  // ── Module state ───────────────────────────────────────────────────

  var svg, gEdges, gCols, gCwords, gTooltip;
  var width, height;
  var currentCode = null;

  // ── Render: initialize SVG structure ───────────────────────────────

  function render(container) {
    container.innerHTML = '';
    var rect = container.getBoundingClientRect();
    width = rect.width || 500;
    height = rect.height || 400;

    svg = d3.select(container)
      .append('svg')
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .style('background', 'transparent');

    // Layer ordering: edges behind nodes
    gEdges = svg.append('g').attr('class', 'graph-edges');
    gCwords = svg.append('g').attr('class', 'graph-codewords');
    gCols = svg.append('g').attr('class', 'graph-columns');

    // Tooltip div (appended to container, positioned absolutely)
    gTooltip = d3.select(container)
      .append('div')
      .attr('class', 'graph-tooltip')
      .style('position', 'absolute')
      .style('pointer-events', 'none')
      .style('background', 'rgba(10,10,20,0.92)')
      .style('border', '1px solid #555')
      .style('border-radius', '4px')
      .style('padding', '6px 10px')
      .style('font-size', '11px')
      .style('font-family', 'monospace')
      .style('color', '#ddd')
      .style('white-space', 'nowrap')
      .style('opacity', 0)
      .style('z-index', 100);

    // Show placeholder
    showPlaceholder();
  }

  // ── Placeholder ────────────────────────────────────────────────────

  function showPlaceholder() {
    svg.selectAll('.placeholder').remove();
    svg.append('text')
      .attr('class', 'placeholder')
      .attr('x', width / 2)
      .attr('y', height / 2)
      .attr('text-anchor', 'middle')
      .attr('fill', '#555')
      .attr('font-size', '14px')
      .attr('font-family', 'sans-serif')
      .text('Select a code to view its bipartite graph');
  }

  // ── Update: rebuild graph for a code ───────────────────────────────

  function update(code, colorScale) {
    if (!svg) return;

    // Refresh dimensions in case container resized
    var viewBox = svg.attr('viewBox').split(' ');
    width = +viewBox[2];
    height = +viewBox[3];

    // Fade out existing content
    gEdges.selectAll('*')
      .transition().duration(TRANSITION_MS / 2).style('opacity', 0).remove();
    gCwords.selectAll('*')
      .transition().duration(TRANSITION_MS / 2).style('opacity', 0).remove();
    gCols.selectAll('*')
      .transition().duration(TRANSITION_MS / 2).style('opacity', 0).remove();
    svg.selectAll('.placeholder')
      .transition().duration(TRANSITION_MS / 2).style('opacity', 0).remove();

    if (!code) {
      setTimeout(showPlaceholder, TRANSITION_MS / 2 + 50);
      return;
    }

    currentCode = code;

    // Parse codewords, skip zero
    var codewords = (code.all_codewords_hex || [])
      .map(parseHex)
      .filter(function (v) { return v !== 0; });

    var k = code.k || 0;

    // Determine which columns are active (at least one codeword has a 1 there)
    var colActive = new Array(N);
    for (var j = 0; j < N; j++) {
      colActive[j] = false;
      for (var ci = 0; ci < codewords.length; ci++) {
        if (hasBit(codewords[ci], j)) {
          colActive[j] = true;
          break;
        }
      }
    }

    // Build edges
    var edges = [];
    for (var ci = 0; ci < codewords.length; ci++) {
      for (var j = 0; j < N; j++) {
        if (hasBit(codewords[ci], j)) {
          edges.push({ cwIdx: ci, colIdx: j });
        }
      }
    }

    // Layout parameters
    var margin = { top: 40, bottom: 20, left: 20, right: 20 };
    var colY = margin.top;
    var colSpacing = (width - margin.left - margin.right) / (N - 1);
    var colX = function (j) { return margin.left + j * colSpacing; };

    // Codeword node sizing
    var cwRadius;
    if (k <= 4) cwRadius = 8;
    else if (k <= 6) cwRadius = 5;
    else cwRadius = 3;

    // Compute codeword positions
    var cwPositions = computeCWPositions(codewords, k, cwRadius, colX, colY, margin);

    // Edge stroke opacity: lighter when more edges
    var edgeOpacity = Math.max(0.06, Math.min(0.25, 4.0 / Math.sqrt(edges.length + 1)));

    // Delay drawing until old content fades
    setTimeout(function () {
      drawEdges(edges, cwPositions, colX, colY, edgeOpacity);
      drawColumnNodes(colActive, colX, colY);
      drawCodewordNodes(codewords, cwPositions, cwRadius, k, edges, colX, colY, edgeOpacity);
    }, TRANSITION_MS / 2 + 20);
  }

  // ── Compute codeword positions ─────────────────────────────────────

  function computeCWPositions(codewords, k, cwRadius, colX, colY, margin) {
    var positions = [];

    if (k <= 4) {
      // Force simulation layout
      positions = forceLayout(codewords, k, cwRadius, colX, colY, margin);
    } else {
      // Row-based layout grouped by Hamming weight
      positions = rowLayout(codewords, cwRadius, margin);
    }

    return positions;
  }

  // Force-directed layout for small codes (k <= 4, up to 15 codewords)
  function forceLayout(codewords, k, cwRadius, colX, colY, margin) {
    var cwAreaTop = colY + 50;
    var cwAreaBottom = height - margin.bottom - 10;
    var cwAreaCenterY = (cwAreaTop + cwAreaBottom) / 2;
    var centerX = width / 2;

    // Initialize positions: spread based on center of mass of connected columns
    var nodes = codewords.map(function (cw, i) {
      var bits = [];
      for (var j = 0; j < N; j++) {
        if (hasBit(cw, j)) bits.push(j);
      }
      var avgX = bits.length > 0
        ? bits.reduce(function (s, j) { return s + colX(j); }, 0) / bits.length
        : centerX;
      return {
        x: avgX + (Math.random() - 0.5) * 30,
        y: cwAreaCenterY + (Math.random() - 0.5) * 40,
        cw: cw
      };
    });

    // Run a quick force simulation synchronously
    var sim = d3.forceSimulation(nodes)
      .force('y', d3.forceY(cwAreaCenterY).strength(0.05))
      .force('x', d3.forceX(centerX).strength(0.02))
      .force('collide', d3.forceCollide(cwRadius + 4))
      .force('charge', d3.forceManyBody().strength(-30))
      .stop();

    // Tick synchronously
    for (var i = 0; i < 120; i++) sim.tick();

    // Clamp positions within bounds
    return nodes.map(function (n) {
      return {
        x: Math.max(margin.left + cwRadius, Math.min(width - margin.right - cwRadius, n.x)),
        y: Math.max(cwAreaTop, Math.min(cwAreaBottom, n.y))
      };
    });
  }

  // Row layout grouped by Hamming weight for larger codes
  function rowLayout(codewords, cwRadius, margin) {
    // Group by weight
    var groups = {};
    var weights = [];
    codewords.forEach(function (cw, idx) {
      var w = weight(cw);
      if (!groups[w]) {
        groups[w] = [];
        weights.push(w);
      }
      groups[w].push(idx);
    });
    weights.sort(function (a, b) { return a - b; });

    var cwAreaTop = margin.top + 55;
    var cwAreaBottom = height - margin.bottom - 10;
    var availHeight = cwAreaBottom - cwAreaTop;
    var rowSpacing = weights.length > 1
      ? availHeight / (weights.length - 1)
      : availHeight / 2;

    // Cap row spacing to avoid overly spread layouts
    rowSpacing = Math.min(rowSpacing, 80);
    var totalRowHeight = (weights.length - 1) * rowSpacing;
    var startY = cwAreaTop + (availHeight - totalRowHeight) / 2;

    var positions = new Array(codewords.length);
    var usableWidth = width - margin.left - margin.right;

    weights.forEach(function (w, rowIdx) {
      var indices = groups[w];
      var rowY = startY + rowIdx * rowSpacing;
      var count = indices.length;
      var spacing = count > 1
        ? Math.min(usableWidth / (count - 1), cwRadius * 3 + 4)
        : 0;
      var totalW = (count - 1) * spacing;
      var startX = (width - totalW) / 2;

      indices.forEach(function (cwIdx, posInRow) {
        positions[cwIdx] = {
          x: startX + posInRow * spacing,
          y: rowY
        };
      });
    });

    return positions;
  }

  // ── Draw edges ─────────────────────────────────────────────────────

  function drawEdges(edges, cwPositions, colX, colY, edgeOpacity) {
    var lines = gEdges.selectAll('line')
      .data(edges);

    lines.enter()
      .append('line')
      .attr('x1', function (d) { return cwPositions[d.cwIdx].x; })
      .attr('y1', function (d) { return cwPositions[d.cwIdx].y; })
      .attr('x2', function (d) { return colX(d.colIdx); })
      .attr('y2', colY)
      .attr('stroke', '#6cf')
      .attr('stroke-width', 0.8)
      .attr('stroke-opacity', 0)
      .attr('data-cw-idx', function (d) { return d.cwIdx; })
      .attr('data-col-idx', function (d) { return d.colIdx; })
      .transition()
      .delay(TRANSITION_MS / 4)
      .duration(TRANSITION_MS)
      .attr('stroke-opacity', edgeOpacity);
  }

  // ── Draw column nodes ──────────────────────────────────────────────

  function drawColumnNodes(colActive, colX, colY) {
    var colData = [];
    for (var j = 0; j < N; j++) {
      colData.push({ idx: j, active: colActive[j] });
    }

    var groups = gCols.selectAll('g.col-node')
      .data(colData)
      .enter()
      .append('g')
      .attr('class', 'col-node')
      .attr('transform', function (d) {
        return 'translate(' + colX(d.idx) + ',' + colY + ')';
      })
      .style('opacity', 0);

    groups.append('circle')
      .attr('r', COL_RADIUS)
      .attr('fill', function (d) { return d.active ? COL_COLOR : COL_COLOR_DIM; })
      .attr('stroke', function (d) { return d.active ? '#4dd0e1' : '#555'; })
      .attr('stroke-width', 1.5);

    groups.append('text')
      .attr('y', -15)
      .attr('text-anchor', 'middle')
      .attr('font-size', '9px')
      .attr('font-family', 'monospace')
      .attr('fill', function (d) { return d.active ? '#aaa' : '#444'; })
      .text(function (d) { return d.idx; });

    groups
      .transition()
      .delay(TRANSITION_MS / 4)
      .duration(TRANSITION_MS)
      .style('opacity', 1);

    // Hover on column: highlight connected edges
    groups
      .on('mouseenter', function (event, d) {
        var colIdx = d.idx;
        gEdges.selectAll('line')
          .attr('stroke-opacity', function (e) {
            return e.colIdx === colIdx ? 0.7 : 0.02;
          })
          .attr('stroke', function (e) {
            return e.colIdx === colIdx ? '#4dd0e1' : '#6cf';
          })
          .attr('stroke-width', function (e) {
            return e.colIdx === colIdx ? 1.5 : 0.8;
          });

        // Highlight connected codeword nodes
        var connectedCwSet = {};
        gEdges.selectAll('line').each(function (e) {
          if (e.colIdx === colIdx) connectedCwSet[e.cwIdx] = true;
        });
        gCwords.selectAll('circle.cw-node')
          .attr('stroke', function (cw) {
            return connectedCwSet[cw.idx] ? '#fff' : 'none';
          })
          .attr('stroke-width', function (cw) {
            return connectedCwSet[cw.idx] ? 1.5 : 0;
          });
      })
      .on('mouseleave', function () {
        resetHighlights();
      });
  }

  // ── Draw codeword nodes ────────────────────────────────────────────

  function drawCodewordNodes(codewords, cwPositions, cwRadius, k, edges, colX, colY, edgeOpacity) {
    var cwData = codewords.map(function (cw, i) {
      return {
        idx: i,
        value: cw,
        weight: weight(cw),
        x: cwPositions[i].x,
        y: cwPositions[i].y
      };
    });

    var circles = gCwords.selectAll('circle.cw-node')
      .data(cwData)
      .enter()
      .append('circle')
      .attr('class', 'cw-node')
      .attr('cx', function (d) { return d.x; })
      .attr('cy', function (d) { return d.y; })
      .attr('r', cwRadius)
      .attr('fill', function (d) { return weightColor(d.weight); })
      .attr('stroke', 'none')
      .attr('stroke-width', 0)
      .style('cursor', 'pointer')
      .style('opacity', 0);

    circles
      .transition()
      .delay(TRANSITION_MS / 4)
      .duration(TRANSITION_MS)
      .style('opacity', 1);

    // Hover interactions
    circles
      .on('mouseenter', function (event, d) {
        var cwIdx = d.idx;

        // Highlight edges connected to this codeword
        gEdges.selectAll('line')
          .attr('stroke-opacity', function (e) {
            return e.cwIdx === cwIdx ? 0.7 : 0.02;
          })
          .attr('stroke', function (e) {
            return e.cwIdx === cwIdx ? weightColor(d.weight) : '#6cf';
          })
          .attr('stroke-width', function (e) {
            return e.cwIdx === cwIdx ? 1.8 : 0.8;
          });

        // Highlight connected column nodes
        var connectedCols = {};
        gEdges.selectAll('line').each(function (e) {
          if (e.cwIdx === cwIdx) connectedCols[e.colIdx] = true;
        });
        gCols.selectAll('g.col-node circle')
          .attr('stroke', function (col) {
            return connectedCols[col.idx] ? '#fff' : (col.active ? '#4dd0e1' : '#555');
          })
          .attr('stroke-width', function (col) {
            return connectedCols[col.idx] ? 2.5 : 1.5;
          });

        // Brighten hovered node
        d3.select(this)
          .attr('stroke', '#fff')
          .attr('stroke-width', 2)
          .attr('r', cwRadius + 2);

        // Show tooltip
        var containerRect = svg.node().parentNode.getBoundingClientRect();
        var svgRect = svg.node().getBoundingClientRect();
        var scaleX = svgRect.width / width;
        var scaleY = svgRect.height / height;
        var tipX = d.x * scaleX + (svgRect.left - containerRect.left) + 15;
        var tipY = d.y * scaleY + (svgRect.top - containerRect.top) - 10;

        gTooltip
          .html(
            '<span style="color:' + weightColor(d.weight) + ';">█</span> ' +
            'wt=' + d.weight +
            '  <span style="color:#888;">' + binaryStr(d.value, N) + '</span>' +
            '  ' + hexStr(d.value)
          )
          .style('left', tipX + 'px')
          .style('top', tipY + 'px')
          .style('opacity', 1);
      })
      .on('mouseleave', function () {
        d3.select(this)
          .attr('stroke', 'none')
          .attr('stroke-width', 0)
          .attr('r', cwRadius);

        resetHighlights();
        gTooltip.style('opacity', 0);
      });
  }

  // ── Reset edge/node highlights ─────────────────────────────────────

  function resetHighlights() {
    // Edges: restore to current base opacity
    var edgeCount = gEdges.selectAll('line').size();
    var baseOpacity = Math.max(0.06, Math.min(0.25, 4.0 / Math.sqrt(edgeCount + 1)));

    gEdges.selectAll('line')
      .attr('stroke-opacity', baseOpacity)
      .attr('stroke', '#6cf')
      .attr('stroke-width', 0.8);

    // Column nodes: restore normal stroke
    gCols.selectAll('g.col-node circle')
      .attr('stroke', function (d) { return d.active ? '#4dd0e1' : '#555'; })
      .attr('stroke-width', 1.5);

    // Codeword nodes: remove highlight stroke
    gCwords.selectAll('circle.cw-node')
      .attr('stroke', 'none')
      .attr('stroke-width', 0);
  }

  // ── Resize handler ─────────────────────────────────────────────────

  function resize(container) {
    if (!svg) return;
    var rect = container.getBoundingClientRect();
    width = rect.width || 500;
    height = rect.height || 400;
    svg.attr('viewBox', '0 0 ' + width + ' ' + height);
    if (currentCode) {
      update(currentCode);
    }
  }

  // ── Export ──────────────────────────────────────────────────────────

  window.GraphPanel = {
    render: render,
    update: update,
    resize: resize
  };
})();
