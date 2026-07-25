const HEADER_BYTES = 20;

export function popcount16(value) {
  value &= 0xffff;
  value = value - ((value >>> 1) & 0x5555);
  value = (value & 0x3333) + ((value >>> 2) & 0x3333);
  value = (value + (value >>> 4)) & 0x0f0f;
  value += value >>> 8;
  return value & 0x1f;
}

export function rref(generators, n) {
  const rows = generators.map((value) => value >>> 0).filter(Boolean);
  let current = 0;
  for (let column = 0; current < rows.length && column < n; column += 1) {
    const bit = 1 << column;
    const pivot = rows.findIndex((row, index) => index >= current && (row & bit) !== 0);
    if (pivot < 0) continue;
    [rows[current], rows[pivot]] = [rows[pivot], rows[current]];
    for (let index = 0; index < rows.length; index += 1) {
      if (index !== current && (rows[index] & bit) !== 0) rows[index] ^= rows[current];
    }
    current += 1;
  }
  return rows.filter(Boolean).sort((a, b) => a - b);
}

export function reduceWord(value, rows, pivots) {
  let reduced = value;
  for (let index = 0; index < rows.length; index += 1) {
    if ((reduced & (1 << pivots[index])) !== 0) reduced ^= rows[index];
  }
  return reduced;
}

export function parseDashingAsset(buffer) {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < HEADER_BYTES) throw new Error('Dashing asset is shorter than its header');
  const magic = String.fromCharCode(...bytes.subarray(0, 4));
  if (magic !== 'AD3D') throw new Error(`Invalid dashing asset magic: ${magic}`);
  const view = new DataView(buffer);
  const version = view.getUint16(4, true);
  if (version !== 1) throw new Error(`Unsupported dashing asset version: ${version}`);
  const n = bytes[6];
  const k = bytes[7];
  const codeIndex = view.getUint16(8, true);
  const vertices = view.getUint32(12, true);
  const edges = view.getUint32(16, true);
  if (bytes.length !== HEADER_BYTES + 2 * edges) {
    throw new Error(`Dashing asset length ${bytes.length} does not match ${edges} edges`);
  }
  const descriptors = bytes.subarray(HEADER_BYTES);
  return { version, n, k, codeIndex, vertices, edges, descriptors };
}

export function edgeIsDashed(base, mask, dashingClass) {
  return (base ^ (popcount16(mask & dashingClass) & 1)) !== 0;
}

export function buildTopology(n, generators, asset) {
  const rows = rref(generators, n);
  const pivots = rows.map((row) => Math.log2(row & -row));
  const pivotSet = new Set(pivots);
  const free = Array.from({ length: n }, (_, index) => index).filter((index) => !pivotSet.has(index));
  const vertexCount = 1 << free.length;
  if (vertexCount !== asset.vertices) {
    throw new Error(`Catalog gives ${vertexCount} vertices; asset gives ${asset.vertices}`);
  }

  const reps = new Uint16Array(vertexCount);
  for (let mask = 0; mask < vertexCount; mask += 1) {
    let value = 0;
    for (let bit = 0; bit < free.length; bit += 1) {
      if ((mask & (1 << bit)) !== 0) value |= 1 << free[bit];
    }
    reps[mask] = reduceWord(value, rows, pivots);
  }
  reps.sort();

  const repToVertex = new Int32Array(1 << n);
  repToVertex.fill(-1);
  for (let index = 0; index < reps.length; index += 1) repToVertex[reps[index]] = index;

  const edgeA = new Uint16Array(asset.edges);
  const edgeB = new Uint16Array(asset.edges);
  const edgeColor = new Uint8Array(asset.edges);
  const edgeIndexByVertexColor = new Uint32Array(vertexCount * n);
  edgeIndexByVertexColor.fill(0xffffffff);
  let edge = 0;

  for (let color = 0; color < n; color += 1) {
    for (let vertex = 0; vertex < vertexCount; vertex += 1) {
      const neighborRep = reduceWord(reps[vertex] ^ (1 << color), rows, pivots);
      const neighbor = repToVertex[neighborRep];
      if (neighbor < 0) throw new Error(`Missing quotient representative ${neighborRep}`);
      if (vertex >= neighbor) continue;
      if (edge >= asset.edges) throw new Error('Catalog topology has more edges than the dashing asset');
      edgeA[edge] = vertex;
      edgeB[edge] = neighbor;
      edgeColor[edge] = color;
      edgeIndexByVertexColor[vertex * n + color] = edge;
      edgeIndexByVertexColor[neighbor * n + color] = edge;
      edge += 1;
    }
  }
  if (edge !== asset.edges) throw new Error(`Catalog topology has ${edge} edges; asset gives ${asset.edges}`);

  return {
    n,
    k: rows.length,
    reps,
    edgeA,
    edgeB,
    edgeColor,
    edgeIndexByVertexColor,
    baseAt: (edgeIndex) => asset.descriptors[2 * edgeIndex],
    maskAt: (edgeIndex) => asset.descriptors[2 * edgeIndex + 1],
  };
}

export function otherEndpoint(topology, edge, vertex) {
  return topology.edgeA[edge] === vertex ? topology.edgeB[edge] : topology.edgeA[edge];
}

export function sampledEdge(edge, totalEdges, budget) {
  if (!Number.isFinite(budget) || budget >= totalEdges) return true;
  const stride = Math.ceil(totalEdges / budget);
  return edge % stride === 0;
}
