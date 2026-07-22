/**
 * rotor7.mjs
 *
 * Pure math for the 7D permutahedron embedding and SO(7) rotor.
 *
 * The S8 permutahedron lives in R^8: each vertex is a point whose
 * coordinates are a permutation of (1..8). All such points satisfy
 * sum = 36, so the object is genuinely 7 dimensional. This module
 * builds an exact orthonormal basis of that hyperplane, embeds every
 * vertex as a 7D coordinate, and composes SO(7) rotations from the
 * 21 coordinate-plane angles (C(7,2) = 21).
 *
 * Embedding convention (important, and verified by the test suite):
 * the dataset's edges are right multiplication by adjacent
 * transpositions (adjacent-position swaps). Embedding a permutation p
 * by the tuple of its INVERSE (the position of value v, for v = 1..n)
 * turns every such edge into a swap of two consecutive values in the
 * embedded tuple, which is exactly a geometric permutohedron edge of
 * Euclidean length sqrt(2). With the direct (non-inverse) tuple the
 * same graph is drawn with wildly unequal edge lengths. So we embed
 * by inverse. This is a relabeling of the same abstract object, not a
 * different object.
 *
 * No rendering code here. Everything is unit-testable in node.
 */

export const DIM = 7;
export const PLANE_COUNT = 21; // C(7,2)

/**
 * Orthonormal (Helmert) basis of the hyperplane { x in R^n : sum x = 0 }.
 * Basis vector k (k = 1..n-1) has first k entries 1/sqrt(k(k+1)),
 * entry k+1 equal to -k/sqrt(k(k+1)), zeros after. Rows are exactly
 * orthonormal, no Gram-Schmidt drift.
 * Returns an array of (n-1) Float64Array rows of length n.
 */
export function hyperplaneBasis(n) {
  const rows = [];
  for (let k = 1; k < n; k += 1) {
    const row = new Float64Array(n);
    const norm = 1 / Math.sqrt(k * (k + 1));
    for (let i = 0; i < k; i += 1) row[i] = norm;
    row[k] = -k * norm;
    rows.push(row);
  }
  return rows;
}

/** Inverse of a permutation given as an array of values 1..n. */
export function inverseTuple(perm) {
  const inv = new Array(perm.length);
  for (let i = 0; i < perm.length; i += 1) inv[perm[i] - 1] = i + 1;
  return inv;
}

/**
 * Embed a list of permutations (strings like "12345678" or arrays of
 * values 1..n) into 7D coordinates. For n < 8 the trailing dimensions
 * are zero, so the same buffers and shaders serve S4 and S8.
 *
 * Steps per vertex: take the inverse tuple, subtract the centroid
 * ((n+1)/2 in every coordinate), then take inner products with the
 * n-1 hyperplane basis rows.
 *
 * Returns { coords: Float64Array(count*7), count, n, dim: n-1 }.
 */
export function embedPermutations(perms, n) {
  const basis = hyperplaneBasis(n);
  const count = perms.length;
  const coords = new Float64Array(count * DIM);
  const centered = new Float64Array(n);
  const mid = (n + 1) / 2;
  for (let v = 0; v < count; v += 1) {
    const raw = perms[v];
    const tuple = typeof raw === "string"
      ? Array.from(raw, ch => Number(ch))
      : raw;
    const inv = inverseTuple(tuple);
    for (let i = 0; i < n; i += 1) centered[i] = inv[i] - mid;
    const base = v * DIM;
    for (let k = 0; k < n - 1; k += 1) {
      const row = basis[k];
      let dot = 0;
      for (let i = 0; i < n; i += 1) dot += row[i] * centered[i];
      coords[base + k] = dot;
    }
    // dims (n-1)..6 stay zero for n < 8
  }
  return { coords, count, n, dim: n - 1 };
}

/** The 21 coordinate planes (i, j), i < j, in a fixed order. */
export function planePairs() {
  const pairs = [];
  for (let i = 0; i < DIM; i += 1) {
    for (let j = i + 1; j < DIM; j += 1) pairs.push([i, j]);
  }
  return pairs;
}

const PAIRS = planePairs();

/** 7x7 identity as Float64Array(49), row major. */
export function identity7() {
  const m = new Float64Array(49);
  for (let i = 0; i < DIM; i += 1) m[i * DIM + i] = 1;
  return m;
}

/**
 * In-place right multiplication of row-major 7x7 matrix m by the
 * Givens rotation G(i, j, theta). Equivalent to rotating columns i, j.
 */
function applyGivensRight(m, i, j, theta) {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  for (let r = 0; r < DIM; r += 1) {
    const a = m[r * DIM + i];
    const b = m[r * DIM + j];
    m[r * DIM + i] = a * c + b * s;
    m[r * DIM + j] = -a * s + b * c;
  }
}

/**
 * Compose the SO(7) rotor from 21 plane angles, one per coordinate
 * plane in planePairs() order. Product of Givens rotations, so the
 * result is exactly orthogonal with determinant +1 (up to float eps).
 * Returns row-major Float64Array(49).
 */
export function composeSO7(angles) {
  if (!angles || angles.length !== PLANE_COUNT) {
    throw new RangeError(`expected ${PLANE_COUNT} plane angles, got ${angles ? angles.length : 0}`);
  }
  const m = identity7();
  for (let k = 0; k < PLANE_COUNT; k += 1) {
    const theta = angles[k];
    if (theta !== 0) applyGivensRight(m, PAIRS[k][0], PAIRS[k][1], theta);
  }
  return m;
}

/** Multiply two row-major 7x7 matrices. */
export function matMul7(a, b) {
  const out = new Float64Array(49);
  for (let r = 0; r < DIM; r += 1) {
    for (let c = 0; c < DIM; c += 1) {
      let sum = 0;
      for (let k = 0; k < DIM; k += 1) sum += a[r * DIM + k] * b[k * DIM + c];
      out[r * DIM + c] = sum;
    }
  }
  return out;
}

/** Transpose of a row-major 7x7 matrix. */
export function transpose7(m) {
  const out = new Float64Array(49);
  for (let r = 0; r < DIM; r += 1) {
    for (let c = 0; c < DIM; c += 1) out[c * DIM + r] = m[r * DIM + c];
  }
  return out;
}

/** Determinant of a row-major 7x7 matrix via LU with partial pivoting. */
export function det7(m) {
  const a = Float64Array.from(m);
  let det = 1;
  for (let col = 0; col < DIM; col += 1) {
    let pivot = col;
    for (let r = col + 1; r < DIM; r += 1) {
      if (Math.abs(a[r * DIM + col]) > Math.abs(a[pivot * DIM + col])) pivot = r;
    }
    if (Math.abs(a[pivot * DIM + col]) < 1e-14) return 0;
    if (pivot !== col) {
      for (let c = 0; c < DIM; c += 1) {
        const t = a[col * DIM + c];
        a[col * DIM + c] = a[pivot * DIM + c];
        a[pivot * DIM + c] = t;
      }
      det = -det;
    }
    const d = a[col * DIM + col];
    det *= d;
    for (let r = col + 1; r < DIM; r += 1) {
      const f = a[r * DIM + col] / d;
      for (let c = col; c < DIM; c += 1) a[r * DIM + c] -= f * a[col * DIM + c];
    }
  }
  return det;
}

/** Max |A - B| entrywise for equal-length arrays. */
export function maxAbsDiff(a, b) {
  let max = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = Math.abs(a[i] - b[i]);
    if (d > max) max = d;
  }
  return max;
}

/** Max |R^T R - I| entrywise, a direct orthogonality residual. */
export function orthogonalityResidual(m) {
  return maxAbsDiff(matMul7(transpose7(m), m), identity7());
}

/**
 * Idle rotor angles at time t (seconds). Incommensurate angular
 * speeds across a handful of the 21 planes give a slow, never
 * repeating drift through SO(7). speed scales everything.
 */
export function idleAngles(t, speed = 1) {
  const angles = new Float64Array(PLANE_COUNT);
  // Small set of planes with pairwise irrational speed ratios.
  angles[0] = 0.031 * speed * t;            // plane (0,1)
  angles[3] = 0.019 * Math.SQRT2 * speed * t; // plane (0,4)
  angles[7] = 0.023 * Math.sqrt(3) * speed * t; // plane (1,3)
  angles[12] = 0.017 * Math.sqrt(5) * speed * t; // plane (2,5)
  angles[18] = 0.013 * Math.sqrt(7) * speed * t; // plane (4,6)
  return angles;
}

/**
 * Per-slice 7D centroids. sliceByRank maps vertex index to slice id.
 * Returns Float64Array(sliceCount * 7).
 */
export function sliceCentroids(coords, sliceByRank, sliceCount) {
  const centroids = new Float64Array(sliceCount * DIM);
  const members = new Uint32Array(sliceCount);
  const count = sliceByRank.length;
  for (let v = 0; v < count; v += 1) {
    const s = sliceByRank[v];
    members[s] += 1;
    for (let k = 0; k < DIM; k += 1) centroids[s * DIM + k] += coords[v * DIM + k];
  }
  for (let s = 0; s < sliceCount; s += 1) {
    if (members[s] === 0) continue;
    for (let k = 0; k < DIM; k += 1) centroids[s * DIM + k] /= members[s];
  }
  return centroids;
}

/** Euclidean length of edge (a, b) in the 7D embedding. */
export function edgeLength7(coords, a, b) {
  let sum = 0;
  for (let k = 0; k < DIM; k += 1) {
    const d = coords[a * DIM + k] - coords[b * DIM + k];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

/**
 * Min and max edge length over an edge list of [source, target, ...]
 * entries. For a genuine permutohedron embedding both should equal
 * sqrt(2) to float precision.
 */
export function edgeLengthStats(coords, edges) {
  let min = Infinity;
  let max = -Infinity;
  for (const edge of edges) {
    const len = edgeLength7(coords, edge[0], edge[1]);
    if (len < min) min = len;
    if (len > max) max = len;
  }
  return { min, max };
}
