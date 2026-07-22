/** Pure permutation utilities used by the standalone permutahedron atlas. */

export function factorial(n) {
  let value = 1;
  for (let k = 2; k <= n; k += 1) value *= k;
  return value;
}

export function inversionCount(permutation) {
  let count = 0;
  for (let i = 0; i < permutation.length; i += 1) {
    for (let j = i + 1; j < permutation.length; j += 1) {
      if (permutation[i] > permutation[j]) count += 1;
    }
  }
  return count;
}

/** Lehmer-code rank in lexicographic order, zero based. */
export function rankPermutation(permutation) {
  const n = permutation.length;
  let rank = 0;
  for (let i = 0; i < n; i += 1) {
    let smaller = 0;
    for (let j = i + 1; j < n; j += 1) {
      if (permutation[j] < permutation[i]) smaller += 1;
    }
    rank += smaller * factorial(n - i - 1);
  }
  return rank;
}

export function unrankPermutation(rank, n) {
  if (!Number.isInteger(rank) || rank < 0 || rank >= factorial(n)) {
    throw new RangeError(`rank ${rank} is outside S_${n}`);
  }
  const available = Array.from({ length: n }, (_, i) => i + 1);
  const result = [];
  let remainder = rank;
  for (let i = n; i >= 1; i -= 1) {
    const block = factorial(i - 1);
    const index = Math.floor(remainder / block);
    remainder %= block;
    result.push(available.splice(index, 1)[0]);
  }
  return result;
}

export function inversePermutation(permutation) {
  const inverse = new Array(permutation.length);
  permutation.forEach((value, index) => { inverse[value - 1] = index + 1; });
  return inverse;
}

/**
 * Exact adjacent-transposition graph distance. This is Kendall tau distance,
 * equivalently Coxeter length of a^{-1}b and weak-Bruhat graph distance.
 */
export function kendallDistance(a, b) {
  if (a.length !== b.length) throw new RangeError("permutation sizes differ");
  const position = new Array(a.length + 1);
  a.forEach((value, index) => { position[value] = index + 1; });
  return inversionCount(b.map(value => position[value]));
}

export function stableId(n, rank) {
  return `S${n}-${String(rank).padStart(Math.max(5, String(factorial(n) - 1).length), "0")}`;
}

export function parseAddress(input, n) {
  const text = String(input ?? "").trim();
  if (!text) return null;
  const stable = text.match(/^S(\d+)[-:#]?0*(\d+)$/i);
  if (stable) {
    if (Number(stable[1]) !== n) return null;
    const rank = Number(stable[2]);
    return rank < factorial(n) ? { rank, permutation: unrankPermutation(rank, n) } : null;
  }
  if (/^\d+$/.test(text) && text.length !== n) {
    const rank = Number(text);
    return rank < factorial(n) ? { rank, permutation: unrankPermutation(rank, n) } : null;
  }
  const tokens = text.includes(",") || /\s/.test(text)
    ? text.split(/[\s,]+/).filter(Boolean).map(Number)
    : [...text].map(Number);
  if (tokens.length !== n || new Set(tokens).size !== n ||
      tokens.some(value => !Number.isInteger(value) || value < 1 || value > n)) return null;
  return { rank: rankPermutation(tokens), permutation: tokens };
}

export function normalizePermutation(entry, n, rank) {
  if (Array.isArray(entry)) return entry.map(Number);
  if (typeof entry === "string") {
    const parsed = parseAddress(entry, n);
    if (parsed) return parsed.permutation;
  }
  if (entry && Array.isArray(entry.permutation)) return entry.permutation.map(Number);
  return unrankPermutation(rank, n);
}

export function validateDataset(data) {
  const n = Number(data?.metadata?.n ?? data?.metadata?.order ?? data?.n);
  if (!Number.isInteger(n) || n < 2 || n > 10) throw new Error("metadata.n is missing or invalid");
  const vertexExpected = factorial(n);
  const edgeExpected = vertexExpected * (n - 1) / 2;
  const vertices = data.permutations?.length ?? 0;
  const edges = data.edges?.length ?? 0;
  const issues = [];
  if (vertices !== vertexExpected) issues.push(`expected ${vertexExpected.toLocaleString()} vertices, found ${vertices.toLocaleString()}`);
  if (edges !== edgeExpected) issues.push(`expected ${edgeExpected.toLocaleString()} edges, found ${edges.toLocaleString()}`);
  if (data.right_slice_by_rank && data.right_slice_by_rank.length !== vertices) issues.push("right_slice_by_rank length differs from vertex count");
  if (data.left_slice_by_rank && data.left_slice_by_rank.length !== vertices) issues.push("left_slice_by_rank length differs from vertex count");
  if (vertices === vertexExpected) {
    for (let rank = 0; rank < vertices; rank += 1) {
      const permutation = normalizePermutation(data.permutations[rank], n, rank);
      if (permutation.length !== n || rankPermutation(permutation) !== rank) {
        issues.push(`permutations are not in lexicographic rank order at rank ${rank}`);
        break;
      }
    }
  }
  if (edges === edgeExpected) {
    const degree = new Uint8Array(vertices);
    const seen = new Set();
    for (let index = 0; index < edges; index += 1) {
      const edge = data.edges[index];
      const source = Number(Array.isArray(edge) ? edge[0] : edge?.source);
      const target = Number(Array.isArray(edge) ? edge[1] : edge?.target);
      if (!Number.isInteger(source) || !Number.isInteger(target) || source < 0 || target < 0 || source >= vertices || target >= vertices || source === target) {
        issues.push(`edge ${index} has invalid endpoints`);
        break;
      }
      const low = Math.min(source, target), high = Math.max(source, target);
      const key = low * vertices + high;
      if (seen.has(key)) { issues.push(`edge ${index} duplicates an earlier edge`); break; }
      seen.add(key); degree[source] += 1; degree[target] += 1;
    }
    if (!issues.some(issue => issue.startsWith("edge"))) {
      const badDegree = degree.findIndex(value => value !== n - 1);
      if (badDegree >= 0) issues.push(`vertex ${badDegree} has degree ${degree[badDegree]}, expected ${n - 1}`);
    }
  }
  return { n, vertices, edges, vertexExpected, edgeExpected, complete: issues.length === 0, issues };
}

export function runSelfChecks() {
  const fixtures = [
    [[1, 2, 3, 4], 0, 0],
    [[1, 2, 4, 3], 1, 1],
    [[4, 3, 2, 1], 23, 6],
    [[2, 4, 1, 3], 10, 3],
  ];
  const checks = [];
  for (const [permutation, rank, inversions] of fixtures) {
    checks.push(rankPermutation(permutation) === rank);
    checks.push(inversionCount(permutation) === inversions);
    checks.push(unrankPermutation(rank, 4).join() === permutation.join());
  }
  checks.push(kendallDistance([1, 2, 3, 4], [4, 3, 2, 1]) === 6);
  checks.push(kendallDistance([2, 1, 3, 4], [1, 3, 2, 4]) === 2);
  checks.push(factorial(4) === 24 && factorial(8) === 40320);
  checks.push(factorial(4) * 3 / 2 === 36 && factorial(8) * 7 / 2 === 141120);
  checks.push(stableId(4, 0) === "S4-00000" && stableId(8, 40319) === "S8-40319");
  if (checks.some(value => !value)) throw new Error("permutation core self-check failed");
  return { passed: checks.length, fixtures: checks.length };
}
