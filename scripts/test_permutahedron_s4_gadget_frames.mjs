#!/usr/bin/env node

// Independent exact census for the fixed-order S4 Gadget-frame artifact.
// This reconstructs all 393,216 Boolean assignments from the six unsigned
// quartets and does not import the Rust implementation or its frame count.

import fs from "node:fs";

const artifactPath = process.argv[2] ?? "data/permutahedron_s4_gadget_frames.json";
const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));

const quartets = [
  [[1,4,2,3],[2,3,1,4],[3,2,4,1],[4,1,3,2]],
  [[1,3,4,2],[2,4,3,1],[3,1,2,4],[4,2,1,3]],
  [[1,3,2,4],[2,4,1,3],[3,1,4,2],[4,2,3,1]],
  [[1,4,3,2],[2,3,4,1],[3,2,1,4],[4,1,2,3]],
  [[1,2,4,3],[2,1,3,4],[3,4,2,1],[4,3,1,2]],
  [[1,2,3,4],[2,1,4,3],[3,4,1,2],[4,3,2,1]],
];

function assert(value, message) {
  if (!value) throw new Error(message);
}

function compose(a, b) {
  // Matrix product b*a, matching the sparse row-to-column convention.
  const permutation = Array(4);
  const signs = Array(4);
  for (let row = 0; row < 4; row++) {
    const middle = b.permutation[row];
    permutation[row] = a.permutation[middle];
    signs[row] = a.signs[middle] * b.signs[row];
  }
  return {permutation, signs};
}

function inverse(matrix) {
  const permutation = Array(4);
  const signs = Array(4);
  for (let row = 0; row < 4; row++) {
    permutation[matrix.permutation[row]] = row;
    signs[matrix.permutation[row]] = matrix.signs[row];
  }
  return {permutation, signs};
}

function negateEqual(a, b) {
  return a.permutation.every((value, row) => value === b.permutation[row]
    && a.signs[row] === -b.signs[row]);
}

function traceProduct(a, b) {
  let trace = 0;
  for (let row = 0; row < 4; row++) {
    const middle = a.permutation[row];
    if (b.permutation[middle] === row) trace += a.signs[row] * b.signs[middle];
  }
  return trace;
}

function signing(sector, packed) {
  const factors = Array.from({length: 4}, (_, color) => (packed >> (4 * color)) & 15);
  const matrices = quartets[sector].map((permutation, color) => ({
    permutation: permutation.map((entry) => entry - 1),
    signs: Array.from({length: 4}, (_, row) => factors[color] & (1 << row) ? -1 : 1),
  }));
  for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) {
    const left = compose(matrices[i], inverse(matrices[j]));
    const right = compose(matrices[j], inverse(matrices[i]));
    if (!negateEqual(left, right)) return null;
  }
  const holoraumy = [];
  for (let i = 1; i < 4; i++) for (let j = 0; j < i; j++) {
    holoraumy.push(compose(inverse(matrices[i]), matrices[j]));
  }
  return {factors, holoraumy};
}

function gadgetNumerator(left, right) {
  return -left.holoraumy.reduce((sum, matrix, index) =>
    sum + traceProduct(matrix, right.holoraumy[index]), 0);
}

const library = quartets.map((_, sector) => {
  const results = [];
  for (let packed = 0; packed < 65536; packed++) {
    const result = signing(sector, packed);
    if (result) results.push(result);
  }
  results.sort((a, b) => a.factors.join(",").localeCompare(b.factors.join(",")));
  return results;
});

assert(library.every((sector) => sector.length === 256), "expected 256 signings per sector");

function profileTypes(sector) {
  const types = new Map();
  library[sector].forEach((candidate, index) => {
    const profile = [];
    library.forEach((other, otherSector) => {
      if (otherSector !== sector) for (const signing of other) {
        profile.push(gadgetNumerator(candidate, signing));
      }
    });
    const key = profile.join(",");
    const current = types.get(key);
    if (current) {
      current.multiplicity++;
      current.members.push(index);
    } else types.set(key, {representative: index, multiplicity: 1n, members: [index]});
  });
  return [...types.values()];
}

const types = library.map((_, sector) => profileTypes(sector));
assert(types.every((sector) => sector.length === 16), "expected sixteen compatibility types per sector");

function countFrames(sector, chosen, weight) {
  if (sector === 6) return weight;
  let total = 0n;
  for (let type = 0; type < types[sector].length; type++) {
    const candidate = library[sector][types[sector][type].representative];
    if (chosen.every(([otherSector, otherType]) =>
      gadgetNumerator(candidate, library[otherSector][types[otherSector][otherType].representative]) === 0)) {
      chosen.push([sector, type]);
      total += countFrames(sector + 1, chosen, weight * types[sector][type].multiplicity);
      chosen.pop();
    }
  }
  return total;
}

const rawFrames = countFrames(0, [], 1n);
const totalFrames = 256n ** 6n;
assert(rawFrames === 28862180229120n, "orthonormal frame count mismatch");
assert(rawFrames * 1024n === totalFrames * 105n, "frame fraction must be 105/1024");
assert(BigInt(artifact.validation.raw_orthonormal_frames) === rawFrames, "artifact frame count mismatch");
assert(BigInt(artifact.validation.common_vertex_switching_orbits) === rawFrames / 128n, "switching orbit count mismatch");
const weightedFactors = [
  [10,12,6,0], [14,4,8,2], [12,10,6,0],
  [6,3,10,0], [12,9,0,10], [12,5,0,6],
];
const weightedFrame = weightedFactors.map((factors, sector) =>
  signing(sector, factors.reduce((packed, factor, color) => packed | (factor << (4 * color)), 0)));
const weightedMatrix = weightedFrame.map((left) => weightedFrame.map((right) => gadgetNumerator(left, right)));
const expectedWeighted = [
  [24,0,0,0,8,0], [0,24,-8,0,0,0], [0,-8,24,0,0,0],
  [0,0,0,24,0,0], [8,0,0,0,24,0], [0,0,0,0,0,24],
];
assert(JSON.stringify(weightedMatrix) === JSON.stringify(expectedWeighted), "literal weighted Table 5 matrix mismatch");

function factorDistance(left, right) {
  return left.reduce((sum, value, color) => sum + (value ^ right[color]).toString(2).replaceAll("0", "").length, 0);
}
let bestRepair = Infinity;
let bestRepairCount = 0n;
function searchRepairs(sector, chosen, cost, count) {
  if (sector === 6) {
    if (cost < bestRepair) {
      bestRepair = cost;
      bestRepairCount = count;
    } else if (cost === bestRepair) bestRepairCount += count;
    return;
  }
  if (cost > bestRepair) return;
  for (let type = 0; type < types[sector].length; type++) {
    const candidate = library[sector][types[sector][type].representative];
    if (!chosen.every(([otherSector, otherType]) =>
      gadgetNumerator(candidate, library[otherSector][types[otherSector][otherType].representative]) === 0)) continue;
    const distances = types[sector][type].members.map((member) =>
      factorDistance(library[sector][member].factors, weightedFactors[sector]));
    const minimum = Math.min(...distances);
    const minimumCount = BigInt(distances.filter((distance) => distance === minimum).length);
    chosen.push([sector, type]);
    searchRepairs(sector + 1, chosen, cost + minimum, count * minimumCount);
    chosen.pop();
  }
}
searchRepairs(0, [], 0, 1n);
assert(bestRepair === 8, "minimum Table 5 repair distance mismatch");
assert(bestRepairCount === 80n, "minimum Table 5 repair count mismatch");
assert(artifact.nearest_orthonormal_repair_of_literal_table5.total_boolean_bit_flips === bestRepair, "artifact repair distance mismatch");
assert(BigInt(artifact.nearest_orthonormal_repair_of_literal_table5.minimum_repair_frames) === bestRepairCount, "artifact repair count mismatch");
assert(artifact.validation.literal_weighted_table5_frame_is_orthonormal === false, "literal Table 5 frame must retain the discrepancy");
assert(artifact.validation.appendix_b_reference_frame_is_orthonormal === true, "Appendix-B reference frame must be orthonormal");
assert(artifact.validation.passed === true, "Rust validation failed");

const mutation = structuredClone(artifact.appendix_b_orthonormal_reference_frame.gadget_numerators_over_24);
mutation[0][1] = 8;
assert(!mutation.every((row, i) => row.every((value, j) => value === (i === j ? 24 : 0))), "mutation must break orthonormality");

console.log(`verified 393216 Boolean assignments and ${rawFrames} orthonormal frames (${Number(rawFrames * 1000000n / totalFrames) / 10000}%)`);
