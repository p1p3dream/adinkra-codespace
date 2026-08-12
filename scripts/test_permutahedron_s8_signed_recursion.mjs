#!/usr/bin/env node

// Independent artifact check for the signed thirty-pair recursion. This file
// does not import the Rust implementation or trust its closure flags.

import fs from "node:fs";

const artifactPath = process.argv[2] ?? "data/permutahedron_s8_signed_recursion.json";
const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));

const signedWords = [
  [[1,-4,2,-3],[2,3,-1,-4],[3,-2,-4,1],[4,1,3,2]],
  [[1,-3,-4,-2],[2,4,-3,1],[3,1,2,-4],[4,-2,1,3]],
  [[1,3,-2,-4],[2,-4,1,-3],[3,-1,-4,2],[4,2,3,1]],
  [[1,-4,-3,2],[-2,-3,4,1],[3,-2,1,-4],[4,1,2,3]],
  [[1,2,-4,-3],[-2,1,3,-4],[3,4,2,1],[4,-3,1,-2]],
  [[1,2,-3,-4],[-2,1,-4,3],[3,4,1,2],[4,-3,-2,1]],
];

const labels = ["CM", "TM", "VM", "VM1", "VM2", "VM3"];
const factors = signedWords.map((sector) => sector.map((word) =>
  word.reduce((factor, entry, row) => factor | (entry < 0 ? 1 << row : 0), 0)
));
const permutations = signedWords.map((sector) => sector.map((word) => word.map(Math.abs)));

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function multiply(left, right) {
  return left.map((_, row) => right[0].map((__, column) =>
    left[row].reduce((sum, value, inner) => sum + value * right[inner][column], 0)
  ));
}

function transpose(matrix) {
  return matrix[0].map((_, column) => matrix.map((row) => row[column]));
}

function dense(permutation, factor) {
  return permutation.map((target, row) => {
    const values = Array(8).fill(0);
    values[target - 1] = (factor & (1 << row)) === 0 ? 1 : -1;
    return values;
  });
}

function closes(candidate) {
  const l = candidate.permutations.map((permutation, color) =>
    dense(permutation, candidate.boolean_factors[color])
  );
  const r = l.map(transpose);
  for (let first = 0; first < 8; first++) {
    for (let second = 0; second < 8; second++) {
      for (const [left, right] of [[l, r], [r, l]]) {
        const a = multiply(left[first], right[second]);
        const b = multiply(left[second], right[first]);
        for (let row = 0; row < 8; row++) {
          for (let column = 0; column < 8; column++) {
            const expected = first === second && row === column ? 2 : 0;
            if (a[row][column] + b[row][column] !== expected) return false;
          }
        }
      }
    }
  }
  return true;
}

function cyclicMask(start) {
  const positions = Array.from({length: 4}, (_, offset) => (start + offset) % 8);
  return positions.reduce((mask, position) => mask | (1 << position), 0);
}

function recursivePermutations(first, second) {
  return Array.from({length: 8}, (_, color) => {
    const local = color % 4;
    if (color < 4) {
      return [...permutations[first][local], ...permutations[second][local].map((x) => x + 4)];
    }
    return [...permutations[first][local].map((x) => x + 4), ...permutations[second][local]];
  });
}

function recursiveFactors(first, second, mask) {
  const base = Array.from({length: 4}, (_, color) =>
    factors[first][color] | (factors[second][color] << 4)
  );
  return [...base, ...base.map((factor) => factor ^ mask)];
}

assert(artifact.schema_version === "permutahedron-s8-signed-recursion-v1", "schema mismatch");
assert(JSON.stringify(artifact.source_s4_labels) === JSON.stringify(labels), "source label mismatch");
assert(JSON.stringify(artifact.source_s4_boolean_factors) === JSON.stringify(factors), "source factor mismatch");
assert(permutations[4].some((p) => p.join("") === "3421"), "VM2 must contain 3421");
assert(!permutations[4].some((p) => p.join("") === "3412"), "VM2 must not contain 3412");
assert(artifact.ordered_pairs.length === 30, "expected thirty ordered pairs");

const observedClosing = [];
for (const pair of artifact.ordered_pairs) {
  const first = labels.indexOf(pair.first_label);
  const second = labels.indexOf(pair.second_label);
  assert(first >= 0 && second >= 0 && first !== second, "invalid ordered pair");
  assert(pair.candidates.length === 8, "each pair must have eight flip candidates");
  const expectedPermutations = recursivePermutations(first, second);
  const closingStarts = [];
  pair.candidates.forEach((candidate, start) => {
    const mask = cyclicMask(start);
    assert(candidate.start_position_one_based === start + 1, "start position mismatch");
    assert(candidate.flip_mask_decimal === mask, "cyclic mask mismatch");
    assert(JSON.stringify(candidate.permutations) === JSON.stringify(expectedPermutations), "recursive permutation mismatch");
    assert(JSON.stringify(candidate.boolean_factors) === JSON.stringify(recursiveFactors(first, second, mask)), "recursive Boolean mismatch");
    const independentClosure = closes(candidate);
    assert(independentClosure === candidate.closure.sparse_garden_passed, "closure flag mismatch");
    assert(independentClosure === (candidate.closure.residual_entries === 0), "closure residual mismatch");
    if (independentClosure) closingStarts.push(start + 1);
  });
  assert(JSON.stringify(closingStarts) === JSON.stringify(pair.closing_start_positions_one_based), "closing mask list mismatch");
  if (closingStarts.length) observedClosing.push(`${pair.first_label}->${pair.second_label}:${closingStarts.join(",")}`);
}

const expectedClosing = [
  "CM->TM:2,6", "CM->VM:2,6", "TM->CM:2,6", "VM->CM:2,6",
  "VM1->VM2:2,6", "VM1->VM3:2,6", "VM2->VM1:2,6", "VM3->VM1:2,6",
];
assert(JSON.stringify(observedClosing) === JSON.stringify(expectedClosing), "closing pair census mismatch");
assert(artifact.validation.closing_candidates === 16, "closing candidate total mismatch");
assert(artifact.validation.passed === true, "Rust validation did not pass");

console.log(`verified ${artifact.ordered_pairs.length} ordered pairs, 240 signed candidates, and 16 exact closures`);
