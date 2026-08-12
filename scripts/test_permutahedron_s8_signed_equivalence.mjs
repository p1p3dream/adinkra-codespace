#!/usr/bin/env node

// Independent check of the relative-color scan and every serialized signed
// equivalence witness. It reconstructs the candidates from the printed S4
// signed words and does not import the Rust implementation.

import fs from "node:fs";

const path = process.argv[2] ?? "data/permutahedron_s8_signed_equivalence.json";
const artifact = JSON.parse(fs.readFileSync(path, "utf8"));

const words = [
  [[1,-4,2,-3],[2,3,-1,-4],[3,-2,-4,1],[4,1,3,2]],
  [[1,-3,-4,-2],[2,4,-3,1],[3,1,2,-4],[4,-2,1,3]],
  [[1,3,-2,-4],[2,-4,1,-3],[3,-1,-4,2],[4,2,3,1]],
  [[1,-4,-3,2],[-2,-3,4,1],[3,-2,1,-4],[4,1,2,3]],
  [[1,2,-4,-3],[-2,1,3,-4],[3,4,2,1],[4,-3,1,-2]],
  [[1,2,-3,-4],[-2,1,-4,3],[3,4,1,2],[4,-3,-2,1]],
];
const labels = ["CM", "TM", "VM", "VM1", "VM2", "VM3"];
const sourcePermutations = words.map((set) => set.map((word) => word.map(Math.abs)));
const sourceFactors = words.map((set) => set.map((word) =>
  word.reduce((value, entry, row) => value | (entry < 0 ? 1 << row : 0), 0)
));

function assert(value, message) {
  if (!value) throw new Error(message);
}

function permutations(values) {
  if (values.length === 1) return [[...values]];
  return values.flatMap((value, index) => permutations(values.filter((_, i) => i !== index))
    .map((tail) => [value, ...tail]));
}

function candidate(first, second, order, start) {
  const mask = Array.from({length: 4}, (_, offset) => (start + offset) % 8)
    .reduce((value, position) => value | (1 << position), 0);
  const base = Array.from({length: 4}, (_, color) =>
    sourceFactors[first][color] | (sourceFactors[second][order[color]] << 4));
  const unsigned = Array.from({length: 8}, (_, color) => {
    const local = color % 4;
    const left = sourcePermutations[first][local];
    const right = sourcePermutations[second][order[local]];
    return color < 4
      ? [...left, ...right.map((entry) => entry + 4)]
      : [...left.map((entry) => entry + 4), ...right];
  });
  return { permutations: unsigned, boolean_factors: [...base, ...base.map((value) => value ^ mask)] };
}

function multiply(left, right) {
  return left.map((_, row) => right[0].map((__, column) =>
    left[row].reduce((sum, value, inner) => sum + value * right[inner][column], 0)));
}

function transpose(matrix) {
  return matrix[0].map((_, column) => matrix.map((row) => row[column]));
}

function closes(rep) {
  const l = rep.permutations.map((permutation, color) => permutation.map((target, row) => {
    const result = Array(8).fill(0);
    result[target - 1] = rep.boolean_factors[color] & (1 << row) ? -1 : 1;
    return result;
  }));
  const r = l.map(transpose);
  for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) {
    for (const [left, right] of [[l, r], [r, l]]) {
      const a = multiply(left[i], right[j]);
      const b = multiply(left[j], right[i]);
      for (let row = 0; row < 8; row++) for (let column = 0; column < 8; column++) {
        if (a[row][column] + b[row][column] !== (i === j && row === column ? 2 : 0)) return false;
      }
    }
  }
  return true;
}

function dual(rep) {
  const output = {permutations: [], boolean_factors: []};
  for (let color = 0; color < 8; color++) {
    const permutation = Array(8);
    let factor = 0;
    for (let row = 0; row < 8; row++) {
      const target = rep.permutations[color][row] - 1;
      permutation[target] = row + 1;
      if (rep.boolean_factors[color] & (1 << row)) factor |= 1 << target;
    }
    output.permutations.push(permutation);
    output.boolean_factors.push(factor);
  }
  return output;
}

function verifyWitness(source, target, witness) {
  if (witness.source_dualized) source = dual(source);
  for (let color = 0; color < 8; color++) for (let boson = 0; boson < 8; boson++) {
    const targetColor = witness.color_map_zero_based[color];
    const sourceFermion = source.permutations[color][boson] - 1;
    const targetBoson = witness.boson_map_zero_based[boson];
    const targetFermion = target.permutations[targetColor][targetBoson] - 1;
    if (witness.fermion_map_zero_based[sourceFermion] !== targetFermion) return false;
    const sourceSign = source.boolean_factors[color] & (1 << boson) ? -1 : 1;
    const targetSign = target.boolean_factors[targetColor] & (1 << targetBoson) ? -1 : 1;
    if (sourceSign * witness.boson_switches[boson]
      * witness.fermion_switches[sourceFermion]
      * witness.supercharge_signs[color] !== targetSign) return false;
  }
  return true;
}

assert(artifact.schema_version === "permutahedron-s8-signed-equivalence-v2", "schema mismatch");
const orders = permutations([0,1,2,3]);
const expectedScan = new Map();
const expectedClosers = new Map();
let checked = 0;
for (let first = 0; first < 6; first++) for (let second = 0; second < 6; second++) {
  if (first === second) continue;
  for (const order of orders) {
    const key = `${labels[first]}->${labels[second]}:${order.map((x) => x + 1).join("")}`;
    const starts = [];
    for (let start = 0; start < 8; start++) {
      checked++;
      const rep = candidate(first, second, order, start);
      if (closes(rep)) {
        starts.push(start + 1);
        expectedClosers.set(`${labels[first]}->${labels[second]}:order-${order.map((x) => x + 1).join("")}:mask-${start + 1}`, rep);
      }
    }
    expectedScan.set(key, starts);
  }
}

assert(checked === 5760, "candidate census mismatch");
assert(expectedClosers.size === 24, "independent closure count mismatch");
assert(artifact.scan.length === 720, "serialized alignment count mismatch");
for (const record of artifact.scan) {
  const key = `${record.first_label}->${record.second_label}:${record.second_color_order_one_based.join("")}`;
  assert(JSON.stringify(record.closing_start_positions_one_based) === JSON.stringify(expectedScan.get(key)), `scan mismatch for ${key}`);
}

const serialized = new Map(artifact.closers.map((record) => [record.id, record]));
assert(serialized.size === expectedClosers.size, "serialized closer count mismatch");
for (const [id, expected] of expectedClosers) {
  const actual = serialized.get(id);
  assert(actual, `missing closer ${id}`);
  assert(JSON.stringify(actual.permutations) === JSON.stringify(expected.permutations), `permutation mismatch for ${id}`);
  assert(JSON.stringify(actual.boolean_factors) === JSON.stringify(expected.boolean_factors), `factor mismatch for ${id}`);
}

const ct = serialized.get("CM->TM:order-1234:mask-6");
const cv = serialized.get("CM->VM:order-2143:mask-4");
assert(ct?.exact_published_system_match === "CT", "CT source anchor missing");
assert(cv?.exact_published_system_match === "CV", "CV source anchor missing");
assert(JSON.stringify(ct.boolean_factors) === JSON.stringify([234,76,134,32,11,173,103,193]), "CT factors mismatch");
assert(JSON.stringify(cv.boolean_factors) === JSON.stringify([170,204,6,96,210,180,126,24]), "CV factors mismatch");

const namedParents = new Set(["CM", "TM", "VM"]);
let namedParentClosers = 0;
let noStatedParentClosers = 0;
for (const id of expectedClosers.keys()) {
  const [pair] = id.split(":");
  const [first, second] = pair.split("->");
  if (namedParents.has(first) && namedParents.has(second)) namedParentClosers++;
  else if (!namedParents.has(first) && !namedParents.has(second)) noStatedParentClosers++;
  else throw new Error(`unexpected mixed-parentage closer ${id}`);
}
assert(namedParentClosers === 12, "named-parent closer count mismatch");
assert(noStatedParentClosers === 12, "no-stated-parent closer count mismatch");
assert(artifact.validation.closers_built_from_two_named_four_dimensional_parents === namedParentClosers, "artifact named-parent count mismatch");
assert(artifact.validation.closers_built_from_sources_with_no_stated_four_dimensional_parent === noStatedParentClosers, "artifact no-stated-parent count mismatch");

let witnessCount = 0;
for (const layer of artifact.equivalence_layers) for (const group of layer.classes) {
  const source = serialized.get(group.representative_id);
  assert(source, `missing class representative ${group.representative_id}`);
  for (const member of group.members) {
    const target = serialized.get(member.id);
    assert(target, `missing class member ${member.id}`);
    assert(verifyWitness(source, target, member.witness_from_representative), `invalid witness ${group.class_id} -> ${member.id}`);
    witnessCount++;
  }
}
assert(artifact.validation.equivalence_class_counts_by_layer.join(",") === "1,1,1,1", "equivalence hierarchy mismatch");
assert(artifact.validation.fixed_color_nodal_class_mixes_source_parentage_categories === true, "fixed-color class must mix source-parentage categories");
assert(artifact.validation.passed === true, "Rust validation failed");

console.log(`verified ${checked} candidates, ${expectedClosers.size} closures, and ${witnessCount} exact signed-equivalence witnesses`);
