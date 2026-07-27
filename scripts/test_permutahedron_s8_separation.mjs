import assert from "node:assert/strict";
import fs from "node:fs";

const artifactPath = "data/permutahedron_s8_separation_probe.json";
const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));

const r8 = [
  [1, 2, 3, 4, 5, 6, 7, 8],
  [2, 1, 4, 3, 6, 5, 8, 7],
  [3, 4, 1, 2, 7, 8, 5, 6],
  [4, 3, 2, 1, 8, 7, 6, 5],
  [5, 6, 7, 8, 1, 2, 3, 4],
  [6, 5, 8, 7, 2, 1, 4, 3],
  [7, 8, 5, 6, 3, 4, 1, 2],
  [8, 7, 6, 5, 4, 3, 2, 1],
];

const s4Quartets = [
  ["1423", "2314", "3241", "4132"],
  ["1342", "2431", "3124", "4213"],
  ["1324", "2413", "3142", "4231"],
  ["1432", "2341", "3214", "4123"],
  ["1243", "2134", "3421", "4312"],
  ["1234", "2143", "3412", "4321"],
];

const s4Sector = new Map(
  s4Quartets.flatMap((quartet, sector) =>
    quartet.map(permutation => [permutation, sector + 1]),
  ),
);

const compose = (left, right) => right.map(value => left[value - 1]);
const key = permutation => permutation.join("");

function permutations(values) {
  if (values.length === 0) return [[]];
  return values.flatMap((value, index) =>
    permutations(values.filter((_, candidate) => candidate !== index)).map(
      suffix => [value, ...suffix],
    ),
  );
}

const s8 = permutations([1, 2, 3, 4, 5, 6, 7, 8]);
const rank = new Map(s8.map((permutation, index) => [key(permutation), index]));

function coset(seed, side) {
  return r8
    .map(element =>
      side === "right" ? compose(element, seed) : compose(seed, element),
    )
    .map(permutation => rank.get(key(permutation)))
    .sort((a, b) => a - b);
}

const covered = new Uint8Array(s8.length);
const cosets = [];
for (let seedRank = 0; seedRank < s8.length; seedRank += 1) {
  if (covered[seedRank]) continue;
  const slice = coset(s8[seedRank], "right");
  slice.forEach(value => {
    covered[value] = 1;
  });
  cosets.push(slice);
}

const mask = values =>
  values.reduce((result, value) => result | (1 << (value - 1)), 0);

const imageMask = (permutation, sourceMask) => {
  let result = 0;
  for (let value = 1; value <= 8; value += 1) {
    if (sourceMask & (1 << (value - 1))) {
      result |= 1 << (permutation[value - 1] - 1);
    }
  }
  return result;
};

const invariantPartitions = [];
for (let sourceMask = 1; sourceMask < 256; sourceMask += 2) {
  if (sourceMask.toString(2).split("1").length - 1 !== 4) continue;
  const complement = sourceMask ^ 255;
  if (
    r8.every(element => {
      const image = imageMask(element, sourceMask);
      return image === sourceMask || image === complement;
    })
  ) {
    const first = Array.from({ length: 8 }, (_, index) => index + 1).filter(
      value => sourceMask & (1 << (value - 1)),
    );
    const second = Array.from({ length: 8 }, (_, index) => index + 1).filter(
      value => complement & (1 << (value - 1)),
    );
    invariantPartitions.push([first, second]);
  }
}

function local(permutation, domain, codomain) {
  return domain.map(value => codomain.indexOf(permutation[value - 1]) + 1);
}

function pairClass(permutation, [first, second]) {
  const firstMask = mask(first);
  const secondMask = mask(second);
  const image = imageMask(permutation, firstMask);
  let firstTarget;
  let secondTarget;
  if (image === firstMask) {
    firstTarget = first;
    secondTarget = second;
  } else if (image === secondMask) {
    firstTarget = second;
    secondTarget = first;
  } else {
    return null;
  }
  const sectors = [
    s4Sector.get(key(local(permutation, first, firstTarget))),
    s4Sector.get(key(local(permutation, second, secondTarget))),
  ].sort((a, b) => a - b);
  assert.ok(sectors.every(Boolean));
  return `P${sectors[0]}+P${sectors[1]}`;
}

assert.equal(cosets.length, 5_040);
assert.equal(covered.reduce((sum, value) => sum + value, 0), 40_320);
assert.equal(invariantPartitions.length, 7);

const histogram = new Map();
const coincidenceHistogram = {
  left_right_coincident: new Map(),
  other: new Map(),
};
const partitionClasses = invariantPartitions.map(() => new Map());
let incidences = 0;
let diagonal = 0;
let mixed = 0;
let compatibleCosets = 0;

for (const slice of cosets) {
  const members = slice.map(value => s8[value]);
  const representative = members[0];
  const coincident =
    coset(representative, "left").join(",") === slice.join(",");
  let compatible = 0;
  for (const [partitionId, partition] of invariantPartitions.entries()) {
    const labels = new Set(
      members.map(member => pairClass(member, partition)).filter(Boolean),
    );
    assert.ok(labels.size === 0 || labels.size === 1);
    if (labels.size === 0) continue;
    const label = [...labels][0];
    compatible += 1;
    incidences += 1;
    const isDiagonal = label.slice(0, 2) === label.slice(3, 5);
    diagonal += Number(isDiagonal);
    mixed += Number(!isDiagonal);
    assert.equal(isDiagonal, coincident);
    partitionClasses[partitionId].set(
      label,
      (partitionClasses[partitionId].get(label) ?? 0) + 1,
    );
  }
  compatibleCosets += Number(compatible > 0);
  histogram.set(compatible, (histogram.get(compatible) ?? 0) + 1);
  const kind = coincident ? "left_right_coincident" : "other";
  coincidenceHistogram[kind].set(
    compatible,
    (coincidenceHistogram[kind].get(compatible) ?? 0) + 1,
  );
}

assert.deepEqual(Object.fromEntries(histogram), {
  0: 4_136,
  1: 854,
  3: 49,
  7: 1,
});
assert.deepEqual(
  Object.fromEntries(coincidenceHistogram.left_right_coincident),
  { 0: 48, 1: 98, 3: 21, 7: 1 },
);
assert.deepEqual(Object.fromEntries(coincidenceHistogram.other), {
  0: 4_088,
  1: 756,
  3: 28,
});
assert.equal(incidences, 1_008);
assert.equal(compatibleCosets, 904);
assert.equal(diagonal, 168);
assert.equal(mixed, 840);
for (const classes of partitionClasses) {
  assert.equal(classes.size, 21);
  assert.equal(
    [...classes.values()].reduce((sum, value) => sum + value, 0),
    144,
  );
}

assert.deepEqual(artifact.validation, {
  r8_cosets: 5_040,
  invariant_partitions: 7,
  compatible_cosets_per_partition: 144,
  total_compatible_incidences: 1_008,
  distinct_compatible_cosets: 904,
  pair_classes_per_partition: 21,
  diagonal_pair_incidences: 168,
  mixed_pair_incidences: 840,
  pair_diagonal_matches_left_right_coincidence: true,
  published_octets_located: 6,
  published_pair_labels_matched: 6,
  passed: true,
});

console.log(
  JSON.stringify(
    {
      artifact: artifactPath,
      cosets_checked: cosets.length,
      vertices_covered: 40_320,
      invariant_partitions: invariantPartitions.length,
      compatible_cosets: compatibleCosets,
      compatible_incidences: incidences,
      diagonal_incidences: diagonal,
      mixed_incidences: mixed,
      passed: true,
    },
    null,
    2,
  ),
);
