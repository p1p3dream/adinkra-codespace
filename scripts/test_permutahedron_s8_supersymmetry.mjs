import assert from "node:assert/strict";
import fs from "node:fs";

const artifactPath = "data/permutahedron_s8_supersymmetry.json";
const viewerPath = "visualizer/permutahedron_s8_supersymmetry.html";
const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
const sourceFactors = [
  [170, 204, 102, 0, 15, 105, 195, 165],
  [234, 76, 134, 32, 11, 173, 103, 193],
  [170, 204, 6, 96, 210, 180, 126, 24],
  [238, 68, 136, 34, 238, 68, 136, 34],
  [174, 196, 8, 98, 174, 196, 8, 98],
  [170, 204, 0, 102, 170, 204, 0, 102],
];

const identity = size =>
  Array.from({ length: size }, (_, row) =>
    Array.from({ length: size }, (_, column) => (row === column ? 1 : 0)),
  );

const transpose = value =>
  value[0].map((_, column) => value.map(row => row[column]));

const multiply = (left, right) =>
  left.map(row =>
    right[0].map((_, column) =>
      row.reduce(
        (sum, value, inner) => sum + value * right[inner][column],
        0,
      ),
    ),
  );

const add = (left, right) =>
  left.map((row, r) => row.map((value, c) => value + right[r][c]));

const matrix = (permutation, factor) =>
  Array.from({ length: 8 }, (_, row) =>
    Array.from({ length: 8 }, (_, column) =>
      column === permutation[row] - 1
        ? factor & (1 << row)
          ? -1
          : 1
        : 0,
    ),
  );

const gamma = (l, r) =>
  Array.from({ length: 16 }, (_, row) =>
    Array.from({ length: 16 }, (_, column) => {
      if (row < 8 && column >= 8) return l[row][column - 8];
      if (row >= 8 && column < 8) return r[row - 8][column];
      return 0;
    }),
  );

const parameterSigns = residue =>
  [
    [1, 1],
    [1, -1],
    [-1, -1],
    [-1, 1],
  ][residue];

function effectiveFactors(sectorIndex, branch) {
  const factors = [...sourceFactors[sectorIndex]];
  if (sectorIndex < 3) return factors;
  const [mPlus, mMinus] = parameterSigns(branch.m_mod_4);
  const [nPlus, nMinus] = parameterSigns(branch.n_mod_4);
  assert.deepEqual([branch.m_plus, branch.m_minus], [mPlus, mMinus]);
  assert.deepEqual([branch.n_plus, branch.n_minus], [nPlus, nMinus]);
  return factors.map((factor, color) => {
    const mSign = color < 4 ? mPlus : mMinus;
    const nSign = color < 4 ? nPlus : nMinus;
    if (nSign < 0) factor ^= 0x0f;
    if (mSign < 0) factor ^= 0xf0;
    return factor;
  });
}

function auditGarden(lMatrices, rMatrices) {
  let checked = 0;
  let residuals = 0;
  let nonclosingPairs = 0;
  for (let i = 0; i < 8; i += 1) {
    for (let j = 0; j < 8; j += 1) {
      for (const [left, right] of [
        [lMatrices, rMatrices],
        [rMatrices, lMatrices],
      ]) {
        const relation = add(
          multiply(left[i], right[j]),
          multiply(left[j], right[i]),
        );
        for (let row = 0; row < 8; row += 1) {
          for (let column = 0; column < 8; column += 1) {
            const expected = i === j && row === column ? 2 : 0;
            checked += 1;
            if (relation[row][column] !== expected) residuals += 1;
          }
        }
      }
      if (i < j) {
        const bosonic = add(
          multiply(lMatrices[i], rMatrices[j]),
          multiply(lMatrices[j], rMatrices[i]),
        );
        const fermionic = add(
          multiply(rMatrices[i], lMatrices[j]),
          multiply(rMatrices[j], lMatrices[i]),
        );
        if (
          bosonic.some(row => row.some(Boolean)) ||
          fermionic.some(row => row.some(Boolean))
        ) {
          nonclosingPairs += 1;
        }
      }
    }
  }
  return { checked, residuals, nonclosingPairs };
}

function auditHymn(lMatrices, rMatrices) {
  let product = identity(16);
  for (let color = 0; color < 8; color += 1) {
    product = multiply(gamma(lMatrices[color], rMatrices[color]), product);
  }
  return product;
}

assert.equal(
  artifact.schema_version,
  "permutahedron-s8-supersymmetry-v1",
);
assert.equal(artifact.validation.passed, true);
assert.equal(artifact.sectors.length, 6);
assert.deepEqual(
  artifact.sectors.map(sector => sector.id),
  ["CC", "CT", "CV", "TT", "TV", "VV"],
);

let branchesChecked = 0;
let entriesChecked = 0;
let residualEntries = 0;
let closingBranches = 0;
let graphsChecked = 0;

for (const [sectorIndex, sector] of artifact.sectors.entries()) {
  assert.equal(sector.permutations.length, 8);
  assert.equal(sector.base_boolean_factors.length, 8);
  assert.deepEqual(sector.base_boolean_factors, sourceFactors[sectorIndex]);
  assert.equal(sector.parameter_branches.length, sector.branch_count);
  for (const branch of sector.parameter_branches) {
    assert.deepEqual(
      branch.effective_boolean_factors,
      effectiveFactors(sectorIndex, branch),
    );
    const lMatrices = sector.permutations.map((permutation, color) =>
      matrix(permutation, branch.effective_boolean_factors[color]),
    );
    const rMatrices = lMatrices.map(transpose);
    assert.deepEqual(
      branch.matrices.map(record => record.l),
      lMatrices,
    );
    assert.deepEqual(
      branch.matrices.map(record => record.r),
      rMatrices,
    );

    const garden = auditGarden(lMatrices, rMatrices);
    assert.equal(garden.checked, 8192);
    assert.equal(
      garden.residuals,
      branch.closure.bosonic_residual_entries +
        branch.closure.fermionic_residual_entries,
    );
    assert.equal(
      garden.nonclosingPairs,
      branch.closure.nonclosing_color_pairs,
    );
    assert.equal(branch.closure.sparse_garden_passed, garden.residuals === 0);

    const hymn = auditHymn(lMatrices, rMatrices);
    assert.deepEqual(
      hymn.map((row, index) => row[index]),
      branch.hymn.diagonal_entries,
    );
    assert.equal(
      hymn.reduce((sum, row, index) => sum + row[index], 0),
      branch.hymn.trace,
    );
    assert.equal(
      hymn.every((row, r) =>
        row.every((value, c) => (r === c ? true : value === 0)),
      ),
      branch.hymn.diagonal,
    );

    assert.equal(branch.graph.bosons.length + branch.graph.fermions.length, 16);
    assert.equal(branch.graph.edge_count, 64);
    assert.equal(branch.graph.square_count, 112);
    assert.equal(
      branch.graph.valid_garden_adinkra,
      garden.residuals === 0 &&
        branch.graph.two_color_squares.every(square => square.odd_dashing),
    );
    assert.equal(
      branch.garden_distance.minimum_edge_sign_flips === 0,
      garden.residuals === 0,
    );

    branchesChecked += 1;
    entriesChecked += garden.checked;
    residualEntries += garden.residuals;
    closingBranches += Number(garden.residuals === 0);
    graphsChecked += 1;
  }
}

assert.equal(branchesChecked, 51);
assert.equal(entriesChecked, 417_792);
assert.equal(residualEntries, 12_416);
assert.equal(closingBranches, 2);
assert.equal(graphsChecked, 51);
assert.equal(
  artifact.validation.dense_closure_entries_checked,
  entriesChecked,
);
assert.equal(
  artifact.validation.dense_closure_residual_entries,
  residualEntries,
);
assert.equal(
  artifact.separation.hymn_class_matches_published_closure_on_every_branch,
  true,
);
assert.equal(
  artifact.separation.unsigned_abnormality_separates_closure,
  false,
);

const viewer = fs.readFileSync(viewerPath, "utf8");
assert.match(viewer, /\.\.\/data\/permutahedron_s8_supersymmetry\.json/);
assert.match(viewer, /Six Signed Eight-Color Systems/);

console.log(
  JSON.stringify(
    {
      artifact: artifactPath,
      viewer: viewerPath,
      sectors: artifact.sectors.length,
      parameter_branches_checked: branchesChecked,
      independently_checked_dense_entries: entriesChecked,
      residual_entries: residualEntries,
      closing_branches: closingBranches,
      graphs_checked: graphsChecked,
      passed: true,
    },
    null,
    2,
  ),
);
