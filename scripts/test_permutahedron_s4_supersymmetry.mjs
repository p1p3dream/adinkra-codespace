import assert from "node:assert/strict";
import fs from "node:fs";

const artifactPath = "data/permutahedron_s4_supersymmetry.json";
const viewerPath = "visualizer/permutahedron_s4_supersymmetry.html";
const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
const viewer = fs.readFileSync(viewerPath, "utf8");

const signs = factor =>
  Array.from({ length: 4 }, (_, row) => (factor & (1 << row) ? -1 : 1));

const matrix = (permutation, factor) => {
  const diagonal = signs(factor);
  return Array.from({ length: 4 }, (_, row) =>
    Array.from({ length: 4 }, (_, column) =>
      column === permutation[row] - 1 ? diagonal[row] : 0,
    ),
  );
};

const transpose = value =>
  Array.from({ length: 4 }, (_, row) =>
    Array.from({ length: 4 }, (_, column) => value[column][row]),
  );

const multiply = (left, right) =>
  Array.from({ length: 4 }, (_, row) =>
    Array.from({ length: 4 }, (_, column) =>
      Array.from(
        { length: 4 },
        (_, inner) => left[row][inner] * right[inner][column],
      ).reduce((sum, value) => sum + value, 0),
    ),
  );

function gardenResiduals(permutations, factors) {
  const l = permutations.map((permutation, color) =>
    matrix(permutation, factors[color]),
  );
  const r = l.map(transpose);
  let checked = 0;
  let residuals = 0;
  for (let i = 0; i < 4; i += 1) {
    for (let j = 0; j < 4; j += 1) {
      const first = multiply(l[i], r[j]);
      const second = multiply(l[j], r[i]);
      for (let row = 0; row < 4; row += 1) {
        for (let column = 0; column < 4; column += 1) {
          const expected = i === j && row === column ? 2 : 0;
          checked += 1;
          if (first[row][column] + second[row][column] !== expected) {
            residuals += 1;
          }
        }
      }
    }
  }
  return { checked, residuals };
}

assert.equal(artifact.schema_version, "permutahedron-s4-supersymmetry-v1");
assert.equal(artifact.validation.passed, true);
assert.equal(artifact.validation.equation_5_10_matrices_matched, true);
assert.equal(artifact.validation.intra_bruhat_spectra_verified, 6);
assert.equal(artifact.validation.intra_bruhat_norm_squared, 224);
assert.equal(artifact.validation.maximum_inter_bruhat_norm_squared, 208);
assert.equal(artifact.validation.intra_norm_exceeds_every_inter_quartet, true);
assert.equal(artifact.permutation_vertices.length, 24);
assert.equal(
  new Set(artifact.permutation_vertices.map(value => value.join(""))).size,
  24,
);
assert.equal(artifact.sectors.length, 6);
assert.equal(
  new Set(
    artifact.sectors.map(sector =>
      sector.invariants.quotient_s3_one_line.join(""),
    ),
  ).size,
  6,
);

const covered = artifact.sectors.flatMap(sector =>
  sector.ordered_permutations.map(value => value.join("")),
);
assert.equal(covered.length, 24);
assert.equal(new Set(covered).size, 24);

let signingsChecked = 0;
let denseEntriesChecked = 0;
for (const sector of artifact.sectors) {
  assert.equal(sector.ordered_permutations.length, 4);
  assert.equal(sector.published_fiducial_signings.length, 16);
  assert.equal(
    new Set(
      sector.published_fiducial_signings.map(signing =>
        signing.boolean_factors.join(","),
      ),
    ).size,
    16,
  );
  assert.equal(sector.adinkra.edge_count, 16);
  assert.equal(sector.adinkra.square_count, 12);
  assert.equal(sector.adinkra.all_squares_odd, true);
  assert.deepEqual(sector.invariants.bruhat_eigenvalues, [12, 0, -4, -8]);
  assert.equal(sector.invariants.bruhat_eigenvalue_norm_squared, 224);
  assert.equal(
    2 *
      sector.invariants.ordered_bruhat_upper_triangle.reduce(
        (sum, value) => sum + value * value,
        0,
      ),
    224,
  );
  assert.equal(
    sector.adinkra.two_color_squares.every(square => square.odd_dashing),
    true,
  );
  for (const signing of sector.published_fiducial_signings) {
    assert.equal(signing.garden_sparse_passed, true);
    assert.equal(signing.dense_residual_entries, 0);
    assert.equal(Math.abs(signing.chi0), 1);
    assert.equal(signing.matrices.length, 4);
    assert.equal(signing.adinkra.edge_count, 16);
    assert.equal(signing.adinkra.square_count, 12);
    assert.equal(signing.adinkra.all_squares_odd, true);
    assert.equal(
      signing.adinkra.two_color_squares.every(square => square.odd_dashing),
      true,
    );
    for (let color = 0; color < 4; color += 1) {
      assert.deepEqual(
        signing.matrices[color].l,
        matrix(
          sector.ordered_permutations[color],
          signing.boolean_factors[color],
        ),
      );
    }
    const independent = gardenResiduals(
      sector.ordered_permutations,
      signing.boolean_factors,
    );
    assert.deepEqual(independent, { checked: 256, residuals: 0 });
    signingsChecked += 1;
    denseEntriesChecked += independent.checked;
  }
}

assert.equal(signingsChecked, 96);
assert.equal(denseEntriesChecked, 24_576);
assert.equal(
  artifact.validation.dense_garden_entries_checked,
  denseEntriesChecked,
);
assert.match(
  viewer,
  /\.\.\/data\/permutahedron_s4_supersymmetry\.json/,
);
assert.match(viewer, /Six Four-Color Supersymmetry Sectors/);

console.log(
  JSON.stringify(
    {
      artifact: artifactPath,
      viewer: viewerPath,
      vertices: 24,
      sectors: 6,
      signings_checked: signingsChecked,
      independently_checked_dense_entries: denseEntriesChecked,
      residual_entries: 0,
      passed: true,
    },
    null,
    2,
  ),
);
