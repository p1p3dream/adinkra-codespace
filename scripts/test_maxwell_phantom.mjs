#!/usr/bin/env node

import fs from "node:fs";

const artifact = JSON.parse(fs.readFileSync("results/maxwell_phantom.json", "utf8"));
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const z = (real = 0, imag = 0) => [real, imag];
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
const mul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const neg = (a) => [-a[0], -a[1]];
const divExact = (a, divisor) => {
  assert(a[0] % divisor === 0 && a[1] % divisor === 0, "nonintegral coefficient");
  return [a[0] / divisor, a[1] / divisor];
};
const I2 = [[z(1), z()], [z(), z(1)]];
const s1 = [[z(), z(1)], [z(1), z()]];
const s2 = [[z(), z(0, -1)], [z(0, 1), z()]];
const s3 = [[z(1), z()], [z(), z(-1)]];
const kron = (a, b) => Array.from({length: 4}, (_, row) =>
  Array.from({length: 4}, (_, column) => mul(a[row >> 1][column >> 1], b[row & 1][column & 1])));
const scale = (matrix, coefficient) => matrix.map((row) => row.map((value) => mul(value, coefficient)));
const matrixAdd = (a, b) => a.map((row, i) => row.map((value, j) => add(value, b[i][j])));
const matrixMul = (a, b) => Array.from({length: 4}, (_, row) => Array.from({length: 4}, (_, column) => {
  let value = z();
  for (let inner = 0; inner < 4; inner++) value = add(value, mul(a[row][inner], b[inner][column]));
  return value;
}));

const gammaUp = [
  scale(kron(s3, s2), z(0, 1)),
  kron(I2, s1),
  kron(s2, s2),
  kron(I2, s3),
];
const gammaDown = gammaUp.map((matrix, mu) => mu === 0 ? scale(matrix, z(-1)) : matrix);
const gamma5 = scale(kron(s1, s2), z(-1));
const C = scale(kron(s3, s2), z(0, -1));
const lower = (matrix) => matrixMul(matrix, C);
const gamma5Gamma = gammaUp.map((matrix) => matrixMul(gamma5, matrix));
const commutator = (mu, nu) => matrixAdd(
  matrixMul(gammaUp[mu], gammaUp[nu]),
  scale(matrixMul(gammaUp[nu], gammaUp[mu]), z(-1)),
);
const epsilon = (a, b, c) => {
  if (a === b || a === c || b === c) return 0;
  const inversions = Number(a > b) + Number(a > c) + Number(b > c);
  return inversions % 2 === 0 ? 1 : -1;
};
const zeroLinkage = () => Array.from({length: 4}, () =>
  Array.from({length: 7}, () => Array.from({length: 4}, () => z())));

const upTranspose = zeroLinkage();
const temporalDown = zeroLinkage();
const spatialDown = Array.from({length: 3}, zeroLinkage);
const magneticPairs = [[2, 3], [3, 1], [1, 2]];
for (let charge = 0; charge < 4; charge++) for (let fermion = 0; fermion < 4; fermion++) {
  for (let spatial = 1; spatial < 4; spatial++) {
    const physical = divExact(mul(lower(commutator(0, spatial))[charge][fermion], z(0, -1)), 2);
    upTranspose[charge][spatial - 1][fermion] = mul(physical, z(0, -1));
    temporalDown[charge][spatial - 1][fermion] = gammaDown[spatial][charge][fermion];
    spatialDown[spatial - 1][charge][spatial - 1][fermion] = neg(gammaDown[0][charge][fermion]);
  }
  upTranspose[charge][3][fermion] = mul(lower(gamma5)[charge][fermion], z(0, -1));
  temporalDown[charge][3][fermion] = mul(gamma5Gamma[0][charge][fermion], z(0, 1));
  for (let spatial = 1; spatial < 4; spatial++) {
    spatialDown[spatial - 1][charge][3][fermion] = mul(gamma5Gamma[spatial][charge][fermion], z(0, 1));
  }
  for (let magnetic = 0; magnetic < 3; magnetic++) {
    const [mu, nu] = magneticPairs[magnetic];
    const physical = divExact(mul(lower(commutator(mu, nu))[charge][fermion], z(0, -1)), 2);
    upTranspose[charge][4 + magnetic][fermion] = mul(physical, z(0, -1));
  }
  for (let derivative = 0; derivative < 3; derivative++) for (let magnetic = 0; magnetic < 3; magnetic++) {
    let coefficient = z();
    for (let electric = 0; electric < 3; electric++) {
      const sign = epsilon(magnetic, derivative, electric);
      coefficient = add(coefficient, mul(temporalDown[charge][electric][fermion], z(sign)));
    }
    spatialDown[derivative][charge][4 + magnetic][fermion] = coefficient;
  }
}

const phantom = upTranspose.map((charge, a) => charge.map((row, b) =>
  row.map((value, f) => add(value, neg(temporalDown[a][b][f])))));
const record = (linkage) => linkage.map((charge) => charge.map((row) =>
  row.map(([real, imag]) => ({real, imag}))));
assert(JSON.stringify(record(upTranspose)) === JSON.stringify(artifact.fermion_up_transpose), "up-link mismatch");
assert(JSON.stringify(record(temporalDown)) === JSON.stringify(artifact.temporal_down), "temporal-down mismatch");
assert(JSON.stringify(record(phantom)) === JSON.stringify(artifact.phantom_matrix), "phantom mismatch");
assert(JSON.stringify(spatialDown.map(record)) === JSON.stringify(artifact.spatial_down), "spatial-down mismatch");
const nonzero = phantom.flat(2).filter(([real, imag]) => real !== 0 || imag !== 0).length;
assert(nonzero === 12 && artifact.nonzero_phantom_entries === 12, "phantom support count mismatch");
assert(artifact.equation_5_8_residual_entries === 0 && artifact.passed, "artifact gate failed");
console.log("verified the 12-entry Maxwell phantom sector and all Eq. 5.8 spatial rows");
