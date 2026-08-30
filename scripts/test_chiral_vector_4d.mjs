#!/usr/bin/env node

import fs from "node:fs";

const artifact = JSON.parse(fs.readFileSync("data/chiral_vector_4d.json", "utf8"));
const assert = (condition, message) => { if (!condition) throw new Error(message); };

const z = (r = 0, i = 0) => [r, i];
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
const mul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const neg = (a) => [-a[0], -a[1]];
const divExact = (a, d) => {
  assert(a[0] % d === 0 && a[1] % d === 0, "nonintegral Gaussian coefficient");
  return [a[0] / d, a[1] / d];
};
const zero = (a) => a[0] === 0 && a[1] === 0;

const I2 = [[z(1), z()], [z(), z(1)]];
const s1 = [[z(), z(1)], [z(1), z()]];
const s2 = [[z(), z(0, -1)], [z(0, 1), z()]];
const s3 = [[z(1), z()], [z(), z(-1)]];
const kron = (a, b) => Array.from({length: 4}, (_, row) =>
  Array.from({length: 4}, (_, column) => mul(a[row >> 1][column >> 1], b[row & 1][column & 1])));
const mscale = (a, c) => a.map((row) => row.map((value) => mul(value, c)));
const madd = (a, b) => a.map((row, i) => row.map((value, j) => add(value, b[i][j])));
const mmul = (a, b) => Array.from({length: 4}, (_, row) => Array.from({length: 4}, (_, column) => {
  let out = z();
  for (let inner = 0; inner < 4; inner++) out = add(out, mul(a[row][inner], b[inner][column]));
  return out;
}));

const gammaUp = [
  mscale(kron(s3, s2), z(0, 1)),
  kron(I2, s1),
  kron(s2, s2),
  kron(I2, s3),
];
const gammaDown = gammaUp.map((matrix, mu) => mu === 0 ? mscale(matrix, z(-1)) : matrix);
const gamma5 = mscale(kron(s1, s2), z(-1));
const C = mscale(kron(s3, s2), z(0, -1));
const lower = (matrix) => mmul(matrix, C);
const gamma5Gamma = gammaUp.map((gamma) => mmul(gamma5, gamma));
const commutator = (mu, nu) => madd(mmul(gammaUp[mu], gammaUp[nu]), mscale(mmul(gammaUp[nu], gammaUp[mu]), z(-1)));

const identity4 = Array.from({length: 4}, (_, row) => Array.from({length: 4}, (_, column) => z(row === column ? 1 : 0)));
for (let mu = 0; mu < 4; mu++) for (let nu = 0; nu < 4; nu++) {
  const expected = mscale(identity4, z(mu === nu ? (mu === 0 ? -2 : 2) : 0));
  assert(JSON.stringify(madd(mmul(gammaUp[mu], gammaUp[nu]), mmul(gammaUp[nu], gammaUp[mu]))) === JSON.stringify(expected), "Clifford mismatch");
}

const fields = ["A", "B", "F", "G", "V0", "V1", "V2", "V3", "d",
  "p0", "p1", "p2", "p3", "l0", "l1", "l2", "l3"];
const key = (field, derivatives) => `${field}|${derivatives.join("")}`;
const parse = (entry) => {
  const [field, digits] = entry.split("|");
  return [field, [...digits].map(Number)];
};
const atom = (field) => new Map([[key(field, [0,0,0,0]), z(1)]]);
const addTerm = (polynomial, field, derivatives, coefficient) => {
  if (zero(coefficient)) return;
  const k = key(field, derivatives);
  const value = add(polynomial.get(k) ?? z(), coefficient);
  if (zero(value)) polynomial.delete(k); else polynomial.set(k, value);
};
const addScaled = (target, source, coefficient) => {
  for (const [k, value] of source) {
    const [field, derivatives] = parse(k);
    addTerm(target, field, derivatives, mul(value, coefficient));
  }
  return target;
};
const derivative = (polynomial, mu) => {
  const result = new Map();
  for (const [k, value] of polynomial) {
    const [field, derivatives] = parse(k);
    derivatives[mu]++;
    addTerm(result, field, derivatives, value);
  }
  return result;
};
const equalPoly = (a, b) => {
  if (a.size !== b.size) return false;
  for (const [k, value] of a) if (JSON.stringify(value) !== JSON.stringify(b.get(k))) return false;
  return true;
};
const strength = (mu, nu) => addScaled(derivative(atom(`V${nu}`), mu), derivative(atom(`V${mu}`), nu), z(-1));
const addRow = (result, matrix, row, prefix, derivativeIndex, scale = z(1)) => {
  for (let component = 0; component < 4; component++) {
    let term = atom(`${prefix}${component}`);
    if (derivativeIndex !== null) term = derivative(term, derivativeIndex);
    addScaled(result, term, mul(matrix[row][component], scale));
  }
};

function chiralFermion(a, b, second) {
  const result = new Map();
  for (let mu = 0; mu < 4; mu++) {
    addScaled(result, derivative(atom("A"), mu), mul(lower(gammaUp[mu])[a][b], z(0, 1)));
    addScaled(result, derivative(atom("B"), mu), neg(lower(gamma5Gamma[mu])[a][b]));
  }
  addScaled(result, atom("F"), mul(C[a][b], z(0, -1)));
  addScaled(result, atom("G"), mul(lower(gamma5)[a][b], z(second ? -1 : 1)));
  return result;
}

function vectorFermion(a, b, second) {
  const result = new Map();
  for (let mu = 0; mu < 4; mu++) for (let nu = mu + 1; nu < 4; nu++) {
    const coefficient = divExact(mul(lower(commutator(mu, nu))[a][b], z(0, second ? 1 : -1)), 2);
    addScaled(result, strength(mu, nu), coefficient);
  }
  addScaled(result, atom("d"), lower(gamma5)[a][b]);
  return result;
}

function delta(susy, a, field) {
  const second = susy === 1;
  const fermion = second ? "l" : "p";
  const other = second ? "p" : "l";
  if (field === "A") return atom(`${fermion}${a}`);
  if (field === "B") { const out = new Map(); addRow(out, gamma5, a, fermion, null, z(0,1)); return out; }
  if (field === "F") { const out = new Map(); for (let mu=0;mu<4;mu++) addRow(out,gammaUp[mu],a,fermion,mu); return out; }
  if (field === "G") { const out = new Map(); for (let mu=0;mu<4;mu++) addRow(out,gamma5Gamma[mu],a,fermion,mu,z(0,second?-1:1)); return out; }
  if (field.startsWith("V")) { const out=new Map(); addRow(out,gammaDown[Number(field[1])],a,other,null,z(second?-1:1)); return out; }
  if (field === "d") { const out=new Map(); for(let mu=0;mu<4;mu++) addRow(out,gamma5Gamma[mu],a,other,mu,z(0,1)); return out; }
  const component = Number(field[1]);
  if (field[0] === "p" && !second) return chiralFermion(a,component,false);
  if (field[0] === "l" && second) return chiralFermion(a,component,true);
  if (field[0] === "l" && !second) return vectorFermion(a,component,false);
  if (field[0] === "p" && second) return vectorFermion(a,component,true);
  throw new Error(`unknown transformation ${susy},${a},${field}`);
}

function applyDelta(susy, a, polynomial) {
  const result = new Map();
  for (const [k, coefficient] of polynomial) {
    const [field, derivatives] = parse(k);
    let transformed = delta(susy,a,field);
    for(let mu=0;mu<4;mu++) for(let n=0;n<derivatives[mu];n++) transformed=derivative(transformed,mu);
    addScaled(result,transformed,coefficient);
  }
  return result;
}

function expectedClosure(left, right, field) {
  const [si,a] = left, [sj,b] = right;
  let result = new Map();
  if (si === sj) {
    for(let mu=0;mu<4;mu++) addScaled(result,derivative(atom(field),mu),mul(lower(gammaUp[mu])[a][b],z(0,2)));
    if(field.startsWith("V")) {
      result=new Map(); const nu=Number(field[1]);
      for(let mu=0;mu<4;mu++) addScaled(result,strength(mu,nu),mul(lower(gammaUp[mu])[a][b],z(0,2)));
    }
  }
  if(field.startsWith("V")) {
    const nu=Number(field[1]), ij=s2[si][sj];
    addScaled(result,derivative(atom("A"),nu),mul(mul(ij,C[a][b]),z(-2)));
    addScaled(result,derivative(atom("B"),nu),mul(mul(ij,lower(gamma5)[a][b]),z(0,-2)));
  }
  return result;
}

const charges=[]; for(let s=0;s<2;s++) for(let a=0;a<4;a++) charges.push([s,a]);
let checked=0;
for(let left=0;left<8;left++) for(let right=left;right<8;right++) for(const field of fields) {
  const actual=applyDelta(...charges[left],delta(...charges[right],field));
  addScaled(actual,applyDelta(...charges[right],delta(...charges[left],field)),z(1));
  assert(equalPoly(actual,expectedClosure(charges[left],charges[right],field)),`closure mismatch ${left},${right},${field}`);
  checked++;
}
assert(checked===612,"relation count mismatch");

const real = (value) => { assert(value[1]===0,"worldline coefficient not real"); return value[0]; };
const reduced=Array.from({length:8},()=>Array.from({length:8},()=>Array(8).fill(0)));
const iGamma5=mscale(gamma5,z(0,1));
const iGamma5Gamma0=mscale(gamma5Gamma[0],z(0,1));
for(let a=0;a<4;a++) {
  const first=[[identity4,0],[iGamma5,0],[gammaUp[0],0],[iGamma5Gamma0,0],[gammaDown[1],4],[gammaDown[2],4],[gammaDown[3],4],[iGamma5Gamma0,4]];
  const second=[[identity4,4,1],[iGamma5,4,1],[gammaUp[0],4,1],[iGamma5Gamma0,4,-1],[gammaDown[1],0,-1],[gammaDown[2],0,-1],[gammaDown[3],0,-1],[iGamma5Gamma0,0,1]];
  for(let row=0;row<8;row++) for(let component=0;component<4;component++) {
    reduced[a][row][first[row][1]+component]=real(first[row][0][a][component]);
    reduced[4+a][row][second[row][1]+component]=second[row][2]*real(second[row][0][a][component]);
  }
}
assert(JSON.stringify(reduced)===JSON.stringify(artifact.published_cv_l_matrices),"reduced CV anchor mismatch");
assert(artifact.report.component_relations_checked===checked,"artifact relation count mismatch");
assert(artifact.report.passed===true,"Rust report failed");
console.log(`verified ${checked} four-dimensional component relations and 512 reduced CV matrix entries`);
