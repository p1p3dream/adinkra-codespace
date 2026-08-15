#!/usr/bin/env node

import fs from "node:fs";

const artifact = JSON.parse(fs.readFileSync("data/chiral_tensor_4d.json", "utf8"));
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const gcd = (a, b) => { a = Math.abs(a); b = Math.abs(b); while (b) [a, b] = [b, a % b]; return a; };
const q = (r = 0, i = 0, d = 1) => {
  if (d < 0) [r, i, d] = [-r, -i, -d];
  const g = gcd(gcd(r, i), d) || 1;
  return [r / g, i / g, d / g];
};
const add = (a, b) => q(a[0] * b[2] + b[0] * a[2], a[1] * b[2] + b[1] * a[2], a[2] * b[2]);
const mul = (a, b) => q(a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0], a[2] * b[2]);
const neg = (a) => q(-a[0], -a[1], a[2]);
const zero = (a) => a[0] === 0 && a[1] === 0;
const equalQ = (a, b) => a[0] === b[0] && a[1] === b[1] && a[2] === b[2];

const I2 = [[q(1), q()], [q(), q(1)]];
const s1 = [[q(), q(1)], [q(1), q()]];
const s2 = [[q(), q(0, -1)], [q(0, 1), q()]];
const s3 = [[q(1), q()], [q(), q(-1)]];
const kron = (a, b) => Array.from({length: 4}, (_, row) =>
  Array.from({length: 4}, (_, column) => mul(a[row >> 1][column >> 1], b[row & 1][column & 1])));
const mscale = (a, c) => a.map((row) => row.map((value) => mul(value, c)));
const madd = (a, b) => a.map((row, i) => row.map((value, j) => add(value, b[i][j])));
const mmul = (a, b) => Array.from({length: 4}, (_, row) => Array.from({length: 4}, (_, column) => {
  let out = q();
  for (let inner = 0; inner < 4; inner++) out = add(out, mul(a[row][inner], b[inner][column]));
  return out;
}));

const gammaUp = [
  mscale(kron(s3, s2), q(0, 1)),
  kron(I2, s1),
  kron(s2, s2),
  kron(I2, s3),
];
const gammaDown = gammaUp.map((matrix, mu) => mu === 0 ? mscale(matrix, q(-1)) : matrix);
const gamma5 = mscale(kron(s1, s2), q(-1));
const C = mscale(kron(s3, s2), q(0, -1));
const lower = (matrix) => mmul(matrix, C);
const gamma5Gamma = gammaUp.map((gamma) => mmul(gamma5, gamma));
const commDown = (mu, nu) => madd(mmul(gammaDown[mu], gammaDown[nu]), mscale(mmul(gammaDown[nu], gammaDown[mu]), q(-1)));

const identity4 = Array.from({length: 4}, (_, row) => Array.from({length: 4}, (_, column) => q(row === column ? 1 : 0)));
for (let mu = 0; mu < 4; mu++) for (let nu = 0; nu < 4; nu++) {
  const expected = mscale(identity4, q(mu === nu ? (mu === 0 ? -2 : 2) : 0));
  const actual = madd(mmul(gammaUp[mu], gammaUp[nu]), mmul(gammaUp[nu], gammaUp[mu]));
  assert(actual.every((row, i) => row.every((value, j) => equalQ(value, expected[i][j]))), "Clifford mismatch");
}

const fields = ["A", "B", "F", "G", "phi", "T01", "T02", "T03", "T12", "T13", "T23",
  "p0", "p1", "p2", "p3", "c0", "c1", "c2", "c3"];
const key = (field, derivatives) => `${field}|${derivatives.join("")}`;
const parse = (entry) => { const [field, digits] = entry.split("|"); return [field, [...digits].map(Number)]; };
const atom = (field) => new Map([[key(field, [0,0,0,0]), q(1)]]);
const addTerm = (polynomial, field, derivatives, coefficient) => {
  if (zero(coefficient)) return;
  const k = key(field, derivatives), value = add(polynomial.get(k) ?? q(), coefficient);
  if (zero(value)) polynomial.delete(k); else polynomial.set(k, value);
};
const addScaled = (target, source, coefficient) => {
  for (const [k, value] of source) { const [field, derivatives] = parse(k); addTerm(target, field, derivatives, mul(value, coefficient)); }
  return target;
};
const derivative = (polynomial, mu) => {
  const result = new Map();
  for (const [k, value] of polynomial) { const [field, derivatives] = parse(k); derivatives[mu]++; addTerm(result, field, derivatives, value); }
  return result;
};
const equalPoly = (a, b) => {
  if (a.size !== b.size) return false;
  for (const [k, value] of a) if (!b.has(k) || !equalQ(value, b.get(k))) return false;
  return true;
};
const metric = (mu) => mu === 0 ? -1 : 1;
const epsilonLower = (indices) => {
  if (new Set(indices).size !== 4) return 0;
  let inversions = 0;
  for (let i=0;i<4;i++) for(let j=i+1;j<4;j++) if(indices[i]>indices[j]) inversions++;
  return inversions % 2 === 0 ? -1 : 1;
};
const tensor = (mu, nu) => {
  if (mu === nu) return new Map();
  const out = new Map(), ordered = mu < nu ? [mu, nu, 1] : [nu, mu, -1];
  addScaled(out, atom(`T${ordered[0]}${ordered[1]}`), q(ordered[2]));
  return out;
};
const strength = (alpha, mu, nu) => {
  const out = derivative(tensor(mu,nu),alpha);
  addScaled(out,derivative(tensor(nu,alpha),mu),q(1));
  addScaled(out,derivative(tensor(alpha,mu),nu),q(1));
  return out;
};
const addRow = (result, matrix, row, prefix, derivativeIndex, scale = q(1)) => {
  for (let component=0;component<4;component++) {
    let term=atom(`${prefix}${component}`); if(derivativeIndex!==null) term=derivative(term,derivativeIndex);
    addScaled(result,term,mul(matrix[row][component],scale));
  }
};

function chiralFermion(a,b,second) {
  const out=new Map();
  for(let mu=0;mu<4;mu++) {
    addScaled(out,derivative(atom("A"),mu),mul(lower(gammaUp[mu])[a][b],q(0,second?-1:1)));
    addScaled(out,derivative(atom("B"),mu),neg(lower(gamma5Gamma[mu])[a][b]));
  }
  addScaled(out,atom("F"),mul(C[a][b],q(0,-1)));
  addScaled(out,atom("G"),lower(gamma5)[a][b]);
  return out;
}

function tensorFermion(a,b,dualSign) {
  const out=new Map();
  for(let mu=0;mu<4;mu++) {
    addScaled(out,derivative(atom("phi"),mu),mul(lower(gammaUp[mu])[a][b],q(0,1)));
    const g5g=lower(gamma5Gamma[mu]);
    for(let rho=0;rho<4;rho++) for(let sigma=0;sigma<4;sigma++) for(let tau=sigma+1;tau<4;tau++) {
      const eps=epsilonLower([mu,rho,sigma,tau]);
      if(eps) addScaled(out,derivative(atom(`T${sigma}${tau}`),rho),mul(g5g[a][b],q(dualSign*2*eps*metric(rho)*metric(sigma)*metric(tau))));
    }
  }
  return out;
}

function delta(susy,a,field) {
  const second=susy===1, own=second?"c":"p";
  if(field==="A") return addScaled(new Map(),atom(`${own}${a}`),q(second?-1:1));
  if(field==="B") { const out=new Map(); addRow(out,gamma5,a,own,null,q(0,1)); return out; }
  if(field==="F") { const out=new Map(); for(let mu=0;mu<4;mu++) addRow(out,gammaUp[mu],a,own,mu); return out; }
  if(field==="G") { const out=new Map(); for(let mu=0;mu<4;mu++) addRow(out,gamma5Gamma[mu],a,own,mu,q(0,1)); return out; }
  if(field==="phi") return atom(`${second?"p":"c"}${a}`);
  if(field.startsWith("T")) {
    const mu=Number(field[1]),nu=Number(field[2]),out=new Map();
    addRow(out,commDown(mu,nu),a,second?"p":"c",null,q(second?1:-1,0,4));
    return out;
  }
  const component=Number(field[1]);
  if(field[0]==="p"&&!second) return chiralFermion(a,component,false);
  if(field[0]==="c"&&second) return chiralFermion(a,component,true);
  if(field[0]==="c"&&!second) return tensorFermion(a,component,1);
  if(field[0]==="p"&&second) return tensorFermion(a,component,-1);
  throw new Error(`unknown transformation ${susy},${a},${field}`);
}

function applyDelta(susy,a,polynomial) {
  const out=new Map();
  for(const [k,coefficient] of polynomial) {
    const [field,derivatives]=parse(k); let transformed=delta(susy,a,field);
    for(let mu=0;mu<4;mu++) for(let n=0;n<derivatives[mu];n++) transformed=derivative(transformed,mu);
    addScaled(out,transformed,coefficient);
  }
  return out;
}

function expectedClosure(left,right,field) {
  const [si,a]=left,[sj,b]=right;
  if(!field.startsWith("T")) {
    const out=new Map();
    if(si===sj) for(let mu=0;mu<4;mu++) addScaled(out,derivative(atom(field),mu),mul(lower(gammaUp[mu])[a][b],q(0,2)));
    return out;
  }
  const mu=Number(field[1]),nu=Number(field[2]),out=new Map();
  if(si===sj) for(let alpha=0;alpha<4;alpha++) addScaled(out,strength(alpha,mu,nu),mul(lower(gammaUp[alpha])[a][b],q(0,2)));
  const addGauge=(derivativeIndex,gamma,sign) => {
    addScaled(out,derivative(atom("A"),derivativeIndex),mul(mul(gamma[a][b],s1[si][sj]),q(0,sign)));
    let gammaGamma5=q(); for(let c=0;c<4;c++) gammaGamma5=add(gammaGamma5,mul(gamma[a][c],gamma5[b][c]));
    addScaled(out,derivative(atom("B"),derivativeIndex),mul(mul(gammaGamma5,s2[si][sj]),q(0,sign)));
    addScaled(out,derivative(atom("phi"),derivativeIndex),mul(mul(gamma[a][b],s3[si][sj]),q(0,-sign)));
  };
  addGauge(nu,lower(gammaDown[mu]),1); addGauge(mu,lower(gammaDown[nu]),-1);
  return out;
}

const charges=[]; for(let s=0;s<2;s++) for(let a=0;a<4;a++) charges.push([s,a]);
let checked=0;
for(let left=0;left<8;left++) for(let right=left;right<8;right++) for(const field of fields) {
  const actual=applyDelta(...charges[left],delta(...charges[right],field));
  addScaled(actual,applyDelta(...charges[right],delta(...charges[left],field)),q(1));
  assert(equalPoly(actual,expectedClosure(charges[left],charges[right],field)),`closure mismatch ${left},${right},${field}`);
  checked++;
}
assert(checked===684,"relation count mismatch");

const real = (value) => { assert(value[1]===0&&value[2]===1,"worldline coefficient not integral real"); return value[0]; };
const reduced=Array.from({length:8},()=>Array.from({length:8},()=>Array(8).fill(0)));
const iGamma5=mscale(gamma5,q(0,1)),iGamma5Gamma0=mscale(gamma5Gamma[0],q(0,1));
const spatial=[[1,2],[2,3],[3,1]];
for(let a=0;a<4;a++) for(let component=0;component<4;component++) {
  const first=[[identity4,0,1],[iGamma5,0,1],[gammaUp[0],0,1],[iGamma5Gamma0,0,1],[identity4,4,1]];
  const second=[[identity4,4,-1],[iGamma5,4,1],[gammaUp[0],4,1],[iGamma5Gamma0,4,1],[identity4,0,1]];
  for(let row=0;row<5;row++) {
    reduced[a][row][first[row][1]+component]=first[row][2]*real(first[row][0][a][component]);
    reduced[4+a][row][second[row][1]+component]=second[row][2]*real(second[row][0][a][component]);
  }
  for(let offset=0;offset<3;offset++) {
    const comm=commDown(...spatial[offset]),half=q(1,0,2);
    reduced[a][5+offset][4+component]=-real(mul(comm[a][component],half));
    reduced[4+a][5+offset][component]=real(mul(comm[a][component],half));
  }
}
assert(JSON.stringify(reduced)===JSON.stringify(artifact.published_ct_l_matrices),"reduced CT anchor mismatch");
assert(artifact.report.component_relations_checked===checked,"artifact relation count mismatch");
assert(artifact.report.passed===true,"Rust report failed");
console.log(`verified ${checked} four-dimensional component relations and 512 reduced CT matrix entries`);
