#!/usr/bin/env node

import fs from "node:fs";

const report = JSON.parse(fs.readFileSync("results/maxwell_worldline_search.json", "utf8"));
const phantom = JSON.parse(fs.readFileSync("results/maxwell_phantom.json", "utf8"));
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const z = (real = 0, imag = 0) => [real, imag];
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
const mul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const scale = (matrix, coefficient) => matrix.map((row) => row.map((value) => mul(value, coefficient)));
const matrixAdd = (a, b) => a.map((row, i) => row.map((value, j) => add(value, b[i][j])));
const matrixMul = (a, b) => Array.from({length: 4}, (_, row) => Array.from({length: 4}, (_, column) => {
  let value = z();
  for (let inner = 0; inner < 4; inner++) value = add(value, mul(a[row][inner], b[inner][column]));
  return value;
}));
const real = (value) => { assert(value[1] === 0, "expected real linkage"); return value[0]; };

const I2 = [[z(1), z()], [z(), z(1)]];
const s1 = [[z(), z(1)], [z(1), z()]];
const s2 = [[z(), z(0, -1)], [z(0, 1), z()]];
const s3 = [[z(1), z()], [z(), z(-1)]];
const kron = (a, b) => Array.from({length: 4}, (_, row) =>
  Array.from({length: 4}, (_, column) => mul(a[row >> 1][column >> 1], b[row & 1][column & 1])));
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
const commutator = (mu, nu) => matrixAdd(
  matrixMul(gammaUp[mu], gammaUp[nu]),
  scale(matrixMul(gammaUp[nu], gammaUp[mu]), z(-1)),
);
const gamma5Gamma = gammaUp.map((matrix) => matrixMul(gamma5, matrix));
const epsilon = (a, b, c) => {
  if (a === b || a === c || b === c) return 0;
  return (Number(a > b) + Number(a > c) + Number(b > c)) % 2 === 0 ? 1 : -1;
};

const sourceMaxwell = phantom.temporal_down.map((charge) => charge.slice(0, 4).map((row) =>
  row.map(({real: value, imag}) => { assert(imag === 0, "complex worldline input"); return value; })));
const identity4 = Array.from({length: 4}, (_, row) => Array.from({length: 4}, (_, column) => z(row === column ? 1 : 0)));
const chiralRows = [identity4, scale(gamma5, z(0, 1)), gammaUp[0], scale(gamma5Gamma[0], z(0, 1))];
const sourceChiral = Array.from({length: 4}, (_, charge) => Array.from({length: 4}, (_, boson) =>
  Array.from({length: 4}, (_, fermion) => real(chiralRows[boson][charge][fermion]))));

function provisional(worldline) {
  const up = Array.from({length: 4}, () => Array.from({length: 7}, () => Array(4).fill(0)));
  const temporal = Array.from({length: 4}, () => Array.from({length: 7}, () => Array(4).fill(0)));
  const spatial = Array.from({length: 3}, () => Array.from({length: 4}, () => Array.from({length: 7}, () => Array(4).fill(0))));
  for (let charge = 0; charge < 4; charge++) for (let boson = 0; boson < 4; boson++) for (let fermion = 0; fermion < 4; fermion++) {
    up[charge][boson][fermion] = temporal[charge][boson][fermion] = worldline[charge][boson][fermion];
  }
  const magneticPairs = [[2,3],[3,1],[1,2]];
  for (let charge=0;charge<4;charge++) for(let fermion=0;fermion<4;fermion++) {
    for(let magnetic=0;magnetic<3;magnetic++) {
      const [mu,nu]=magneticPairs[magnetic];
      up[charge][4+magnetic][fermion]=real(lower(commutator(mu,nu))[charge][fermion])/(-2);
    }
  }
  for(let derivative=0;derivative<3;derivative++) {
    const gg=matrixMul(gammaDown[0],gammaDown[1+derivative]);
    for(let charge=0;charge<4;charge++) for(let fermion=0;fermion<4;fermion++) {
      for(let visible=0;visible<4;visible++) {
        let value=0;
        for(let mixed=0;mixed<4;mixed++) value-=real(gg[charge][mixed])*temporal[mixed][visible][fermion];
        if(visible<3) value+=real(lower(commutator(1+derivative,1+visible))[charge][fermion])/2;
        spatial[derivative][charge][visible][fermion]=value;
      }
      for(let magnetic=0;magnetic<3;magnetic++) {
        let value=0;
        for(let electric=0;electric<3;electric++) value+=epsilon(magnetic,derivative,electric)*temporal[charge][electric][fermion];
        spatial[derivative][charge][4+magnetic][fermion]=value;
      }
    }
  }
  return {up,down:[temporal,...spatial]};
}

function gate(worldline) {
  const {up,down}=provisional(worldline);
  let bosonicResidual=0,fermionicResidual=0;
  for(let left=0;left<4;left++) for(let right=left;right<4;right++) {
    const omega=Array.from({length:4},()=>Array.from({length:7},()=>Array(7).fill(0)));
    for(let mu=0;mu<4;mu++) {
      const fermionic=Array.from({length:4},(_,row)=>Array.from({length:4},(_,column)=>
        Array.from({length:7},(_,boson)=>up[left][boson][row]*down[mu][right][boson][column]+up[right][boson][row]*down[mu][left][boson][column]).reduce((a,b)=>a+b,0)));
      const lambda=fermionic[0][0];
      for(let row=0;row<4;row++) for(let column=0;column<4;column++) fermionicResidual+=Number(fermionic[row][column] !== (row===column?lambda:0));
      for(let row=0;row<7;row++) for(let column=0;column<7;column++) {
        const product=Array.from({length:4},(_,f)=>down[mu][left][row][f]*up[right][column][f]+down[mu][right][row][f]*up[left][column][f]).reduce((a,b)=>a+b,0);
        omega[mu][row][column]=product-(row===column?lambda:0);
      }
    }
    const timePhantom=Array.from({length:7},(_,row)=>omega[0][row].slice(4));
    for(let row=0;row<7;row++) for(let magnetic=0;magnetic<3;magnetic++) omega[0][row][4+magnetic]=0;
    for(let derivative=0;derivative<3;derivative++) for(let electric=0;electric<3;electric++) for(let row=0;row<7;row++) for(let magnetic=0;magnetic<3;magnetic++) {
      omega[1+derivative][row][electric]+=epsilon(derivative,electric,magnetic)*timePhantom[row][magnetic];
    }
    const pivot=Array.from({length:7},(_,row)=>omega[1][row][4]);
    for(let row=0;row<7;row++) { omega[1][row][4]=0; omega[2][row][5]-=pivot[row]; omega[3][row][6]-=pivot[row]; }
    bosonicResidual+=omega.flat(2).filter((value)=>value!==0).length;
  }
  return bosonicResidual===0&&fermionicResidual===0;
}

const permutations=[];
for(let a=0;a<4;a++) for(let b=0;b<4;b++) for(let c=0;c<4;c++) for(let d=0;d<4;d++) if(new Set([a,b,c,d]).size===4) permutations.push([a,b,c,d]);
const frames=[];
for(const permutation of permutations) for(let mask=0;mask<16;mask++) frames.push({permutation,signs:Array.from({length:4},(_,i)=>(mask&(1<<i))?-1:1)});
assert(frames.length===384,"frame inventory mismatch");
const transform=(input,boson,fermion)=>Array.from({length:4},(_,charge)=>Array.from({length:4},(_,row)=>Array.from({length:4},(_,column)=>
  boson.signs[row]*fermion.signs[column]*input[charge][boson.permutation[row]][fermion.permutation[column]])));
const equalMatrix=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
function search(input) {
  let examined=0,normalized=0,passed=0;
  for(const boson of frames) for(const fermion of frames) {
    examined++;
    const candidate=transform(input,boson,fermion);
    if(!equalMatrix(candidate[0],sourceMaxwell[0])) continue;
    normalized++;
    passed+=Number(gate(candidate));
  }
  return {examined,normalized,passed};
}
const scrambled=transform(sourceMaxwell,{permutation:[2,0,3,1],signs:[-1,1,-1,1]},{permutation:[1,3,0,2],signs:[1,-1,1,-1]});
const sourceResult=search(sourceMaxwell),scrambledResult=search(scrambled),chiralResult=search(sourceChiral);
assert(sourceResult.examined===report.maxwell_source_basis.frame_pairs_examined&&sourceResult.normalized===384&&sourceResult.passed===8,"source search mismatch");
assert(scrambledResult.examined===report.maxwell_scrambled_basis.frame_pairs_examined&&scrambledResult.normalized===384&&scrambledResult.passed===8,"scrambled search mismatch");
assert(chiralResult.examined===report.chiral_negative_control.frame_pairs_examined&&chiralResult.normalized===384&&chiralResult.passed===0,"chiral search mismatch");
assert(report.passed,"Rust search report failed");
console.log("verified 442,368 signed frame pairs: Maxwell 8 passers in two bases, chiral 0");
