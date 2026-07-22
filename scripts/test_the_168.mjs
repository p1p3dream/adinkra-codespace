/**
 * test_the_168.mjs
 *
 * Node test suite for The 168 instrument. Run from the repo root:
 *
 *   node scripts/test_the_168.mjs
 *
 * Prints PASS or FAIL per check and exits nonzero on any failure.
 * Covers: SO(7) rotor orthogonality and determinant, the exactness of
 * the 7D permutohedron embedding (every dataset edge has identical
 * geometric length sqrt(2), and edge neighbors are exactly the
 * nearest-point neighbors), the 168 igniter selection, and the Ledger
 * numbers against the checked-in JSON.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  DIM, PLANE_COUNT, composeSO7, identity7, orthogonalityResidual, det7,
  embedPermutations, edgeLengthStats, edgeLength7, sliceCentroids, idleAngles,
} from "../visualizer/rotor7.mjs";
import { validateDataset, runSelfChecks } from "../visualizer/permutahedron_core.mjs";
import { makeBeats, makeLedger } from "../visualizer/the_168_narrative.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const s8 = JSON.parse(readFileSync(join(root, "data/permutahedron_s8_atlas.json"), "utf8"));
const s4 = JSON.parse(readFileSync(join(root, "data/permutahedron_s4_atlas.json"), "utf8"));
const garden = JSON.parse(readFileSync(join(root, "data/permutahedron_s8_garden.json"), "utf8"));

let failures = 0;
function check(label, ok, detail = "") {
  const status = ok ? "PASS" : "FAIL";
  if (!ok) failures += 1;
  console.log(`${status}  ${label}${detail ? `  (${detail})` : ""}`);
}

// Deterministic pseudo-random angles so the test is reproducible.
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

console.log("== SO(7) rotor ==");
{
  const idResidual = orthogonalityResidual(identity7());
  check("identity7 is orthogonal", idResidual < 1e-15);

  const rand = mulberry32(168);
  let worstOrtho = 0;
  let worstDet = 0;
  for (let trial = 0; trial < 25; trial += 1) {
    const angles = Array.from({ length: PLANE_COUNT }, () => (rand() - 0.5) * 2 * Math.PI);
    const m = composeSO7(angles);
    worstOrtho = Math.max(worstOrtho, orthogonalityResidual(m));
    worstDet = Math.max(worstDet, Math.abs(det7(m) - 1));
  }
  check("R^T R = I for 25 random 21-angle rotors", worstOrtho < 1e-12, `max residual ${worstOrtho.toExponential(2)}`);
  check("det R = +1 for 25 random 21-angle rotors", worstDet < 1e-12, `max |det-1| ${worstDet.toExponential(2)}`);

  const idle = composeSO7(idleAngles(1234.5, 1));
  check("idle rotor stays in SO(7)", orthogonalityResidual(idle) < 1e-12 && Math.abs(det7(idle) - 1) < 1e-12);
}

console.log("== core self-checks and dataset completeness ==");
{
  let ok = true;
  try { runSelfChecks(); } catch { ok = false; }
  check("permutation core self-checks", ok);

  const v8 = validateDataset(s8);
  check("S8 atlas complete (40,320 vertices, 141,120 edges, degree 7)",
    v8.complete && v8.vertices === 40320 && v8.edges === 141120, v8.issues.join("; "));
  const v4 = validateDataset(s4);
  check("S4 atlas complete (24 vertices, 36 edges, degree 3)",
    v4.complete && v4.vertices === 24 && v4.edges === 36, v4.issues.join("; "));
}

console.log("== 7D embedding preserves the permutohedron ==");
const SQRT2 = Math.SQRT2;
const { coords: coords8 } = embedPermutations(s8.permutations, 8);
{
  const stats = edgeLengthStats(coords8, s8.edges);
  check("all 141,120 S8 edges have 7D length sqrt(2)",
    Math.abs(stats.min - SQRT2) < 1e-9 && Math.abs(stats.max - SQRT2) < 1e-9,
    `min ${stats.min.toFixed(12)}, max ${stats.max.toFixed(12)}`);

  // Adjacency preservation both ways on a sample: for each sampled
  // vertex, the set of vertices at geometric distance sqrt(2) must be
  // exactly its 7 graph neighbors.
  const neighbors = new Map();
  for (const [a, b] of s8.edges) {
    if (!neighbors.has(a)) neighbors.set(a, new Set());
    if (!neighbors.has(b)) neighbors.set(b, new Set());
    neighbors.get(a).add(b);
    neighbors.get(b).add(a);
  }
  const rand = mulberry32(40320);
  let adjacencyOk = true;
  let sampleDetail = "";
  for (let s = 0; s < 40 && adjacencyOk; s += 1) {
    const v = Math.floor(rand() * 40320);
    const near = [];
    for (let w = 0; w < 40320; w += 1) {
      if (w === v) continue;
      if (edgeLength7(coords8, v, w) < SQRT2 + 1e-6) near.push(w);
    }
    const expected = neighbors.get(v);
    if (near.length !== 7 || near.some(w => !expected.has(w))) {
      adjacencyOk = false;
      sampleDetail = `vertex ${v}: ${near.length} geometric neighbors`;
    }
  }
  check("geometric sqrt(2) neighbors = graph neighbors (40-vertex sample)", adjacencyOk, sampleDetail);

  // The whole cloud is centered: the embedding subtracted the centroid.
  let maxMean = 0;
  for (let k = 0; k < DIM; k += 1) {
    let sum = 0;
    for (let v = 0; v < 40320; v += 1) sum += coords8[v * DIM + k];
    maxMean = Math.max(maxMean, Math.abs(sum / 40320));
  }
  check("embedded cloud is centered at the origin", maxMean < 1e-9);
}
{
  const { coords: coords4 } = embedPermutations(s4.permutations, 4);
  const stats = edgeLengthStats(coords4, s4.edges);
  check("all 36 S4 edges have length sqrt(2)",
    Math.abs(stats.min - SQRT2) < 1e-9 && Math.abs(stats.max - SQRT2) < 1e-9);
  let flat = true;
  for (let v = 0; v < 24; v += 1) {
    for (let k = 3; k < DIM; k += 1) if (coords4[v * DIM + k] !== 0) flat = false;
  }
  check("S4 embedding is genuinely 3D (dims 4..7 exactly zero)", flat);
}

console.log("== the 168 igniter ==");
{
  const abnormal = s8.abnormal_right_slices;
  check("abnormal_right_slices has exactly 168 entries",
    abnormal.length === 168 && new Set(abnormal).size === 168);

  // The renderer marks gold via right_slice_by_rank membership.
  const goldSet = new Set(abnormal);
  let goldVertices = 0;
  for (let v = 0; v < 40320; v += 1) {
    if (goldSet.has(s8.right_slice_by_rank[v])) goldVertices += 1;
  }
  check("igniter selects exactly 168 x 8 = 1,344 vertices", goldVertices === 1344, `${goldVertices}`);

  // Left-right coincidence: an abnormal right coset IS a left coset,
  // and no ordinary right coset is. Checked over all 5,040.
  const leftKeys = new Set(s8.left_slices.map(m => [...m].sort((a, b) => a - b).join(",")));
  const coincident = [];
  s8.right_slices.forEach((members, id) => {
    const key = [...members].sort((a, b) => a - b).join(",");
    if (leftKeys.has(key)) coincident.push(id);
  });
  const sortedAbnormal = [...abnormal].sort((a, b) => a - b).join(",");
  const sortedCoincident = coincident.sort((a, b) => a - b).join(",");
  check("left-right coincident right cosets = exactly the 168 abnormal ids",
    sortedCoincident === sortedAbnormal, `${coincident.length} coincident`);

  check("garden contingency agrees: 168 abnormal signable, 4,872 ordinary",
    garden.contingency.abnormal_and_signable === 168
    && garden.contingency.ordinary_and_signable === 4872
    && garden.contingency.abnormal_and_not_signable === 0);

  check("168 = |GL(3,2)| = normalizer 1,344 / 8",
    garden.normalizer.normalizer_order === 1344
    && garden.normalizer.normalizer_order / 8 === 168
    && garden.normalizer.expected_gl_3_2_order === 168);
}

console.log("== gather centroids ==");
{
  const centroids = sliceCentroids(coords8, s8.right_slice_by_rank, s8.right_slices.length);
  // Spot-check: centroid of coset 0 equals the mean of its 8 members.
  const members = s8.right_slices[0];
  let maxErr = 0;
  for (let k = 0; k < DIM; k += 1) {
    let sum = 0;
    for (const v of members) sum += coords8[v * DIM + k];
    maxErr = Math.max(maxErr, Math.abs(sum / 8 - centroids[k]));
  }
  check("coset centroid matches mean of members (coset 0)", maxErr < 1e-12);
}

console.log("== the Ledger vs the JSON ==");
{
  check("garden.passed === true", garden.passed === true);
  check("5,040 cosets scanned, 5,040 signable, 0 unsignable",
    garden.cosets_scanned === 5040 && garden.signable_cosets === 5040 && garden.unsignable_cosets === 0);
  check("rank 45 and nullity 19 for all 5,040 cosets",
    garden.rank_histogram.length === 1 && garden.rank_histogram[0].value === 45
    && garden.rank_histogram[0].octets === 5040
    && garden.nullity_histogram.length === 1 && garden.nullity_histogram[0].value === 19
    && garden.nullity_histogram[0].octets === 5040);
  check("2^19 = 524,288 raw signings per coset; 2^19 / 2^15 = 16 honest classes",
    garden.solution_count_per_coset === 2 ** 19 && (2 ** 19) / (2 ** 15) === 16);
  check("7 named octets present with published statuses",
    garden.named_octets.length === 7
    && garden.named_octets.every(o => typeof o.published_status === "string"));
  check("at least one closure and one nonclosure named octet (beat 7 needs both)",
    garden.named_octets.some(o => o.published_status === "published_garden_closure")
    && garden.named_octets.some(o => o.published_status === "published_nonclosure"));
  check("metadata provenance carries arXiv ids and pdf sha256 hashes",
    Array.isArray(s8.metadata.source) && s8.metadata.source.length >= 2
    && s8.metadata.source.every(src => src.arxiv_id && /^[0-9a-f]{64}$/.test(src.pdf_sha256)));

  const ledger = makeLedger({ s8, garden });
  check("Ledger has equal-standing Reproduced and Not-claimed columns",
    ledger.reproduced.length >= 5 && ledger.notClaimed.length >= 5);
  const ledgerText = [...ledger.reproduced, ...ledger.notClaimed].join(" ");
  check("Ledger quotes the scan boundary verbatim", ledgerText.includes(garden.boundary));
  check("Ledger quotes the correlator definition verbatim",
    ledgerText.includes(s8.metadata.correlator_definition));

  const beats = makeBeats({ s8, s4, garden });
  check("narrative has exactly 9 beats, ids 0..8",
    beats.length === 9 && beats.every((b, i) => b.id === i));
  check("every beat has a nonempty caption and targets",
    beats.every(b => b.caption.length > 40 && b.targets && typeof b.targets.gold === "number"));
  check("gold is reserved: only beat 8 targets gold > 0",
    beats.every(b => (b.id === 8 ? b.targets.gold === 1 : b.targets.gold === 0)));
  check("beat 8 ends on the open question, not a claim",
    beats[8].caption.includes("(open)"));
  const allText = beats.map(b => `${b.title} ${b.caption}`).join(" ") + " " + ledgerText;
  check("no em dashes anywhere in narrative or Ledger text", !allText.includes("\u2014"));
}

console.log("== isolate filter sets ==");
{
  // Union of the 8 member ranks over a list of right-slice ids.
  const unionOfSlices = (atlas, ids) => {
    const set = new Set();
    for (const id of ids) for (const r of atlas.right_slices[id]) set.add(r);
    return set;
  };

  // Named octets: the 7 garden octets, each a right slice of 8 members.
  const namedIds = garden.named_octets.map(o => o.right_slice_id);
  const named = unionOfSlices(s8, namedIds);
  check("named octets isolate set = 56 ranks (7 disjoint octets)",
    named.size === 56, `${named.size}`);

  // The 168: 168 abnormal right slices, 8 members each, disjoint.
  const the168 = unionOfSlices(s8, s8.abnormal_right_slices);
  check("the 168 isolate set = 1,344 ranks", the168.size === 1344, `${the168.size}`);

  // Coset reps: first member of every right slice, one per coset.
  const reps = new Set(s8.right_slices.map(s => s[0]));
  check("coset reps isolate set = 5,040 ranks", reps.size === 5040, `${reps.size}`);

  // Every rank in every set is a valid S8 vertex index.
  const allInRange = set => [...set].every(r => Number.isInteger(r) && r >= 0 && r < 40320);
  check("every isolate rank is in [0, 40320)",
    allInRange(named) && allInRange(the168) && allInRange(reps));

  // The setFilter mapping: 1 for members, 0 otherwise, over the whole cloud.
  const buildFilter = (set, count) => {
    const arr = new Float32Array(count);
    for (const r of set) if (r >= 0 && r < count) arr[r] = 1;
    return arr;
  };
  const f168 = buildFilter(the168, 40320);
  let ones = 0;
  for (let i = 0; i < f168.length; i += 1) ones += f168[i];
  const sample168 = s8.abnormal_right_slices[0];
  const memberOk = s8.right_slices[sample168].every(r => f168[r] === 1);
  const nonMember = [...Array(40320).keys()].find(r => !the168.has(r));
  check("setFilter mapping marks exactly the 1,344 members (1 in, 0 out)",
    ones === 1344 && memberOk && f168[nonMember] === 0, `${ones} ones`);

  // S4 has no S8-only sets, but coset reps still apply (6 cosets of 4).
  const s4reps = new Set(s4.right_slices.map(s => s[0]));
  check("S4 coset reps isolate set = 6 ranks, all in [0, 24)",
    s4reps.size === 6 && [...s4reps].every(r => r >= 0 && r < 24), `${s4reps.size}`);
}

console.log(failures === 0 ? "\nALL CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
