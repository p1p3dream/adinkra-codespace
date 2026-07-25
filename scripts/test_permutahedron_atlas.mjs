import { readFile } from "node:fs/promises";
import {
  normalizePermutation,
  runSelfChecks,
  validateDataset,
} from "../visualizer/permutahedron_core.mjs";

const load = async path => JSON.parse(await readFile(path, "utf8"));
const requireCondition = (condition, message) => {
  if (!condition) throw new Error(message);
};

const [s4, s8, garden] = await Promise.all([
  load("data/permutahedron_s4_atlas.json"),
  load("data/permutahedron_s8_atlas.json"),
  load("data/permutahedron_s8_garden.json"),
]);

const selfChecks = runSelfChecks();
const s4Audit = validateDataset(s4);
const s8Audit = validateDataset(s8);
requireCondition(s4Audit.complete, `S4 dataset: ${s4Audit.issues.join("; ")}`);
requireCondition(s8Audit.complete, `S8 dataset: ${s8Audit.issues.join("; ")}`);
s8.permutations.forEach((entry, rank) => normalizePermutation(entry, 8, rank));

requireCondition(garden.passed, "Garden scan is not marked passed");
requireCondition(garden.cosets_scanned === 5040, "Garden scan does not cover 5,040 cosets");
requireCondition(garden.signable_cosets === 5040, "Not every R8 coset is signable");
requireCondition(garden.cosets.length === 5040, "Garden coset records are incomplete");
requireCondition(garden.dense_residual_entries === 0, "Garden scan contains a dense residual");

const atlasAbnormal = new Set(s8.abnormal_right_slices);
let recordAbnormal = 0;
for (let id = 0; id < garden.cosets.length; id++) {
  const record = garden.cosets[id];
  requireCondition(record.right_slice_id === id, `Garden slice order mismatch at ${id}`);
  requireCondition(record.garden_signing_exists, `Slice ${id} has no Garden signing`);
  requireCondition(record.equation_rank === 45 && record.nullity === 19, `Slice ${id} has a different affine dimension`);
  requireCondition(/^[0-9a-f]{16}$/.test(record.canonical_sign_mask), `Slice ${id} has an invalid sign mask`);
  requireCondition(record.abnormal === atlasAbnormal.has(id), `Slice ${id} abnormal flag disagrees with the atlas`);
  recordAbnormal += Number(record.abnormal);
}
requireCondition(recordAbnormal === 168, "Garden records do not contain 168 ab-normal cosets");

const namedById = new Map(garden.named_octets.map(record => [record.id, record]));
for (const representation of s8.representations) {
  const scan = namedById.get(representation.id);
  requireCondition(scan, `Missing Garden result for ${representation.id}`);
  requireCondition(scan.right_slice_id === representation.right_slice_id, `${representation.id} slice mismatch`);
  requireCondition(scan.published_status === representation.garden_status, `${representation.id} publication-status mismatch`);
}

console.log(JSON.stringify({
  self_checks: selfChecks.passed,
  s4_vertices: s4Audit.vertices,
  s8_vertices: s8Audit.vertices,
  garden_cosets: garden.cosets_scanned,
  abnormal_cosets: recordAbnormal,
  issues: 0,
}));
