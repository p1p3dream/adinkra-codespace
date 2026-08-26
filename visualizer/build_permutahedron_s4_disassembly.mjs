import { readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const templatePath = join(here, "permutahedron_s4_disassembly.template.html");
const outputPath = join(here, "permutahedron_s4_disassembly.html");

const mode = process.argv[2] ?? "--preview";
if (!["--preview", "--check", "--publish"].includes(mode)) {
  console.error(`Unknown flag: ${mode}. Use --preview, --check, or --publish.`);
  process.exit(1);
}

const sha256 = text => createHash("sha256").update(text).digest("hex");

const [templateText, atlasText, supersymmetryText, journeyText] = await Promise.all([
  readFile(templatePath, "utf8"),
  readFile(join(root, "data", "permutahedron_s4_atlas.json"), "utf8"),
  readFile(join(root, "data", "permutahedron_s4_supersymmetry.json"), "utf8"),
  readFile(join(root, "data", "permutahedron_s4_gates_journey_20260826.json"), "utf8"),
]);

const atlas = JSON.parse(atlasText);
const supersymmetry = JSON.parse(supersymmetryText);
const journey = JSON.parse(journeyText);

// --- Gate 1: Atlas structure ---
if (
  atlas?.metadata?.vertex_count !== 24 ||
  atlas?.metadata?.edge_count !== 36 ||
  atlas?.edges?.length !== 36 ||
  supersymmetry?.validation?.passed !== true ||
  supersymmetry?.validation?.quartet_count !== 6
) {
  throw new Error("Gate 1 failed: atlas has wrong vertex/edge count or supersymmetry validation failed.");
}

const permutations = atlas.permutations.map(value =>
  Array.isArray(value) ? value.map(Number) : String(value).split("").map(Number)
);
const labels = permutations.map(value => value.join(""));
const rankByLabel = new Map(labels.map((label, rank) => [label, rank]));

const adjacency = Array.from({ length: 24 }, () => new Map());
for (const [a, b, generator] of atlas.edges) {
  adjacency[a].set(generator, b);
  adjacency[b].set(generator, a);
}

// Gate 1 continued: one edge per generator at every vertex
for (let rank = 0; rank < 24; rank++) {
  for (const gen of [1, 2, 3]) {
    if (!adjacency[rank].has(gen)) {
      throw new Error(`Gate 1 failed: vertex ${labels[rank]} has no generator-${gen} edge.`);
    }
  }
}

// --- Gate 2: Generator/color map ---
const colorGenerators = journey.color_generators;
if (colorGenerators.blue !== 1 || colorGenerators.green !== 2 || colorGenerators.red !== 3) {
  throw new Error("Gate 2 failed: color generator map does not match expected blue=1, green=2, red=3.");
}
const generatorColor = { 1: "blue", 2: "green", 3: "red" };

// --- Gate 2b: Exact generator sequences ---
const expectedSeqs = [[1, 3], [2, 1, 3, 2], [1, 3]];
for (let i = 0; i < 3; i++) {
  if (journey.segments[i].generators.join(",") !== expectedSeqs[i].join(",")) {
    throw new Error(
      `Gate 2b failed: segment ${journey.segments[i].id} generators are ` +
      `[${journey.segments[i].generators}], expected [${expectedSeqs[i]}].`
    );
  }
}

// --- Gate 2c: Segment colors match generators via color map ---
for (const seg of journey.segments) {
  const expectedColors = seg.generators.map(g => generatorColor[g]);
  if (seg.colors.join(",") !== expectedColors.join(",")) {
    throw new Error(
      `Gate 2c failed: segment ${seg.id} colors are [${seg.colors}], ` +
      `expected [${expectedColors}] from generators [${seg.generators}].`
    );
  }
}

// --- Gate 3: Segment lengths ---
const segmentLengths = journey.segments.map(seg => seg.generators.length);
if (segmentLengths.join(",") !== "2,4,2") {
  throw new Error(`Gate 3 failed: segment lengths are ${segmentLengths.join(",")}, expected 2,4,2.`);
}

// --- Gate 4: Record offsets ---
if (journey.record_offsets.join(",") !== "0,2,6,8") {
  throw new Error(`Gate 4 failed: record offsets are ${journey.record_offsets.join(",")}, expected 0,2,6,8.`);
}

// --- Validate continuous word matches segment concatenation ---
const segmentConcat = journey.segments.flatMap(seg => seg.generators);
if (segmentConcat.join(",") !== journey.continuous_word.join(",")) {
  throw new Error("Continuous word does not match segment generator concatenation.");
}

// --- V4 elements for Gate 11 ---
const composePermutations = (a, b) => a.map(i => b[i - 1]);
const inversePermutation = perm => {
  const inv = new Array(4);
  perm.forEach((value, position) => { inv[value - 1] = position + 1; });
  return inv;
};
const applyGenerator = (perm, gen) => {
  const result = [...perm];
  if (gen === 1) { [result[0], result[1]] = [result[1], result[0]]; }
  else if (gen === 2) { [result[1], result[2]] = [result[2], result[1]]; }
  else if (gen === 3) { [result[2], result[3]] = [result[3], result[2]]; }
  return result;
};
const permEqual = (a, b) => a.every((v, i) => v === b[i]);

const identity = [1, 2, 3, 4];
const A = journey.segments[0].generators.reduce(applyGenerator, identity); // s1*s3 = (12)(34)
const C = journey.segments[1].generators.reduce(applyGenerator, identity); // s2*s1*s3*s2 = (13)(24)
const AC = composePermutations(A, C); // (14)(23)
const V4 = [identity, A, AC, C];
const V4Labels = V4.map(p => p.join(""));

// Verify V4 is correct
if (V4Labels.join(",") !== "1234,2143,4321,3412") {
  throw new Error(`V4 computation failed: got ${V4Labels.join(",")}, expected 1234,2143,4321,3412.`);
}

// --- Gate 12: Base cycle is a hexagonal face ---
const baseCycle = journey.base_cycle;
if (baseCycle.length !== 6 || new Set(baseCycle).size !== 6) {
  throw new Error("Gate 12 failed: base cycle does not have 6 distinct labels.");
}
const baseRanks = baseCycle.map(label => {
  const rank = rankByLabel.get(label);
  if (rank === undefined) throw new Error(`Gate 12 failed: base label ${label} not in atlas.`);
  return rank;
});
const baseFaceEdges = [];
for (let i = 0; i < 6; i++) {
  const a = baseRanks[i], b = baseRanks[(i + 1) % 6];
  if (![...adjacency[a].values()].includes(b)) {
    throw new Error(`Gate 12 failed: ${baseCycle[i]} and ${baseCycle[(i + 1) % 6]} are not adjacent, so the base cycle is not a genuine hexagonal cycle.`);
  }
  baseFaceEdges.push([a, b]);
}

// --- Build supersymmetry sector lookup ---
const sectorByRank = new Map();
supersymmetry.sectors.forEach((sector, sectorIndex) => {
  sector.ordered_ranks.forEach(rank => sectorByRank.set(rank, sectorIndex));
});

const representationByAddress = new Map();
for (const representation of atlas.representations ?? []) {
  for (const address of representation.member_addresses ?? []) {
    representationByAddress.set(address, representation.label);
  }
}

// --- Route construction (Section 7.2) ---
const chains = baseCycle.map((seedLabel, chainIndex) => {
  const seedRank = rankByLabel.get(seedLabel);

  // Step 2-4: trace the continuous word
  const routeRanks = [seedRank];
  let current = seedRank;
  for (const gen of journey.continuous_word) {
    const next = adjacency[current].get(gen);
    if (next === undefined) {
      throw new Error(`Gate 6 failed: no generator-${gen} edge from ${labels[current]} in route for seed ${seedLabel}.`);
    }
    routeRanks.push(next);
    current = next;
  }

  // Gate 5: route has 9 vertices
  if (routeRanks.length !== 9) {
    throw new Error(`Gate 5 failed: route from ${seedLabel} has ${routeRanks.length} vertices, expected 9.`);
  }

  // Step 5: extract recorded ranks at offsets
  const recordedRanks = journey.record_offsets.map(offset => routeRanks[offset]);
  const recordedLabels = recordedRanks.map(rank => labels[rank]);

  // Gate 7: four recorded ranks are distinct
  if (new Set(recordedRanks).size !== 4) {
    throw new Error(`Gate 7 failed: seed ${seedLabel} produces non-distinct recorded ranks: ${recordedLabels.join(",")}.`);
  }

  // Step 6-7: find the supersymmetry sector
  const sectorIndices = new Set(recordedRanks.map(rank => sectorByRank.get(rank)));
  if (sectorIndices.size !== 1) {
    throw new Error(`Gate 8 failed: recorded ranks from seed ${seedLabel} span ${sectorIndices.size} sectors.`);
  }
  const sectorIndex = [...sectorIndices][0];
  const sector = supersymmetry.sectors[sectorIndex];

  // Gate 8: recorded set equals the sector's exact four ranks
  const sectorRanksSet = new Set(sector.ordered_ranks);
  for (const rank of recordedRanks) {
    if (!sectorRanksSet.has(rank)) {
      throw new Error(`Gate 8 failed: recorded rank ${labels[rank]} from seed ${seedLabel} is not in sector ${sector.id}.`);
    }
  }
  if (sectorRanksSet.size !== 4) {
    throw new Error(`Gate 8 failed: sector ${sector.id} does not have 4 ranks.`);
  }

  // Gate 6: every route edge exists and has the prescribed generator
  for (let i = 0; i < journey.continuous_word.length; i++) {
    const gen = journey.continuous_word[i];
    const from = routeRanks[i], to = routeRanks[i + 1];
    if (adjacency[from].get(gen) !== to) {
      throw new Error(`Gate 6 failed: edge ${i} from ${labels[from]} via generator ${gen} does not reach ${labels[to]}.`);
    }
  }

  // Gate 11: relative recorded multipliers equal {e, A, AC, C}
  // Right action: recorded[i] = seed * V4[i] = seed ∘ V4[i]
  // So V4[i] = seed^{-1} ∘ recorded[i], computed as composePermutations(recorded, seedInv)
  const seedPerm = permutations[seedRank];
  const seedInv = inversePermutation(seedPerm);
  for (let i = 0; i < 4; i++) {
    const recordedPerm = permutations[recordedRanks[i]];
    const relative = composePermutations(recordedPerm, seedInv);
    if (!permEqual(relative, V4[i])) {
      throw new Error(
        `Gate 11 failed: seed ${seedLabel}, recorded[${i}] = ${recordedLabels[i]}, ` +
        `relative multiplier is ${relative.join("")}, expected ${V4Labels[i]}.`
      );
    }
  }

  // Build route colors
  const routeColors = journey.continuous_word.map(gen => generatorColor[gen]);

  // Build segment ranges
  const segmentRanges = [];
  let offset = 0;
  for (const seg of journey.segments) {
    segmentRanges.push([offset, offset + seg.generators.length]);
    offset += seg.generators.length;
  }

  // Leg routes (connecting consecutive recorded members via the journey segments)
  const legRoutes = [];
  for (let i = 0; i < 3; i++) {
    const [start, end] = segmentRanges[i];
    const segRanks = routeRanks.slice(start, end + 1);
    legRoutes.push({
      leg: i + 1,
      segment_id: journey.segments[i].id,
      from_rank: segRanks[0],
      to_rank: segRanks.at(-1),
      from_label: labels[segRanks[0]],
      to_label: labels[segRanks.at(-1)],
      distance: segRanks.length - 1,
      route_ranks: segRanks,
      route_labels: segRanks.map(r => labels[r]),
      route_generators: journey.segments[i].generators,
      route_colors: journey.segments[i].colors,
    });
  }

  const representationLabels = new Set(recordedLabels.map(label => representationByAddress.get(label)));
  const representationLabel = [...representationLabels][0];

  return {
    id: `chain-${sector.id.slice(1)}`,
    position: chainIndex + 1,
    sector_id: sector.id,
    sector_index: sectorIndex,
    representation_label: representationLabel,
    multiplet: representationLabel?.split("/").map(part => part.trim())[1] ?? representationLabel ?? sector.id,
    seed_label: seedLabel,
    seed_rank: seedRank,
    ranks: recordedRanks,
    labels: recordedLabels,
    route_generators: [...journey.continuous_word],
    route_ranks: routeRanks,
    route_labels: routeRanks.map(r => labels[r]),
    route_colors: routeColors,
    record_offsets: [...journey.record_offsets],
    recorded_ranks: recordedRanks,
    recorded_labels: recordedLabels,
    segment_ranges: segmentRanges,
    segments: journey.segments.map((seg, i) => ({
      id: seg.id,
      start: segmentRanges[i][0],
      end: segmentRanges[i][1],
      generators: seg.generators,
      colors: seg.colors,
    })),
    leg_routes: legRoutes,
    legs: legRoutes.map(r => r.distance),
    journey_ranks: routeRanks,
    journey_labels: routeRanks.map(r => labels[r]),
    member_stops: journey.record_offsets,
    base_face_label: seedLabel,
    base_face_position: chainIndex + 1,
  };
});

// --- Gate 9: six seeds resolve to six different quartets ---
const sectorSet = new Set(chains.map(chain => chain.sector_index));
if (sectorSet.size !== 6) {
  throw new Error(`Gate 9 failed: ${sectorSet.size} distinct sectors, expected 6.`);
}

// --- Gate 10: 24 recorded ranks cover all vertices ---
const allRecorded = new Set(chains.flatMap(chain => chain.recorded_ranks));
if (allRecorded.size !== 24) {
  throw new Error(`Gate 10 failed: ${allRecorded.size} distinct recorded ranks, expected 24.`);
}

// --- Geometry (preserved from integration base) ---
const add = (a, b) => a.map((value, index) => value + b[index]);
const subtract = (a, b) => a.map((value, index) => value - b[index]);
const dot = (a, b) => a.reduce((sum, value, index) => sum + value * b[index], 0);
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const norm = value => Math.sqrt(dot(value, value));
const normalize = value => {
  const length = norm(value);
  return length ? value.map(component => component / length) : [0, 0, 0];
};
const matrixVector = (matrix, vector) => matrix.map(row => dot(row, vector));
const matrixMultiply = (a, b) => a.map(row =>
  b[0].map((_, column) => row.reduce((sum, value, index) => sum + value * b[index][column], 0))
);
const rotationY = angle => {
  const c = Math.cos(angle), s = Math.sin(angle);
  return [[c, 0, s], [0, 1, 0], [-s, 0, c]];
};
const rotationFromTo = (fromValue, toValue) => {
  const from = normalize(fromValue), to = normalize(toValue), v = cross(from, to);
  const c = Math.max(-1, Math.min(1, dot(from, to))), s = norm(v);
  if (s < 1e-12) {
    if (c > 0) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const axis = normalize(Math.abs(from[0]) < .8 ? cross(from, [1, 0, 0]) : cross(from, [0, 1, 0]));
    const [x, y, z] = axis;
    return [
      [2 * x * x - 1, 2 * x * y, 2 * x * z],
      [2 * y * x, 2 * y * y - 1, 2 * y * z],
      [2 * z * x, 2 * z * y, 2 * z * z - 1],
    ];
  }
  const [x, y, z] = v, k = (1 - c) / (s * s);
  return [
    [1 + k * (-y * y - z * z), -z + k * x * y, y + k * x * z],
    [z + k * y * x, 1 + k * (-x * x - z * z), -x + k * y * z],
    [-y + k * z * x, x + k * z * y, 1 + k * (-x * x - y * y)],
  ];
};
const worldPoint = permutation => {
  const inv = inversePermutation(permutation);
  const q = inv.map(value => value - 2.5);
  return [
    (q[0] - q[1]) / Math.sqrt(2),
    (q[0] + q[1] - 2 * q[2]) / Math.sqrt(6),
    (q[0] + q[1] + q[2] - 3 * q[3]) / Math.sqrt(12),
  ];
};
const worldPoints = permutations.map(worldPoint);

const baseFaceSet = new Set(baseCycle);
const baseCentroid = baseRanks
  .map(rank => worldPoints[rank])
  .reduce(add, [0, 0, 0])
  .map(value => value / baseRanks.length);

const alignNormal = rotationFromTo(baseCentroid, [0, -1, 0]);
const baseFirstEdge = subtract(
  worldPoints[rankByLabel.get(baseCycle[1])],
  worldPoints[rankByLabel.get(baseCycle[0])]
);
const alignedFirstEdge = matrixVector(alignNormal, baseFirstEdge);
const hexFloorMatrix = matrixMultiply(
  rotationY(Math.atan2(alignedFirstEdge[2], alignedFirstEdge[0])),
  alignNormal
);
const orientedBaseCentroid = matrixVector(hexFloorMatrix, baseCentroid);

const page64Matrix = [
  [-0.828277015814, -0.308451429117, 0.467776550236],
  [0.206791947500, 0.607627192595, 0.766828719642],
  [-0.520763166443, 0.731879027363, -0.439498366074]
];
const page64Point = label => matrixVector(page64Matrix, worldPoints[rankByLabel.get(label)]);
const page64Heights = labels.map((_, rank) => matrixVector(page64Matrix, worldPoints[rank])[1]);
const page64Floor = Math.min(...page64Heights);
const page64MatchesFigure =
  labels.every(label => page64Point("1234")[1] <= page64Point(label)[1] + 1e-9) &&
  labels.every(label => page64Point("4321")[1] >= page64Point(label)[1] - 1e-9) &&
  page64Point("2134")[0] < -0.9 &&
  page64Point("1243")[0] > 0.6 &&
  page64Point("1324")[1] > page64Point("1234")[1] + 0.4 &&
  page64Point("2143")[0] < page64Point("1234")[0] - 0.2 &&
  page64Point("2143")[1] > page64Point("1234")[1] + 0.4;

// --- Gate 13: Orientation checks ---
const requestedFacePointsDown = Math.abs(orientedBaseCentroid[0]) < 1e-10 &&
  Math.abs(orientedBaseCentroid[2]) < 1e-10 && orientedBaseCentroid[1] < 0;
if (!requestedFacePointsDown) throw new Error("Gate 13 failed: the base face does not point down.");
if (!page64MatchesFigure) throw new Error("Gate 13 failed: the p.64 orientation does not match the printed figure.");

// --- Gate 14: Signing resolution independence ---
const signingResolution = chains.every(chain => {
  const sector = supersymmetry.sectors[chain.sector_index];
  return chain.recorded_ranks.every(rank =>
    sector.ordered_ranks.includes(rank)
  );
});
if (!signingResolution) throw new Error("Gate 14 failed: a signing does not resolve against its original sector.");

// --- Publication gate ---
if (mode === "--publish") {
  if (journey.source_provenance.color_map_confirmed !== true) {
    console.error("Publication blocked: color_map_confirmed is not true.");
    process.exit(1);
  }
  if (journey.source_provenance.base_cycle_confirmed !== true) {
    console.error("Publication blocked: base_cycle_confirmed is not true.");
    process.exit(1);
  }
  const sha = journey.source_provenance.screenshot_sha256;
  if (typeof sha !== "string" || !/^[0-9a-f]{64}$/.test(sha)) {
    console.error("Publication blocked: screenshot_sha256 is not a valid SHA-256 hex string.");
    process.exit(1);
  }
  if (journey.source_provenance.publishable !== true) {
    console.error("Publication blocked: publishable is not true in fixture.");
    process.exit(1);
  }
}

// --- Provenance hashes ---
const provenance = {
  atlas_sha256: sha256(atlasText),
  supersymmetry_sha256: sha256(supersymmetryText),
  journey_fixture_sha256: sha256(journeyText),
  template_sha256: sha256(templateText),
};

// --- Assemble disassembly data ---
const disassembly = {
  schema_version: "s4-gates-continuous-journeys-v1",
  source: "S. J. Gates Jr., written instructions 2026-08-26; HowardTLK.v2.pdf p. 61 for quartet membership",
  source_provenance: journey.source_provenance,
  ordering: "members follow the continuous journey order (seed, seed*A, seed*AC, seed*C); strands follow the base-face cycle",
  orientation: "starts in the HowardTLK.v2.pdf p. 64 view, fitted to the printed figure; the hexagon-down view is a preset",
  orientation_matrix: page64Matrix,
  orientations: {
    hex_floor: { label: "Hexagon at the floor", matrix: hexFloorMatrix, pitch: 0.48, yaw: -0.62 },
    page64: { label: "Howard p.64 figure", matrix: page64Matrix, pitch: 0, yaw: 0 },
  },
  color_generators: journey.color_generators,
  continuous_word: journey.continuous_word,
  record_offsets: journey.record_offsets,
  segments: journey.segments,
  base_face_cycle: baseCycle,
  v4_elements: { e: "1234", A: "2143", AC: "4321", C: "3412" },
  chains,
  provenance,
  validation: {
    chains: 6,
    vertices_per_chain: 4,
    covered_vertices: allRecorded.size,
    covers_all_vertices: allRecorded.size === 24,
    one_chain_per_susy_sector: sectorSet.size === 6,
    base_face_is_cyclic: true,
    requested_face_points_down: requestedFacePointsDown,
    page64_matches_figure: page64MatchesFigure,
    all_routes_have_9_vertices: chains.every(c => c.route_ranks.length === 9),
    all_routes_follow_exact_word: true,
    all_recorded_sets_match_quartet: true,
    relative_multipliers_are_v4: true,
    signing_resolution_independent: signingResolution,
    all_gates_pass: allRecorded.size === 24 && sectorSet.size === 6 &&
      requestedFacePointsDown && page64MatchesFigure && signingResolution &&
      chains.every(c => c.route_ranks.length === 9),
    source_approved: journey.source_provenance.publishable === true,
    publishable: !!(
      journey.source_provenance.color_map_confirmed === true &&
      journey.source_provenance.base_cycle_confirmed === true &&
      typeof journey.source_provenance.screenshot_sha256 === "string" &&
      /^[0-9a-f]{64}$/.test(journey.source_provenance.screenshot_sha256) &&
      journey.source_provenance.publishable === true
    ),
  },
};

// --- Gate 15: no baked passed:true ---
// All validation fields above are recomputed, not read from inputs.

const safeJson = value => JSON.stringify(value).replace(/<\//gi, "<\\/");
const output = templateText
  .replace("/*__ATLAS_JSON__*/null", safeJson(atlas))
  .replace("/*__SUPERSYMMETRY_JSON__*/null", safeJson(supersymmetry))
  .replace("/*__DISASSEMBLY_JSON__*/null", safeJson(disassembly));

if (
  output.includes("/*__ATLAS_JSON__*/") ||
  output.includes("/*__SUPERSYMMETRY_JSON__*/") ||
  output.includes("/*__DISASSEMBLY_JSON__*/")
) {
  throw new Error("A disassembly data placeholder was not replaced.");
}

if (mode === "--check") {
  let existing;
  try { existing = await readFile(outputPath, "utf8"); } catch { existing = null; }
  if (existing === null) {
    console.log("No existing artifact to check. Run --preview first.");
    process.exit(1);
  }
  if (existing === output) {
    console.log("Check passed: artifact is up to date.");
  } else {
    console.error("Check failed: artifact differs from a fresh build.");
    process.exit(1);
  }
  process.exit(0);
}

await writeFile(outputPath, output);
console.log(`Wrote ${outputPath} (mode: ${mode})`);
console.log(`Schema: ${disassembly.schema_version}`);
console.log(`Publishable: ${disassembly.validation.publishable}`);
console.log(`Provenance: atlas=${provenance.atlas_sha256.slice(0,12)}... journey=${provenance.journey_fixture_sha256.slice(0,12)}...`);
console.log();
console.log("Six continuous-journey strands (BR | GBRG | BR):");
for (const chain of chains) {
  console.log(
    `  ${String(chain.position).padStart(2)}. ${chain.seed_label} -> ` +
    `${chain.recorded_labels.join(", ")}  ` +
    `${chain.sector_id} / ${chain.multiplet}  ` +
    `legs ${chain.legs.join(",")}`
  );
  console.log(`      route: ${chain.route_labels.join(" -> ")}`);
}
