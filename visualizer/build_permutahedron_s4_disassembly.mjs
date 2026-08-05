import { readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const templatePath = join(here, "permutahedron_s4_disassembly.template.html");
const outputPath = join(here, "permutahedron_s4_disassembly.html");

const [template, atlasText, supersymmetryText] = await Promise.all([
  readFile(templatePath, "utf8"),
  readFile(join(root, "data", "permutahedron_s4_atlas.json"), "utf8"),
  readFile(join(root, "data", "permutahedron_s4_supersymmetry.json"), "utf8"),
]);

const atlas = JSON.parse(atlasText);
const supersymmetry = JSON.parse(supersymmetryText);
const baseFaceLabels = ["1234", "1324", "3124", "3214", "2314", "2134"];
const chainSeeds = ["1423", "1342", "1324", "1432", "1243", "1234"];
const expectedSectorOrder = ["P1", "P2", "P3", "P4", "P5", "P6"];
// HowardTLK.v2.pdf, p. 73. Products act rightmost first, so these are the
// adjacent-transposition steps actually followed from each H1 vertex.
const hoppers = [
  { id: "H1", generators: [] },
  { id: "H2", generators: [3, 1] },
  { id: "H3", generators: [2, 3, 1, 2] },
  { id: "H4", generators: [3, 1, 2, 3, 1, 2] },
];
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

const shortestDistance = (start, target) => {
  const distance = new Array(24).fill(-1);
  const queue = [start];
  distance[start] = 0;
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const rank = queue[cursor];
    if (rank === target) return distance[rank];
    for (const next of adjacency[rank].values()) {
      if (distance[next] !== -1) continue;
      distance[next] = distance[rank] + 1;
      queue.push(next);
    }
  }
  return -1;
};
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
const inversePermutation = permutation => {
  const inverse = new Array(4);
  permutation.forEach((value, position) => { inverse[value - 1] = position + 1; });
  return inverse;
};
const worldPoint = permutation => {
  const q = inversePermutation(permutation).map(value => value - 2.5);
  return [
    (q[0] - q[1]) / Math.sqrt(2),
    (q[0] + q[1] - 2 * q[2]) / Math.sqrt(6),
    (q[0] + q[1] + q[2] - 3 * q[3]) / Math.sqrt(12),
  ];
};
const worldPoints = permutations.map(worldPoint);
const followGenerators = (start, generators) => {
  const route = [start];
  for (const generator of generators) {
    const next = adjacency[route.at(-1)].get(generator);
    if (next === undefined) throw new Error(`Missing generator ${generator} from ${labels[route.at(-1)]}.`);
    route.push(next);
  }
  return route;
};

if (
  atlas?.metadata?.vertex_count !== 24 ||
  atlas?.metadata?.edge_count !== 36 ||
  supersymmetry?.validation?.passed !== true ||
  supersymmetry?.validation?.quartet_count !== 6
) {
  throw new Error("The S4 disassembly input artifacts failed validation.");
}

const sectorByRank = new Map();
supersymmetry.sectors.forEach((sector, sectorIndex) => {
  sector.ordered_ranks.forEach(rank => sectorByRank.set(rank, sectorIndex));
});

const chains = chainSeeds.map((seedLabel, chainIndex) => {
  const seedRank = rankByLabel.get(seedLabel);
  if (seedRank === undefined) throw new Error(`Missing H1 vertex ${seedLabel}.`);
  const sectorIndex = sectorByRank.get(seedRank);
  if (sectorIndex === undefined) throw new Error(`H1 vertex ${seedLabel} has no SUSY quartet.`);
  const sector = supersymmetry.sectors[sectorIndex];
  if (sector.id !== expectedSectorOrder[chainIndex]) {
    throw new Error(`${seedLabel} resolved to ${sector.id}, expected ${expectedSectorOrder[chainIndex]}.`);
  }
  const hopperRoutes = hoppers.map(hopper => {
    const route = followGenerators(seedRank, hopper.generators);
    const target = route.at(-1);
    const distance = shortestDistance(seedRank, target);
    if (route.length - 1 !== distance || new Set(route).size !== route.length) {
      throw new Error(`${sector.id} ${hopper.id} is not a shortest no-backtracking route.`);
    }
    return {
      hopper: hopper.id,
      from_rank: seedRank,
      to_rank: target,
      distance,
      route_ranks: route,
      route_labels: route.map(member => labels[member]),
      generators: hopper.generators,
    };
  });
  const hopperRanks = hopperRoutes.map(route => route.to_rank);
  if (new Set(hopperRanks).size !== 4 ||
      hopperRanks.some(rank => !sector.ordered_ranks.includes(rank))) {
    throw new Error(`${sector.id} hopper endpoints do not reproduce its published quartet.`);
  }
  return {
    id: `chain-${chainIndex + 1}`,
    sector_id: sector.id,
    sector_index: sectorIndex,
    h1_label: seedLabel,
    hopper_ranks: hopperRanks,
    labels: hopperRanks.map(rank => labels[rank]),
    hopper_routes: hopperRoutes,
    root_distances: hopperRoutes.map(route => route.distance),
    consecutive_distances: hopperRanks.slice(0, -1).map((rank, index) =>
      shortestDistance(rank, hopperRanks[index + 1])
    ),
  };
});
const covered = new Set(chains.flatMap(chain => chain.hopper_ranks));
const baseRanks = new Set(baseFaceLabels.map(label => rankByLabel.get(label)));
const baseEdges = atlas.edges.filter(([a, b]) => baseRanks.has(a) && baseRanks.has(b));
const baseDegrees = new Map([...baseRanks].map(rank => [rank, 0]));
for (const [a, b] of baseEdges) {
  baseDegrees.set(a, baseDegrees.get(a) + 1);
  baseDegrees.set(b, baseDegrees.get(b) + 1);
}
const baseCentroid = [...baseRanks]
  .map(rank => worldPoints[rank])
  .reduce(add, [0, 0, 0])
  .map(value => value / baseRanks.size);
const alignNormal = rotationFromTo(baseCentroid, [0, -1, 0]);
const baseFirstEdge = subtract(worldPoints[rankByLabel.get(baseFaceLabels[1])], worldPoints[rankByLabel.get(baseFaceLabels[0])]);
const alignedFirstEdge = matrixVector(alignNormal, baseFirstEdge);
const orientationMatrix = matrixMultiply(rotationY(Math.atan2(alignedFirstEdge[2], alignedFirstEdge[0])), alignNormal);
const orientedBaseCentroid = matrixVector(orientationMatrix, baseCentroid);

const coversAllVertices = covered.size === 24;
const requestedBaseIsHexFace = baseRanks.size === 6 &&
  baseEdges.length === 6 && [...baseDegrees.values()].every(degree => degree === 2);
const oneChainPerSector = new Set(chains.map(chain => chain.sector_index)).size === 6;
const orderedByHopperLength = chains.every(chain =>
  chain.root_distances.join(",") === "0,2,4,6"
);
const shortestHopperRoutes = chains.every(chain =>
  chain.hopper_routes.every(route => route.route_ranks.length - 1 === route.distance)
);
const noBacktracking = chains.every(chain =>
  new Set(chain.hopper_ranks).size === 4 &&
  chain.hopper_routes.every(route => new Set(route.route_ranks).size === route.route_ranks.length)
);
const uniformConsecutiveDistances = chains.every(chain => chain.consecutive_distances.join(",") === "2,6,2");
const guideSegmentsAreNotEdges = chains.every(chain => chain.consecutive_distances.every(distance => distance > 1));
const requestedFacePointsDown = Math.abs(orientedBaseCentroid[0]) < 1e-10 &&
  Math.abs(orientedBaseCentroid[2]) < 1e-10 && orientedBaseCentroid[1] < 0;

if (!coversAllVertices || !oneChainPerSector) throw new Error("The six H1-H4 quartet chains do not partition S4.");
if (!requestedBaseIsHexFace) throw new Error("The requested down-facing base is not a hexagonal face.");
if (!orderedByHopperLength) throw new Error("A quartet is not ordered H1 through H4 by distances 0, 2, 4, 6.");
if (!shortestHopperRoutes || !noBacktracking) throw new Error("A hopper lacks a shortest no-backtracking route from H1.");
if (!uniformConsecutiveDistances) throw new Error("The H1-H4 endpoint order does not have the expected 2, 6, 2 spacing.");
if (!guideSegmentsAreNotEdges) throw new Error("A quartet guide was incorrectly classified as one permutahedron edge.");
if (!requestedFacePointsDown) throw new Error("The requested hexagonal face was not oriented downward.");

const disassembly = {
  schema_version: "s4-six-hopper-ordered-quartets-v3",
  source: "HowardTLK.v2.pdf pp. 61, 64, 73, 77, 80-81; Gates screen share 2026-08-04",
  orientation: "requested hexagonal face down; permutations ending in 4 form the base",
  orientation_matrix: orientationMatrix,
  base_face_labels: baseFaceLabels,
  hopper_definitions: hoppers,
  chains,
  validation: {
    chains: 6,
    vertices_per_chain: 4,
    covered_vertices: covered.size,
    covers_all_vertices: coversAllVertices,
    one_chain_per_susy_sector: oneChainPerSector,
    requested_base_is_hex_face: requestedBaseIsHexFace,
    ordered_by_hopper_length: orderedByHopperLength,
    shortest_hopper_routes: shortestHopperRoutes,
    no_backtracking: noBacktracking,
    uniform_consecutive_distances_2_6_2: uniformConsecutiveDistances,
    guide_segments_are_not_single_edges: guideSegmentsAreNotEdges,
    requested_face_points_down: requestedFacePointsDown,
  },
};

const safeJson = value => JSON.stringify(value).replaceAll("</script", "<\\/script");
const output = template
  .replace("/*__ATLAS_JSON__*/null", safeJson(atlas))
  .replace("/*__SUPERSYMMETRY_JSON__*/null", safeJson(supersymmetry))
  .replace("/*__DISASSEMBLY_JSON__*/null", safeJson(disassembly));

if (output.includes("/*__ATLAS_JSON__*/") || output.includes("/*__SUPERSYMMETRY_JSON__*/") || output.includes("/*__DISASSEMBLY_JSON__*/")) {
  throw new Error("A disassembly data placeholder was not replaced.");
}

await writeFile(outputPath, output);
console.log(`Wrote ${outputPath}`);
console.log("Verified six SUSY quartets in H1-H4 order with shortest no-backtracking hopper routes.");
