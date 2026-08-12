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
// HowardTLK.v2.pdf, p. 61, transcribed line for line. Gates named that page as
// the authority for how these quartets should be listed (2026-08-04, L199-201).
//
// Two separate rules are at work and they are easy to conflate. Within a
// quartet the members ascend as four-digit numbers. The quartets themselves run
// P[1] through P[6], which is NOT ascending by smallest member: those run 1423,
// 1342, 1324, 1432, 1243, 1234. Sorting the quartets by smallest member looks
// tidier and is wrong.
//
// Legs are the link distances between consecutive listed members. Six distinct
// patterns, not the uniform 2,6,2 the earlier hopper ordering produced.
const expectedQuartets = [
  { sector: "P1", multiplet: "CM",  members: ["1423", "2314", "3241", "4132"], legs: [4, 2, 4] },
  { sector: "P2", multiplet: "TM",  members: ["1342", "2431", "3124", "4213"], legs: [6, 4, 6] },
  { sector: "P3", multiplet: "VM",  members: ["1324", "2413", "3142", "4231"], legs: [4, 6, 4] },
  { sector: "P4", multiplet: "VM1", members: ["1432", "2341", "3214", "4123"], legs: [6, 2, 6] },
  { sector: "P5", multiplet: "VM2", members: ["1243", "2134", "3421", "4312"], legs: [2, 4, 2] },
  { sector: "P6", multiplet: "VM3", members: ["1234", "2143", "3412", "4321"], legs: [2, 6, 2] },
];
// The six permutations ending in 4, in cyclic order around the face. Verified
// against the atlas edges by baseFaceIsCyclic below: consecutive entries, and
// the wrap from last to first, are all genuine permutahedron edges.
const baseFaceCycle = ["1234", "1324", "3124", "3214", "2314", "2134"];
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
const bfsDistances = start => {
  const distance = new Array(24).fill(-1);
  const queue = [start];
  distance[start] = 0;
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const rank = queue[cursor];
    for (const next of adjacency[rank].values()) {
      if (distance[next] !== -1) continue;
      distance[next] = distance[rank] + 1;
      queue.push(next);
    }
  }
  return distance;
};
// Every shortest route between two members, generated in a deterministic order:
// at each step the lower-numbered generator is explored first, so the list is
// stable across builds.
const allShortestRoutes = (start, target) => {
  const toTarget = bfsDistances(target);
  const routes = [];
  const walk = path => {
    const current = path.at(-1);
    if (current === target) { routes.push([...path]); return; }
    const steps = [...adjacency[current].entries()]
      .filter(([, next]) => toTarget[next] === toTarget[current] - 1)
      .sort((a, b) => a[0] - b[0]);
    for (const [, next] of steps) { path.push(next); walk(path); path.pop(); }
  };
  walk([start]);
  if (!routes.length) throw new Error(`No shortest route from ${labels[start]} to ${labels[target]}.`);
  return routes;
};

// Gates calls the traversal a self-avoiding random walk: shortest path, never
// backtrack (2026-08-04 call, L147-161). He means the whole journey through the
// quartet, not each leg in isolation. Picking each leg greedily satisfies the
// per-leg condition but revisits a node in two of the six quartets, so the legs
// are chosen jointly here: the first combination of per-leg geodesics whose
// concatenation visits no node twice. All six quartets admit one.
const selfAvoidingJourney = ranks => {
  const options = ranks.slice(0, -1).map((rank, index) => allShortestRoutes(rank, ranks[index + 1]));
  let solution = null;
  const search = (index, walk, chosen) => {
    if (solution) return;
    if (index === options.length) { solution = { walk: [...walk], routes: [...chosen] }; return; }
    for (const route of options[index]) {
      const next = walk.concat(index ? route.slice(1) : route);
      if (new Set(next).size !== next.length) continue;
      chosen.push(route);
      search(index + 1, next, chosen);
      chosen.pop();
      if (solution) return;
    }
  };
  search(0, [], []);
  if (!solution) {
    throw new Error(`No self-avoiding journey through ${ranks.map(rank => labels[rank]).join(", ")}.`);
  }
  return solution;
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

const baseFaceSet = new Set(baseFaceCycle);
// The atlas carries the physics names for each quartet. Gates recognises
// CM, TM, VM, VM1, VM2 and VM3; the internal P1-P6 sector ids mean nothing to him.
const representationByAddress = new Map();
for (const representation of atlas.representations ?? []) {
  for (const address of representation.member_addresses ?? []) {
    representationByAddress.set(address, representation.label);
  }
}
if (representationByAddress.size !== 24) {
  throw new Error(`The atlas maps ${representationByAddress.size} addresses to representations, expected 24.`);
}
// Each quartet comes from its published coset. Members are sorted by reading the
// label as a four-digit number; the quartets themselves follow p. 61's P[1] to
// P[6] sequence, which the atlas sector ids already encode. Nothing here consults
// the base face; that is a highlight, not the ordering rule.
const sectorOrder = expectedQuartets.map(quartet => quartet.sector);
const chains = supersymmetry.sectors
  .map((sector, sectorIndex) => ({
    sector,
    sectorIndex,
    ranks: [...sector.ordered_ranks].sort((a, b) => Number(labels[a]) - Number(labels[b])),
  }))
  .sort((a, b) => sectorOrder.indexOf(a.sector.id) - sectorOrder.indexOf(b.sector.id))
  .map(({ sector, sectorIndex, ranks }, chainIndex) => {
    const memberLabels = ranks.map(rank => labels[rank]);
    const expected = expectedQuartets[chainIndex];
    if (sector.id !== expected.sector) {
      throw new Error(
        `Strand ${chainIndex + 1} is ${sector.id}, expected ${expected.sector} per HowardTLK.v2.pdf p. 61.`
      );
    }
    if (memberLabels.join(",") !== expected.members.join(",")) {
      throw new Error(
        `Quartet ${chainIndex + 1} is ${memberLabels.join(",")}, expected ${expected.members.join(",")} per HowardTLK.v2.pdf p. 61.`
      );
    }
    const journey = selfAvoidingJourney(ranks);
    const legRoutes = ranks.slice(0, -1).map((rank, index) => {
      const target = ranks[index + 1];
      const route = journey.routes[index];
      const distance = shortestDistance(rank, target);
      if (route.length - 1 !== distance || new Set(route).size !== route.length) {
        throw new Error(`${sector.id} leg ${index + 1} is not a shortest no-backtracking route.`);
      }
      return {
        leg: index + 1,
        from_rank: rank,
        to_rank: target,
        from_label: labels[rank],
        to_label: labels[target],
        distance,
        route_ranks: route,
        route_labels: route.map(member => labels[member]),
      };
    });
    const legs = legRoutes.map(route => route.distance);
    if (legs.join(",") !== expected.legs.join(",")) {
      throw new Error(`${sector.id} legs are ${legs.join(",")}, expected ${expected.legs.join(",")}.`);
    }
    const baseMembers = memberLabels.filter(label => baseFaceSet.has(label));
    if (baseMembers.length !== 1) {
      throw new Error(`${sector.id} has ${baseMembers.length} base-face members, expected exactly one.`);
    }
    const representationLabels = new Set(memberLabels.map(label => representationByAddress.get(label)));
    if (representationLabels.size !== 1) {
      throw new Error(`${sector.id} spans ${representationLabels.size} representations, expected one.`);
    }
    const representationLabel = [...representationLabels][0];
    return {
      id: `chain-${chainIndex + 1}`,
      position: chainIndex + 1,
      sector_id: sector.id,
      sector_index: sectorIndex,
      representation_label: representationLabel,
      multiplet: representationLabel.split("/").map(part => part.trim())[1] ?? representationLabel,
      ranks,
      labels: memberLabels,
      leg_routes: legRoutes,
      legs,
      // The three legs concatenated: one continuous walk visiting the four
      // members in order and no node twice. This is what the glow animates.
      journey_ranks: journey.walk,
      journey_labels: journey.walk.map(rank => labels[rank]),
      // Index into journey_ranks where each member sits, so the animation knows
      // where to rest between members.
      member_stops: ranks.map(rank => journey.walk.indexOf(rank)),
      base_face_label: baseMembers[0],
      base_face_position: memberLabels.indexOf(baseMembers[0]) + 1,
    };
  });
const covered = new Set(chains.flatMap(chain => chain.ranks));
const baseRanks = new Set(baseFaceCycle.map(label => rankByLabel.get(label)));
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
// Orientation A, the hexagon flat on the floor. Gates asked for this on
// 2026-08-04 (L177) so the six permutations ending in 4 form the base.
const alignNormal = rotationFromTo(baseCentroid, [0, -1, 0]);
const baseFirstEdge = subtract(worldPoints[rankByLabel.get(baseFaceCycle[1])], worldPoints[rankByLabel.get(baseFaceCycle[0])]);
const alignedFirstEdge = matrixVector(alignNormal, baseFirstEdge);
const hexFloorMatrix = matrixMultiply(rotationY(Math.atan2(alignedFirstEdge[2], alignedFirstEdge[0])), alignNormal);
const orientedBaseCentroid = matrixVector(hexFloorMatrix, baseCentroid);

// Orientation B: the view printed on HowardTLK.v2.pdf p. 64.
// This matrix was not hand-derived. It was fitted to the printed figure itself by
// five independent methods that agree to 0.63 degrees, among them disc detection
// with label OCR, normalised cross-correlation template matching with OCR, and
// reading the 24 labels visually off tiled crops. Reprojection RMS came out at 1.75 to 3.30 px
// against a disc radius of about 55 px, which is the detection noise floor.
//
// Two findings from that fit, both of which defeated every hand attempt:
// 1. The printed PNG is stretched horizontally by about 1.14. Measured directly:
//    the discs are ellipses of mean width over height 1.112. That is a print
//    artifact, so it is deliberately NOT reproduced here.
// 2. The solid does NOT rest flat on the square face {1234, 2134, 2143, 1243}.
//    Fitted heights are 1234 -2.219, 2134 -1.925, 1243 -1.828, 2143 -1.534: the
//    face is tilted, with the camera looking down on it by about 20 degrees.
//    Forcing that face horizontal makes the view symmetric about the
//    1234-to-2143 diagonal, and the printed figure is 29 degrees off that.
const page64Matrix = [
  [-0.828277015814, -0.308451429117, 0.467776550236],
  [0.206791947500, 0.607627192595, 0.766828719642],
  [-0.520763166443, 0.731879027363, -0.439498366074]
];
const page64Point = label => matrixVector(page64Matrix, worldPoints[rankByLabel.get(label)]);
const page64Heights = labels.map((_, rank) => matrixVector(page64Matrix, worldPoints[rank])[1]);
const page64Floor = Math.min(...page64Heights);
const page64MatchesFigure =
  // Facts readable straight off the printed figure.
  labels.every(label => page64Point("1234")[1] <= page64Point(label)[1] + 1e-9) &&
  labels.every(label => page64Point("4321")[1] >= page64Point(label)[1] - 1e-9) &&
  page64Point("2134")[0] < -0.9 &&
  page64Point("1243")[0] > 0.6 &&
  page64Point("1324")[1] > page64Point("1234")[1] + 0.4 &&
  // 2143 sits above and to the LEFT, about 29 degrees off vertical. If this ever
  // comes out near zero the view has been forced symmetric again.
  page64Point("2143")[0] < page64Point("1234")[0] - 0.2 &&
  page64Point("2143")[1] > page64Point("1234")[1] + 0.4;

const coversAllVertices = covered.size === 24;
const requestedBaseIsHexFace = baseRanks.size === 6 &&
  baseEdges.length === 6 && [...baseDegrees.values()].every(degree => degree === 2);
const oneChainPerSector = new Set(chains.map(chain => chain.sector_index)).size === 6;
// Recomputed from the emitted chains, not read back from a flag set during the build.
const ascendingWithinQuartets = chains.every(chain =>
  chain.labels.every((label, index) => index === 0 || Number(chain.labels[index - 1]) < Number(label))
);
// p. 61 runs P[1] to P[6]. Deliberately not a smallest-member sort: that would
// give 1234, 1243, 1324, 1342, 1423, 1432 and disagree with the printed page.
const matchesPage61Order = chains.every((chain, index) =>
  chain.sector_id === expectedQuartets[index].sector &&
  chain.multiplet === expectedQuartets[index].multiplet
);
const quartetsAreNotSmallestMemberSorted = chains.some((chain, index) =>
  index > 0 && Number(chains[index - 1].labels[0]) > Number(chain.labels[0])
);
const legsMatchPage61 = chains.every((chain, index) =>
  chain.legs.join(",") === expectedQuartets[index].legs.join(",")
);
const legPatternsAreDistinct =
  new Set(chains.map(chain => chain.legs.join(","))).size === 6;
const shortestLegRoutes = chains.every(chain =>
  chain.leg_routes.every(route => route.route_ranks.length - 1 === route.distance)
);
const noBacktracking = chains.every(chain =>
  new Set(chain.ranks).size === 4 &&
  chain.leg_routes.every(route => new Set(route.route_ranks).size === route.route_ranks.length)
);
const guideSegmentsAreNotEdges = chains.every(chain => chain.legs.every(distance => distance > 1));
const journeysAreSelfAvoiding = chains.every(chain =>
  new Set(chain.journey_ranks).size === chain.journey_ranks.length
);
const journeysUseRealEdges = chains.every(chain =>
  chain.journey_ranks.slice(0, -1).every((rank, index) =>
    [...adjacency[rank].values()].includes(chain.journey_ranks[index + 1])
  )
);
const journeysVisitMembersInOrder = chains.every(chain =>
  chain.member_stops.every((stop, index) =>
    stop >= 0 &&
    chain.journey_ranks[stop] === chain.ranks[index] &&
    (index === 0 || stop > chain.member_stops[index - 1])
  )
);
const baseFaceIsCyclic = baseFaceCycle.every((label, index) => {
  const a = rankByLabel.get(label);
  const b = rankByLabel.get(baseFaceCycle[(index + 1) % baseFaceCycle.length]);
  return [...adjacency[a].values()].includes(b);
});
const oneBaseMemberPerQuartet = chains.every(chain =>
  chain.labels.filter(label => baseFaceSet.has(label)).length === 1
);
const requestedFacePointsDown = Math.abs(orientedBaseCentroid[0]) < 1e-10 &&
  Math.abs(orientedBaseCentroid[2]) < 1e-10 && orientedBaseCentroid[1] < 0;

if (!coversAllVertices || !oneChainPerSector) throw new Error("The six quartets do not partition S4.");
if (!requestedBaseIsHexFace) throw new Error("The requested down-facing base is not a hexagonal face.");
if (!baseFaceIsCyclic) throw new Error("The base face labels are not in cyclic order, so the hexagon self-intersects.");
if (!ascendingWithinQuartets) throw new Error("A quartet is not listed in ascending four-digit order.");
if (!matchesPage61Order) throw new Error("The quartets are not in the P[1] to P[6] order printed on HowardTLK.v2.pdf p. 61.");
if (!quartetsAreNotSmallestMemberSorted) throw new Error("The quartets came out sorted by smallest member, which is the error p. 61 rules out.");
if (!legsMatchPage61) throw new Error("A quartet's legs do not match HowardTLK.v2.pdf p. 61.");
if (!legPatternsAreDistinct) throw new Error("The six quartets no longer have six distinct leg patterns.");
if (!shortestLegRoutes || !noBacktracking) throw new Error("A leg lacks a shortest no-backtracking route.");
if (!guideSegmentsAreNotEdges) throw new Error("A quartet guide was incorrectly classified as one permutahedron edge.");
if (!journeysAreSelfAvoiding) throw new Error("A quartet journey revisits a node, so it is not a self-avoiding walk.");
if (!journeysUseRealEdges) throw new Error("A quartet journey traverses something that is not a permutahedron edge.");
if (!journeysVisitMembersInOrder) throw new Error("A quartet journey does not pass through its four members in listed order.");
if (!oneBaseMemberPerQuartet) throw new Error("A quartet does not contain exactly one base-face member.");
if (!requestedFacePointsDown) throw new Error("The requested hexagonal face was not oriented downward.");
if (!page64MatchesFigure) throw new Error("The p.64 orientation is wrong: expected the solid tipped onto the 1234 corner (lowest, at the front) with 4321 at the top, 2134 left, 1243 right, and 2143 above and to the left of 1234; the bottom square face is tilted about 20 degrees, not level.");

const disassembly = {
  schema_version: "s4-six-ascending-weight-quartets-v4",
  source: "HowardTLK.v2.pdf p. 61 for the quartet listing, p. 64 for link weight; Gates call 2026-08-04",
  ordering: "members ascend as four-digit numbers within each quartet; the quartets themselves follow p. 61 order P[1] to P[6]",
  orientation: "starts in the HowardTLK.v2.pdf p. 64 view, fitted to the printed figure; the hexagon-down view is a preset",
  orientation_matrix: page64Matrix,
  // Each preset carries the camera tilt it is meant to be viewed at. The
  // hexagon-down view wants a three-quarter angle so the solid reads as a solid;
  // the p. 64 figure is drawn flat on, so it takes no extra rotation.
  orientations: {
    hex_floor: { label: "Hexagon at the floor", matrix: hexFloorMatrix, pitch: 0.48, yaw: -0.62 },
    // pitch and yaw MUST stay zero. The matrix is fitted to the printed figure, so
    // any extra camera rotation here silently rotates the view away from the fit.
    page64: { label: "Howard p.64 figure", matrix: page64Matrix, pitch: 0, yaw: 0 },
  },
  base_face_cycle: baseFaceCycle,
  chains,
  validation: {
    chains: 6,
    vertices_per_chain: 4,
    covered_vertices: covered.size,
    covers_all_vertices: coversAllVertices,
    one_chain_per_susy_sector: oneChainPerSector,
    requested_base_is_hex_face: requestedBaseIsHexFace,
    base_face_is_cyclic: baseFaceIsCyclic,
    ascending_within_quartets: ascendingWithinQuartets,
    matches_page_61_order: matchesPage61Order,
    legs_match_page_61: legsMatchPage61,
    leg_patterns_are_distinct: legPatternsAreDistinct,
    shortest_leg_routes: shortestLegRoutes,
    no_backtracking: noBacktracking,
    guide_segments_are_not_single_edges: guideSegmentsAreNotEdges,
    journeys_are_self_avoiding: journeysAreSelfAvoiding,
    journeys_use_real_edges: journeysUseRealEdges,
    journeys_visit_members_in_order: journeysVisitMembersInOrder,
    one_base_member_per_quartet: oneBaseMemberPerQuartet,
    requested_face_points_down: requestedFacePointsDown,
    page64_matches_figure: page64MatchesFigure,
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
console.log("Verified six SUSY quartets in ascending four-digit order with shortest no-backtracking legs.");
for (const chain of chains) {
  console.log(
    `  ${String(chain.position).padStart(2)}. ${chain.multiplet.padEnd(4)} ` +
    `${chain.labels.join(", ")}  legs ${chain.legs.join(",")}` +
    `  journey ${chain.journey_ranks.length - 1} links, self-avoiding` +
    `  base ${chain.base_face_label} at ${chain.base_face_position}`
  );
}
