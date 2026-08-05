// Acceptance check for the S4 disassembly build.
//
// This deliberately recomputes every claim from data/permutahedron_s4_atlas.json
// and never reads the validation booleans the builder emits. A build that sets
// its own flags to true cannot pass this script; only the geometry can.
//
// Expected quartets are HowardTLK.v2.pdf p. 61, the listing Gates named as
// authoritative on the 2026-08-04 call.

import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");

const expected = [
  { members: ["1234", "2143", "3412", "4321"], legs: [2, 6, 2], base: "1234", basePosition: 1, multiplet: "VM3" },
  { members: ["1243", "2134", "3421", "4312"], legs: [2, 4, 2], base: "2134", basePosition: 2, multiplet: "VM2" },
  { members: ["1324", "2413", "3142", "4231"], legs: [4, 6, 4], base: "1324", basePosition: 1, multiplet: "VM" },
  { members: ["1342", "2431", "3124", "4213"], legs: [6, 4, 6], base: "3124", basePosition: 3, multiplet: "TM" },
  { members: ["1423", "2314", "3241", "4132"], legs: [4, 2, 4], base: "2314", basePosition: 2, multiplet: "CM" },
  { members: ["1432", "2341", "3214", "4123"], legs: [6, 2, 6], base: "3214", basePosition: 3, multiplet: "VM1" },
];

// arXiv:2408.09342 Tables 3 and 5 print 3412 for VM2. That address belongs to VM3.
// The correct VM2 member is 3421, which is what arXiv:1210.0478 Eq. 5.2 lists and
// what the Garden algebra admits: 3412 gives 0 signings of 65,536, and 3421 gives 256.
const erratum = { wrong: "3412", belongsTo: "VM3", right: "3421", inMultiplet: "VM2" };

const failures = [];
const check = (condition, message) => { if (!condition) failures.push(message); };

const [html, atlasText] = await Promise.all([
  readFile(join(here, "permutahedron_s4_disassembly.html"), "utf8"),
  readFile(join(root, "data", "permutahedron_s4_atlas.json"), "utf8"),
]);

const atlas = JSON.parse(atlasText);
const permutations = atlas.permutations.map(value =>
  Array.isArray(value) ? value.map(Number) : String(value).split("").map(Number)
);
const labels = permutations.map(value => value.join(""));
const rankByLabel = new Map(labels.map((label, rank) => [label, rank]));
const adjacency = Array.from({ length: 24 }, () => new Set());
for (const [a, b] of atlas.edges) { adjacency[a].add(b); adjacency[b].add(a); }

const distances = start => {
  const distance = new Array(24).fill(-1);
  const queue = [start];
  distance[start] = 0;
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    for (const next of adjacency[queue[cursor]]) {
      if (distance[next] !== -1) continue;
      distance[next] = distance[queue[cursor]] + 1;
      queue.push(next);
    }
  }
  return distance;
};

const embedded = html.match(/const DISASSEMBLY\s*=\s*(\{.*?\});?\s*$/m)?.[1]
  ?? html.match(/const DISASSEMBLY\s*=\s*(\{.*\})/)?.[1];
if (!embedded) throw new Error("Embedded disassembly artifact missing from the built HTML.");
const data = JSON.parse(embedded);

check(Array.isArray(data.chains) && data.chains.length === 6,
  `Expected 6 chains, found ${data.chains?.length}.`);

// Quartet membership, ordering, legs, and routes, all recomputed.
data.chains.forEach((chain, index) => {
  const want = expected[index];
  const got = chain.labels;
  check(got.join(",") === want.members.join(","),
    `Quartet ${index + 1} is ${got.join(",")}, expected ${want.members.join(",")}.`);

  // Ascending four-digit order, recomputed from the labels themselves.
  check(got.every((label, i) => i === 0 || Number(got[i - 1]) < Number(label)),
    `Quartet ${index + 1} is not in ascending four-digit order: ${got.join(",")}.`);

  // Legs recomputed by BFS, not read from the artifact.
  const legs = got.slice(0, -1).map((label, i) =>
    distances(rankByLabel.get(label))[rankByLabel.get(got[i + 1])]
  );
  check(legs.join(",") === want.legs.join(","),
    `Quartet ${index + 1} legs are ${legs.join(",")}, expected ${want.legs.join(",")}.`);
  check(legs.join(",") === chain.legs.join(","),
    `Quartet ${index + 1} emitted legs ${chain.legs.join(",")} disagree with recomputed ${legs.join(",")}.`);

  // Every emitted route must be a genuine shortest, non-repeating walk.
  chain.leg_routes.forEach(route => {
    const ranks = route.route_ranks;
    const trueDistance = distances(ranks[0])[ranks.at(-1)];
    check(ranks.length - 1 === trueDistance,
      `Quartet ${index + 1} leg ${route.leg} has length ${ranks.length - 1}, shortest is ${trueDistance}.`);
    check(new Set(ranks).size === ranks.length,
      `Quartet ${index + 1} leg ${route.leg} revisits a node.`);
    ranks.slice(0, -1).forEach((rank, i) => {
      check(adjacency[rank].has(ranks[i + 1]),
        `Quartet ${index + 1} leg ${route.leg} step ${i + 1} is not a permutahedron edge.`);
    });
  });

  // The journey is the three legs concatenated. Gates calls the traversal a
  // self-avoiding walk, so it must visit no node twice across the whole quartet,
  // not merely within each leg.
  const journey = chain.journey_ranks ?? [];
  check(journey.length > 0, `Quartet ${index + 1} has no journey.`);
  check(new Set(journey).size === journey.length,
    `Quartet ${index + 1} journey revisits a node, so it is not self-avoiding.`);
  journey.slice(0, -1).forEach((rank, i) => {
    check(adjacency[rank]?.has(journey[i + 1]),
      `Quartet ${index + 1} journey step ${i + 1} is not a permutahedron edge.`);
  });
  // Recompute the concatenation from the legs and require it to match.
  const rebuilt = chain.leg_routes.reduce((walk, route, i) =>
    walk.concat(i ? route.route_ranks.slice(1) : route.route_ranks), []);
  check(rebuilt.join(",") === journey.join(","),
    `Quartet ${index + 1} journey does not equal its legs concatenated.`);
  // The four members must appear in listed order along the journey.
  const stops = got.map(label => journey.indexOf(rankByLabel.get(label)));
  check(stops.every((stop, i) => stop >= 0 && (i === 0 || stop > stops[i - 1])),
    `Quartet ${index + 1} journey does not pass through its members in order.`);
  check((chain.member_stops ?? []).join(",") === stops.join(","),
    `Quartet ${index + 1} member_stops ${chain.member_stops} disagree with recomputed ${stops}.`);

  // Base-face membership is a highlight, so verify it without letting it order anything.
  check(chain.base_face_label === want.base,
    `Quartet ${index + 1} base member is ${chain.base_face_label}, expected ${want.base}.`);
  check(chain.base_face_position === want.basePosition,
    `Quartet ${index + 1} base position is ${chain.base_face_position}, expected ${want.basePosition}.`);
});

// Multiplet identity, so a reorder cannot silently relabel the physics.
data.chains.forEach((chain, index) => {
  check(chain.multiplet === expected[index].multiplet,
    `Quartet ${index + 1} is multiplet ${chain.multiplet}, expected ${expected[index].multiplet}.`);
});

// Guard the published erratum directly: 3412 must sit in VM3, not VM2.
const holder = data.chains.find(chain => chain.labels.includes(erratum.wrong));
check(holder?.multiplet === erratum.belongsTo,
  `${erratum.wrong} is in ${holder?.multiplet}, but it belongs to ${erratum.belongsTo}.`);
const vm2 = data.chains.find(chain => chain.multiplet === erratum.inMultiplet);
check(vm2?.labels.includes(erratum.right),
  `${erratum.inMultiplet} does not contain ${erratum.right}.`);
check(!vm2?.labels.includes(erratum.wrong),
  `${erratum.inMultiplet} contains ${erratum.wrong}, reintroducing the published erratum.`);

// The quartets must partition all 24 vertices.
const covered = new Set(data.chains.flatMap(chain => chain.labels));
check(covered.size === 24, `Quartets cover ${covered.size} vertices, expected 24.`);

// The base face must close as a real hexagon in the emitted cyclic order.
const cycle = data.base_face_cycle ?? [];
check(cycle.length === 6, `Base face cycle has ${cycle.length} entries, expected 6.`);
check(cycle.every(label => label.endsWith("4")),
  "A base face member does not end in 4.");
cycle.forEach((label, i) => {
  const a = rankByLabel.get(label);
  const b = rankByLabel.get(cycle[(i + 1) % cycle.length]);
  check(adjacency[a]?.has(b),
    `Base face ${label} to ${cycle[(i + 1) % cycle.length]} is not an edge, so the hexagon self-intersects.`);
});

// The six leg patterns must stay distinct; collapsing to one pattern is the
// signature of the ordering reverting to hopper order.
check(new Set(data.chains.map(chain => chain.legs.join(","))).size === 6,
  "The six quartets no longer have six distinct leg patterns.");

if (failures.length) {
  console.error(`FAILED: ${failures.length} acceptance check(s)`);
  failures.forEach(message => console.error(`  - ${message}`));
  process.exit(1);
}
console.log("PASS: quartets, ascending order, legs, routes, base hexagon, and partition all recomputed from the atlas.");
