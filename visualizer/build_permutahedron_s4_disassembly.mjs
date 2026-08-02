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
const hopperWords = [[], [1, 3], [2, 1, 3, 2], [3, 1, 2, 1, 3, 2]];
const adjacency = Array.from({ length: 24 }, () => new Map());
for (const [a, b, generator] of atlas.edges) {
  adjacency[a].set(generator, b);
  adjacency[b].set(generator, a);
}
const applyWord = (start, word) => word.reduce((rank, generator) => {
  const next = adjacency[rank].get(generator);
  if (next === undefined) throw new Error(`Missing generator ${generator} at rank ${rank}`);
  return next;
}, start);

if (
  atlas?.metadata?.vertex_count !== 24 ||
  atlas?.metadata?.edge_count !== 36 ||
  supersymmetry?.validation?.passed !== true ||
  supersymmetry?.validation?.quartet_count !== 6
) {
  throw new Error("The S4 disassembly input artifacts failed validation.");
}

const sectors = supersymmetry.sectors.map(sector => {
  const hopperOrderedRanks = hopperWords.map(word => applyWord(sector.ordered_ranks[0], word));
  const expected = [...sector.ordered_ranks].sort((a, b) => a - b);
  const actual = [...hopperOrderedRanks].sort((a, b) => a - b);
  if (new Set(actual).size !== 4 || actual.some((rank, index) => rank !== expected[index])) {
    throw new Error(`${sector.id} is not the four-member orbit of the common hopper path.`);
  }
  return { id: sector.id, hopper_ordered_ranks: hopperOrderedRanks };
});
const covered = new Set(sectors.flatMap(sector => sector.hopper_ordered_ranks));
if (covered.size !== 24) throw new Error("The six hopper paths do not partition S4.");

const disassembly = {
  schema_version: "s4-six-quartet-disassembly-v1",
  hopper_labels: ["H1", "H2", "H3", "H4"],
  hopper_expressions: ["()", "(12)(34)", "(23)(12)(34)(23)", "(34)(12)(23)(12)(34)(23)"],
  sectors,
  validation: {
    paths: 6,
    vertices_per_path: 4,
    covered_vertices: covered.size,
    common_hopper_path_verified: true,
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
console.log("Verified six common hopper paths, four permutations each, covering all 24 S4 vertices.");
