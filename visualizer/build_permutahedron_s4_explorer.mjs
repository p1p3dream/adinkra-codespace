import { readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const templatePath = join(here, "permutahedron_s4_explorer.template.html");
const outputPath = join(here, "permutahedron_s4_explorer.html");

const [template, atlasText, supersymmetryText] = await Promise.all([
  readFile(templatePath, "utf8"),
  readFile(join(root, "data", "permutahedron_s4_atlas.json"), "utf8"),
  readFile(join(root, "data", "permutahedron_s4_supersymmetry.json"), "utf8"),
]);

const atlas = JSON.parse(atlasText);
const supersymmetry = JSON.parse(supersymmetryText);

if (
  atlas?.metadata?.n !== 4 ||
  atlas?.metadata?.vertex_count !== 24 ||
  atlas?.metadata?.edge_count !== 36 ||
  atlas?.metadata?.complete_vertex_and_edge_dataset !== true
) {
  throw new Error("The S4 atlas failed its completeness gate.");
}
if (
  supersymmetry?.validation?.passed !== true ||
  supersymmetry?.validation?.quartet_count !== 6 ||
  supersymmetry?.validation?.garden_representations_passed !== 96
) {
  throw new Error("The S4 supersymmetry artifact failed its validation gate.");
}

const minifiedAtlas = JSON.stringify(atlas).replaceAll("</script", "<\\/script");
const minifiedSupersymmetry = JSON.stringify(supersymmetry).replaceAll(
  "</script",
  "<\\/script",
);

const output = template
  .replace("/*__ATLAS_JSON__*/null", minifiedAtlas)
  .replace("/*__SUPERSYMMETRY_JSON__*/null", minifiedSupersymmetry);

if (output.includes("/*__ATLAS_JSON__*/") || output.includes("/*__SUPERSYMMETRY_JSON__*/")) {
  throw new Error("A data placeholder was not replaced.");
}

await writeFile(outputPath, output);
console.log(
  `Wrote ${outputPath}\n` +
    `Embedded ${atlas.metadata.vertex_count} vertices, ${atlas.metadata.edge_count} edges, ` +
    `${supersymmetry.validation.quartet_count} SUSY quartets, and ` +
    `${supersymmetry.validation.garden_representations_passed} signed representatives.`,
);
