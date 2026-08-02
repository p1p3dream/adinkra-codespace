#!/usr/bin/env node

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const templatePath = join(here, "permutahedron_s8_explorer.template.html");
const corePath = join(here, "permutahedron_s8_webgl_core.mjs");
const outputPath = join(here, "permutahedron_s8_explorer.html");

const readJson = path => JSON.parse(readFileSync(path, "utf8"));
const atlas = readJson(join(root, "data", "permutahedron_s8_atlas.json"));
const garden = readJson(join(root, "data", "permutahedron_s8_garden.json"));
const template = readFileSync(templatePath, "utf8");
const core = readFileSync(corePath, "utf8")
  .replace(/^export\s+/gm, "")
  .replace(/^\/\/\# sourceMappingURL=.*$/gm, "");

const output = template
  .replace("/*__S8_WEBGL_CORE__*/", core)
  .replace("/*__S8_ATLAS__*/", JSON.stringify(atlas))
  .replace("/*__S8_GARDEN__*/", JSON.stringify(garden));

if (output.includes("/*__S8_")) {
  throw new Error("S8 explorer template still contains an unresolved placeholder");
}

writeFileSync(outputPath, output);
console.log(`Wrote ${outputPath}`);
console.log(
  `Embedded ${atlas.metadata.vertex_count.toLocaleString()} vertices, ` +
  `${atlas.metadata.edge_count.toLocaleString()} edges, ` +
  `${atlas.right_slices.length.toLocaleString()} R8 octets, and ` +
  `${garden.named_octets.length} named systems.`,
);
