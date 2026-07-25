import assert from 'node:assert/strict';
import fs from 'node:fs';

import {
  buildTopology,
  edgeIsDashed,
  otherEndpoint,
  parseDashingAsset,
} from '../visualizer/adinkra_graph_core.mjs';

const catalog = JSON.parse(fs.readFileSync('adinkra_codes_n16.json', 'utf8'));
const manifest = JSON.parse(fs.readFileSync('visualizer/adinkra_dashing/manifest.json', 'utf8'));

assert.equal(catalog.codes.length, 145);
assert.equal(manifest.code_count, 145);
assert.equal(manifest.assets.length, 145);
assert.equal(manifest.total_dashing_classes, 5128);

function readAsset(record) {
  const bytes = fs.readFileSync(`visualizer/adinkra_dashing/${record.file}`);
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return parseDashingAsset(buffer);
}

for (const record of manifest.assets) {
  const asset = readAsset(record);
  assert.equal(asset.codeIndex, record.code_index);
  assert.equal(asset.k, record.k);
  assert.equal(asset.vertices, record.vertices);
  assert.equal(asset.edges, record.edges);
  assert.equal(asset.descriptors.length, 2 * record.edges);
}

for (const codeIndex of [0, 75, 76, 144]) {
  const code = catalog.codes.find((item) => item.index === codeIndex);
  const record = manifest.assets.find((item) => item.code_index === codeIndex);
  const topology = buildTopology(16, code.generators_raw, readAsset(record));
  assert.equal(topology.reps.length, record.vertices);
  assert.equal(topology.edgeA.length, record.edges);
  for (const edge of topology.edgeIndexByVertexColor) assert.notEqual(edge, 0xffffffff);
}

const code = catalog.codes.find((item) => item.index === 75);
const record = manifest.assets.find((item) => item.code_index === 75);
const asset = readAsset(record);
const topology = buildTopology(16, code.generators_raw, asset);

for (let dashing = 0; dashing < record.dashing_classes; dashing += 1) {
  for (let vertex = 0; vertex < topology.reps.length; vertex += 1) {
    for (let first = 0; first < 16; first += 1) {
      const edge1 = topology.edgeIndexByVertexColor[16 * vertex + first];
      const firstVertex = otherEndpoint(topology, edge1, vertex);
      for (let second = first + 1; second < 16; second += 1) {
        const edge2 = topology.edgeIndexByVertexColor[16 * firstVertex + second];
        const edge3 = topology.edgeIndexByVertexColor[16 * vertex + second];
        const secondVertex = otherEndpoint(topology, edge3, vertex);
        const edge4 = topology.edgeIndexByVertexColor[16 * secondVertex + first];
        const parity = [edge1, edge2, edge3, edge4]
          .map((edge) => edgeIsDashed(topology.baseAt(edge), topology.maskAt(edge), dashing))
          .filter(Boolean).length % 2;
        assert.equal(parity, 1, `even face at dashing ${dashing}, vertex ${vertex}, colors ${first},${second}`);
      }
    }
  }
}

console.log('3D catalog assets: 145 classes, 5,128 dashings, topology samples, and odd faces verified');
