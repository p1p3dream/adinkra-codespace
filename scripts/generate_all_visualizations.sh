#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PIPELINE_JSON="${ADINKRA_VIZ_PIPELINE_JSON:-/tmp/n16_full_pipeline.json}"
if [[ "$PIPELINE_JSON" != "/tmp/n16_full_pipeline.json" ]]; then
  echo "The current figure generators require /tmp/n16_full_pipeline.json." >&2
  exit 2
fi

echo "Generating the complete N=16 source dataset with Rust..."
cargo run --release --quiet -- pipeline adinkra_codes_n16.json > "$PIPELINE_JSON"

echo "Exporting catalog-wide 3D dashing assets with Rust..."
cargo run --release --quiet -- export-3d-assets \
  adinkra_codes_n16.json visualizer/adinkra_dashing > /tmp/adinkra_3d_export_manifest.json

echo "Rendering checked-in figures with the pinned Python visualization environment..."
python3 viz/gadget_heatmaps.py
python3 viz/gadget_distributions.py
python3 viz/eigenvalue_spectra.py
python3 viz/diffusion_maps.py
python3 viz/code_clustering.py
python3 viz/strata_dashboard.py
python3 viz/adinkra_3d_graph.py

echo "Validating catalog-wide browser assets..."
node scripts/test_adinkra_catalog_3d.mjs

figure_count="$(find viz -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')"
if [[ "$figure_count" != "42" ]]; then
  echo "Expected 42 checked-in figures, found $figure_count." >&2
  exit 1
fi

echo "Visualization generation complete: $figure_count figures, 145 code classes, 5,128 dashing classes."
