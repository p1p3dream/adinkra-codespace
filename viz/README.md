# Visualization pipeline

The visualization pipeline separates mathematical computation from rendering.

- Rust constructs the catalog, Garden representations, Gadget matrices, and
  catalog-wide dashing assets.
- Python renders the checked-in analytical and graph figures from the Rust
  output.
- JavaScript provides the interactive browser views.

## Reproduce all retained outputs

Create a Python environment and install the pinned rendering dependencies:

```sh
python3 -m venv .venv-viz
. .venv-viz/bin/activate
python3 -m pip install -r viz/requirements.txt
```

Then run:

```sh
scripts/generate_all_visualizations.sh
```

The command regenerates the full Rust pipeline dataset for all 145 code classes
and all eight code dimensions, exports the 145 catalog-wide three-dimensional
assets covering 5,128 dashing classes, renders the 42 checked-in figures, and
runs the browser-data validation.

The full pipeline JSON is approximately 143 MB and is written to
`/tmp/n16_full_pipeline.json`. It is reproducible from the catalog and is not
checked into the repository.

Some individual figures intentionally present selected strata. Their titles and
documentation must state that restriction. The complete-dataset audit and any
additional per-stratum figures are separate from this reproduction command.
