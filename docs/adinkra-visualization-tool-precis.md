# Adinkra visualization suite: précis

## Purpose

The maintained browser suite comprises five interfaces and 42 checked-in
figures for the doubly-even binary codes and Garden representations used in the
Adinkra search. The primary application is the **Adinkra Chromotopology Explorer** in
`visualizer/index.html`. It displays the 145 permutation-equivalence classes of
positive-dimensional doubly-even length-16 codes in the catalog.

The intended users are researchers checking the N=16 code catalog, developers
debugging enumeration output, and collaborators who need to compare candidate
chromotopologies without reading raw JSON.

## Primary explorer

The explorer loads `adinkra_codes_n16.json` and offers:

- filtering by code dimension `k=1,...,8`;
- filters for self-duality, self-orthogonality, and indecomposability;
- sorting by dimension, minimum distance, codeword count, zero columns,
  automorphism-group size, or catalog index;
- an overview scatter plot with selectable vertical metrics;
- live summaries of the filtered catalog, including the `k` distribution,
  decomposable versus indecomposable counts, and a zero-column heatmap;
- inspection of a selected code's generator matrix, weight enumerator, column
  weight profile, automorphism-group size, and structural badges;
- a bipartite incidence graph connecting nonzero codewords to the 16 coordinate
  positions in their support, with hover highlighting and tooltips.

The application is static HTML and JavaScript using D3 v7. It has no application
build step or server-side state.

## Additional browser interfaces

The repository also contains four additional browser interfaces:

- `visualizer/adinkra_catalog_3d.html` constructs a user-selected N=16 catalog
  chromotopology. It provides all 145 code classes and all 5,128
  pairs consisting of a code class and a dashing cohomology class, generator
  filters, dashing selection, node inspection, and full or capped edge display.
- `visualizer/adinkra_interactive.html` is a Three.js fly-through viewer for four
  selected colored and dashed graphs at N=4, N=8, and the two N=16
  self-dual topologies, with free navigation and node inspection.
- `visualizer/k8_gadget_atlas.html` is a Plotly interface for the 512
  catalogued `k=8` representations, split between E16/E8xE8 and D16, with Gadget,
  spectral, and strong-edge views.
- `visualizer/adinkra_gallery.html` displays all 42 checked-in analytical and
  graph figures, grouped by subject.

The seven figure-generation programs under `viz/` generated 42 checked-in figures: 10 3D
graph renders, 14 Gadget summaries, four spectral or rank-nullity plots, four
clustering or code-comparison views, three diffusion-map views, and seven
catalog or scaling dashboard panels. The gallery presents this complete set.

These interfaces use different input files and display different mathematical
objects. The codeword-incidence graph, for example, is not an Adinkra graph.

## What is validated

The primary explorer's input file declares `n=16`, `total_classes=145`, and
contains 145 records. Their dimensions are distributed as follows:

```text
k:       1   2   3   4   5   6   7   8
classes: 4  10  23  38  36  23   9   2
```

Every record includes the fields consumed by the four D3 panels: generators,
all codewords, weight distribution, column profile, minimum distance,
decomposability, duality properties, zero-column count, and automorphism-group
size. The catalog contains two self-dual classes and 104 classes marked
indecomposable.

The computational backend has tests for code construction,
canonicalization, chromotopology formation, dashing classes, rankings, Garden
relations, and known low-N counts and gadget values. The visualizer reads the
resulting catalog rather than recomputing these invariants in the browser.

The `export-3d-assets` command produces one compact dashing file per code
class. The manifest contains 145 files and 5,128 dashing classes. A JavaScript
test verifies all headers, constructs sample topologies from four strata, and
checks the odd-face condition for all 256 dashings of catalog class 75. The
legacy four-example graph file records each node, colored edge, and dashing
sign for four reference examples.

## Reproduction

Repository: <https://github.com/p1p3dream/adinkra-codespace>

From the repository root, serve the files over HTTP:

```sh
cargo run --release -- export-3d-assets
node scripts/test_adinkra_catalog_3d.mjs
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000/visualizer/
http://localhost:8000/visualizer/adinkra_catalog_3d.html
http://localhost:8000/visualizer/adinkra_interactive.html
http://localhost:8000/visualizer/k8_gadget_atlas.html
http://localhost:8000/visualizer/adinkra_gallery.html
```

Serving from the repository root matters because the explorer fetches
`../adinkra_codes_n16.json`. Opening `index.html` directly with a `file:` URL may
fail under normal browser fetch restrictions. The primary explorer also loads
D3 v7 from `d3js.org`, so it requires network access unless that dependency is
vendored locally.

The rendering dependencies are pinned in `viz/requirements.txt`. Running
`scripts/generate_all_visualizations.sh` regenerates the catalog assets and all
42 figures, then validates their counts.

## Limitations

- The main explorer visualizes code classes, not all dashings, rankings, or
  supersymmetry transformation matrices.
- Its bipartite panel is a codeword-to-coordinate incidence graph. It is not the
  usual boson-to-fermion colored and dashed Adinkra graph.
- The catalog 3D viewer supports every class and dashing. Its default edge
  budget limits the displayed edges for large low-k graphs. Full graph display
  remains available. Dense low-k graphs remain difficult to interpret.
- The interactive fly-through contains four selected graphs. It is not a second
  catalog-wide viewer.
- The `k=8` interface displays checked-in results. It does not calculate
  arbitrary Gadget values in the browser.
- The browser code has no automated visual-regression or end-to-end test suite.
  Validation covers the catalog and algebraic backend, not pixel-level rendering
  or every interaction path.
- The browser views do not establish dimensional enhancement, Lorentz
  covariance, closure modulo gauge transformations, or the existence of a
  finite off-shell supermultiplet.
- Checked-in JSON files and figures are generated files. The repository does
  not yet provide one command that regenerates the catalog, every calculated
  quantity, and every
  browser asset from scratch.
