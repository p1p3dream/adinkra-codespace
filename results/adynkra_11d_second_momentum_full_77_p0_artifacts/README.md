# Full 77-column second-momentum artifact bundle

This directory is the local, content-verified copy of every binary column and
companion report used by the committed rank certificate:

```text
results/adynkra_11d_second_momentum_full_77_rank_p0.json
```

The bundle contains 77 `ADFXGPU3` binaries and 77 companion JSON reports at
prime `1073741783`. `manifest.json` binds every local filename to its binary,
companion, semantic, and exact source-stream hashes. The complete matrix has
25,344 functional rows, rank 77 over the Gaussian finite-field extension, and
nullity zero. The recorded columns contain 161,538,963,966 emitted source
terms in total.

The bundle was copied from the production paths recorded in the rank
certificate. Each binary SHA-256 and every companion identity field was
revalidated locally against that certificate before commit. Transient logs,
PID files, checkpoints, and machine-specific status snapshots are excluded.

Regenerate and validate the rank report from this directory with:

```bash
target/release/adinkra-codespace \
  adynkra-11d-second-momentum-full-gpu-rank \
  1073741783 \
  /tmp/adynkra_11d_second_momentum_full_77_rank_p0.json \
  results/adynkra_11d_second_momentum_full_77_p0_artifacts
```

The regenerated report has bundle-local input and artifact paths, so its JSON
bytes differ from the production certificate. Its rank, matrix SHA-256,
static semantic SHA-256, ordinals, and per-column content identities must
match exactly. The committed `manifest.json` provides the local-path binding.
