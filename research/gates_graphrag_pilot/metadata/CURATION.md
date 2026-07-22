# Gates literature GraphRAG pilot: corpus curation

## Scope

The pilot contains nine papers:

1. Four papers designated as the original Adynkra sequence: arXiv:1911.00807, 2002.08502, 2006.03609, and 2007.07390.
2. Three distinct papers explicitly linked by Gates: arXiv:2012.14015, 2304.09830, and 2311.06842.
3. The confirmed fourth paper from the supplied set: arXiv:2012.13308.
4. One later Adynkra continuation: arXiv:2407.09334.

All nine records have verified local PDFs. `manifest.json` is the canonical machine-readable manifest. `manifest.csv` is a flattened equivalent for table loading.

## Verification method

Metadata was checked against:

- each local PDF title page;
- the INSPIRE-derived collection manifest at `/Users/brandon/Documents/S_James_Gates_Publications/MANIFEST.csv`;
- the arXiv Atom API response retrieved on 2026-07-20.

The local PDF title page controls the display title and author list. arXiv and the local collection manifest supply identifier and date cross-checks. Every PDF SHA-256 digest was recomputed and matched the digest in the collection manifest. Every recorded local path exists and was parsed successfully by `pdfinfo`.

## Fourth linked paper

arXiv:2012.13308 is the confirmed fourth paper from the supplied set.

Evidence:

- The paper studies the `S4` permutahedron with 24 vertices and the four-color formulation of minimal 4D, `N = 1` off-shell supersymmetry representations.
- arXiv:2012.14015 cites arXiv:2012.13308.
- arXiv:2304.09830 cites both arXiv:2012.13308 and arXiv:2012.14015 as prior permutahedron work.
- The subject matches Gates's comparison between the 24-node and 40,320-node problems.

Its `S4` and 24-vertex subject supplies the smaller permutahedron analysis used
by arXiv:2012.14015 and arXiv:2304.09830.

## Series boundaries

The corpus keeps three lines of work distinct:

- `adynkra_sequence`: higher-dimensional component decompositions, Adynkras, Adynkrafields, libraries, and the later genome formulation;
- `permutahedron_sequence`: the `S4` and `S8` permutation and permutahedron constructions;
- `unfolded_adinkra`: infinite unfolded Adinkras and their net-centric quantities.

arXiv:2311.06842 was explicitly linked, but its principal subject is unfolded Adinkras rather than the finite permutahedron atlas. It is not relabeled as part of the permutahedron sequence merely because its keywords and discussion include permutahedra.

## Import constraints

- Use `arxiv_id` as the stable pilot paper key.
- Keep DOI and INSPIRE identifiers as alternate identifiers.
- Treat the arXiv and publisher versions as artifacts of one paper, not separate paper nodes.
- Retain `selection_confidence` on every imported paper node.
- Preserve page numbers and section names during later text chunking.
- Require source paper, page or section, extraction method, and review status on inferred entity and relation records.
- Do not derive mathematical claims from title or abstract metadata alone.

## Validation status

- JSON syntax: validated.
- CSV structure: validated.
- Records: 9.
- Unique arXiv identifiers: 9.
- Verified local PDFs: 9.
- SHA-256 matches against the collection manifest: 9.
- Inferred corpus memberships: 0.
