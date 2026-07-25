# Gates literature GraphRAG pilot import

This directory contains a safe, corpus-scoped import layer for the focused
Gates literature pilot. It does not modify the shared `graphrag_*` graph.

## Isolation decision

The existing shared graph tables are not safe for corpus isolation:

- They do not have a `corpus_id` column.
- Vertex IDs are derived from entity name and type, so matching names from
  different corpora merge.
- Edge uniqueness is global across source, target, and relationship.
- Existing graph search and analytics do not apply corpus predicates.
- Build state, communities, and neighborhood calculations are global.

Encoding the corpus only in JSON properties or identifier prefixes would
prevent some key collisions but would not make search and analytics isolated.
The pilot therefore uses dedicated `gates_pilot_*` tables. Every table carries
`corpus_id`, and every graph node and edge has a separate evidence record.

## Safety properties

- Dry-run is the default.
- Database writes require `--apply`.
- Schema creation adds only `gates_pilot_*` tables and indexes.
- The importer never runs `DROP`, `TRUNCATE`, `DELETE`, or `ALTER`.
- An import is one transaction guarded by a corpus-specific advisory lock.
- Upserts are idempotent and scoped by `corpus_id`.
- Reimporting a pending automated edge does not overwrite an existing accepted
  or rejected review decision.
- Tests use temporary files and never connect to the shared database.
- Automated relationships start as `pending`, not established facts.

## Files

- `schema.sql`: dedicated PostgreSQL and pgvector tables.
- `import_pilot.py`: input normalization, deduplication, validation, dry-run,
  and transactional import.
- `FORMAT.md`: manifest and JSONL contract.
- `tests/`: database-free unit tests.

## Safe run procedure

Set paths to the focused manifest and section-aware extraction output:

```bash
MANIFEST=/path/to/pilot_manifest.json
RAW_EXTRACTED=/path/to/pilot_extracted.jsonl
EXTRACTED=/path/to/pilot_extracted_with_citations.jsonl
```

1. Add exact within-pilot arXiv citation records:

```bash
python3 research/gates_graphrag_pilot/graph/enrich_citations.py \
  --manifest "$MANIFEST" \
  --input "$RAW_EXTRACTED" \
  --output "$EXTRACTED"
```

2. Run validation without database access:

```bash
python3 research/gates_graphrag_pilot/graph/import_pilot.py \
  --manifest "$MANIFEST" \
  --extracted "$EXTRACTED" \
  --validate-only
```

3. Inspect a dry-run report:

```bash
python3 research/gates_graphrag_pilot/graph/import_pilot.py \
  --manifest "$MANIFEST" \
  --extracted "$EXTRACTED" \
  --dry-run \
  --report /tmp/gates-pilot-import-report.json
```

Confirm the paper count, node types, relationship types, warnings, provenance
counts, and the number of `pending` relationships before continuing.

4. Create only the dedicated pilot tables:

```bash
psql "$GATES_GRAPHRAG_DSN" -v ON_ERROR_STOP=1 \
  -f research/gates_graphrag_pilot/graph/schema.sql
```

Review `schema.sql` before running it. Do not run the shared graph reset or drop
commands for this pilot.

5. Apply the validated import:

```bash
python3 research/gates_graphrag_pilot/graph/import_pilot.py \
  --manifest "$MANIFEST" \
  --extracted "$EXTRACTED" \
  --apply \
  --report /tmp/gates-pilot-applied-report.json
```

`--apply` reads the connection from `GATES_GRAPHRAG_DSN`. The DSN is not stored
in the repository.

6. Populate the chunk vectors:

```bash
python3 research/gates_graphrag_pilot/graph/embed_pilot.py
```

7. Search the isolated pilot:

```bash
python3 research/gates_graphrag_pilot/graph/search_pilot.py \
  "How do permutahedra enter Adinkra constructions?"
```

Explore the graph neighborhood of a concept, method, result or paper:

```bash
python3 research/gates_graphrag_pilot/graph/explore_pilot.py "hopping operator"
```

## Semantic relationships

Validate the combined passage-backed proposals:

```bash
python3 research/gates_graphrag_pilot/enrichment/validate_proposals.py \
  --chunks "$EXTRACTED" \
  --input research/gates_graphrag_pilot/enrichment/combined/proposals.jsonl \
  --output /tmp/gates-semantic-proposals.jsonl \
  --report /tmp/gates-semantic-validation.json
```

Inspect the transactional dry run, then apply:

```bash
python3 research/gates_graphrag_pilot/enrichment/import_proposals.py \
  --input /tmp/gates-semantic-proposals.jsonl

python3 research/gates_graphrag_pilot/enrichment/import_proposals.py \
  --input /tmp/gates-semantic-proposals.jsonl --apply
```

The semantic importer never deletes rows. New relationships begin as
`pending`, and reimport does not overwrite accepted or rejected decisions.

5. Validate isolation and provenance with read-only queries:

```sql
SELECT corpus_id, count(*) FROM gates_pilot_papers GROUP BY corpus_id;
SELECT node_type, count(*) FROM gates_pilot_nodes
WHERE corpus_id = 'gates_literature_pilot' GROUP BY node_type;
SELECT review_status, count(*) FROM gates_pilot_edges
WHERE corpus_id = 'gates_literature_pilot' GROUP BY review_status;

SELECT count(*) AS nodes_without_evidence
FROM gates_pilot_nodes n
LEFT JOIN gates_pilot_node_evidence e
  ON e.corpus_id = n.corpus_id AND e.node_id = n.node_id
WHERE n.corpus_id = 'gates_literature_pilot' AND e.evidence_id IS NULL;

SELECT count(*) AS edges_without_evidence
FROM gates_pilot_edges g
LEFT JOIN gates_pilot_edge_evidence e
  ON e.corpus_id = g.corpus_id AND e.edge_id = g.edge_id
WHERE g.corpus_id = 'gates_literature_pilot' AND e.evidence_id IS NULL;
```

Both provenance queries must return zero.

Two read-only views simplify inspection:

- `gates_pilot_entity_catalog`
- `gates_pilot_relationship_catalog`

## Test

```bash
python3 -m unittest discover \
  -s research/gates_graphrag_pilot/graph/tests -v
```
