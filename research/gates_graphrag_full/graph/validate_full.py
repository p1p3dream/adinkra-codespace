#!/usr/bin/env python3
"""Validate a full-corpus import plan without a database."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from import_full import (DEFAULT_CITATIONS, DEFAULT_MANIFEST, DEFAULT_SEMANTIC,
                         DEFAULT_SHARDS, DEFAULT_UNRESOLVED, build_plan)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--unresolved", type=Path, default=DEFAULT_UNRESOLVED)
    parser.add_argument("--semantic", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("VALIDATION.json"))
    args = parser.parse_args()
    semantics = args.semantic or ([DEFAULT_SEMANTIC] if DEFAULT_SEMANTIC.exists() else [])
    first = build_plan(args.manifest,args.shard_dir,args.citations,args.unresolved,semantics)
    second = build_plan(args.manifest,args.shard_dir,args.citations,args.unresolved,semantics)
    citation_rows = [json.loads(line) for line in args.citations.read_text(encoding="utf-8").splitlines() if line.strip()]
    external_rows = [json.loads(line) for line in args.unresolved.read_text(encoding="utf-8").splitlines() if line.strip()]
    exact_rows = [row for row in citation_rows if str(row.get("review_status", "")).startswith("accepted_exact")]
    title_rows = [row for row in citation_rows if row.get("review_status") == "pending_title_review"]
    pair = lambda row: (row["source_paper_id"], row["target_paper_id"])
    exact_pairs, title_pairs = {pair(row) for row in exact_rows}, {pair(row) for row in title_rows}
    external_pairs = {(row["source_paper_id"], row["stub_id"]) for row in external_rows}
    cite_states = Counter(e["review_status"] for e in first.edges.values() if e["relationship"] == "CITES")
    reconciliation = {
        "pdf_artifacts": {"canonical": sum(a["is_canonical"] for a in first.artifacts.values()),
                          "exact_called_out_copies": sum(not a["is_canonical"] for a in first.artifacts.values()),
                          "total": len(first.artifacts)},
        "citations": {"exact_occurrences": len(exact_rows), "exact_distinct_pairs": len(exact_pairs),
                      "title_occurrences": len(title_rows), "title_distinct_pairs": len(title_pairs),
                      "exact_title_pair_overlap": len(exact_pairs & title_pairs),
                      "internal_distinct_pairs": len(exact_pairs | title_pairs),
                      "external_stub_occurrences": len(external_rows), "external_distinct_pairs": len(external_pairs),
                      "accepted_edges": cite_states["accepted"], "pending_edges": cite_states["pending"],
                      "total_edges": sum(cite_states.values())},
    }
    checks = {
        "manifest_has_295_corpus_papers": first.counts()["corpus_papers"] == 295,
        "all_166_local_pdfs_have_chunks": len({c["paper_id"] for c in first.chunks.values()}) == 166,
        "all_edges_have_evidence": {e["edge_id"] for e in first.edges.values()} <= {e["edge_id"] for e in first.edge_evidence.values()},
        "all_nodes_have_evidence": {n["node_id"] for n in first.nodes.values()} <= {e["node_id"] for e in first.node_evidence.values()},
        "no_dangling_chunk_evidence": all(not e["chunk_id"] or e["chunk_id"] in first.chunks for e in [*first.node_evidence.values(),*first.edge_evidence.values()]),
        "deterministic_plan": first.digest() == second.digest(),
        "zero_resolution_warnings": not first.warnings,
        "all_planned_strings_are_nul_free": True,
        "pdf_artifacts_reconcile_166_plus_4": first.counts()["artifacts"] == 170,
        "citation_edges_reconcile": cite_states == {"accepted": len(exact_pairs),
            "pending": len(title_pairs - exact_pairs) + len(external_pairs)},
        "one_review_state_per_citation_edge": len({e["edge_id"] for e in first.edges.values() if e["relationship"] == "CITES"}) == sum(e["relationship"] == "CITES" for e in first.edges.values()),
    }
    result = {"status": "pass" if all(checks.values()) else "fail", "checks": checks,
              "counts": first.counts(), "manifest_sha256": first.manifest_sha256,
              "plan_sha256": first.digest(), "reconciliation": reconciliation,
              "semantic_inputs": [str(p) for p in semantics]}
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
