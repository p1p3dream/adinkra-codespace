CREATE INDEX IF NOT EXISTS idx_gates_full_papers_title
    ON gates_full_papers (corpus_id, title);
CREATE INDEX IF NOT EXISTS idx_gates_full_chunks_paper
    ON gates_full_chunks (corpus_id, paper_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_gates_full_nodes_type
    ON gates_full_nodes (corpus_id, node_type);
CREATE INDEX IF NOT EXISTS idx_gates_full_edges_src
    ON gates_full_edges (corpus_id, src_node_id);
CREATE INDEX IF NOT EXISTS idx_gates_full_edges_dst
    ON gates_full_edges (corpus_id, dst_node_id);
CREATE INDEX IF NOT EXISTS idx_gates_full_edges_review
    ON gates_full_edges (corpus_id, review_status, relationship);
CREATE INDEX IF NOT EXISTS idx_gates_full_chunks_fts
    ON gates_full_chunks USING gin (to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_gates_full_nodes_fts
    ON gates_full_nodes USING gin (
        to_tsvector('english', coalesce(name, '') || ' ' || coalesce(description, ''))
    );
CREATE INDEX IF NOT EXISTS idx_gates_full_chunks_embedding_hnsw
    ON gates_full_chunks USING hnsw (content_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_gates_full_nodes_embedding_hnsw
    ON gates_full_nodes USING hnsw (description_embedding vector_cosine_ops);

CREATE OR REPLACE VIEW gates_full_relationship_catalog AS
SELECT e.corpus_id, e.edge_id,
       s.node_type AS source_type, s.canonical_key AS source_key, s.name AS source_name,
       e.relationship,
       t.node_type AS target_type, t.canonical_key AS target_key, t.name AS target_name,
       e.review_status, e.confidence,
       ev.paper_id AS evidence_paper_id, ev.chunk_id AS evidence_chunk_id,
       ev.locator, ev.excerpt, ev.extraction_method
FROM gates_full_edges e
JOIN gates_full_nodes s
  ON s.corpus_id=e.corpus_id AND s.node_id=e.src_node_id
JOIN gates_full_nodes t
  ON t.corpus_id=e.corpus_id AND t.node_id=e.dst_node_id
LEFT JOIN gates_full_edge_evidence ev
  ON ev.corpus_id=e.corpus_id AND ev.edge_id=e.edge_id;
