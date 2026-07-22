-- Dedicated, corpus-scoped storage for the Gates literature GraphRAG pilot.
--
-- This file creates new gates_pilot_* tables only. It does not alter the
-- shared graphrag_* or embeddings tables.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS gates_pilot_corpora (
    corpus_id          TEXT PRIMARY KEY,
    description        TEXT,
    manifest_sha256    TEXT NOT NULL,
    extraction_sha256  TEXT NOT NULL,
    properties         JSONB NOT NULL DEFAULT '{}',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS gates_pilot_papers (
    corpus_id          TEXT NOT NULL REFERENCES gates_pilot_corpora(corpus_id),
    paper_id           TEXT NOT NULL,
    stable_identifier  TEXT NOT NULL,
    title              TEXT NOT NULL,
    publication_year   INTEGER,
    abstract           TEXT,
    is_stub            BOOLEAN NOT NULL DEFAULT false,
    properties         JSONB NOT NULL DEFAULT '{}',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (corpus_id, paper_id),
    UNIQUE (corpus_id, stable_identifier)
);

CREATE TABLE IF NOT EXISTS gates_pilot_identifiers (
    corpus_id          TEXT NOT NULL,
    identifier_type    TEXT NOT NULL,
    identifier_value   TEXT NOT NULL,
    paper_id           TEXT NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (corpus_id, identifier_type, identifier_value),
    FOREIGN KEY (corpus_id, paper_id)
        REFERENCES gates_pilot_papers(corpus_id, paper_id)
);

CREATE TABLE IF NOT EXISTS gates_pilot_artifacts (
    corpus_id          TEXT NOT NULL,
    artifact_id        TEXT NOT NULL,
    paper_id           TEXT NOT NULL,
    artifact_type      TEXT NOT NULL,
    local_path         TEXT,
    source_url         TEXT,
    sha256             TEXT,
    properties         JSONB NOT NULL DEFAULT '{}',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (corpus_id, artifact_id),
    FOREIGN KEY (corpus_id, paper_id)
        REFERENCES gates_pilot_papers(corpus_id, paper_id),
    UNIQUE NULLS NOT DISTINCT (corpus_id, paper_id, artifact_type, sha256, local_path)
);

CREATE TABLE IF NOT EXISTS gates_pilot_chunks (
    corpus_id          TEXT NOT NULL,
    chunk_id           TEXT NOT NULL,
    paper_id           TEXT NOT NULL,
    section_title      TEXT,
    section_path       TEXT,
    page_start         INTEGER,
    page_end           INTEGER,
    chunk_index        INTEGER NOT NULL DEFAULT 0,
    content            TEXT NOT NULL,
    content_sha256     TEXT NOT NULL,
    content_embedding  vector(768),
    properties         JSONB NOT NULL DEFAULT '{}',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (corpus_id, chunk_id),
    FOREIGN KEY (corpus_id, paper_id)
        REFERENCES gates_pilot_papers(corpus_id, paper_id)
);

CREATE TABLE IF NOT EXISTS gates_pilot_nodes (
    corpus_id          TEXT NOT NULL,
    node_id            TEXT NOT NULL,
    node_type          TEXT NOT NULL CHECK (
        node_type IN ('paper', 'author', 'concept', 'claim', 'result', 'series')
    ),
    canonical_key      TEXT NOT NULL,
    name               TEXT NOT NULL,
    description        TEXT,
    description_embedding vector(768),
    properties         JSONB NOT NULL DEFAULT '{}',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (corpus_id, node_id),
    UNIQUE (corpus_id, node_type, canonical_key)
);

CREATE TABLE IF NOT EXISTS gates_pilot_edges (
    corpus_id          TEXT NOT NULL,
    edge_id            TEXT NOT NULL,
    src_node_id        TEXT NOT NULL,
    dst_node_id        TEXT NOT NULL,
    relationship       TEXT NOT NULL,
    description        TEXT,
    basis              TEXT NOT NULL CHECK (
        basis IN ('bibliographic', 'explicit_text', 'automated_inference', 'manual')
    ),
    review_status      TEXT NOT NULL CHECK (
        review_status IN ('observed', 'pending', 'accepted', 'rejected')
    ),
    confidence         DOUBLE PRECISION CHECK (confidence >= 0 AND confidence <= 1),
    properties         JSONB NOT NULL DEFAULT '{}',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (corpus_id, edge_id),
    FOREIGN KEY (corpus_id, src_node_id)
        REFERENCES gates_pilot_nodes(corpus_id, node_id),
    FOREIGN KEY (corpus_id, dst_node_id)
        REFERENCES gates_pilot_nodes(corpus_id, node_id),
    UNIQUE (corpus_id, src_node_id, dst_node_id, relationship)
);

CREATE TABLE IF NOT EXISTS gates_pilot_node_evidence (
    corpus_id          TEXT NOT NULL,
    evidence_id        TEXT NOT NULL,
    node_id            TEXT NOT NULL,
    paper_id           TEXT NOT NULL,
    chunk_id           TEXT,
    source_kind        TEXT NOT NULL,
    locator            TEXT,
    excerpt            TEXT,
    extraction_method  TEXT NOT NULL,
    confidence         DOUBLE PRECISION CHECK (confidence >= 0 AND confidence <= 1),
    properties         JSONB NOT NULL DEFAULT '{}',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (corpus_id, evidence_id),
    FOREIGN KEY (corpus_id, node_id)
        REFERENCES gates_pilot_nodes(corpus_id, node_id),
    FOREIGN KEY (corpus_id, paper_id)
        REFERENCES gates_pilot_papers(corpus_id, paper_id),
    FOREIGN KEY (corpus_id, chunk_id)
        REFERENCES gates_pilot_chunks(corpus_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS gates_pilot_edge_evidence (
    corpus_id          TEXT NOT NULL,
    evidence_id        TEXT NOT NULL,
    edge_id            TEXT NOT NULL,
    paper_id           TEXT NOT NULL,
    chunk_id           TEXT,
    source_kind        TEXT NOT NULL,
    locator            TEXT,
    excerpt            TEXT,
    extraction_method  TEXT NOT NULL,
    confidence         DOUBLE PRECISION CHECK (confidence >= 0 AND confidence <= 1),
    properties         JSONB NOT NULL DEFAULT '{}',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (corpus_id, evidence_id),
    FOREIGN KEY (corpus_id, edge_id)
        REFERENCES gates_pilot_edges(corpus_id, edge_id),
    FOREIGN KEY (corpus_id, paper_id)
        REFERENCES gates_pilot_papers(corpus_id, paper_id),
    FOREIGN KEY (corpus_id, chunk_id)
        REFERENCES gates_pilot_chunks(corpus_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_gates_pilot_papers_title
    ON gates_pilot_papers (corpus_id, title);
CREATE INDEX IF NOT EXISTS idx_gates_pilot_chunks_paper
    ON gates_pilot_chunks (corpus_id, paper_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_gates_pilot_nodes_type
    ON gates_pilot_nodes (corpus_id, node_type);
CREATE INDEX IF NOT EXISTS idx_gates_pilot_edges_src
    ON gates_pilot_edges (corpus_id, src_node_id);
CREATE INDEX IF NOT EXISTS idx_gates_pilot_edges_dst
    ON gates_pilot_edges (corpus_id, dst_node_id);
CREATE INDEX IF NOT EXISTS idx_gates_pilot_edges_review
    ON gates_pilot_edges (corpus_id, review_status, relationship);
CREATE INDEX IF NOT EXISTS idx_gates_pilot_chunks_fts
    ON gates_pilot_chunks USING gin (to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_gates_pilot_nodes_fts
    ON gates_pilot_nodes USING gin (
        to_tsvector('english', coalesce(name, '') || ' ' || coalesce(description, ''))
    );
CREATE INDEX IF NOT EXISTS idx_gates_pilot_chunks_embedding_hnsw
    ON gates_pilot_chunks USING hnsw (content_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_gates_pilot_nodes_embedding_hnsw
    ON gates_pilot_nodes USING hnsw (description_embedding vector_cosine_ops);

CREATE OR REPLACE VIEW gates_pilot_entity_catalog AS
SELECT n.corpus_id,
       n.node_id,
       coalesce(n.properties->>'semantic_type', n.node_type) AS semantic_type,
       n.canonical_key,
       n.name,
       n.description,
       n.properties
FROM gates_pilot_nodes n;

CREATE OR REPLACE VIEW gates_pilot_relationship_catalog AS
SELECT e.corpus_id,
       e.edge_id,
       coalesce(s.properties->>'semantic_type', s.node_type) AS source_type,
       s.canonical_key AS source_key,
       s.name AS source_name,
       e.relationship,
       coalesce(t.properties->>'semantic_type', t.node_type) AS target_type,
       t.canonical_key AS target_key,
       t.name AS target_name,
       e.review_status,
       e.confidence,
       ev.paper_id AS evidence_paper_id,
       ev.chunk_id AS evidence_chunk_id,
       ev.locator,
       ev.excerpt,
       ev.extraction_method
FROM gates_pilot_edges e
JOIN gates_pilot_nodes s
  ON s.corpus_id=e.corpus_id AND s.node_id=e.src_node_id
JOIN gates_pilot_nodes t
  ON t.corpus_id=e.corpus_id AND t.node_id=e.dst_node_id
LEFT JOIN gates_pilot_edge_evidence ev
  ON ev.corpus_id=e.corpus_id AND ev.edge_id=e.edge_id;
