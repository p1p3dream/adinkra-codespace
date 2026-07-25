-- Isolated storage for the full Gates literature corpus.
-- This migration does not read or alter gates_pilot_* or shared graphrag_* tables.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS gates_full_schema_migrations (
    migration_id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS gates_full_corpora (
    corpus_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    description TEXT,
    manifest_sha256 TEXT NOT NULL,
    plan_sha256 TEXT NOT NULL,
    properties JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS gates_full_ingest_sources (
    corpus_id TEXT NOT NULL REFERENCES gates_full_corpora(corpus_id),
    source_id TEXT NOT NULL,
    source_role TEXT NOT NULL,
    local_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    properties JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (corpus_id, source_id)
);

CREATE TABLE IF NOT EXISTS gates_full_papers (
    corpus_id TEXT NOT NULL REFERENCES gates_full_corpora(corpus_id),
    paper_id TEXT NOT NULL,
    stable_identifier TEXT NOT NULL,
    title TEXT NOT NULL,
    publication_year INTEGER,
    document_type TEXT,
    is_external_stub BOOLEAN NOT NULL DEFAULT false,
    full_text_status TEXT NOT NULL,
    properties JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (corpus_id, paper_id),
    UNIQUE (corpus_id, stable_identifier)
);

CREATE TABLE IF NOT EXISTS gates_full_identifiers (
    corpus_id TEXT NOT NULL,
    identifier_type TEXT NOT NULL,
    identifier_value TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    is_canonical BOOLEAN NOT NULL DEFAULT false,
    properties JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (corpus_id, identifier_type, identifier_value, paper_id),
    FOREIGN KEY (corpus_id, paper_id)
        REFERENCES gates_full_papers(corpus_id, paper_id)
);

CREATE TABLE IF NOT EXISTS gates_full_artifacts (
    corpus_id TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    local_path TEXT,
    source_url TEXT,
    sha256 TEXT,
    byte_count BIGINT,
    page_count INTEGER,
    is_canonical BOOLEAN NOT NULL DEFAULT false,
    properties JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (corpus_id, artifact_id),
    FOREIGN KEY (corpus_id, paper_id)
        REFERENCES gates_full_papers(corpus_id, paper_id)
);

CREATE TABLE IF NOT EXISTS gates_full_chunks (
    corpus_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    section_title TEXT,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    content_embedding vector(768),
    properties JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (corpus_id, chunk_id),
    FOREIGN KEY (corpus_id, paper_id)
        REFERENCES gates_full_papers(corpus_id, paper_id)
);

CREATE TABLE IF NOT EXISTS gates_full_nodes (
    corpus_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    node_type TEXT NOT NULL,
    canonical_key TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    description_embedding vector(768),
    properties JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (corpus_id, node_id),
    UNIQUE (corpus_id, node_type, canonical_key)
);

CREATE TABLE IF NOT EXISTS gates_full_edges (
    corpus_id TEXT NOT NULL,
    edge_id TEXT NOT NULL,
    src_node_id TEXT NOT NULL,
    dst_node_id TEXT NOT NULL,
    relationship TEXT NOT NULL,
    description TEXT,
    basis TEXT NOT NULL CHECK (basis IN
        ('bibliographic', 'explicit_text', 'automated_inference', 'manual')),
    review_status TEXT NOT NULL CHECK (review_status IN
        ('observed', 'pending', 'accepted', 'rejected')),
    confidence DOUBLE PRECISION CHECK (confidence >= 0 AND confidence <= 1),
    properties JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (corpus_id, edge_id),
    FOREIGN KEY (corpus_id, src_node_id)
        REFERENCES gates_full_nodes(corpus_id, node_id),
    FOREIGN KEY (corpus_id, dst_node_id)
        REFERENCES gates_full_nodes(corpus_id, node_id),
    UNIQUE (corpus_id, src_node_id, dst_node_id, relationship)
);

CREATE TABLE IF NOT EXISTS gates_full_node_evidence (
    corpus_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    chunk_id TEXT,
    source_kind TEXT NOT NULL,
    locator TEXT,
    excerpt TEXT,
    extraction_method TEXT NOT NULL,
    confidence DOUBLE PRECISION CHECK (confidence >= 0 AND confidence <= 1),
    properties JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (corpus_id, evidence_id),
    FOREIGN KEY (corpus_id, node_id)
        REFERENCES gates_full_nodes(corpus_id, node_id),
    FOREIGN KEY (corpus_id, paper_id)
        REFERENCES gates_full_papers(corpus_id, paper_id),
    FOREIGN KEY (corpus_id, chunk_id)
        REFERENCES gates_full_chunks(corpus_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS gates_full_edge_evidence (
    corpus_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    edge_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    chunk_id TEXT,
    source_kind TEXT NOT NULL,
    locator TEXT,
    excerpt TEXT,
    extraction_method TEXT NOT NULL,
    confidence DOUBLE PRECISION CHECK (confidence >= 0 AND confidence <= 1),
    properties JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (corpus_id, evidence_id),
    FOREIGN KEY (corpus_id, edge_id)
        REFERENCES gates_full_edges(corpus_id, edge_id),
    FOREIGN KEY (corpus_id, paper_id)
        REFERENCES gates_full_papers(corpus_id, paper_id),
    FOREIGN KEY (corpus_id, chunk_id)
        REFERENCES gates_full_chunks(corpus_id, chunk_id)
);
