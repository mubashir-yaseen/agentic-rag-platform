-- ==============================================================================
-- Migration 001: Multi-Document Knowledge Base Schema & Vector Search RPC
-- ==============================================================================

-- 1. Enable the pgvector extension to work with dense vector embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create documents table for multi-document management
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    filename TEXT NOT NULL,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    doc_type TEXT NOT NULL DEFAULT 'pdf',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Index on document upload date and filename
CREATE INDEX IF NOT EXISTS idx_documents_uploaded_at ON documents (uploaded_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents (filename);

-- 3. Create chunks table for multimodal chunk embeddings & JSONB metadata
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes on chunks for foreign key lookup and JSONB metadata filtering
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON chunks USING gin (metadata);

-- Vector similarity index (HNSW for fast approximate nearest neighbor search)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
ON chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 4. Match Chunks RPC Function supporting multi-attribute metadata filtering
CREATE OR REPLACE FUNCTION match_chunks (
    query_embedding vector(384),
    filter_doc_id text DEFAULT NULL,
    filter_content_type text DEFAULT NULL,
    filter_start_date timestamptz DEFAULT NULL,
    filter_end_date timestamptz DEFAULT NULL,
    match_count int DEFAULT 5
)
RETURNS TABLE (
    chunk_id text,
    doc_id text,
    content text,
    metadata jsonb,
    similarity float,
    filename text,
    uploaded_at timestamptz
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.chunk_id,
        c.doc_id,
        c.content,
        c.metadata,
        1 - (c.embedding <=> query_embedding) AS similarity,
        d.filename,
        d.uploaded_at
    FROM chunks c
    JOIN documents d ON c.doc_id = d.doc_id
    WHERE
        (filter_doc_id IS NULL OR c.doc_id = filter_doc_id)
        AND (filter_content_type IS NULL OR c.metadata->>'content_type' = filter_content_type)
        AND (filter_start_date IS NULL OR d.uploaded_at >= filter_start_date)
        AND (filter_end_date IS NULL OR d.uploaded_at <= filter_end_date)
        AND c.embedding IS NOT NULL
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
