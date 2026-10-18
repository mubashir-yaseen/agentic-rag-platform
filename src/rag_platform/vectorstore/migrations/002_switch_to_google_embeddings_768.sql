-- ============================================================================
-- Migration 002: store 768-number embeddings (Google gemini-embedding-001)
-- instead of 384 (MiniLM).
--
-- Run once in Supabase: SQL Editor > New query > paste > Run.
-- Safe to run again: if the column is already vector(768), the data is kept.
--
-- WARNING (first run only): existing chunks are deleted, because their old
-- 384-number vectors cannot be converted. Document rows are kept, but you
-- must upload your PDFs again afterwards.
-- ============================================================================

DO $$
DECLARE
    current_type text;
BEGIN
    SELECT format_type(a.atttypid, a.atttypmod)
      INTO current_type
      FROM pg_attribute a
     WHERE a.attrelid = 'public.chunks'::regclass
       AND a.attname = 'embedding'
       AND NOT a.attisdropped;

    IF current_type IS DISTINCT FROM 'vector(768)' THEN
        DELETE FROM public.chunks;
        DROP INDEX IF EXISTS public.idx_chunks_embedding_hnsw;
        ALTER TABLE public.chunks ALTER COLUMN embedding TYPE vector(768);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
ON public.chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Remove the old 384-number search function (a leftover would confuse the API).
DROP FUNCTION IF EXISTS match_chunks(vector(384), text, text, timestamptz, timestamptz, int);

CREATE OR REPLACE FUNCTION match_chunks (
    query_embedding vector(768),
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

-- Check (should show: vector(768))
-- SELECT format_type(atttypid, atttypmod) FROM pg_attribute
--  WHERE attrelid = 'public.chunks'::regclass AND attname = 'embedding';
