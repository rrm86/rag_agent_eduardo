-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schema if not exists (use public if you don't have a specific schema)
CREATE SCHEMA IF NOT EXISTS public;

-- Create the table for storing embeddings with the structure expected by our code
CREATE TABLE IF NOT EXISTS public.eduardo_embeddings (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding VECTOR(768), -- Dimension 768 is for text-embedding-004
    collection_name TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Create an index for the collection_name field for faster filtering
CREATE INDEX IF NOT EXISTS idx_collection_name ON public.eduardo_embeddings (collection_name);

-- Create a vector index for faster similarity searches
CREATE INDEX IF NOT EXISTS eduardo_embeddings_vector_idx ON public.eduardo_embeddings 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100); -- You may adjust the number of lists based on your data size

