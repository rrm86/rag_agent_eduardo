CREATE OR REPLACE FUNCTION match_documents(
  query_embedding vector(768),
  filter_collection_name text,
  match_count int DEFAULT 5
) RETURNS TABLE (
  id bigint,
  content text,
  metadata jsonb,
  embedding vector(768),
  collection_name text,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    t.id,
    t.content,
    t.metadata,
    t.embedding,
    t.collection_name,
    -- Computa similaridade de cosseno
    1 - (t.embedding <=> query_embedding) as similarity
  FROM
    eduardo_embeddings t
  WHERE
    t.collection_name = filter_collection_name
  ORDER BY
    t.embedding <=> query_embedding
  LIMIT
    match_count;
END;
$$;