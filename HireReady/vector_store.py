from endee import Endee

client = Endee()
client.set_base_url("http://localhost:8080/api/v1")

INDEX_NAME = "interview_index"

# Ensure index exists
try:
    index = client.get_index(name=INDEX_NAME)
except Exception:
    print("Index not found. Creating new index...")
    index = client.create_index(
        name=INDEX_NAME,
        dimension=384,   # must match embedding model
        metric="cosine"
    )


def upsert_vector(vector_id, embedding, metadata):
    index.upsert([
        {
            "id": str(vector_id),
            "vector": embedding,
            "meta": metadata,
            "filter": {}
        }
    ])


def search_vector(query_embedding, top_k=3):
    results = index.query(
        vector=query_embedding,
        top_k=top_k
    )
    return results