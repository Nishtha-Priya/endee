from endee import Endee

client = Endee()
client.set_base_url("http://localhost:8080/api/v1")

INDEX_NAME = "interview_index"
index = client.get_index(name=INDEX_NAME)


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
