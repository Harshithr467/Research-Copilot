from elasticsearch import Elasticsearch

def create_research_index():
    # Initialize the Elasticsearch client (v8 compatibility)
    es = Elasticsearch(
        ["http://localhost:9200"],
        verify_certs=False,
        request_timeout=30
    )

    index_name = "research_papers"

    # Define the mapping
    index_mapping = {
        "properties": {
            "title": {
                "type": "text",
                "analyzer": "standard"
            },
            "content": {
                "type": "text",
                "analyzer": "standard"
            },
            "embedding": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "author": {"type": "keyword"},
                    "year": {"type": "integer"},
                    "source_file": {"type": "keyword"}
                }
            }
        }
    }

    try:
        # Check if index already exists
        if es.indices.exists(index=index_name):
            print(f"Index '{index_name}' already exists. Deleting it to re-create with new mapping...")
            es.indices.delete(index=index_name)

        # Create the index
        es.indices.create(index=index_name, mappings=index_mapping)
        print(f"Successfully created index: {index_name}")

    except Exception as e:
        print(f"Error creating index: {e}")

if __name__ == "__main__":
    create_research_index()
