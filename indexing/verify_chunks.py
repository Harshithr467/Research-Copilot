from elasticsearch import Elasticsearch

def verify_chunks():
    """
    Retrieves and displays a summary of indexed chunks from Elasticsearch.
    """
    # Initialize the Elasticsearch client
    es = Elasticsearch(
        ["http://localhost:9200"],
        verify_certs=False,
        request_timeout=30
    )

    index_name = "research_papers"

    # Define the search query
    # We use _source to exclude 'embedding' and sort by parent_id/chunk_id
    query = {
        "query": {
            "match_all": {}
        },
        "_source": {
            "excludes": ["embedding"]
        },
        "sort": [
            {"metadata.parent_id": "asc"},
            {"metadata.chunk_id": "asc"}
        ],
        "size": 400
    }

    try:
        # Check if index exists
        if not es.indices.exists(index=index_name):
            print(f"Error: Index '{index_name}' does not exist.")
            return

        # Execute search
        response = es.search(index=index_name, body=query)
        hits = response['hits']['hits']

        if not hits:
            print(f"No documents found in index '{index_name}'.")
            return

        print(f"\n{'='*100}")
        print(f"{'INDEX VERIFICATION REPORT: ' + index_name:^100}")
        print(f"{'='*100}\n")

        for hit in hits:
            source = hit['_source']
            metadata = source.get('metadata', {})
            
            parent_id = metadata.get('parent_id', 'N/A')
            chunk_id = metadata.get('chunk_id', 'N/A')
            title = source.get('title', 'N/A')
            author = metadata.get('author', 'N/A')
            year = metadata.get('year', 'N/A')
            content_snippet = source.get('content', '')[:75].replace('\n', ' ') + "..."

            print(f"Parent ID : {parent_id}")
            print(f"Chunk ID  : {chunk_id}")
            print(f"Title     : {title}")
            print(f"Author     : {author}")
            print(f"Year     : {year}")
            print(f"Content   : {content_snippet}")
            print("-" * 50)

        print(f"\nTotal chunks displayed: {len(hits)}")

    except Exception as e:
        print(f"Error during verification: {e}")

if __name__ == "__main__":
    verify_chunks()
