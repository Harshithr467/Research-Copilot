from elasticsearch import Elasticsearch

def check_cluster_health():
    # Initialize the Elasticsearch client
    # Security is disabled as per requirements
    # Now using elasticsearch-py v8.x to match server v8.12.0
    es = Elasticsearch(
        ["http://localhost:9200"],
        verify_certs=False,
        request_timeout=30
    )

    try:
        # Check cluster health
        health = es.cluster.health()
        
        cluster_name = health.get('cluster_name')
        status = health.get('status')
        nodes = health.get('number_of_nodes')

        print(f"Connected to cluster: {cluster_name}")
        print(f"Cluster Status: {status}")
        print(f"Number of Nodes: {nodes}")
        
        if status == 'green':
            print("Cluster is healthy.")
        elif status == 'yellow':
            print("Cluster is yellow (all primary shards are allocated, but some replicas are not).")
        else:
            print("Cluster is red (some primary shards are not allocated).")

    except Exception as e:
        print(f"Error connecting to Elasticsearch: {e}")

if __name__ == "__main__":
    check_cluster_health()
