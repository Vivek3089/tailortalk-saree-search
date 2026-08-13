from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from src.config import Config

class QdrantService:
    def __init__(self):
        Config.validate()
        print(f"Connecting to Qdrant Cloud at {Config.QDRANT_URL}...")
        self.client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
            timeout=120.0
        )
        self.collection_name = Config.COLLECTION_NAME
        self.vector_size = Config.VECTOR_SIZE
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Creates the Qdrant collection if it does not already exist."""
        collections = [col.name for col in self.client.get_collections().collections]
        if self.collection_name not in collections:
            print(f"Creating collection '{self.collection_name}' with vector size {self.vector_size}...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Collection '{self.collection_name}' created successfully!")
        else:
            print(f"✅ Collection '{self.collection_name}' already exists.")

    def search_similar(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        """Queries Qdrant for top-K visually similar items across client versions."""
        if hasattr(self.client, "query_points"):
            # Modern Qdrant Client (v1.10.0+)
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k
            )
            search_results = response.points
        elif hasattr(self.client, "search"):
            # Legacy Qdrant Client
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
        else:
            raise AttributeError("Installed QdrantClient does not support search or query_points methods.")

        results = []
        for hit in search_results:
            results.append({
                "id": hit.id,
                "score": round(hit.score, 4),
                "payload": hit.payload
            })
        return results