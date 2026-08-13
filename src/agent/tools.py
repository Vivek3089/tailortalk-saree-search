from langchain.tools import tool
from pydantic import BaseModel, Field
from src.embeddings.fashion_embedder import FashionEmbedder
from src.vector_db.qdrant_service import QdrantService
from src.utils.image_helpers import load_image_from_input

# Singletons to reuse loaded model and database connection
embedder = FashionEmbedder()
db_service = QdrantService()

class VisualSearchInput(BaseModel):
    image_input: str = Field(
        description="URL, base64 string, or file path of the query image to find similar sarees for."
    )
    top_k: int = Field(
        default=5, 
        description="Number of visually similar sarees to return (default: 5)."
    )

@tool("search_similar_sarees", args_schema=VisualSearchInput)
def search_similar_sarees(image_input: str, top_k: int = 5) -> list[dict]:
    """Finds visually similar sarees from the catalogue based on weave, border, pallu, color, and pattern."""
    try:
        # Load and preprocess image
        image = load_image_from_input(image_input)
        
        # Generate vector embedding
        query_vector = embedder.embed_image(image)
        
        # Perform cosine similarity search in Qdrant
        results = db_service.search_similar(query_vector=query_vector, top_k=top_k)
        return results
    except Exception as e:
        return [{"error": f"Failed to perform visual search: {str(e)}"}]