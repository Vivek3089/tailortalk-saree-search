import os
from dotenv import load_dotenv

# Load environment variables from the local .env file
load_dotenv()

class Config:
    # API Credentials
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    
    # Qdrant Database Collection Settings
    COLLECTION_NAME: str = "saree_catalog"
    VECTOR_SIZE: int = 512  # Fashion-CLIP embedding dimension size
    
    # Embedding Model Name
    MODEL_NAME: str = "patrickjohncyh/fashion-clip"

    @classmethod
    def validate(cls) -> None:
        """Ensures all required API keys exist before running the application."""
        missing = []
        if not cls.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        if not cls.QDRANT_URL:
            missing.append("QDRANT_URL")
        if not cls.QDRANT_API_KEY:
            missing.append("QDRANT_API_KEY")
            
        if missing:
            raise ValueError(f"Missing environment variables in .env file: {', '.join(missing)}")