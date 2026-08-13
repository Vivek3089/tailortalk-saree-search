import os
from dotenv import load_dotenv

# Load local .env file if available
load_dotenv()

def get_secret(key_name: str, default: str = "") -> str:
    """Retrieves secret from Streamlit secrets (for cloud) or environment variables (for local)."""
    # 1. Check Streamlit Secrets (Cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key_name in st.secrets:
            return str(st.secrets[key_name])
    except Exception:
        pass

    # 2. Check Environment Variables / .env file
    return os.getenv(key_name, default)

class Config:
    GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
    QDRANT_URL = get_secret("QDRANT_URL")
    QDRANT_API_KEY = get_secret("QDRANT_API_KEY")
    
    COLLECTION_NAME = get_secret("COLLECTION_NAME", "saree_catalog")
    VECTOR_SIZE = int(get_secret("VECTOR_SIZE", "512"))

    @classmethod
    def validate(cls):
        """Ensures all required configuration keys are present."""
        missing = []
        if not cls.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        if not cls.QDRANT_URL:
            missing.append("QDRANT_URL")
        if not cls.QDRANT_API_KEY:
            missing.append("QDRANT_API_KEY")

        if missing:
            raise ValueError(
                f"Missing environment variables: {', '.join(missing)}. "
                "Please set them in your local .env file or Streamlit Cloud Secrets."
            )