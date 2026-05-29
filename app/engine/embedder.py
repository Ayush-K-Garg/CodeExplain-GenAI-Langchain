import os
from langchain_core.embeddings import Embeddings
from google import genai
from app.core.config import settings

class GeminiEmbeddings(Embeddings):
    """
    An ultra-lightweight, high-performance embedder that leverages Google's Cloud API.
    Bypasses PyTorch, sentence-transformers, and local HuggingFace weights,
    fully resolving memory (RAM) footprint constraints on cloud deployments.
    """
    def __init__(self):
        # Retrieve your verified API key from Pydantic config state
        api_key = getattr(settings, "GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set in your cloud environment variables.")
        
        self.client = genai.Client(api_key=api_key)
        # Using Google's state-of-the-art enterprise text embedding model
        self.model_name = "text-embedding-004"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds a list of document strings using the cloud API endpoint."""
        try:
            # Batch embedding requests cleanly
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=texts
            )
            return [embedding.values for embedding in response.embeddings]
        except Exception as e:
            raise RuntimeError(f"Cloud embedding generation failed: {str(e)}")

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single developer prompt question string."""
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            return response.embeddings[0].values
        except Exception as e:
            raise RuntimeError(f"Cloud query embedding generation failed: {str(e)}")

# Singleton instantiation for vector store index generation and retrieval
embedding_engine = GeminiEmbeddings()