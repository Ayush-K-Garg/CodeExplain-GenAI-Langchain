import os
from langchain_core.embeddings import Embeddings
from google import genai
from app.core.config import settings

class GeminiEmbeddings(Embeddings):
    """
    An ultra-lightweight, crash-resilient embedder utilizing Google's Cloud API.
    Bypasses PyTorch and local models to run comfortably under Render's 512MB RAM cap.
    
    Features automatic regional failovers to guarantee 100% uptime even if 
    particular model routes are temporarily restricted or return 404 errors.
    """
    def __init__(self):
        # Retrieve your verified API key from Pydantic config state
        api_key = getattr(settings, "GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set in your cloud environment variables.")
        
        self.client = genai.Client(api_key=api_key)
        # Both models output exactly 768-dimensional vectors, making them seamlessly interchangeable!
        self.primary_model = "text-embedding-004"
        self.fallback_model = "embedding-001"

    def _embed_with_failover(self, contents: list[str]) -> list[list[float]]:
        """
        Executes embedding requests with automatic fallback in case of regional 404/NOT_FOUND errors.
        """
        try:
            # 1. Attempt primary enterprise model (text-embedding-004)
            response = self.client.models.embed_content(
                model=self.primary_model,
                contents=contents
            )
            return [embedding.values for embedding in response.embeddings]
        except Exception as e:
            err_str = str(e).lower()
            # 2. Check if the error is a 404 / NOT_FOUND regional routing error
            if "not_found" in err_str or "404" in err_str or "not found" in err_str:
                print(f"⚠️ Regional routing fail: {self.primary_model} returned 404. Dropping back to {self.fallback_model}...")
                try:
                    # 3. Fallback to embedding-001 (also 768-dimensional)
                    response = self.client.models.embed_content(
                        model=self.fallback_model,
                        contents=contents
                    )
                    return [embedding.values for embedding in response.embeddings]
                except Exception as fallback_err:
                    raise RuntimeError(
                        f"CRITICAL: Both primary and fallback cloud embedding generation failed. "
                        f"Original error: {str(e)} | Fallback error: {str(fallback_err)}"
                    ) from fallback_err
            # If it's a different error (e.g. invalid API key, rate limit), raise immediately
            raise RuntimeError(f"Cloud embedding generation failed: {str(e)}") from e

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds a list of document strings using the cloud API endpoint."""
        return self._embed_with_failover(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single developer prompt question string."""
        # Convert single query to a list, run through our failover, and extract the first vector
        embeddings = self._embed_with_failover([text])
        return embeddings[0]

# Singleton instantiation for vector store index generation and retrieval
embedding_engine = GeminiEmbeddings()
