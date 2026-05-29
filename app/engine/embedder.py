import os
import requests
from langchain_core.embeddings import Embeddings
from app.core.config import settings

class GeminiEmbeddings(Embeddings):
    """
    An ultra-lightweight, high-performance embedder utilizing Google's STABLE v1 REST API.
    Bypasses the google-genai SDK's regional v1beta routing bugs to prevent 404 errors,
    while running comfortably under Render's 512MB RAM budget.
    """
    def __init__(self):
        # Retrieve your verified API key from Pydantic config state
        self.api_key = getattr(settings, "GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be set in your cloud environment variables.")
        
        # Primary stable 768-dimensional model and standard backup
        self.primary_model = "models/text-embedding-004"
        self.fallback_model = "models/embedding-001"

    def _embed_with_rest_failover(self, contents: list[str]) -> list[list[float]]:
        """
        Performs batch embedding requests against Google's stable v1 endpoint,
        with automated fallback options.
        """
        # Determine the target endpoint using the primary stable model
        url = f"https://generativelanguage.googleapis.com/v1/{self.primary_model}:batchEmbedContents?key={self.api_key}"
        
        payload = {
            "requests": [
                {
                    "model": self.primary_model,
                    "content": {"parts": [{"text": t}]}
                } for t in contents
            ]
        }

        try:
            # 1. Attempt primary REST request (v1/text-embedding-004)
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return [emb["values"] for emb in data["embeddings"]]
            
        except Exception as e:
            print(f"⚠️ Stable text-embedding-004 REST request failed: {str(e)}. Attempting v1 backup fallback model...")
            
            # 2. Regional Failover: Swap to stable v1/embedding-001
            fallback_url = f"https://generativelanguage.googleapis.com/v1/{self.fallback_model}:batchEmbedContents?key={self.api_key}"
            fallback_payload = {
                "requests": [
                    {
                        "model": self.fallback_model,
                        "content": {"parts": [{"text": t}]}
                    } for t in contents
                ]
            }
            
            try:
                response = requests.post(fallback_url, json=fallback_payload, timeout=30)
                response.raise_for_status()
                data = response.json()
                return [emb["values"] for emb in data["embeddings"]]
            except Exception as fallback_err:
                raise RuntimeError(
                    f"CRITICAL EMBEDDING FAIL: Both stable primary and fallback routes failed. "
                    f"Primary Error: {str(e)} | Fallback Error: {str(fallback_err)}"
                ) from fallback_err

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds a list of document strings using the cloud REST API endpoint."""
        # Batch requests in groups of 100 to stay within standard HTTP payload sizes safely
        embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_vectors = self._embed_with_rest_failover(batch_texts)
            embeddings.extend(batch_vectors)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single developer prompt question string."""
        url = f"https://generativelanguage.googleapis.com/v1/{self.primary_model}:embedContent?key={self.api_key}"
        payload = {
            "model": self.primary_model,
            "content": {"parts": [{"text": text}]}
        }
        try:
            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data["embedding"]["values"]
        except Exception as e:
            # Fallback for individual queries
            print(f"⚠️ Primary query embedding failed. Falling back to {self.fallback_model}...")
            fallback_url = f"https://generativelanguage.googleapis.com/v1/{self.fallback_model}:embedContent?key={self.api_key}"
            fallback_payload = {
                "model": self.fallback_model,
                "content": {"parts": [{"text": text}]}
            }
            try:
                response = requests.post(fallback_url, json=fallback_payload, timeout=15)
                response.raise_for_status()
                data = response.json()
                return data["embedding"]["values"]
            except Exception as fallback_err:
                raise RuntimeError(f"Cloud query embedding generation failed: {str(fallback_err)}") from fallback_err

# Singleton instantiation for vector store index generation and retrieval
embedding_engine = GeminiEmbeddings()
