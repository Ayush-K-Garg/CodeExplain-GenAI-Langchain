import os
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.embeddings import Embeddings
from app.core.config import settings

class GeminiEmbeddings(Embeddings):
    """
    An ultra-lightweight, high-performance embedder utilizing Google's STABLE v1 REST API.
    Bypasses SDK regional v1beta routing bugs and the non-stable :batchEmbedContents REST endpoints
    by executing concurrent singular :embedContent calls.
    
    Fully self-healing and thread-safe: dynamically falls back to stable alternative models
    if the primary text-embedding-004 model is restricted in your region.
    """
    def __init__(self):
        # Retrieve your verified API key from Pydantic config state
        self.api_key = getattr(settings, "GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be set in your cloud environment variables.")
        
        # Thread safety lock for model configuration updates
        self._lock = threading.Lock()
        
        # Models
        self.active_model = "models/text-embedding-004"
        self.fallback_model = "models/embedding-001"
        self._model_locked = False

    def _embed_single(self, text: str) -> list[float]:
        """
        Executes a singular REST API request to embed text.
        Thread-safe fallback switching to embedding-001 if text-embedding-004 is unavailable.
        """
        with self._lock:
            model = self.active_model

        url = f"https://generativelanguage.googleapis.com/v1/{model}:embedContent?key={self.api_key}"
        payload = {
            "model": model,
            "content": {"parts": [{"text": text}]}
        }

        try:
            response = requests.post(url, json=payload, timeout=15)
            
            # Check for regional 404/Not Found exceptions
            if response.status_code == 404:
                with self._lock:
                    if not self._model_locked and model == self.active_model:
                        print(f"⚠️ {model} returned 404. Dynamically locking fallback {self.fallback_model}...")
                        self.active_model = self.fallback_model
                        self._model_locked = True
                # Recursive retry using the newly assigned working model
                return self._embed_single(text)

            response.raise_for_status()
            data = response.json()
            return data["embedding"]["values"]

        except Exception as e:
            with self._lock:
                if not self._model_locked and model == self.active_model:
                    print(f"⚠️ Embedding with {model} failed: {str(e)}. Swapping to fallback {self.fallback_model}...")
                    self.active_model = self.fallback_model
                    self._model_locked = True
            
            # Verify if the active model was updated during execution failure
            with self._lock:
                current_model = self.active_model
            if current_model != model:
                return self._embed_single(text)
                
            raise RuntimeError(f"REST embedding generation failed for model {model}: {str(e)}") from e

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds a list of document strings concurrently using a ThreadPoolExecutor.
        Preserves original array order for FAISS alignment.
        """
        embeddings = [None] * len(texts)
        
        # Keep max_workers conservative (10) to respect basic rate limits gracefully
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Map future object to its original list position index
            future_to_index = {
                executor.submit(self._embed_single, text): i 
                for i, text in enumerate(texts)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    embeddings[index] = future.result()
                except Exception as e:
                    raise RuntimeError(f"Parallel batch embedding failed at index {index}: {str(e)}") from e
                    
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single query string using the currently active model."""
        return self._embed_single(text)

# Singleton instantiation for vector store index generation and retrieval
embedding_engine = GeminiEmbeddings()
