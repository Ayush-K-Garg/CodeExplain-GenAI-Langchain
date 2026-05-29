import os
import gc
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.embeddings import Embeddings
from app.core.config import settings

class GeminiEmbeddings(Embeddings):
    """
    An ultra-lightweight, memory-optimized embedder utilizing Google's STABLE v1 REST API.
    Bypasses SDK regional routing bugs by executing concurrent singular :embedContent calls.
    
    Features Dynamic Live Model Discovery on startup to inspect and automatically lock onto
    the best working embedding and fallback models authorized for your specific Google key.
    """
    def __init__(self):
        # Retrieve your verified API key from Pydantic config state
        self.api_key = getattr(settings, "GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be set in your cloud environment variables.")
        
        # Thread safety lock for model configuration updates
        self._lock = threading.Lock()
        
        # Default safety fallbacks (overwritten dynamically during discovery)
        self.active_model = "models/text-embedding-004"
        self.fallback_model = "models/embedding-001"
        self._model_locked = False
        
        # Thread-local storage to hold distinct requests.Session pools per thread
        self._thread_local = threading.local()

        # 🔍 Run Dynamic Live Model Discovery on startup to find what your key has access to!
        self._discover_authorized_models()

    def _discover_authorized_models(self):
        """
        Queries Google's API to list all models authorized for this key,
        filtering for those that support embedding.
        """
        url = f"https://generativelanguage.googleapis.com/v1/models?key={self.api_key}"
        print("🕵️‍♂️ [Model Discovery] Querying Google AI Studio to inspect authorized models for your key...")
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                
                # Filter for models that support embedding content
                embed_capable_models = []
                for m in models:
                    methods = m.get("supportedGenerationMethods", [])
                    if "embedContent" in methods or "batchEmbedContents" in methods:
                        embed_capable_models.append(m.get("name"))
                
                if embed_capable_models:
                    print(f"🟢 [Model Discovery] Success! Your API key has access to these embedding models: {embed_capable_models}")
                    
                    # 1. Lock onto preferred active model
                    active_assigned = False
                    for preferred in ["models/text-embedding-004", "models/embedding-001"]:
                        if preferred in embed_capable_models:
                            self.active_model = preferred
                            active_assigned = True
                            print(f"🎯 [Model Discovery] Locked active model: {self.active_model}")
                            break
                    
                    if not active_assigned:
                        self.active_model = embed_capable_models[0]
                        print(f"🎯 [Model Discovery] Locked active model: {self.active_model}")

                    # 2. Dynamically assign a valid fallback model from your authorized list!
                    remaining_models = [m for m in embed_capable_models if m != self.active_model]
                    if remaining_models:
                        self.fallback_model = remaining_models[0]
                    else:
                        self.fallback_model = self.active_model
                    print(f"🎯 [Model Discovery] Locked fallback model: {self.fallback_model}")
                    
                else:
                    print("⚠️ [Model Discovery] WARNING: Google returned models, but NONE of them support 'embedContent'.")
                    print(f"📋 Here is the list of all models your key is allowed to see:")
                    for m in models[:10]:
                        print(f"  - {m.get('name')} (Methods: {m.get('supportedGenerationMethods')})")
            else:
                print(f"❌ [Model Discovery] Failed to list models. HTTP {response.status_code}: {response.text}")
        except Exception as e:
            print(f"❌ [Model Discovery] Connection error during startup discovery: {str(e)}")

    @property
    def session(self) -> requests.Session:
        """
        Returns a persistent, thread-safe HTTP connection pool session.
        Bypasses redundant connection/SSL handshakes, cutting RAM overhead and latency.
        """
        if not hasattr(self._thread_local, "session"):
            self._thread_local.session = requests.Session()
        return self._thread_local.session

    def _embed_single(self, text: str) -> list[float]:
        """
        Executes a singular REST API request to embed text.
        """
        with self._lock:
            model = self.active_model

        url = f"https://generativelanguage.googleapis.com/v1/{model}:embedContent?key={self.api_key}"
        payload = {
            "model": model,
            "content": {"parts": [{"text": text}]}
        }

        try:
            # Utilize the persistent thread-safe session connection pool
            response = self.session.post(url, json=payload, timeout=15)
            
            # Print detailed diagnostics if the REST API rejects the request
            if response.status_code != 200:
                print(f"❌ [API Error] HTTP {response.status_code} received from Gemini REST API during embedding.")
                print(f"🔍 Target Endpoint: {url.split('?')[0]} (API Key omitted for security)")
                print(f"🔍 Server Response Body: {response.text}")
            
            # Check for regional 404/Not Found exceptions
            if response.status_code == 404:
                with self._lock:
                    # ESCAPE CONDITION: If the fallback model also returns a 404, raise immediately.
                    if model == self.fallback_model:
                        print(f"❌ Both primary and fallback embedding models returned 404.")
                        response.raise_for_status()

                    if not self._model_locked and model == self.active_model:
                        print(f"⚠️ {model} returned 404. Dynamically locking fallback {self.fallback_model}...")
                        self.active_model = self.fallback_model
                        self._model_locked = True
                
                # Safe recursive retry using the fallback model
                return self._embed_single(text)

            response.raise_for_status()
            data = response.json()
            return data["embedding"]["values"]

        except Exception as e:
            with self._lock:
                if model == self.fallback_model:
                    raise RuntimeError(f"REST embedding generation failed for fallback {model}: {str(e)}") from e
                
                if not self._model_locked and model == self.active_model:
                    print(f"⚠️ Embedding with {model} failed: {str(e)}. Swapping to fallback {self.fallback_model}...")
                    self.active_model = self.fallback_model
                    self._model_locked = True
            
            return self._embed_single(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds a list of document strings concurrently using a ThreadPoolExecutor.
        Processed in small, garbage-collected batches to stay within 512MB RAM budgets.
        """
        embeddings = [None] * len(texts)
        batch_size = 15  # Process only 15 items at a time to prevent RAM spikes
        
        # Using 2 workers keeps thread-stack allocation low (~16MB total overhead)
        with ThreadPoolExecutor(max_workers=2) as executor:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Submit ONLY the current batch of 15 to the thread pool
                future_to_index = {
                    executor.submit(self._embed_single, text): idx 
                    for idx, text in enumerate(batch_texts, start=i)
                }
                
                # Wait for the current batch to finish before submitting the next
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        embeddings[idx] = future.result()
                    except Exception as e:
                        raise RuntimeError(f"Parallel batch embedding failed at index {idx}: {str(e)}") from e
                
                # Explicitly clear temporary batch variables and force garbage collection
                future_to_index.clear()
                gc.collect()
                    
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single query string using the currently active model."""
        return self._embed_single(text)

# Singleton instantiation for vector store index generation and retrieval
embedding_engine = GeminiEmbeddings()