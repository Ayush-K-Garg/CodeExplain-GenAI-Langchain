import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.core.config import settings
from app.engine.parser import code_parser
from app.engine.embedder import embedding_engine
from app.services.repo_service import repo_service

class VectorStoreManager:
    def get_index_path(self, repo_id: str) -> str:
        """Returns the absolute file-system path where a repo's FAISS index is saved."""
        return os.path.join(settings.FAISS_STORE_DIR, repo_id)

    def create_vector_store(self, repo_id: str, local_repo_path: str) -> dict:
        """
        Orchestrates the entire RAG pipeline: scans files, parses them structurally,
        generates vector embeddings locally, and saves the index to disk.
        """
        index_path = self.get_index_path(repo_id)
        
        try:
            # 1. Gather all valid code/doc files using our repo_service scanner
            all_files = repo_service.scan_repo_files(local_repo_path)
            if not all_files:
                return {"success": False, "error": "No supported source code files discovered."}

            # 2. Iterate through files and break them down into structured chunks
            langchain_documents = []
            for file_path in all_files:
                structured_chunks = code_parser.chunk_file(file_path, local_repo_path)
                
                # Convert our custom chunks into official LangChain Document objects
                for chunk in structured_chunks:
                    doc = Document(
                        page_content=chunk["page_content"],
                        metadata=chunk["metadata"]
                    )
                    langchain_documents.append(doc)

            if not langchain_documents:
                return {"success": False, "error": "Repository parsing resulted in zero structural blocks."}

            # 3. Feed the documents into FAISS along with our HuggingFace Embedder.
            # This runs completely locally on your CPU, turning code into numbers.
            vector_store = FAISS.from_documents(langchain_documents, embedding_engine)

            # 4. Save the compiled vector indices as physical files in our data directory
            os.makedirs(index_path, exist_ok=True)
            vector_store.save_local(index_path)

            return {
                "success": True,
                "total_chunks": len(langchain_documents),
                "storage_path": index_path
            }

        except Exception as e:
            # Clean up the vector directory if the indexing job breaks midway
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
            return {"success": False, "error": str(e)}

    def load_vector_store(self, repo_id: str) -> FAISS:
        """
        Instantly hydrates a previously saved local FAISS index back into system memory
        so your chat loop can execute rapid similarity searches against it.
        """
        index_path = self.get_index_path(repo_id)
        if not os.path.exists(index_path):
            raise ValueError(f"No indexed database found for repository ID: {repo_id}")
            
        # Read the local files back into memory instantly with dangerous deserialization explicitly allowed
        return FAISS.load_local(index_path, embedding_engine, allow_dangerous_deserialization=True)

# Singleton instance to import across our routes
vector_store_manager = VectorStoreManager()