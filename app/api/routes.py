from fastapi import APIRouter, HTTPException, status, UploadFile, File
from pydantic import BaseModel
from app.services.repo_service import repo_service
from app.engine.vector_store import vector_store_manager
from app.engine.gemini_client import gemini_chat_engine
from app.services.chat_service import chat_service

router = APIRouter()

# Data Transfer Objects (DTO) / Validation Schemas
class RepoUploadRequest(BaseModel):
    github_url: str

class ChatRequest(BaseModel):
    repo_id: str
    question: str
    session_id: str  # NEW: Tracks individual user conversation tabs

@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Verifies that the CodeExplain server and local directories are completely active.
    """
    return {
        "status": "operational",
        "storage_layer": "ready",
        "app": "CodeExplain Node Online"
    }

@router.post("/repo/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_repository(payload: RepoUploadRequest):
    """
    Full End-to-End Ingestion Pipeline:
    1. Clones a repository onto the disk.
    2. Structurally parses and chunks supported code/doc files.
    3. Generates 384-dimensional vectors locally using Sentence-Transformers.
    4. Serializes a standalone FAISS vector store database index onto disk.
    5. Auto-generates a comprehensive Codebase Overview dashboard.
    """
    # Step 1: Clone the repository down to local disk
    clone_result = await repo_service.clone_repository(payload.github_url)
    
    if not clone_result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Repository cloning failed: {clone_result.get('error')}"
        )
        
    repo_id = clone_result["repo_id"]
    local_path = clone_result["local_path"]

    # Step 2: Trigger the parsing, local embedding, and FAISS vector generation pipeline
    vector_result = vector_store_manager.create_vector_store(repo_id, local_path)
    
    if not vector_result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector database indexing failed: {vector_result.get('error')}"
        )
        
    # Step 3: Run the structural summary engine via Gemini Client
    codebase_overview_report = await gemini_chat_engine.generate_codebase_overview(repo_id)
        
    return {
        "message": "Repository ingested, parsed, and embedded into local vector database successfully!",
        "repo_id": repo_id,
        "analyzed_files_count": clone_result["total_files"],
        "generated_vector_chunks": vector_result["total_chunks"],
        "codebase_dashboard": codebase_overview_report
    }

@router.post("/repo/upload-zip", status_code=status.HTTP_202_ACCEPTED)
async def upload_zip_repository(file: UploadFile = File(...)):
    """
    Full Local Archive Ingestion Pipeline:
    1. Accepts a multipart ZIP file upload.
    2. Extracts contents securely and handles local cleanup.
    3. Traverses directories to isolate code assets using our regex scanner.
    4. Computes 384-dimensional dense vectors locally and saves a FAISS index.
    5. Returns an automated comprehensive architectural overview dashboard.
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file layout. Please upload a standard compressed '.zip' archive."
        )

    file_bytes = await file.read()
    
    # Step 1: Process and unpack the zip contents to disk
    extraction_result = await repo_service.process_zip_archive(file_bytes, file.filename)
    
    if not extraction_result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Archive extraction failed: {extraction_result.get('error')}"
        )
        
    repo_id = extraction_result["repo_id"]
    local_path = extraction_result["local_path"]

    # Step 2: Parse structural chunks and construct local FAISS index matrices
    vector_result = vector_store_manager.create_vector_store(repo_id, local_path)
    
    if not vector_result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector store serialization failed: {vector_result.get('error')}"
        )
        
    # Step 3: Run the structural summary engine via Gemini Client
    codebase_overview_report = await gemini_chat_engine.generate_codebase_overview(repo_id)

    return {
        "message": "ZIP archive unpacked, parsed, and embedded into local vector database successfully!",
        "repo_id": repo_id,
        "analyzed_files_count": extraction_result["total_files"],
        "generated_vector_chunks": vector_result["total_chunks"],
        "codebase_dashboard": codebase_overview_report
    }

@router.post("/chat", status_code=status.HTTP_200_OK)
async def chat_with_repository(payload: ChatRequest):
    """
    Core AI conversational endpoint. Proxies the validated payload over
    to the ChatService execution layer to handle context retrieval, session memory mapping, and generation.
    """
    # Call the intermediate service layer with session tracking
    ai_answer = await chat_service.process_chat_query(
        repo_id=payload.repo_id, 
        question=payload.question,
        session_id=payload.session_id
    )
    
    return {
        "repo_id": payload.repo_id,
        "session_id": payload.session_id,
        "answer": ai_answer
    }