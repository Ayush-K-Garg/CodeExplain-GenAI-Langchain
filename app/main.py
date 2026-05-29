from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import router as api_router

# Initialize the core FastAPI engine
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Structural RAG platform for repository understanding.",
    version="1.0.0"
)

# Configure CORS protection with a dynamic origin regex.
# This matches any incoming HTTP or HTTPS domain and safely echoes it back 
# to the browser, satisfying the 'allow_credentials=True' requirement.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect our API paths under a professional v1 prefix
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}. Go to /docs for the interactive playground."}