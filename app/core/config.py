import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # These match your variables in the .env file
    PROJECT_NAME: str = "CodeExplain API"
    GEMINI_API_KEY: str
    ALLOWED_HOSTS: str = "*"
    
    # Setting up absolute file-system storage paths for your local databases
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    REPO_STORE_DIR: str = os.path.join(DATA_DIR, "repos")
    FAISS_STORE_DIR: str = os.path.join(DATA_DIR, "faiss_indices")

    # Tells Pydantic to cleanly read from your local .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Create a single settings object to use across the entire app
settings = Settings()

# Make sure our data storage folders exist locally right on startup
os.makedirs(settings.REPO_STORE_DIR, exist_ok=True)
os.makedirs(settings.FAISS_STORE_DIR, exist_ok=True)