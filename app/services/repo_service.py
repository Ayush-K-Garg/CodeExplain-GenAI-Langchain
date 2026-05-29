import os
import uuid
import shutil
import asyncio
import zipfile
import gc
from git import Repo
from app.core.config import settings

class RepoService:
    # Production ignore lists to avoid processing irrelevant files or heavy build artifacts
    BANNED_DIRECTORIES = {
        '.git', '.github', 'node_modules', 'venv', '__pycache__', 
        'dist', 'build', '.next', '.idea', '.vscode', '.dart_tool', 'ios', 'android'
    }
    
    # Complete list of supported language extensions for parsing
    SUPPORTED_EXTENSIONS = {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.cpp', '.h', '.hpp', '.c', 
        '.java', '.go', '.rs', '.kt', '.swift', '.html', '.css', '.md', '.dart'
    }

    @staticmethod
    def generate_repo_id() -> str:
        """Generates a clean, unique 8-character identifier for each repository."""
        return str(uuid.uuid4())[:8]

    async def clone_repository(self, github_url: str) -> dict:
        """
        Asynchronously clones a public git repository to local storage.
        Runs the blocking GitPython clone operation inside a thread pool with depth=1 (shallow clone).
        """
        repo_id = self.generate_repo_id()
        target_dir = os.path.join(settings.REPO_STORE_DIR, repo_id)
        
        try:
            loop = asyncio.get_running_loop()
            
            # Perform memory-optimized shallow clone in an async executor
            def _clone():
                print(f"📥 Shallow cloning {github_url} (depth=1) to prevent RAM spikes...")
                repo = Repo.clone_from(github_url, target_dir, depth=1)
                repo.close()
                del repo
                gc.collect()

            await loop.run_in_executor(None, _clone)
            
            # Scan the cloned directory to see how many valid code files we found
            valid_files = self.scan_repo_files(target_dir)
            
            return {
                "success": True,
                "repo_id": repo_id,
                "local_path": target_dir,
                "total_files": len(valid_files)
            }
            
        except Exception as e:
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            return {
                "success": False,
                "error": str(e)
            }

    def scan_repo_files(self, base_path: str) -> list[str]:
        """
        Walks through the repository directory tree and collects 
        the absolute paths of valid source files based on extensions.
        """
        valid_files = []
        
        for root, dirs, files in os.walk(base_path):
            # Modifying dirs in-place tells os.walk to skip banned directories entirely
            dirs[:] = [d for d in dirs if d not in self.BANNED_DIRECTORIES]
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    full_path = os.path.join(root, file)
                    valid_files.append(full_path)
                    
        return valid_files
    
    async def process_zip_archive(self, file_contents: bytes, original_filename: str) -> dict:
        """
        Saves an uploaded raw byte stream to disk, extracts its nested folders into 
        the standard data/repos location inside an async thread executor, and runs a file inventory scan.
        """
        repo_id = self.generate_repo_id()
        target_dir = os.path.join(settings.REPO_STORE_DIR, repo_id)
        os.makedirs(target_dir, exist_ok=True)
        
        temp_zip_path = f"{target_dir}_temp.zip"
        
        try:
            # 1. Write the raw bytes down to disk
            with open(temp_zip_path, "wb") as f:
                f.write(file_contents)

            # 2. Helper function to unpack the zip archive safely
            def extract_zip():
                print(f"Extracting uploaded archive: {original_filename}...")
                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)

            # 3. Offload the heavy file unzipping task to an async thread executor
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, extract_zip)

            # 4. Housekeeping: Delete the temporary .zip file immediately
            if os.path.exists(temp_zip_path):
                os.remove(temp_zip_path)

            # 5. Scan the newly extracted directory for code files
            total_files = len(self.scan_repo_files(target_dir))

            return {
                "success": True,
                "repo_id": repo_id,
                "local_path": target_dir,
                "total_files": total_files
            }

        except Exception as e:
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            if os.path.exists(temp_zip_path):
                os.remove(temp_zip_path)
            return {"success": False, "error": str(e)}

# Create a single instance to import across our routes
repo_service = RepoService()