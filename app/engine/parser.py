import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

class CodeParser:
    def __init__(self):
        # 1. Map typical code extensions to LangChain's structural Language enums
        self.LANGUAGE_MAP = {
            '.py': Language.PYTHON,
            '.js': Language.JS,
            '.jsx': Language.JS,
            '.ts': Language.TS,
            '.tsx': Language.TS,
            '.cpp': Language.CPP,
            '.h': Language.CPP,
            '.hpp': Language.CPP,
            '.java': Language.JAVA,
            '.html': Language.HTML,
            # Dart falls back gracefully to JS bracket rules to preserve structural blocks
            '.dart': Language.JS  
        }

    def get_language_string(self, file_path: str) -> str:
        """Helper to return a clean language label for our metadata."""
        ext = os.path.splitext(file_path)[1].lower()
        mapping = {
            '.py': 'python', '.js': 'javascript', '.jsx': 'javascript',
            '.ts': 'typescript', '.tsx': 'typescript', '.cpp': 'cpp',
            '.h': 'cpp', '.java': 'java', '.dart': 'dart',
            '.html': 'html', '.css': 'css', '.md': 'markdown'
        }
        return mapping.get(ext, 'text')

    def chunk_file(self, file_path: str, repo_base_path: str) -> list[dict]:
        """
        Reads a single file, selects the best chunking strategy based on 
        extension, slices it safely, and returns chunks loaded with metadata.
        """
        ext = os.path.splitext(file_path)[1].lower()

        # Securely read the file contents
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            return [] # Skip files that are locked or corrupted

        # 2. Select the optimal Splitting Strategy
        if ext == '.md':
            # Markdown specific splitting strategy
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=800,
                chunk_overlap=50
            )
        elif ext in self.LANGUAGE_MAP:
            # Code structure syntax splitting strategy
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=self.LANGUAGE_MAP[ext],
                chunk_size=1000,
                chunk_overlap=100
            )
        else:
            # Standard line-by-line breakdown for styling files (.css) or plain text
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=50
            )

        # 3. Chop the text content into chunks
        raw_chunks = splitter.split_text(content)

        # 4. Standardize the file path format (e.g., convert 'lib\main.dart' to 'lib/main.dart')
        relative_path = os.path.relpath(file_path, repo_base_path).replace("\\", "/")

        # 5. Pack everything into clean dictionary blocks containing your core metadata
        structured_chunks = []
        for chunk in raw_chunks:
            structured_chunks.append({
                "page_content": chunk,
                "metadata": {
                    "file_path": relative_path,
                    "language": self.get_language_string(file_path)
                }
            })

        return structured_chunks

# Singleton instance to import inside our orchestrator services
code_parser = CodeParser()