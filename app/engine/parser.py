import os
from langchain_core.documents import Document
# Import directly from the modern, lightweight splitters library
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

class CodeParser:
    def __init__(self):
        # We build this mapping dynamically on startup to prevent ANY AttributeError
        # due to library updates or missing compiler configurations on cloud hosts!
        self.LANGUAGE_MAP = {}
        
        # Structural mapping configurations
        target_mappings = {
            '.py': 'PYTHON',
            '.js': 'JS',
            '.jsx': 'JS',
            '.ts': 'TS',
            '.tsx': 'TS',
            '.cpp': 'CPP',
            '.h': 'CPP',
            '.hpp': 'CPP',
            '.c': 'CPP',      # Fallback: parse C files using standard C++ syntax rules
            '.java': 'JAVA',
            '.go': 'GO',
            '.rs': 'RUST',
            '.kt': 'KOTLIN',
            '.swift': 'SWIFT',
            '.html': 'HTML',
            '.md': 'MARKDOWN',
            '.dart': 'DART'   # Will fall back dynamically below if DART enum doesn't exist
        }

        # Build our active map safely
        for ext, enum_name in target_mappings.items():
            lang_enum = getattr(Language, enum_name, None)
            if lang_enum is not None:
                self.LANGUAGE_MAP[ext] = lang_enum
            else:
                # Smart fallbacks for less common language configurations
                if ext == '.dart':
                    self.LANGUAGE_MAP[ext] = getattr(Language, 'JS', None) or getattr(Language, 'TS', None)
                elif ext == '.c':
                    self.LANGUAGE_MAP[ext] = getattr(Language, 'CPP', None)

    def get_language_string(self, file_path: str) -> str:
        """Helper to return a clean language label for our metadata."""
        ext = os.path.splitext(file_path)[1].lower()
        mapping = {
            '.py': 'python', '.js': 'javascript', '.jsx': 'javascript',
            '.ts': 'typescript', '.tsx': 'typescript', '.cpp': 'cpp',
            '.h': 'cpp', '.hpp': 'cpp', '.c': 'c', '.java': 'java', 
            '.dart': 'dart', '.html': 'html', '.css': 'css', 
            '.md': 'markdown', '.go': 'go', '.rs': 'rust',
            '.kt': 'kotlin', '.swift': 'swift'
        }
        return mapping.get(ext, 'text')

    def chunk_file(self, file_path: str, repo_base_path: str) -> list[dict]:
        """
        Reads a single file, selects the best chunking strategy based on 
        extension, slices it safely, and returns a list of dictionary blocks 
        loaded with metadata.
        """
        ext = os.path.splitext(file_path)[1].lower()

        # Securely read the file contents
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"⚠️ Error reading file {file_path}: {str(e)}")
            return [] # Skip files that are locked or corrupted

        if not content.strip():
            return []

        # 2. Select the optimal Splitting Strategy
        md_lang = self.LANGUAGE_MAP.get('.md', getattr(Language, 'MARKDOWN', None))
        
        if ext == '.md' and md_lang is not None:
            # Markdown specific splitting strategy
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=md_lang,
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

    def parse_file(self, file_path: str, root_dir: str) -> list[Document]:
        """
        Alternative interface wrapper: chunks a file and returns LangChain Document objects.
        This provides perfect compatibility with the repository services.
        """
        raw_chunks = self.chunk_file(file_path, root_dir)
        documents = []
        for idx, chunk in enumerate(raw_chunks):
            doc = Document(
                page_content=chunk["page_content"],
                metadata={
                    "file_path": chunk["metadata"]["file_path"],
                    "language": chunk["metadata"]["language"],
                    "chunk_index": idx,
                    "source": chunk["metadata"]["file_path"]
                }
            )
            documents.append(doc)

        return documents

# Singleton instance to import inside our orchestrator services
code_parser = CodeParser()