import os
from langchain_core.documents import Document
# FIX: Migrated from deprecated 'langchain.text_splitter' to 'langchain_text_splitters'
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

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
            '.c': Language.C,
            '.java': Language.JAVA,
            '.go': Language.GO,
            '.rs': Language.RUST,
            '.kt': Language.KOTLIN,
            '.swift': Language.SWIFT,
            '.html': Language.HTML,
            '.css': Language.CSS,
            '.md': Language.MARKDOWN,
            # Use Dart explicitly, with a graceful fallback if needed
            '.dart': Language.DART if hasattr(Language, 'DART') else Language.JS  
        }

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

    def parse_file(self, file_path: str, root_dir: str) -> list[Document]:
        """
        Reads a single file, selects the best chunking strategy based on 
        extension, slices it safely, and returns a list of Document objects 
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
        relative_path = os.path.relpath(file_path, root_dir).replace("\\", "/")

        # 5. Pack everything into clean LangChain Document objects containing your core metadata
        documents = []
        for idx, chunk in enumerate(raw_chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "file_path": relative_path,
                    "language": self.get_language_string(file_path),
                    "chunk_index": idx,
                    "source": relative_path
                }
            )
            documents.append(doc)

        return documents

# Singleton instance to import inside our orchestrator services
code_parser = CodeParser()