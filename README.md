# CodeExplain

Production-Grade AI-Powered Repository Tutor & Architectural Mapping Platform

CodeExplain is an AI-powered codebase comprehension platform that helps developers understand unfamiliar repositories through architectural analysis, semantic code search, and repository-aware question answering.

Instead of treating source files as isolated text documents, CodeExplain analyzes an entire repository as a connected system, generates architecture reports, builds vector indices for semantic retrieval, and provides grounded answers based on the actual codebase.

---

## Features

### Repository Analysis

* Import public GitHub repositories
* Upload and analyze ZIP archives
* Automatic source code discovery
* Multi-language repository support

### Architecture Mapping

* Generate project architecture reports
* Visualize repository structure
* Create ASCII-based system flow diagrams
* Identify service and dependency relationships

### AI-Powered Code Understanding

* Semantic code search using vector embeddings
* Repository-aware chat assistant
* Context-grounded code explanations
* Multi-turn conversational memory

### Retrieval-Augmented Generation (RAG)

* FAISS-based vector retrieval
* Gemini-powered answer generation
* Context-aware code reasoning
* Source-grounded responses

---

## System Architecture

```text
+-------------------------+
|     React Frontend      |
|   (Vite + Tailwind)     |
+------------+------------+
             |
             | HTTP REST
             |
             v
+------------+------------+
|        FastAPI API      |
|      Application Layer  |
+------------+------------+
             |
    +--------+--------+
    |                 |
    v                 v

+---------+     +-----------+
| Ingest  |     |   Chat    |
| Service |     |  Service  |
+----+----+     +-----+-----+
     |                |
     v                v

+----+----------------+----+
|      FAISS Vector Index  |
+----+----------------+----+
     |
     v

+-------------------------+
|      Gemini Models      |
| Embeddings + Generation |
+-------------------------+
```

---

## Processing Pipeline

### 1. Repository Ingestion

1. Clone repository using shallow checkout (`depth=1`)
2. Traverse repository files
3. Ignore generated directories:

   * `.git`
   * `node_modules`
   * `build`
   * `.dart_tool`
4. Extract supported source files
5. Parse code using language-aware splitters
6. Generate embeddings
7. Build FAISS vector index
8. Persist index to disk

---

### 2. Semantic Retrieval

1. User submits a question
2. Query is converted into embeddings
3. FAISS performs similarity search
4. Top relevant code chunks are retrieved
5. Chat history is loaded
6. Context is assembled
7. Gemini generates grounded responses
8. Answer is returned to the client

---

## Project Structure

```text
CodeExplain/
│
├── app/
│   ├── api/
│   │   └── routes.py
│   │
│   ├── core/
│   │   └── config.py
│   │
│   ├── engine/
│   │   ├── parsers/
│   │   ├── embeddings/
│   │   └── retrieval/
│   │
│   └── services/
│       ├── repo_service.py
│       └── chat_service.py
│
├── data/
│   ├── repositories/
│   └── faiss_indices/
│
├── codeexplain-web/
│
├── requirements.txt
├── .env
└── README.md
```

---

## Tech Stack

### Backend

* FastAPI
* Uvicorn
* Pydantic
* GitPython
* Python Dotenv

### AI & Retrieval

* Google Gemini 2.5 Flash
* Gemini Embeddings
* LangChain
* FAISS
* Retrieval-Augmented Generation (RAG)

### Frontend

* React
* Vite
* Tailwind CSS
* Axios

### DevOps

* Docker
* Render
* Railway
* Persistent Volume Storage

---

## Production Optimizations

### Memory-Efficient Embeddings

Instead of loading local embedding models that consume significant RAM, CodeExplain uses Google's hosted embedding APIs. This keeps memory consumption low and enables deployment on free cloud tiers.

### Concurrent Processing

Embedding generation runs through a controlled thread pool with connection reuse, reducing latency and avoiding socket exhaustion.

### Dynamic Model Discovery

At startup, the application automatically discovers accessible Gemini models for the configured API key and selects the best available embedding model.

### Fault-Tolerant Retries

All external AI requests use exponential backoff retry strategies to handle temporary network failures and rate limits.

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd CodeExplain
```

### Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key
GEMINI_MODEL_NAME=gemini-2.5-flash
```

### Run Backend

```bash
uvicorn app.main:app --reload
```

---

## Frontend Setup

```bash
cd codeexplain-web

npm install

npm run dev
```

---

## Example Workflow

1. Paste a GitHub repository URL
2. Start repository analysis
3. Generate embeddings and vector index
4. Open the architecture report
5. Ask repository-specific questions
6. Receive grounded AI-generated explanations

---

## Future Enhancements

* Repository dependency graphs
* Mermaid architecture diagrams
* Multi-repository indexing
* Code ownership analysis
* Pull request summarization
* Repository change impact analysis
* Team onboarding workflows

---


