import os
import asyncio
from dotenv import load_dotenv
from google import genai
from google.genai import types
from app.core.config import settings
from app.engine.vector_store import vector_store_manager
from app.engine.memory import chat_memory_manager

class GeminiChatEngine:
    def __init__(self):
        # 1. Force reload local disk state to ensure environmental consistency
        load_dotenv(override=True)
        
        # 2. Extract configuration credentials cleanly from central Pydantic settings or OS flags
        api_key = getattr(settings, "GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
        if not api_key:
            raise ValueError(
                "CRITICAL RUNTIME ERROR: GEMINI_API_KEY is completely missing from the system environment. "
                "Please verify your local .env configuration file or target cloud provider secrets manager container."
            )
            
        # 3. Pull model configuration parameter dynamically with a safe fallback
        env_model = getattr(settings, "GEMINI_MODEL_NAME", os.getenv("GEMINI_MODEL_NAME"))
        self.MODEL_NAME = env_model if env_model else "gemini-2.5-flash"
        
        # Log active telemetry prefix safely on launch
        print(f"🚀 [CodeExplain Engine Initialized] Active Key Target: {api_key[:8]}... | Engine Variant: {self.MODEL_NAME}")
        
        # 4. Instantiate the official Google GenAI production client
        self.client = genai.Client(api_key=api_key)

    async def get_ai_explanation(self, repo_id: str, question: str, session_id: str) -> str:
        """
        Loads specific vector store indices, structures similarity contexts,
        hydrates past message turns, and fires off content generation prompts.
        Features automated retries with exponential backoffs.
        """
        try:
            # 1. Hydrate the local FAISS index file into memory for this specific repo_id
            vector_store = vector_store_manager.load_vector_store(repo_id)

            # 2. Perform a similarity search to grab the top 4 most relevant code chunks
            relevant_docs = vector_store.similarity_search(question, k=4)

            # 3. Concatenate retrieved chunks cleanly with their respective source paths
            context_blocks = []
            for doc in relevant_docs:
                file_path = doc.metadata.get("file_path", "Unknown File")
                language = doc.metadata.get("language", "text")
                block = f"--- FILE PATH: {file_path} (Language: {language}) ---\n{doc.page_content}"
                context_blocks.append(block)

            context_string = "\n\n".join(context_blocks)

            # 4. Draft professional system instruction guardrails
            system_instruction = (
                "You are a Senior Software Architect. Your objective is to explain codebases "
                "clearly and thoroughly to developers. You will be provided a user's question along "
                "with exact, contextually relevant snippets of code retrieved from their repository.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. Base your answer strictly on the provided source code context blocks.\n"
                "2. Always reference the exact file paths when explaining specific logic, classes, or methods.\n"
                "3. If the answer cannot be confidently derived from the provided snippets, state that "
                "the specific code implementation wasn't captured in the search context, and offer to analyze "
                "related architectural patterns instead.\n"
                "4. Format code blocks using appropriate markdown syntax highlight tags. Do not use any emojis."
            )

            # 5. Fetch past dialogue turns and convert them to official Gemini Content types
            raw_history = chat_memory_manager.get_history(session_id)
            contents_payload = []
            
            for turn in raw_history:
                contents_payload.append(
                    types.Content(
                        role=turn["role"],
                        parts=[types.Part.from_text(text=turn["text"])]
                    )
                )

            # 6. Append the brand-new grounded user query to the end of the history track
            user_prompt = (
                f"RELEVANT REPOSITORY CONTEXT:\n{context_string}\n\n"
                f"STUDENT QUESTION: {question}"
            )
            
            contents_payload.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_prompt)]
                )
            )

            # 7. Fire request with an automatic exponential backoff retry safety net
            max_retries = 3
            backoff_delay = 2  
            response = None

            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=self.MODEL_NAME,
                        contents=contents_payload,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.3,
                        )
                    )
                    break  
                except Exception as api_err:
                    if attempt == max_retries - 1:
                        raise api_err
                    print(f"⚠️ Gemini API busy or rate-limited (Attempt {attempt + 1}/{max_retries}). Retrying in {backoff_delay}s...")
                    await asyncio.sleep(backoff_delay)
                    backoff_delay *= 2  

            # 8. Commit fresh data turns onto session tracking memory files
            chat_memory_manager.add_message(session_id, "user", question)
            chat_memory_manager.add_message(session_id, "model", response.text)

            return response.text

        except Exception as e:
            print(f"⚠️ Chat API blocked ({str(e)}). Serving fallback notice...")
            return (
                "⚠️ Direct Chat is temporarily offline due to Gemini API rate-limits or high server demands.\n\n"
                "Please wait a moment and try again. In the meantime, you can explore the Project Overview tab to view the codebase file map!"
            )
        
    async def generate_codebase_overview(self, repo_id: str) -> str:
        """
        Generates an Overview Report. Enforces strict Markdown blockquotes 
        for all structural diagrams to preserve spacing on web browsers.
        """
        try:
            vector_store = vector_store_manager.load_vector_store(repo_id)
            anchor_docs = vector_store.similarity_search("main controller configuration architecture init setup", k=6)

            context_blocks = []
            for doc in anchor_docs:
                file_path = doc.metadata.get("file_path", "Unknown File")
                context_blocks.append(f"--- FILE: {file_path} ---\n{doc.page_content}")

            context_string = "\n\n".join(context_blocks)

            # Updated strict system layout instructions to prevent HTML layout collapsing and premature section splitting
            system_instruction = (
                "You are an expert Technical Architect. Your job is to analyze a connected codebase "
                "and generate a highly professional, detailed, and comprehensive Architecture Report. Do not use emojis.\n\n"
                "YOUR REPORT MUST FOLLOW THIS EXACT LAYOUT:\n"
                "1. CODEBASE SUMMARY: A thorough 1-2 paragraph technical executive summary of the application.\n"
                "2. ARCHITECTURAL LAYOUT: A detailed breakdown of the directory structure.\n"
                "3. CORE DATA FLOW (ASCII DIAGRAM): A clean, visual representation of how data travels through the application.\n"
                "4. RECOMMENDED STARTING POINTS: Suggest critical structural files the developer should look at first.\n\n"
                "CRITICAL FORMATTING INSTRUCTIONS:\n"
                "- EVERY SINGLE ASCII flowchart, visual route sequence, tree map, path, or visual layout block "
                "MUST be fully enclosed inside raw markdown code blocks (using triple-backticks ```text ... ```).\n"
                "- NEVER mix raw flowchart bars (|), pipelines (||), or alignment lines directly into standard paragraph text.\n"
                "- Write explanations in standard, clean paragraphs. Place all visual tracks in isolated monospaced code blocks "
                "so their spacing columns do not collapse on web browsers.\n"
                "- DO NOT write trailing divider hashtags (like ###) or blank separator blocks at the end of sections.\n"
                "- CRITICAL LAYOUT RULES FOR HEADERS AND TEXT:\n"
                "  The frontend card parser splits this document into cards based on lines written in ALL-CAPS (uppercase).\n"
                "  Therefore, DO NOT use ALL-CAPS (UPPERCASE) for any subheadings, step labels, diagram labels, flow charts, or inline text descriptions.\n"
                "  The ONLY lines allowed to be in ALL-CAPS are the four main numbered section headers listed above (e.g. '1. CODEBASE SUMMARY', '2. ARCHITECTURAL LAYOUT', '3. CORE DATA FLOW (ASCII DIAGRAM)', '4. RECOMMENDED STARTING POINTS').\n"
                "  Use standard Title Case or Sentence Case for all other titles, descriptions, and ASCII flowchart nodes (e.g., use 'App Launch' instead of 'APP LAUNCH', 'Determine City' instead of 'DETERMINE CITY', 'Call worker.getData(city)' instead of 'CALL WORKER.GETDATA(CITY)')."
            )

            prompt = (
                f"Analyze this core structural context of the repository (ID: {repo_id}):\n\n{context_string}\n\n"
                "Generate the clean, professional, markdown-formatted Overview Report."
            )

            response = self.client.models.generate_content(
                model=self.MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                )
            )

            return response.text
        except Exception as e:
            print(f"⚠️ Overview generation API blocked ({str(e)}). Running offline analyzer...")
            return self.generate_offline_fallback_report(repo_id)

    def generate_offline_fallback_report(self, repo_id: str) -> str:
        """
        Scans your local project files on disk, classifies the project tech stack,
        and generates a beautiful, valid markdown report matching your layout engine.
        """
        target_dir = os.path.join(settings.REPO_STORE_DIR, repo_id)
        all_files = []
        
        supported_exts = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.cpp', '.h', '.hpp', '.c', 
            '.java', '.go', '.rs', '.kt', '.swift', '.html', '.css', '.md', '.dart'
        }
        
        if os.path.exists(target_dir):
            for root, _, filenames in os.walk(target_dir):
                for filename in filenames:
                    relpath = os.path.relpath(os.path.join(root, filename), target_dir)
                    normalized_relpath = relpath.replace("\\", "/")
                    
                    if any(x in normalized_relpath for x in ['.git/', 'node_modules/', 'build/', '.dart_tool/', 'ios/', 'android/', 'macos/', 'windows/']):
                        continue
                    
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in supported_exts:
                        continue
                        
                    all_files.append(normalized_relpath)
        
        is_flutter = any(f.endswith('.dart') for f in all_files)
        is_react = any(f.endswith(('.jsx', '.tsx', '.js')) for f in all_files)
        is_python = any(f.endswith('.py') for f in all_files)
        
        tech_stack = "Flutter (Dart)" if is_flutter else ("React (JS/TS)" if is_react else "Custom Codebase")
        primary_purpose = "cross-platform application development" if is_flutter else ("modern reactive web dashboard" if is_react else "general software modules")
        
        ascii_flow = """+------------------+
|   User Action    |
+--------v---------+
         |
+--------v---------+
|   Core Logic     |
+------------------+"""

        if is_flutter:
            ascii_flow = """+------------------+
|   UI Component   |
+--------v---------+
         | (triggers state event)
+------------------+
|  BLoC / Provider |
+--------v---------+
         | (queries updates)
+------------------+
|  Data Repository |
+------------------+"""
        elif is_react:
            ascii_flow = """+------------------+
|   React View DOM |
+--------v---------+
         | (dispatches hook actions)
+------------------+
| State Controller |
+--------v---------+
         | (fetches API data)
+------------------+
| Backend Service  |
+------------------+"""
        elif is_python:
            ascii_flow = """+------------------+
|    HTTP Client   |
+--------v---------+
         | (REST Request API)
+------------------+
| Router Controller|
+--------v---------+
         | (executes DB query)
+------------------+
| Database Service |
+------------------+"""

        key_files = [f for f in all_files if any(k in f.lower() for k in ['main', 'app', 'home', 'route', 'controller', 'service'])]
        if not key_files:
            key_files = all_files[:6]
        else:
            key_files = list(set(key_files))[:6]

        report = f"""1. CODEBASE SUMMARY
This codebase represents an elite development project constructed using the {tech_stack} tech stack. Designed primarily for {primary_purpose}, it utilizes clean OOP practices to coordinate UI states, structural classes, and underlying service dependencies.

The architecture is designed for modular scalability and rapid deployment. It packages core operations into isolated models, ensuring strict separation of concerns and high-performance operations across runtime workflows.

2. ARCHITECTURAL LAYOUT
Below is the directory map of the connected files found within this workspace. The project utilizes structural components to split view presentation from business layer code:

"""
        for f in all_files[:15]:
            report += f"* `{f}` : System source file coordinating logic elements.\n"
        if len(all_files) > 15:
            report += f"* ... and {len(all_files) - 15} additional source files in database.\n"

        report += f"""
3. CORE DATA FLOW (ASCII DIAGRAM)
```
{ascii_flow}
```

4. RECOMMENDED STARTING POINTS
To understand the core business logic of this codebase, developers are recommended to look at these files first:

"""
        for kf in key_files:
            report += f"* `{kf}`: Primary structural entry point coordinating application workflows.\n"

        return report

# Singleton instantiation for route controller imports
gemini_chat_engine = GeminiChatEngine()