import time

class ChatMemoryManager:
    def __init__(self):
        # Dictionary structure: { session_id: [ {"role": "user/model", "text": "..."}, ... ] }
        self._registry = {}
        # Keep track of when sessions were last accessed so we can clear old ones if needed
        self._ttl_tracker = {}

    def get_history(self, session_id: str, limit: int = 6) -> list[dict]:
        """
        Retrieves the last 'limit' messages for a given session 
        formatted cleanly for the Google GenAI SDK.
        """
        if session_id not in self._registry:
            return []
        
        # Update last accessed time
        self._ttl_tracker[session_id] = time.time()
        
        # Return the last few rounds of conversation to prevent token bloating
        return self._registry[session_id][-limit:]

    def add_message(self, session_id: str, role: str, text: str):
        """
        Appends a new conversational turn into the session registry.
        role must be either 'user' or 'model' to match Gemini's API schema.
        """
        if session_id not in self._registry:
            self._registry[session_id] = []
        
        self._registry[session_id].append({
            "role": role,
            "text": text
        })
        self._ttl_tracker[session_id] = time.time()

# Singleton instance to import across services
chat_memory_manager = ChatMemoryManager()