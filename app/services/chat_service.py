from app.engine.gemini_client import gemini_chat_engine

class ChatService:
    async def process_chat_query(self, repo_id: str, question: str, session_id: str) -> str:
        """
        Acts as the business logic intermediary for the conversational pipeline.
        Passes the query along with the session tracking ID down to the Gemini engine layer.
        """
        # Forward the execution down to our updated engine layer with session tracking
        ai_response = await gemini_chat_engine.get_ai_explanation(
            repo_id=repo_id, 
            question=question,
            session_id=session_id
        )
        return ai_response

# Singleton instance to import inside our API routes
chat_service = ChatService()