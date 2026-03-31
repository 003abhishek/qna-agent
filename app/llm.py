"""LLM Service: OpenAI chat completion with RAG."""

from openai import OpenAI
from app.config import get_settings
from app import rag

settings = get_settings()

# OpenAI client for Ollama
client = OpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_api_base
)

# System prompt
SYSTEM_PROMPT = """You are a helpful QnA assistant. Answer questions based on the provided context.

Rules:
1. Use ONLY the provided context to answer questions
2. If the context doesn't contain relevant information, say "I don't have information about that"
3. Be concise and accurate
4. Cite the source file when possible

Context:
{context}
"""

async def generate_response(user_message: str, conversation_history: list[dict]) -> str:
    """Generate response using RAG."""
    context = await rag.get_relevant_context(user_message)
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(context=context)
        }
    ]
    
    for msg in conversation_history[-10:]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    messages.append({
        "role": "user",
        "content": user_message
    })
    
    # Sync call
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.7,
        max_tokens=1024
    )
    
    return response.choices[0].message.content