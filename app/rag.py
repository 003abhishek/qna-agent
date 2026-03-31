"""RAG: Embeddings and Vector Similarity Search."""

import uuid
import numpy as np
from pathlib import Path
from openai import OpenAI

from app.config import get_settings
from app import database as db

settings = get_settings()

# OpenAI client for Ollama
client = OpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_api_base
)

# ============ Embeddings ============

def get_embedding(text: str) -> list[float]:
    """Generate embedding for text."""
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ============ Document Processing ============

def chunk_text(text: str, chunk_size: int = 2000) -> list[str]:
    """Split text into chunks."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            para_break = text.rfind('\n\n', start, end)
            if para_break > start:
                end = para_break
            else:
                sent_break = text.rfind('. ', start, end)
                if sent_break > start:
                    end = sent_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end
    
    return chunks

async def index_file(file_path: Path) -> int:
    """Index a single file: chunk and create embeddings."""
    content = file_path.read_text(encoding='utf-8')
    file_name = file_path.name
    
    chunks = chunk_text(content, settings.chunk_size)
    
    for idx, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)  # sync call
        emb_id = str(uuid.uuid4())
        
        await db.save_embedding(
            emb_id=emb_id,
            file_name=file_name,
            chunk_index=idx,
            content=chunk,
            embedding=embedding
        )
    
    return len(chunks)

async def index_knowledge_base() -> dict:
    """Index all files in knowledge base directory."""
    kb_path = settings.knowledge_path
    
    if not kb_path.exists():
        return {"indexed_files": 0, "total_chunks": 0, "status": "no_directory"}
    
    await db.clear_embeddings()
    
    txt_files = list(kb_path.glob("*.txt"))
    total_chunks = 0
    
    for file_path in txt_files:
        chunks = await index_file(file_path)
        total_chunks += chunks
    
    return {
        "indexed_files": len(txt_files),
        "total_chunks": total_chunks,
        "status": "completed"
    }

# ============ Similarity Search ============

async def search_similar(query: str, top_k: int = 2) -> list[dict]:
    """Find most similar documents to query."""
    query_embedding = get_embedding(query)  # sync call
    
    all_embeddings = await db.get_all_embeddings()
    if not all_embeddings:
        return []
    
    results = []
    for doc in all_embeddings:
        similarity = cosine_similarity(query_embedding, doc["embedding"])
        results.append({
            "file_name": doc["file_name"],
            "chunk_index": doc["chunk_index"],
            "content": doc["content"],
            "similarity": similarity
        })
    
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

async def get_relevant_context(query: str) -> str:
    """Get relevant context for a query."""
    similar_docs = await search_similar(query, top_k=settings.max_context_docs)
    
    if not similar_docs:
        return "No relevant information found in knowledge base."
    
    context_parts = []
    for doc in similar_docs:
        context_parts.append(f"[Source: {doc['file_name']}]\n{doc['content']}")
    
    return "\n\n---\n\n".join(context_parts)