"""
FastAPI web API layer for the TED Talks RAG System.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import os
from rag_system import TEDTalksRAG
from utils.consts import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, DEFAULT_RETRIEVE_TOP_K

# Global RAG instance (initialized on startup)
rag: Optional[TEDTalksRAG] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    global rag
    
    # Load API keys from environment variables
    pinecone_api_key = os.environ["PINECONE_API_KEY"]
    llmod_api_key = os.environ["LLMOD_API_KEY"]
    
    # Initialize RAG system
    rag = TEDTalksRAG(
        pinecone_api_key=pinecone_api_key,
        llmod_api_key=llmod_api_key
    )
    print("RAG system initialized successfully")
    
    # Check indexing status
    stats = rag.get_index_stats()
    vector_count = stats.get("total_vectors", 0)
    print(f"Indexing status: {vector_count} vectors in index")
    
    if vector_count == 0:
        print("⚠️  WARNING: Index is empty! No vectors found.")
        print("   Please run indexing before using /api/prompt:")
        print("   python rag_system.py --mode index --csv ted_talks_en.csv")
    
    yield
    
    # Shutdown
    rag = None
    print("RAG system shut down successfully")


app = FastAPI(title="TED Talks RAG API", lifespan=lifespan)


class PromptRequest(BaseModel):
    question: str


class ContextItem(BaseModel):
    talk_id: Optional[str]
    title: Optional[str]
    chunk: Optional[str]
    score: Optional[float]


class AugmentedPrompt(BaseModel):
    System: str
    User: str


class PromptResponse(BaseModel):
    response: str
    context: List[ContextItem]
    Augmented_prompt: AugmentedPrompt


class StatsResponse(BaseModel):
    chunk_size: int
    overlap_ratio: float
    top_k: int


@app.post("/api/prompt", response_model=PromptResponse)
async def post_prompt(request: PromptRequest):
    """
    Process a user question and return the response with context and augmented prompts.
    
    Args:
        request: PromptRequest containing the user's question
        
    Returns:
        PromptResponse with response, context, and augmented prompt
    """
    if rag is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Query the RAG system
        result = rag.query(request.question, top_k=DEFAULT_RETRIEVE_TOP_K)
        
        # Convert context to ContextItem objects
        context_items = [
            ContextItem(
                talk_id=ctx.get("talk_id"),
                title=ctx.get("title"),
                chunk=ctx.get("chunk_text"),
                score=ctx.get("score")
            )
            for ctx in result.get("context", [])
        ]
        
        # Build augmented prompt
        augmented_prompt = AugmentedPrompt(
            System=result.get("augmented_prompt", {}).get("System", ""),
            User=result.get("augmented_prompt", {}).get("User", "")
        )
        
        return PromptResponse(
            response=result.get("answer", ""),
            context=context_items,
            Augmented_prompt=augmented_prompt
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing prompt: {str(e)}")


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    overlap_ratio = (DEFAULT_OVERLAP / DEFAULT_CHUNK_SIZE) if DEFAULT_CHUNK_SIZE > 0 else 0.0
    return StatsResponse(
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap_ratio=round(overlap_ratio, 4),
        top_k=DEFAULT_RETRIEVE_TOP_K
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "TED Talks RAG API",
        "endpoints": {
            "POST /api/prompt": "Process a user prompt",
            "GET /api/stats": "Get RAG system statistics"
        }
    }

