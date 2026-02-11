"""
FastAPI application for Multi-Document RAG system.

Endpoints:
- POST /ingest: Upload and ingest documents
- POST /query: Query the RAG system via LangGraph workflow
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.ingestion.vector_store import ingest_to_qdrant
from src.agents.langgraph_workflow import run_rag_query

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Multi-Doc RAG API",
    description="REST API for Multi-Document RAG with LangGraph workflow",
    version="1.0.0"
)

# Pydantic Models

class IngestResponse(BaseModel):
    """Response from document ingestion."""
    document_ids: List[str]
    chunk_count: int
    status: str

class QueryRequest(BaseModel):
    """Request body for /query endpoint."""
    question: str = Field(..., min_length=1, description="The question to ask")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Optional query parameters")

class Source(BaseModel):
    """A source citation in the query response."""
    document: str
    chunk_id: str
    relevance_score: float
    content_snippet: str

class QueryResponse(BaseModel):
    """Response from the /query endpoint."""
    answer: str
    sources: List[Source]
    confidence: float
    
# Endpoints

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    """
    Ingest multiple documents (PDF, TXT, DOCX).
    Uploads documents, chunks them, and stores in vector database.
    """
    logger.info(f"Received request to ingest {len(files)} files")
    
    try:
        # Load documents
        documents = await load_documents(files)
        if not documents:
            raise HTTPException(status_code=400, detail="No documents could be loaded")
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Chunk documents
        chunks = chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Ingest to Qdrant
        doc_ids = ingest_to_qdrant(chunks)
        logger.info(f"Ingested {len(doc_ids)} chunks to vector store")
        
        return IngestResponse(
            document_ids=doc_ids,
            chunk_count=len(chunks),
            status="success"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system.
    
    Invokes the LangGraph workflow with Query Analyzer → Retrieval → Synthesizer.
    """
    logger.info(f"Received query: {request.question[:100]}...")
    
    try:
        # Run LangGraph workflow
        result = run_rag_query(request.question)
        
        # Transform retrieved_chunks to sources format
        sources = []
        for chunk in result.get("retrieved_chunks", []):
            sources.append(Source(
                document=chunk.get("document", "Unknown"),
                chunk_id=chunk.get("chunk_id", ""),
                relevance_score=chunk.get("relevance_score", 0.0),
                content_snippet=chunk.get("content", "")[:500]  # Limit snippet length
            ))
        
        response = QueryResponse(
            answer=result.get("answer", "Unable to generate answer"),
            sources=sources,
            confidence=result.get("confidence_score", 0.0)
        )
        
        logger.info(f"Query completed with confidence: {response.confidence}")
        return response

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_knowledge_base():
    """
    Clear all ingested documents from the vector store.
    Useful for starting fresh without restarting the server.
    """
    from src.utils.qdrant import get_qdrant_client
    
    try:
        client = get_qdrant_client()
        # Check if collection exists first
        collections = client.get_collections().collections
        if any(c.name == "multi_doc_rag" for c in collections):
            client.delete_collection("multi_doc_rag")
            logger.info("Collection 'multi_doc_rag' deleted.")
            return {"status": "success", "message": "Knowledge base cleared."}
        else:
             return {"status": "success", "message": "Knowledge base was already empty."}

    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
