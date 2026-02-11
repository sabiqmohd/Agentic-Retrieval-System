"""State schema for the Multi-Document RAG LangGraph workflow."""

from typing import TypedDict, List, Optional, Literal


class RetrievedChunk(TypedDict):
    """A single retrieved document chunk with metadata."""
    document: str  # Source document filename
    chunk_id: str
    content: str
    relevance_score: float


class Citation(TypedDict):
    """Citation for a piece of information in the answer."""
    document: str
    chunk_id: str
    quote: str  # Relevant snippet from the chunk


class RAGState(TypedDict):
    """
    State that flows through the LangGraph RAG workflow.
    
    Example JSON state after Query Analyzer:
    {
        "query": "Compare the revenue of Company A vs Company B in 2023",
        "query_type": "comparative",
        "entities": ["Company A", "Company B"],
        "multi_entity": true,
        "requires_calculation": false,
        "retrieved_chunks": [],
        "calculation_result": null,
        "answer": "",
        "citations": [],
        "confidence_score": 0.0,
        "verification_passed": false
    }
    
    Example JSON state after Synthesizer (calculation query):
    {
        "query": "What is the total revenue if Company A made $5M and Company B made $3M?",
        "query_type": "calculation",
        "entities": ["Company A", "Company B"],
        "multi_entity": false,
        "requires_calculation": true,
        "retrieved_chunks": [
            {"document": "financials.pdf", "chunk_id": "c1", "content": "Company A revenue: $5M", "relevance_score": 0.95},
            {"document": "financials.pdf", "chunk_id": "c2", "content": "Company B revenue: $3M", "relevance_score": 0.92}
        ],
        "calculation_result": {"expression": "5000000 + 3000000", "result": 8000000},
        "answer": "The total revenue is $8M (Company A: $5M + Company B: $3M = $8M).",
        "citations": [
            {"document": "financials.pdf", "chunk_id": "c1", "quote": "Company A revenue: $5M"},
            {"document": "financials.pdf", "chunk_id": "c2", "quote": "Company B revenue: $3M"}
        ],
        "confidence_score": 0.92,
        "verification_passed": true
    }
    """
    # Input
    query: str
    
    # Query Analysis
    query_type: Literal["factual", "comparative", "summarization", "calculation"]
    entities: List[str]
    multi_entity: bool
    requires_calculation: bool
    
    # Retrieval
    retrieved_chunks: List[RetrievedChunk]
    
    # Calculation (optional)
    calculation_result: Optional[dict]  # {"expression": str, "result": float}
    
    # Synthesis
    answer: str
    citations: List[Citation]
    confidence_score: float
    verification_passed: bool
