"""
LangGraph v1 Multi-Document RAG Workflow

This module implements a three-agent workflow for RAG:
1. Query Analyzer - Classifies query and extracts entities
2. Retrieval Agent - Performs vector search with multi-hop reasoning
3. Answer Synthesizer - Synthesizes answer with citations and verification
"""

import os
import json
import re
from typing import Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from src.agents.state import RAGState, RetrievedChunk, Citation
from src.agents.tools import safe_calculator
from src.retrieval.hybrid import HybridRetriever
from src.agents.safety import validate_input, detect_hallucination

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize retriever
retriever = HybridRetriever()



# PROMPTS (inline to avoid Docker path issues)

QUERY_ANALYZER_PROMPT = """Analyze the following user query and extract structured information.

Query: {query}

Respond with a JSON object containing:
1. "query_type": One of "factual", "comparative", "summarization", "calculation"
2. "entities": List of key entities/topics to search for
3. "requires_calculation": true if the answer needs arithmetic computation

Respond ONLY with the JSON object, no additional text."""

SYNTHESIS_PROMPT = """Based on the following retrieved context, answer the user's question.

Question: {query}
Query Type: {query_type}

Retrieved Context:
{context}

{calculation_instruction}

Instructions:
1. Synthesize a comprehensive answer using ONLY the provided context
2. For comparative queries, structure the answer to clearly compare the entities
3. Include specific quotes or data points from the context
4. If the context doesn't contain enough information, say so
5. Rate your confidence in the answer from 0.0 to 1.0 based on, How well the context covers the question (0.3 = poor coverage, 1.0 = complete coverage)
6. DO NOT add information not present in the context - this is critical for hallucination prevention

Respond with a JSON object:
{{
    "answer": "Your synthesized answer here",
    "citations": [
        {{"document": "source.pdf", "chunk_id": "c1", "quote": "relevant quote"}}
    ],
    "confidence_score": 0.85,
    "has_sufficient_context": true
}}

Respond ONLY with the JSON object."""

CALCULATION_INSTRUCTION = """
IMPORTANT: This query requires calculation. 
The calculator has computed: {expression} = {result}
Include this computed result in your answer.
"""


def query_analyzer(state: RAGState) -> RAGState:
    """
    Analyzes user query to determine intent, classify type, and extract entities.
    Sets multi_entity=True for comparative queries.
    """
    query = state["query"]
    
    # Call LLM for analysis
    response = llm.invoke(QUERY_ANALYZER_PROMPT.format(query=query))
    
    try:
        # Parse JSON response
        content = response.content.strip()
        # Handle markdown code blocks if present
        if content.startswith("```"):
            content = re.sub(r'^```(?:json)?\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
        
        analysis = json.loads(content)
        
        query_type = analysis.get("query_type", "factual")
        entities = analysis.get("entities", [])
        requires_calculation = analysis.get("requires_calculation", False)
        
    except (json.JSONDecodeError, KeyError) as e:
        # Fallback if LLM doesn't return valid JSON
        print(f"⚠️  Query analyzer JSON parse failed: {e}")
        print(f"Raw LLM response: {response.content[:200]}")
        
        # Simple fallback: treat as factual query
        query_type = "factual"
        entities = []
        requires_calculation = False
    
    # Set multi_entity flag for comparative queries
    multi_entity = query_type == "comparative"
    
    return {
        **state,
        "query_type": query_type,
        "entities": entities,
        "multi_entity": multi_entity,
        "requires_calculation": requires_calculation,
    }


# AGENT 2: Retrieval & Reasoning Agent

def retrieval_agent(state: RAGState) -> RAGState:
    """
    Performs vector search based on analyzed query.
    For comparative queries (multi_entity=True), retrieves info for each entity separately.
    For calculation queries, focuses on retrieving numeric content.
    """
    query = state["query"]
    entities = state["entities"]
    multi_entity = state["multi_entity"]
    requires_calculation = state["requires_calculation"]
    
    all_chunks: list[RetrievedChunk] = []
    
    if multi_entity and len(entities) >= 2:
        # Multi-hop retrieval: search for each entity separately
        for entity in entities[:3]:  # Limit to top 3 entities
            # Search for just the entity name for focused retrieval
            results = retriever.retrieve(entity, top_k=3)
            
            for r in results:
                chunk: RetrievedChunk = {
                    "document": r.get("document", "Unknown"),
                    "chunk_id": r.get("chunk_id", ""),
                    "content": r.get("content_snippet", ""),
                    "relevance_score": r.get("relevance_score", 0.0)
                }
                # Deduplicate by chunk_id
                if not any(c["chunk_id"] == chunk["chunk_id"] for c in all_chunks):
                    all_chunks.append(chunk)
    else:
        # Standard single-query retrieval
        results = retriever.retrieve(query, top_k=5)
        
        for r in results:
            chunk: RetrievedChunk = {
                "document": r.get("document", "Unknown"),
                "chunk_id": r.get("chunk_id", ""),
                "content": r.get("content_snippet", ""),
                "relevance_score": r.get("relevance_score", 0.0)
            }
            all_chunks.append(chunk)
    
    # For calculation queries, prioritize chunks with numbers
    if requires_calculation:
        # Sort by presence of numbers (chunks with numbers first)
        def has_numbers(chunk: RetrievedChunk) -> int:
            return -len(re.findall(r'\d+\.?\d*', chunk["content"]))
        all_chunks.sort(key=has_numbers)
    
    return {
        **state,
        "retrieved_chunks": all_chunks,
    }


# AGENT 3: Answer Synthesizer & Verifier

def synthesizer_agent(state: RAGState) -> RAGState:
    """
    Synthesizes final answer from retrieved context.
    Calls calculator if needed, verifies quality, and adds citations.
    """
    query = state["query"]
    query_type = state["query_type"]
    retrieved_chunks = state["retrieved_chunks"]
    requires_calculation = state["requires_calculation"]
    
    # Format context for LLM
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(
            f"[{i+1}] Source: {chunk['document']} (ID: {chunk['chunk_id']})\n"
            f"Content: {chunk['content']}\n"
            f"Relevance: {chunk['relevance_score']:.2f}"
        )
    context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
    
    # Handle calculation if needed
    calculation_result = None
    calculation_instruction = ""
    
    if requires_calculation:
        # Extract numbers from context and query
        all_text = query + " " + " ".join(c["content"] for c in retrieved_chunks)
        numbers = re.findall(r'\$?([\d,]+(?:\.\d+)?)\s*(?:million|M|billion|B)?', all_text)
        
        if len(numbers) >= 2:
            # Clean numbers
            clean_numbers = []
            for n in numbers[:2]:
                clean_n = n.replace(',', '')
                clean_numbers.append(clean_n)
            
            # Determine operation from query
            query_lower = query.lower()
            if any(word in query_lower for word in ["total", "sum", "add", "combined"]):
                expression = f"{clean_numbers[0]} + {clean_numbers[1]}"
            elif any(word in query_lower for word in ["difference", "subtract", "minus"]):
                expression = f"{clean_numbers[0]} - {clean_numbers[1]}"
            elif any(word in query_lower for word in ["multiply", "times", "product"]):
                expression = f"{clean_numbers[0]} * {clean_numbers[1]}"
            elif any(word in query_lower for word in ["divide", "ratio", "per"]):
                expression = f"{clean_numbers[0]} / {clean_numbers[1]}"
            elif any(word in query_lower for word in ["average", "mean"]):
                expression = f"({clean_numbers[0]} + {clean_numbers[1]}) / 2"
            else:
                expression = f"{clean_numbers[0]} + {clean_numbers[1]}"
            
            calculation_result = safe_calculator(expression)
            
            if calculation_result["success"]:
                calculation_instruction = CALCULATION_INSTRUCTION.format(
                    expression=calculation_result["expression"],
                    result=calculation_result["result"]
                )
    
    # Call LLM for synthesis
    prompt = SYNTHESIS_PROMPT.format(
        query=query,
        query_type=query_type,
        context=context,
        calculation_instruction=calculation_instruction
    )
    
    response = llm.invoke(prompt)
    
    try:
        content = response.content.strip()
        # Handle markdown code blocks
        if content.startswith("```"):
            content = re.sub(r'^```(?:json)?\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
        
        synthesis = json.loads(content)
        
        answer = synthesis.get("answer", "Unable to generate answer.")
        citations_raw = synthesis.get("citations", [])
        confidence_score = float(synthesis.get("confidence_score", 0.5))
        has_sufficient_context = synthesis.get("has_sufficient_context", True)
        
        # Convert citations to proper format
        citations: list[Citation] = []
        for c in citations_raw:
            citation: Citation = {
                "document": c.get("document", "Unknown"),
                "chunk_id": c.get("chunk_id", ""),
                "quote": c.get("quote", "")
            }
            citations.append(citation)
        
    except (json.JSONDecodeError, KeyError):
        # Fallback response
        answer = response.content
        citations = []
        confidence_score = 0.3
        has_sufficient_context = len(retrieved_chunks) > 0
    
    # Verification: basic hallucination check
    verification_passed = has_sufficient_context and confidence_score >= 0.4
    
    return {
        **state,
        "calculation_result": calculation_result,
        "answer": answer,
        "citations": citations,
        "confidence_score": confidence_score,
        "verification_passed": verification_passed,
    }

# LANGGRAPH WORKFLOW
def build_rag_workflow() -> StateGraph:
    """
    Builds and returns the LangGraph RAG workflow.
    
    Flow:
        START -> query_analyzer -> retrieval_agent -> synthesizer_agent -> END
    
    Conditional behavior is handled within agents based on state flags:
    - multi_entity: Triggers multi-hop retrieval in retrieval_agent
    - requires_calculation: Triggers calculator in synthesizer_agent
    """
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("query_analyzer", query_analyzer)
    workflow.add_node("retrieval_agent", retrieval_agent)
    workflow.add_node("synthesizer_agent", synthesizer_agent)
    
    # Add edges
    workflow.set_entry_point("query_analyzer")
    workflow.add_edge("query_analyzer", "retrieval_agent")
    workflow.add_edge("retrieval_agent", "synthesizer_agent")
    workflow.add_edge("synthesizer_agent", END)
    
    return workflow


def compile_workflow():
    """Compile the workflow into an executable graph."""
    workflow = build_rag_workflow()
    return workflow.compile()


def run_rag_query(query: str, enable_safety: bool = True) -> RAGState:
    """
    Execute a RAG query through the workflow with optional safety checks.
    
    Args:
        query: User's question or request
        enable_safety: If True, performs input validation and hallucination detection
        
    Returns:
        Final RAGState with answer, citations, metadata, and safety flags
    """
    # Input validation
    if enable_safety:
        validation_result = validate_input(query, use_llm=False)
        
        if validation_result["should_block"]:
            # Return error state for malicious queries
            return {
                "query": query,
                "query_type": "blocked",
                "entities": [],
                "multi_entity": False,
                "requires_calculation": False,
                "retrieved_chunks": [],
                "calculation_result": None,
                "answer": f"Query blocked: {validation_result['reason']}",
                "citations": [],
                "confidence_score": 0.0,
                "verification_passed": False,
            }
    
    app = compile_workflow()
    
    # Initialize state
    initial_state: RAGState = {
        "query": query,
        "query_type": "factual",
        "entities": [],
        "multi_entity": False,
        "requires_calculation": False,
        "retrieved_chunks": [],
        "calculation_result": None,
        "answer": "",
        "citations": [],
        "confidence_score": 0.0,
        "verification_passed": False,
    }
    
    # Run workflow
    final_state = app.invoke(initial_state)
    
    # Output guardrails - hallucination detection
    if enable_safety:
        hallucination_result = detect_hallucination(
            answer=final_state["answer"],
            retrieved_chunks=final_state["retrieved_chunks"],
            confidence_score=final_state["confidence_score"])
        # Update confidence score based on hallucination detection
        final_state["confidence_score"] = hallucination_result["confidence"]
        final_state["verification_passed"] = not hallucination_result["hallucination_flag"]
    
    return final_state
