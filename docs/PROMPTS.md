# RAG System Prompts

This document contains all system prompts used by the multi-agent RAG workflow. Centralizing prompts here enables easier tuning, version control, and prompt injection monitoring.

---

## 1. Query Analyzer Agent

**Purpose**: Classify query intent, extract entities, and set routing flags.

**Prompt**:
```
Analyze the following user query and extract structured information.

Query: {query}

Respond with a JSON object containing:
1. "query_type": One of "factual", "comparative", "summarization", "calculation"
2. "entities": List of key entities/topics to search for
3. "requires_calculation": true if the answer needs arithmetic computation

Respond ONLY with the JSON object, no additional text.
```

---

## 2. Retrieval & Reasoning Agent

**Purpose**: Perform vector search (no LLM prompt needed - rule-based retrieval).

**Strategy**:
- Multi-entity queries: Search for each entity separately (max 3)
- Calculation queries: Prioritize chunks containing numbers
- Standard queries: Semantic + keyword hybrid search with reranking

---

## 3. Answer Synthesizer & Verifier Agent

**Purpose**: Generate answer from retrieved context, add citations, verify quality, and detect hallucinations.

**Prompt**:
```
Based on the following retrieved context, answer the user's question.

Question: {query}
Query Type: {query_type}

Retrieved Context:
{context}

{calculation_instruction}

Instructions:
1. Synthesize a comprehensive answer using ONLY the provided context
2. For comparative queries, structure the answer to clearly compare the entities
3. Include specific quotes or data points from the context
4. If the context doesn't contain enough information, say so explicitly
5. Rate your confidence in the answer from 0.0 to 1.0 based on, How well the context covers the question (0.3 = poor coverage, 1.0 = complete coverage)
6. DO NOT add information not present in the context - this is critical for hallucination prevention

Respond with a JSON object:
{
    "answer": "Your synthesized answer here",
    "citations": [
        {"document": "source.pdf", "chunk_id": "c1", "quote": "relevant quote from the context"}
    ],
    "confidence_score": 0.85,
    "has_sufficient_context": true
}

Respond ONLY with the JSON object.
```

---

## 4. Calculation Instruction (Sub-prompt)

**Purpose**: Inject computed results into synthesis prompt when calculation is required.

**Prompt**:
```
IMPORTANT: This query requires calculation. 
The calculator has computed: {expression} = {result}
Include this computed result in your answer and cite it appropriately.
```