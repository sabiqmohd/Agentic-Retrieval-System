# Multi-Document RAG Agentic Workflow

A multi-agent RAG system built with **LangGraph**, **FastAPI**, and **Qdrant**. Upload documents, ask questions, and get cited answers — powered by three specialized AI agents.

## What It Does

1. **Upload documents** (PDF, TXT, DOCX) → auto-chunks and indexes them
2. **Ask questions** → AI agents analyze, retrieve, and synthesize answers with citations
3. **Safety built-in** → prompt injection detection + hallucination checks

## Quick Start (< 10 minutes)

```bash
# 1. Clone and navigate
cd Multi_doc_rag_agentic_workflow

# 2. Add your API keys
cp .env
# Edit .env and add your OPENAI_API_KEY and COHERE_API_KEY

# 3. Start everything
docker-compose up --build

# 4. Open API docs
# http://localhost:8000/docs
```

## API Endpoints

| Endpoint | Method | What It Does |
|----------|--------|-------------|
| `/ingest` | POST | Upload and index documents |
| `/query` | POST | Ask a question, get answer with sources |
| `/reset` | POST | Clear all indexed documents |
| `/health` | GET | Check if server is running |
| `/docs` | GET | Swagger UI (interactive API docs) |

## Sample curl Commands

### Upload a document
```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "files=@your_document.pdf"
```

### Ask a question
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings in the document?"}'
```

Response:
```json
{
  "answer": "The key findings include...",
  "sources": [
    {
      "document": "your_document.pdf",
      "chunk_id": "abc123",
      "relevance_score": 0.95,
      "content_snippet": "..."
    }
  ],
  "confidence": 0.88
}
```

### Clear the knowledge base
```bash
curl -X POST "http://localhost:8000/reset"
```

## How It Works

Three AI agents work using LangGraph:

```
User Query → [Query Analyzer] → [Retrieval Agent] → [Answer Synthesizer] → Answer + Citations
```

1. **Query Analyzer** — Figures out what type of question it is (factual, comparative, calculation)
2. **Retrieval Agent** — Searches the vector database using hybrid search (semantic + keyword + reranking)
3. **Answer Synthesizer** — Generates the answer with citations and runs a hallucination check

For more details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Project Structure

```
├── src/
│   ├── agents/          # LangGraph agents (analyzer, synthesizer, safety)
│   ├── retrieval/       # Hybrid retriever (semantic + keyword + Cohere rerank)
│   ├── ingestion/       # Document loading, chunking, vector store
│   ├── api/             # FastAPI endpoints
│   └── utils/           # Qdrant client, logging
├── docs/
│   ├── ARCHITECTURE.md  # System design with Mermaid diagrams
│   └── PROMPTS.md       # All agent prompts + token optimization
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4o-mini |
| `COHERE_API_KEY` | Yes | Cohere API key for search reranking |
| `QDRANT_URL` | No | Qdrant server URL (auto-set in Docker, uses local storage if not set) |

```