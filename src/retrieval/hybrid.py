import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from langchain_openai import OpenAIEmbeddings
import cohere
from src.utils.qdrant import get_qdrant_client

class HybridRetriever:
    def __init__(self):
        load_dotenv()
        # Use singleton client
        self.qdrant = get_qdrant_client()
        self.embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.cohere = cohere.Client(os.getenv("COHERE_API_KEY"))
        self.collection = "multi_doc_rag"

    def normalize_payload(self, payload):
        """Extract chunk_id and text/content robustly from different structures."""
        if not payload: return None
        
        # 1. Try top-level
        chunk_id = payload.get('chunk_id')
        text = payload.get('text') or payload.get('page_content')
        
        # 2. Try nested metadata (common in LangChain)
        if not chunk_id and 'metadata' in payload:
            meta = payload['metadata']
            chunk_id = meta.get('chunk_id')
            # Text might still be at top level as page_content
            if not text:
                text = meta.get('text')
        
        if chunk_id and text:
            return {"chunk_id": chunk_id, "text": text, "filename": payload.get('filename') or payload.get('metadata', {}).get('filename')}
        return None

    def retrieve(self, query: str, top_k: int = 5):
        # ... (keep diagnostic check) ...
        try:
            count = self.qdrant.get_collection(self.collection).points_count
            if count == 0:
                print(f"⚠️  WARNING: Collection '{self.collection}' is EMPTY! retrieval will fail. Run ingestion first.")
                return []
        except Exception:
            print(f"⚠️  WARNING: Collection '{self.collection}' does not exist!")
            return []

        # 1. Semantic Search
        vector = self.embedding.embed_query(query)
        results = self.qdrant.query_points(
            collection_name=self.collection, query=vector, limit=20, with_payload=True
        )
        semantic = results.points if hasattr(results, 'points') else results
        print(f"DEBUG: Semantic search found {len(semantic)} raw results.")

        # 2. Keyword Search
        keyword, _ = self.qdrant.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(must=[models.FieldCondition(key="text", match=models.MatchText(text=query))]),
            limit=20, with_payload=True, with_vectors=False
        )
        print(f"DEBUG: Keyword search found {len(keyword)} raw results.")

        # 3. Merge & Deduplicate
        candidates = {}
        for h in semantic + keyword:
            clean_data = self.normalize_payload(h.payload)
            if clean_data:
                candidates[clean_data['chunk_id']] = clean_data
        
        unique_docs = list(candidates.values())
        print(f"DEBUG: Merged into {len(unique_docs)} unique candidates.")

        if not unique_docs:
            if semantic:
                print("DEBUG: Raw payload of first result:", semantic[0].payload)
            return []

        # 4. Rerank
        try:
            reranked = self.cohere.rerank(
                model="rerank-english-v3.0", query=query, documents=[d['text'] for d in unique_docs], top_n=top_k
            )
            
            # 5. Format Results
            return [{
                "document": unique_docs[r.index].get("filename", "Unknown"),
                "chunk_id": unique_docs[r.index]["chunk_id"],
                "relevance_score": r.relevance_score,
                "content_snippet": unique_docs[r.index]["text"][:1500]
            } for r in reranked.results]
            
        except Exception as e:
            print(f"Error during reranking: {e}")
            return []
