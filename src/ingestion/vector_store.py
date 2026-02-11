import os
import uuid
from typing import List
from datetime import datetime
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

load_dotenv()

from src.utils.qdrant import get_qdrant_client

def ingest_to_qdrant(chunks: List[Document]) -> List[str]:
    # Use singleton client to avoid lock issues
    client = get_qdrant_client()
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    collection = "multi_doc_rag"

    try:
        client.get_collection(collection)
    except Exception:
        client.create_collection(
            collection, vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )

    ids = [str(uuid.uuid4()) for _ in chunks]
    for chunk, id_ in zip(chunks, ids):
        chunk.metadata.update({
            "doc_id": str(uuid.uuid4()),
            "chunk_id": id_,
            "text": chunk.page_content,
            "upload_date": datetime.now().isoformat()
        })

    Qdrant(client, collection, embeddings).add_documents(chunks, ids=ids)
    return ids
