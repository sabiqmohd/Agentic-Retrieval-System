import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

_client = None

def get_qdrant_client():
    global _client
    if _client is None:
        qdrant_url = os.getenv("QDRANT_URL")
        
        if qdrant_url:
            # Server mode (Docker / remote Qdrant)
            _client = QdrantClient(url=qdrant_url)
        else:
            # Local storage mode (development)
            try:
                _client = QdrantClient(path="./qdrant_data")
            except RuntimeError as e:
                if "already accessed" in str(e):
                    raise RuntimeError(
                        "Qdrant Error: './qdrant_data' is locked by another process.\n"
                        "Fix: Stop the other process (notebook/terminal) and try again.\n"
                        "Or set QDRANT_URL=http://localhost:6333 in .env to use Qdrant server."
                    ) from e
                raise e
    return _client
