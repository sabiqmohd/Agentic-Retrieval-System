import os
import tempfile
from typing import List
from fastapi import UploadFile
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document

LOADERS = {".pdf": PyMuPDFLoader, ".txt": TextLoader, ".docx": Docx2txtLoader}

async def load_documents(files: List[UploadFile]) -> List[Document]:
    documents = []
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in LOADERS: continue
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            loader = LOADERS[ext](tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({"filename": file.filename, "doc_type": ext})
            documents.extend(docs)
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)
            
    return documents
