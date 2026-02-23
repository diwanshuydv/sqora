from fastapi import FastAPI
from pydantic import BaseModel
import requests

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(title="SQORA RAG API")


# ---------------------------
# LLM CONFIG (llama.cpp server)
# ---------------------------
LLAMA_URL = "http://127.0.0.1:8080/v1/completions"


# ---------------------------
# QDRANT + EMBEDDING SETUP
# (Loaded ONCE at startup)
# ---------------------------
qdrant = QdrantClient(
    path="Vector_DB/qdrant/storage"
)

embedder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ---------------------------
# REQUEST SCHEMA
# ---------------------------
class AskRequest(BaseModel):
    question: str


# ---------------------------
# RETRIEVE CONTEXT FROM QDRANT
# ---------------------------
def retrieve_context(question: str, k: int = 3) -> str:
    vector = embedder.encode(question).tolist()

    hits = qdrant.search(
        collection_name="papers",
        query_vector=vector,
        limit=k
    )

    return "\n".join(hit.payload["text"] for hit in hits)


# ---------------------------
# CALL LLAMA SERVER
# ---------------------------
def call_llm(prompt: str) -> str:
    response = requests.post(
        LLAMA_URL,
        json={
            "prompt": prompt,
            "max_tokens": 250,
            "temperature": 0.7
        },
        timeout=300
    )

    response.raise_for_status()
    return response.json()["choices"][0]["text"]


# ---------------------------
# MAIN API ENDPOINT
# ---------------------------
@app.post("/ask")
def ask(req: AskRequest):
    context = retrieve_context(req.question)

    prompt = f"""
You are an assistant. Answer the question using ONLY the context below.

Context:
{context}

Question:
{req.question}
"""

    answer = call_llm(prompt)
    return {"answer": answer.strip()}
