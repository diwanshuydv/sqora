from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct

# Connect to local Qdrant storage
client = QdrantClient(path="Vector_DB/qdrant/storage")

# Create embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION_NAME = "papers"

# Create collection (safe if already exists)
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "size": 384,
        "distance": "Cosine"
    }
)

# Dummy documents
documents = [
    "SQORA is a project that combines Qdrant and llama.cpp for RAG.",
    "Qdrant is a vector database used for similarity search.",
    "llama.cpp allows running LLMs locally without GPUs."
]

points = []

for idx, doc in enumerate(documents):
    vector = embedder.encode(doc).tolist()
    points.append(
        PointStruct(
            id=idx,
            vector=vector,
            payload={"text": doc}
        )
    )

# Insert into Qdrant
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print("✅ Dummy data inserted into Qdrant")
