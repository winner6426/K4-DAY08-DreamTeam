"""Task 4 — chunk Markdown documents and index them in persistent ChromaDB."""

from pathlib import Path
from typing import Any

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Recursive splitting keeps paragraphs and sentences together whenever possible.
# 800 characters retains useful policy context; 100 characters protects facts that
# lie at a chunk boundary.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

# Multilingual BGE-M3 is suitable for the Vietnamese/English support corpus.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

VECTOR_STORE = "chromadb"
# The collection name mirrors the corpus domain and is shared by retrieval tasks.
COLLECTION_NAME = "ecommerce_policy_support"


def load_documents() -> list[dict[str, Any]]:
    """Read non-empty Markdown files from ``data/standardized``."""
    if not STANDARDIZED_DIR.exists():
        raise FileNotFoundError(f"Standardized directory not found: {STANDARDIZED_DIR}")

    documents: list[dict[str, Any]] = []
    for markdown_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = markdown_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        relative_path = markdown_file.relative_to(STANDARDIZED_DIR)
        document_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": markdown_file.name,
                    "type": document_type,
                    "path": relative_path.as_posix(),
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split documents with RecursiveCharacterTextSplitter (800/100)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict[str, Any]] = []
    for document in documents:
        for chunk_index, content in enumerate(splitter.split_text(document["content"])):
            chunks.append(
                {
                    "content": content,
                    "metadata": {**document["metadata"], "chunk_index": chunk_index},
                }
            )
    return chunks


def get_embedding_model():
    """Load the one SentenceTransformer model used for index and retrieval."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create normalized, 1024-dimensional BGE-M3 embeddings for every chunk."""
    if not chunks:
        return []

    model = get_embedding_model()
    embeddings = model.encode(
        [chunk["content"] for chunk in chunks],
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Expected {EMBEDDING_DIM}-dimensional embeddings from {EMBEDDING_MODEL}; "
            f"received {embeddings.shape[1]}."
        )
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()
    return chunks


def get_collection():
    """Open the persistent Chroma collection for e-commerce support policies."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "domain": "ecommerce_policy_support"},
    )


def index_to_vectorstore(chunks: list[dict[str, Any]]):
    """Upsert chunk content, metadata, and precomputed embeddings into ChromaDB."""
    collection = get_collection()
    if not chunks:
        return collection

    collection.upsert(
        ids=[f"{chunk['metadata']['path']}::{chunk['metadata']['chunk_index']}" for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    return collection


def run_pipeline():
    """Run load → chunk → embed → persistent ChromaDB upsert."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Collection: {COLLECTION_NAME}")
    print("=" * 50)

    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")
    embedded_chunks = embed_chunks(chunks)
    print(f"Embedded {len(embedded_chunks)} chunks")
    collection = index_to_vectorstore(embedded_chunks)
    print(f"Indexed {collection.count()} chunks to {CHROMA_DIR}")
    return collection


if __name__ == "__main__":
    run_pipeline()
