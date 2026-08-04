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

<<<<<<< HEAD
# Multilingual BGE-M3 is suitable for the Vietnamese/English support corpus.
EMBEDDING_MODEL = "BAAI/bge-m3"
=======
# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 800        # Đủ giữ trọn một đoạn chính sách nhưng không quá dài
CHUNK_OVERLAP = 100      # Đủ để hạn chế câu hoặc ý bị cắt ở ranh giới
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"
# Đơn giản, ổn định với tài liêu legal dài và bài hướng dẫn ngắn

# TODO: Chọn embedding model và giải thích
EMBEDDING_MODEL = "BAAI/bge-m3"  # Multilingual, tốt cho tiếng Việt lẫn tiếng Anh
>>>>>>> dd5d6a5bf38f262848364a22d43fbbd268fc9cff
EMBEDDING_DIM = 1024

VECTOR_STORE = "chromadb"
# The collection name mirrors the corpus domain and is shared by retrieval tasks.
COLLECTION_NAME = "ecommerce_policy_support"


def load_documents() -> list[dict[str, Any]]:
    """Read non-empty Markdown files from ``data/standardized``."""
    if not STANDARDIZED_DIR.exists():
        raise FileNotFoundError(f"Standardized directory not found: {STANDARDIZED_DIR}")

<<<<<<< HEAD
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
=======
_embedding_model = None


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    return _embedding_model


def get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name=COLLECTION_NAME)

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    # TODO: Iterate qua STANDARDIZED_DIR, đọc .md files
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents
    #raise NotImplementedError("Implement load_documents")
>>>>>>> dd5d6a5bf38f262848364a22d43fbbd268fc9cff


def chunk_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split documents with RecursiveCharacterTextSplitter (800/100)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

<<<<<<< HEAD
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
=======
    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    # TODO: Implement chunking
    #
    # Ví dụ với RecursiveCharacterTextSplitter:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks
    # raise NotImplementedError("Implement chunk_documents")
>>>>>>> dd5d6a5bf38f262848364a22d43fbbd268fc9cff


def get_embedding_model():
    """Load the one SentenceTransformer model used for index and retrieval."""
    from sentence_transformers import SentenceTransformer

<<<<<<< HEAD
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
=======
    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    # TODO: Implement embedding
    #
    # Ví dụ với sentence-transformers (local, mặc định):
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks
    #
    # Nâng cao (optional): nếu muốn cho cả nhóm chọn được provider qua .env, viết
    # 1 hàm embed_texts(texts) dispatch theo os.getenv("EMBEDDING_PROVIDER") sang
    # sentence-transformers | Google (genai.embed_content) | OpenAI (client.embeddings.create)
    # rồi gọi lại hàm đó ở đây và ở Task 5 — tránh viết logic embed lặp lại 2 nơi.
    # raise NotImplementedError("Implement embed_chunks")


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    # TODO: Implement indexing
    #
    # Ví dụ với ChromaDB:
    import chromadb
    
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    
    ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
    collection.upsert(
        ids=ids,
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    # raise NotImplementedError("Implement index_to_vectorstore")
>>>>>>> dd5d6a5bf38f262848364a22d43fbbd268fc9cff


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
