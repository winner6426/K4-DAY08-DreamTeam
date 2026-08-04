"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import re
from pathlib import Path

# TODO: Load corpus từ data/standardized/ hoặc từ vector store
# CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}

from .task4_chunking_indexing import load_documents, chunk_documents

CORPUS = chunk_documents(load_documents())


# BM25 only matches exact terms, so expand the English benchmark vocabulary
# with Vietnamese equivalents used by the Shopee corpus.
QUERY_EXPANSIONS = {
    "return": "trả hàng",
    "refund": "hoàn tiền",
    "evidence": "bằng chứng",
    "policy": "chính sách",
    "payment": "thanh toán",
    "method": "phương thức",
    "methods": "phương thức",
    "seller": "người bán",
    "listing": "đăng bán",
    "regulation": "quy định",
    "regulations": "quy định",
    "order": "đơn hàng",
    "tracking": "theo dõi",
    "guide": "hướng dẫn",
}


def tokenize(text: str) -> list[str]:
    """Lowercase Unicode-aware tokenization without attached punctuation."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def expand_query(query: str) -> list[str]:
    """Keep original query terms and append domain-specific translations."""
    tokens = tokenize(query)
    expanded = list(tokens)
    for token in tokens:
        translated = QUERY_EXPANSIONS.get(token)
        if translated:
            expanded.extend(tokenize(translated))
    return expanded


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    # TODO: Implement BM25 index
    #
    from rank_bm25 import BM25Okapi
    
    # Tokenize - có thể đơn giản split(), hoặc dùng underthesea cho tiếng Việt
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    # TODO: Implement lexical search
    bm25 = build_bm25_index(CORPUS)
    tokenized_query = expand_query(query)
    scores = bm25.get_scores(tokenized_query)
    
    # Get top_k indices
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
