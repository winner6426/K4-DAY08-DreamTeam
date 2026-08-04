"""Task 5 — cosine semantic search with a lightweight local HyDE expansion."""

from __future__ import annotations

import re
from typing import Any

from .task4_chunking_indexing import get_collection, get_embedding_model


def _generate_hypothetical_doc(query: str) -> str:
    """Create a retrieval-oriented hypothetical support article for a short query.

    Classic HyDE asks an LLM to draft a likely answer, then retrieves using that
    draft's embedding. This project runs locally without an LLM/API key, so this
    deterministic generator supplies the same useful policy-document context and
    expands common e-commerce intents. It never claims the generated text is a
    source of truth; only ChromaDB results are returned as evidence.
    """
    normalized_query = " ".join(query.strip().split())
    lowered = normalized_query.casefold()

    intent_expansions = []
    intent_map = {
        ("return", "refund", "hoàn tiền", "trả hàng", "đổi trả"): (
            "return and refund policy, eligibility conditions, return request, "
            "required evidence, inspection, processing timeline and refund method"
        ),
        ("payment", "pay", "thanh toán", "chi trả"): (
            "payment methods, supported payment options, payment failure, "
            "payment verification, order payment and transaction security"
        ),
        ("order", "tracking", "delivery", "shipping", "đơn hàng", "vận chuyển", "giao hàng"): (
            "order tracking, delivery status, shipment process, delivery issue, "
            "estimated delivery time and customer support steps"
        ),
        ("seller", "listing", "product", "người bán", "đăng bán", "sản phẩm"): (
            "seller product-listing regulations, prohibited products, listing "
            "requirements, product information and seller responsibilities"
        ),
        ("privacy", "personal data", "bảo mật", "dữ liệu cá nhân", "riêng tư"): (
            "privacy policy, personal data collection, data use, account security "
            "and customer privacy rights"
        ),
    }
    for keywords, expansion in intent_map.items():
        if any(keyword in lowered for keyword in keywords):
            intent_expansions.append(expansion)

    topical_context = "; ".join(intent_expansions) or (
        "e-commerce customer-support policy, conditions, procedures, exceptions, "
        "customer rights, seller responsibilities and official support guidance"
    )
    return (
        "Hypothetical e-commerce support article. "
        f"Customer question: {normalized_query}. "
        f"This policy document explains {topical_context}. "
        "It provides the official steps a customer should follow, required details "
        "or evidence, important eligibility conditions, and relevant time limits."
    )


def _validate_request(query: str, top_k: int) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    return re.sub(r"\s+", " ", query).strip()


def semantic_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Retrieve the most similar indexed chunks by cosine similarity using HyDE.

    The hypothetical document—not the original query—is embedded. ChromaDB is
    configured with cosine distance in Task 4; it is transformed to a familiar
    similarity score via ``1 - distance`` and returned in descending order.
    """
    normalized_query = _validate_request(query, top_k)
    collection = get_collection()
    available = collection.count()
    if available == 0:
        return []

    hypothetical_doc = _generate_hypothetical_doc(normalized_query)
    query_embedding = get_embedding_model().encode(
        hypothetical_doc,
        normalize_embeddings=True,
    ).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, available),
        include=["documents", "metadatas", "distances"],
    )

    ranked = []
    for content, metadata, distance in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        ranked.append(
            {
                "content": content,
                "score": round(max(0.0, 1.0 - float(distance)), 4),
                "metadata": metadata,
            }
        )
    return sorted(ranked, key=lambda result: result["score"], reverse=True)


if __name__ == "__main__":
    for result in semantic_search("quy định trả hàng hoàn tiền", top_k=5):
        print(f"[{result['score']:.3f}] {result['metadata']['source']}: {result['content'][:120]}...")
