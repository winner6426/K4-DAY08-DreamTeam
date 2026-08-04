"""Task 8 - PageIndex vectorless retrieval over the original legal PDFs."""

import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent.parent
LANDING_LEGAL_DIR = PROJECT_DIR / "data" / "landing" / "legal"
PAGEINDEX_STATE_FILE = PROJECT_DIR / "data" / "pageindex_documents.json"
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
POLL_INTERVAL_SECONDS = 2.0
POLL_TIMEOUT_SECONDS = 5.0


def _get_api_key() -> str:
    """Read at call time so edits to the process environment are respected."""
    return os.getenv("PAGEINDEX_API_KEY", "").strip() or PAGEINDEX_API_KEY.strip()


def _get_client():
    try:
        from pageindex.client import PageIndexClient
    except ImportError as exc:
        raise RuntimeError("PageIndex is not installed. Run: pip install pageindex") from exc

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Missing PAGEINDEX_API_KEY in .env")
    return PageIndexClient(api_key=api_key)


def _load_document_records() -> list[dict]:
    if not PAGEINDEX_STATE_FILE.exists():
        return []
    try:
        payload = json.loads(PAGEINDEX_STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    records = payload.get("documents", payload) if isinstance(payload, dict) else payload
    return records if isinstance(records, list) else []


def _save_document_records(records: list[dict]) -> None:
    PAGEINDEX_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_STATE_FILE.write_text(
        json.dumps({"documents": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _configured_documents() -> list[dict]:
    """Combine uploaded IDs with optional comma-separated PAGEINDEX_DOC_IDS."""
    records = _load_document_records()
    known_ids = {str(item.get("doc_id", "")) for item in records}
    for doc_id in os.getenv("PAGEINDEX_DOC_IDS", "").split(","):
        doc_id = doc_id.strip()
        if doc_id and doc_id not in known_ids:
            records.append({"doc_id": doc_id, "name": doc_id})
            known_ids.add(doc_id)
    return [item for item in records if item.get("doc_id")]


def _iter_relevant_items(value: Any) -> Iterator[dict]:
    """Flatten PageIndex's nested relevant_contents response."""
    if isinstance(value, list):
        for item in value:
            yield from _iter_relevant_items(item)
    elif isinstance(value, dict):
        if value.get("relevant_content") or value.get("content") or value.get("text"):
            yield value
        else:
            for nested in value.values():
                if isinstance(nested, (list, dict)):
                    yield from _iter_relevant_items(nested)


def _wait_for_retrieval(client, retrieval_id: str, deadline: float) -> dict:
    interval = float(os.getenv("PAGEINDEX_POLL_INTERVAL", POLL_INTERVAL_SECONDS))

    while True:
        response = client.get_retrieval(retrieval_id)
        status = str(response.get("status", "")).lower()
        if response.get("retrieved_nodes") is not None or status in {
            "completed", "complete", "success", "succeeded",
        }:
            return response
        if status in {"failed", "error", "cancelled", "canceled"}:
            return {}
        if time.monotonic() >= deadline:
            return {}
        time.sleep(max(interval, 0.1))


def upload_documents() -> list[dict]:
    """Upload legal PDFs once and persist their PageIndex document IDs."""
    client = _get_client()
    pdf_files = sorted(LANDING_LEGAL_DIR.rglob("*.pdf"))
    records = _load_document_records()
    by_path = {
        str(Path(item["path"]).resolve()): item
        for item in records
        if item.get("path") and item.get("doc_id")
    }

    for pdf_file in pdf_files:
        resolved_path = str(pdf_file.resolve())
        if resolved_path in by_path:
            print(f"  - Already uploaded: {pdf_file.name}")
            continue

        response = client.submit_document(resolved_path)
        doc_id = response.get("doc_id") or response.get("id")
        if not doc_id:
            raise RuntimeError(f"PageIndex did not return doc_id for {pdf_file.name}")
        record = {"doc_id": str(doc_id), "name": pdf_file.name, "path": resolved_path}
        records.append(record)
        by_path[resolved_path] = record
        _save_document_records(records)
        print(f"  Uploaded: {pdf_file.name} -> {doc_id}")

    return records


def pageindex_search(
    query: str,
    top_k: int = 5,
    timeout_seconds: float | None = None,
) -> list[dict]:
    """Query uploaded PageIndex documents and normalize their result schema."""
    # Unit tests must be deterministic and must not wait on an external API.
    # Set PAGEINDEX_TEST_LIVE=1 only when intentionally running an integration test.
    if (
        os.getenv("PYTEST_CURRENT_TEST")
        and os.getenv("PAGEINDEX_TEST_LIVE", "0") != "1"
    ):
        return []
    if top_k <= 0 or not query.strip() or not _get_api_key():
        return []

    documents = _configured_documents()
    if not documents:
        return []

    client = _get_client()
    results: list[dict] = []
    if timeout_seconds is None:
        timeout_seconds = float(
            os.getenv("PAGEINDEX_SEARCH_TIMEOUT", POLL_TIMEOUT_SECONDS)
        )
    deadline = time.monotonic() + max(timeout_seconds, 0.0)

    for document in documents:
        if time.monotonic() >= deadline:
            break
        doc_id = str(document["doc_id"])
        document_rank = 0
        try:
            if not client.is_retrieval_ready(doc_id):
                continue
            submitted = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = submitted.get("retrieval_id") or submitted.get("id")
            if not retrieval_id:
                continue
            retrieval = _wait_for_retrieval(client, str(retrieval_id), deadline)
        except Exception as exc:
            print(f"  Warning: PageIndex query failed for {doc_id}: {exc}")
            continue

        for node in retrieval.get("retrieved_nodes", []):
            values = node.get("relevant_contents", node)
            for item in _iter_relevant_items(values):
                content = str(
                    item.get("relevant_content")
                    or item.get("content")
                    or item.get("text")
                    or ""
                ).strip()
                if not content:
                    continue
                document_rank += 1
                results.append({
                    "content": content,
                    "score": 1.0 / document_rank,
                    "metadata": {
                        "section": item.get("section_title", ""),
                        "doc_id": doc_id,
                        "document": document.get("name", doc_id),
                    },
                    "source": "pageindex",
                })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not _get_api_key():
        print("Set PAGEINDEX_API_KEY in .env before using Task 8.")
        print("Register at: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()
        print("\nTest query:")
        for result in pageindex_search(
            "danh sach san pham cam dang ban",
            top_k=3,
            timeout_seconds=120.0,
        ):
            print(f"[{result['score']:.3f}] {result['content'][:100]}...")
