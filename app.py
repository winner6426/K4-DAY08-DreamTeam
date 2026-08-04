"""Streamlit UI for the E-commerce Policy & Support RAG chatbot."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="E-commerce support RAG",
    page_icon=":material/support_agent:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl="30m", max_entries=30, show_spinner=False)
def answer_question(query: str, top_k: int) -> dict[str, Any]:
    """Cache serializable RAG responses; Task 4 caches the embedding model itself."""
    from src.task10_generation import generate_with_citation

    return generate_with_citation(query, top_k=top_k)


def show_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"Nguồn tham khảo ({len(sources)} chunks)", icon=":material/source:"):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata") or {}
            name = metadata.get("source", "Không rõ nguồn")
            kind = metadata.get("type", "unknown")
            score = float(source.get("score", 0.0))
            st.markdown(f"**[{index}] {name}** · `{kind}` · cosine/RRF score `{score:.4f}`")
            st.caption(source.get("content", "")[:500])


st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending_query", None)

SUGGESTIONS = [
    "Thời hạn yêu cầu trả hàng hoặc hoàn tiền là bao lâu?",
    "Shopee hỗ trợ những phương thức thanh toán nào?",
    "Tôi cần bằng chứng gì khi yêu cầu hoàn tiền?",
    "Làm thế nào để theo dõi đơn hàng?",
    "Người bán không được đăng những nội dung nào?",
]

with st.sidebar:
    st.title("E-commerce support RAG")
    st.caption("Trợ lý tra cứu chính sách Shopee và hướng dẫn hỗ trợ khách hàng.")
    st.badge("ChromaDB + BGE-M3", icon=":material/database:", color="blue")

    top_k = st.slider("Số nguồn truy xuất", min_value=3, max_value=10, value=5, key="top_k")
    st.subheader("Câu hỏi gợi ý")
    for index, suggestion in enumerate(SUGGESTIONS):
        if st.button(suggestion, key=f"suggestion_{index}", width="stretch"):
            st.session_state.pending_query = suggestion
            st.rerun()

    if st.button("Xóa cuộc trò chuyện", icon=":material/delete_sweep:", width="stretch"):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    st.caption("Pipeline: Hybrid retrieval → RRF → optional PageIndex fallback → cited answer")

st.title("Hỏi đáp chính sách e-commerce")
st.caption("Câu trả lời được tạo từ tài liệu đã index; mở nguồn để kiểm tra evidence.")

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=":material/smart_toy:" if message["role"] == "assistant" else None):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_sources(message.get("sources", []))

prompt = st.chat_input(
    "Nhập câu hỏi về thanh toán, đổi trả, giao hàng hoặc quy định người bán",
    key="chat_input",
    submit_mode="disable",
)
query = prompt or st.session_state.pending_query

if query:
    st.session_state.pending_query = None
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        with st.status("Đang truy xuất tài liệu và tạo câu trả lời…", expanded=False) as status:
            try:
                response = answer_question(query, top_k)
                answer = response.get("answer", "Tôi không thể xác minh thông tin này từ nguồn hiện có.")
                sources = response.get("sources", [])
                status.update(label="Đã hoàn thành", state="complete", expanded=False)
            except Exception as exc:
                answer = f"Không thể chạy RAG pipeline: {type(exc).__name__}: {exc}"
                sources = []
                status.update(label="Pipeline gặp lỗi", state="error", expanded=True)

        st.markdown(answer)
        show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
