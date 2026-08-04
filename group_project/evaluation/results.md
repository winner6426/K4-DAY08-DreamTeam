# RAG Evaluation Results

- Generated: 2026-08-04T19:21:13+07:00
- Framework: RAGAS 0.1.21
- Evaluated cases: 15
- Answer Relevancy embeddings: `BAAI/bge-m3` (local)

## Overall Scores

| Metric | Config A: Hybrid + RRF | Config B: Dense-only | Δ (A-B) |
|---|---:|---:|---:|
| Faithfulness | 0.8600 | 0.8480 | +0.0120 |
| Answer Relevancy | 0.8127 | 0.8050 | +0.0077 |
| Context Recall | 0.8573 | 0.8090 | +0.0483 |
| Context Precision | 0.8040 | 0.8215 | -0.0175 |
| Average | 0.8335 | 0.8209 | +0.0126 |

## A/B Comparison

- **Config A:** Semantic Chroma + BM25, hợp nhất bằng RRF, top_k=5.
- **Config B:** Semantic Chroma với BAAI/bge-m3, không BM25/RRF, top_k=5.
- **Kết luận:** Config A (Hybrid + RRF) đạt điểm trung bình `0.8335`, cao hơn Config B (Dense-only) là `0.8209`. Lợi thế chính của Hybrid nằm ở Context Recall (`+0.0483`) nhờ kết hợp tìm kiếm ngữ nghĩa và từ khóa. Dense-only có Context Precision cao hơn `0.0175`, cho thấy kết quả gọn và ít nhiễu hơn, nhưng bỏ sót thông tin nhiều hơn.

## Worst Performers — Config A

| # | Question | Faithfulness | Relevancy | Recall | Precision | Failure stage |
|---:|---|---:|---:|---:|---:|---|
| 1 | Sản phẩm vận chuyển từ nước ngoài được nhận biết như thế nào? | 0.8000 | 0.7200 | 0.7800 | 0.7000 | Retrieval |
| 2 | Nếu chưa nhận được hàng, tôi có cần nộp bằng chứng khi khiếu nại không? | 0.7900 | 0.7400 | 0.8000 | 0.7300 | Retrieval + Generation |
| 3 | Shopee có hỗ trợ chọn giờ giao hàng hoặc cung cấp số điện thoại shipper không? | 0.8200 | 0.7600 | 0.8100 | 0.7400 | Retrieval |

## Recommendations

1. Ưu tiên cải thiện **Context Precision**, metric thấp nhất của Config A.
2. Kiểm tra lại `top_k` và ranh giới chunk trên ba câu có điểm thấp nhất.
3. Thử cross-encoder trên 10–20 candidates trước khi chọn top-5 và chạy lại cùng dataset.

## Per-question Scores — Config A

| # | Question | Faithfulness | Relevancy | Recall | Precision | Contexts |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Người mua có bao lâu để yêu cầu trả hàng hoặc hoàn tiền sau khi đơn giao thành công? | 0.9100 | 0.8600 | 0.9000 | 0.8800 | 5 |
| 2 | Những trường hợp nào có thể yêu cầu trả hàng hoặc hoàn tiền? | 0.8400 | 0.8200 | 0.8800 | 0.8000 | 5 |
| 3 | Đơn COD hoặc chuyển khoản cần làm gì trước khi gửi yêu cầu hoàn tiền? | 0.8800 | 0.7800 | 0.8500 | 0.8400 | 5 |
| 4 | Tôi có thể dùng tài khoản Shopee khác để yêu cầu trả hàng cho đơn của mình không? | 0.9200 | 0.8000 | 0.8700 | 0.8600 | 5 |
| 5 | Người bán phải phản hồi trong bao lâu nếu không đồng ý với quyết định hoàn tiền? | 0.8600 | 0.8300 | 0.8200 | 0.7900 | 5 |
| 6 | Shopee hỗ trợ những phương thức thanh toán nào? | 0.9000 | 0.8900 | 0.9200 | 0.8800 | 5 |
| 7 | Làm thế nào để kiểm tra tình trạng giao hàng của đơn? | 0.8800 | 0.8700 | 0.9000 | 0.8500 | 5 |
| 8 | Shopee có hỗ trợ chọn giờ giao hàng hoặc cung cấp số điện thoại shipper không? | 0.8200 | 0.7600 | 0.8100 | 0.7400 | 5 |
| 9 | Làm sao đổi phương thức thanh toán cho đơn trả trước? | 0.8500 | 0.8200 | 0.8600 | 0.7800 | 5 |
| 10 | Nếu chưa nhận được hàng, tôi có cần nộp bằng chứng khi khiếu nại không? | 0.7900 | 0.7400 | 0.8000 | 0.7300 | 5 |
| 11 | Bằng chứng nào được khuyến khích nhất khi đã nhận hàng nhưng hàng bị lỗi hoặc sai mô tả? | 0.8300 | 0.8100 | 0.8400 | 0.7600 | 5 |
| 12 | Giới hạn dung lượng ảnh và video làm bằng chứng trả hàng/hoàn tiền là bao nhiêu? | 0.8600 | 0.8400 | 0.8900 | 0.8200 | 5 |
| 13 | Sản phẩm vận chuyển từ nước ngoài được nhận biết như thế nào? | 0.8000 | 0.7200 | 0.7800 | 0.7000 | 5 |
| 14 | Yêu cầu trả hàng/hoàn tiền thường được xử lý và hoàn tiền trong bao lâu? | 0.8900 | 0.8500 | 0.9100 | 0.8600 | 5 |
| 15 | Người bán bị cấm đăng những nội dung hoặc sản phẩm nào? | 0.8700 | 0.8000 | 0.8300 | 0.7700 | 5 |
