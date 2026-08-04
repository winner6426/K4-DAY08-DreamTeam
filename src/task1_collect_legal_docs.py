"""
Task 1 — Thu thập văn bản chính sách thương mại điện tử / hỗ trợ khách hàng.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang chính thức của một sàn TMĐT.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Gợi ý nguồn (ví dụ trang công khai Shopee Vietnam — help.shopee.vn):
    - https://help.shopee.vn/portal/4/article/77251 (Chính sách trả hàng và hoàn tiền)
    - https://help.shopee.vn/portal/4/article/79198 (Phương thức thanh toán)
    - https://help.shopee.vn/portal/4/article/77244 (Chính sách bảo mật)

Gợi ý văn bản (chủ đề chính sách thương mại điện tử):
    - Chính sách đổi trả/hoàn tiền (Returns/Refund Policy)
    - Phương thức thanh toán (Payment Methods)
    - Chính sách bảo mật (Privacy Policy)
    - Quy định đăng bán sản phẩm cho người bán (Seller Listing Regulations)

Nhớ gắn metadata `customer_role` (`buyer`/`seller`/`both`) cho từng tài liệu — yêu cầu riêng
của K4 Variant (kế thừa từ Lab 07), cần thiết để viết benchmark query dùng metadata_filter.

Lưu ý: một số trang help center dùng JavaScript render nội dung (SPA) — crawl về chỉ thấy
tiêu đề mà không có nội dung thật. Đổi sang bài viết khác cùng domain thay vì cố xử lý,
và chỉ dùng nguồn công khai/được phép chia sẻ.
"""

from pathlib import Path
import sys

from bs4 import BeautifulSoup
from fpdf import FPDF
from fpdf.enums import WrapMode

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


# TODO: Tải file PDF/DOCX về DATA_DIR
# Có thể tải thủ công hoặc viết script download nếu có direct link.
#
# Ví dụ nếu có direct link:
#
import requests

def download_file(url: str, filename: str):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    filepath = DATA_DIR / filename
    filepath.write_bytes(response.content)
    print(f"✓ Đã tải: {filepath}")
def html_to_pdf(input_path: Path, output_path: Path) -> None:
    """Convert a browser-saved HTML file to a Vietnamese Unicode PDF."""
    soup = BeautifulSoup(input_path.read_bytes(), "html.parser")

    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()

    lines = [line.strip() for line in soup.get_text("\n").splitlines() if line.strip()]
    text = "\n".join(lines)
    if not text:
        raise ValueError(f"No readable text found in {input_path.name}")
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Unicode font not found: {FONT_PATH}")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.add_font("ArialUnicode", fname=str(FONT_PATH))
    pdf.set_font("ArialUnicode", size=11)
    pdf.set_title(input_path.name)
    pdf.multi_cell(w=0, h=6, text=text, wrapmode=WrapMode.CHAR)
    pdf.output(str(output_path))
    print(f"Created PDF: {output_path}")


def convert_saved_html_to_pdf() -> int:
    """Convert extensionless HTML files in DATA_DIR to PDF."""
    converted = 0
    for input_path in sorted(DATA_DIR.iterdir()):
        if input_path.is_file() and input_path.suffix == "" and input_path.stat().st_size > 0:
            html_to_pdf(input_path, input_path.with_suffix(".pdf"))
            converted += 1
    return converted


#
# Nếu trang là HTML thuần (không phải PDF sẵn), có thể convert nội dung text
# thành PDF đơn giản bằng thư viện fpdf2 (đã có trong requirements.txt).


if __name__ == "__main__":
    setup_directory()
    total = convert_saved_html_to_pdf()
    print(f"Completed: converted {total} files to PDF.")
