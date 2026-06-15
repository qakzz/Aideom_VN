# AIDEOM-VN - Hệ thống hỗ trợ ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI

Repo này là bản sẵn sàng đưa lên GitHub cho bài cuối kỳ môn **Các mô hình ra quyết định**.

## Yêu cầu đã đáp ứng

- Mã nguồn Python tối thiểu 1.500 dòng.
- Dashboard Streamlit chạy local.
- Dữ liệu CSV được nhập lại từ đề bài.
- Báo cáo nghiên cứu có tối thiểu 4 bảng kết quả và 5 hình minh họa.
- Có `requirements.txt`, `README.md`, `.gitignore`, `tests/` và `run_all.py`.

## Cài đặt local

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy toàn bộ mô hình

```bash
python run_all.py
```

## Chạy dashboard local

```bash
streamlit run dashboard/app.py
```

Mở trình duyệt theo địa chỉ Streamlit hiển thị, thường là `http://localhost:8501`.

## Chạy test

```bash
pytest -q
```

## Đưa lên GitHub

```bash
git init
git add .
git commit -m "Initial AIDEOM-VN decision support prototype"
git branch -M main
git remote add origin https://github.com/<your-username>/aideom-vn.git
git push -u origin main
```

## Cấu trúc thư mục

```text
data/        Dữ liệu CSV
src/         Mã nguồn Python các module M1-M6
dashboard/   Dashboard Streamlit chạy local
outputs/     Bảng và hình đầu ra
reports/     Báo cáo nghiên cứu
notebooks/   Vị trí để bổ sung notebook nếu cần
tests/       Kiểm thử tự động
```
