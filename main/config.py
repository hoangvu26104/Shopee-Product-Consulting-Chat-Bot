#Cài đặt kết nối đến Qdrant
QDRANT_HOST = "localhost" 
QDRANT_PORT = 6333  

#Cài đặt cho việc xử lý dữ liệu
COLLECTION_NAME = "Thay_doi_ten_collection" 
EMBED_MODEL = "intfloat/multilingual-e5-small" 
GROQ_API_KEY = "Thay_doi_bang_api_key_cua_ban"
LLM_MODEL = "llama3-70b-8192" 
# Các từ khóa liên quan đến sản phẩm
PRODUCT_KEYWORDS = {
    "giá đỡ": ["chắc", "trượt", "cứng", "nặng", "kê", "cao", "gập"],
    "tai nghe": ["pin", "ồn", "âm", "bluetooth", "mic"],
    "micro": ["hút", "khò", "ồn", "rè"],
    "kệ": ["bền", "gọn", "chắc"],
    "ốp điện thoại": ["mềm", "dẻo", "chống sốc", "mỏng", "cứng", "trong suốt"],
    "chuột máy tính": ["êm", "mượt", "chạy tốt", "pin", "nhạy", "bền", "kết nối", "cắm", "nhỏ gọn"],
    "bàn phím": ["cơ", "rgb", "bluetooth", "êm", "game"],
    "kính cường lực": ["dày", "chống nhìn trộm", "xước", "vỡ", "lóa", "chói", "bụi bẩn"]
}
# Các từ khóa mặc định nếu không nhận diện được loại sản phẩm
DEFAULT_KEYWORDS = ["tốt", "không tốt", "giá", "đáng", "nên", "không nên"]