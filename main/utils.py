from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq
from datetime import datetime
from uuid import uuid4

# Import các config
import config

# --- CACHE ---
_embedding_model = None
_qdrant_client = None
# --- Các hàm khởi tạo và lấy client (có cache) ---
def get_llm():
    return ChatGroq(
        model_name=config.LLM_MODEL, # Lấy tên model từ config
        api_key=config.GROQ_API_KEY, # Lấy API key từ config
        temperature=0.2, # Giảm nhiệt độ để câu trả lời của AI bớt sáng tạo, tập trung vào dữ liệu
        max_tokens=512 # Giới hạn độ dài của câu trả lời
    )

def get_embedding_model():
    """Khởi tạo model embedding nếu chưa có, sau đó trả về."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
    return _embedding_model

def get_qdrant_client():
    """Khởi tạo kết nối đến Qdrant nếu chưa có, sau đó trả về."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    return _qdrant_client

# --- HELPER FUNCTIONS ---
# --- Các hàm trợ giúp cho việc xử lý prompt và dữ liệu ---

def fuzzy_match(product_type: str, text: str) -> bool:
    """Kiểm tra xem các từ trong `product_type` có xuất hiện trong `text` không (cho phép sai 1 từ)."""
    if not product_type or not text:
        return False
    words = product_type.lower().split()
    matched = sum(1 for w in words if w in text.lower())
    return matched >= len(words) - 1

def generate_prompt_context(results, product_type):
    """Tạo ra một chuỗi ngữ cảnh từ các kết quả tìm kiếm để đưa cho LLM."""
    keywords = config.PRODUCT_KEYWORDS.get(product_type, config.DEFAULT_KEYWORDS)
    context_lines = []
    for idx, r in enumerate(results[:5]): # Giới hạn 5 sản phẩm
        product_name = r.get("name_product", f"Sản phẩm {idx+1}")
        comment_text = r.get("comment", "") or r.get("describe", "") or r.get("detail", "") or ""
        if not comment_text.strip():
            continue
        context_lines.append(f"Tên: {product_name}\nBình luận: {comment_text.strip()}\n")
    return "\n".join(context_lines)

def base_prompt(task: str, context: str, query: str):
    """Tạo ra cấu trúc prompt hoàn chỉnh để gửi đến LLM."""
    if not context.strip():
        return None
    return f"""
Bạn là trợ lý AI tư vấn sản phẩm. Trả lời hoàn toàn bằng tiếng Việt.
Người dùng hỏi: \"{query}\"
Dưới đây là các bình luận thật:
{context}
🌟 Nhiệm vụ:
{task}
"""

# --- QDRANT MEMORY FUNCTIONS ---
# --- Các hàm quản lý "trí nhớ dài hạn" của bot ---
def save_chat_history_to_qdrant(state, user_id: str = "default_user"):
    """Lưu lại cuộc hội thoại hiện tại vào Qdrant để tham khảo trong tương lai."""
    embedding_model = get_embedding_model()
    client = get_qdrant_client()
    vector = embedding_model.embed_query(state["query"])
    products = [
        {k: p.get(k) for k in ["name_product", "price", "link_product", "link_image", "describe", "comment"]}
        for p in state.get("last_recommended_products", [])
    ]
    payload = {
        "query": state["query"],
        "response": state.get("recommendations", ""),
        "chat_history": state.get("chat_history", []),
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "products": products
    }
    client.upsert(
        collection_name="chat_memory",
        points=[{"id": str(uuid4()), "vector": vector, "payload": payload}]
    )

def retrieve_related_chats(query: str, top_k=3):
    """Lấy ra các cuộc hội thoại cũ có liên quan đến câu hỏi hiện tại."""
    embedding_model = get_embedding_model()
    client = get_qdrant_client()
    query_vector = embedding_model.embed_query(query)
    results = client.search(
        collection_name="chat_memory",
        query_vector=query_vector,
        limit=top_k,
        with_payload=True
    )
    return [r.payload for r in results]