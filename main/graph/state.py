from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    query: str # Câu hỏi hiện tại của người dùng
    chat_history: List[str] # Lịch sử trò chuyện
    user_intent: Optional[dict] # Ý định của người dùng sau khi được AI phân tích
    question_type: Optional[str] # Loại câu hỏi (recommend, compare, find_by_name, summary)
    results: Optional[List[dict]] # Kết quả tìm kiếm thô từ Qdrant
    recommendations: Optional[str] # Câu trả lời cuối cùng được tạo ra
    last_recommended_products: Optional[List[dict]] # "Giỏ hàng" chứa các sản phẩm đã xem