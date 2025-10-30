from langgraph.graph import StateGraph
from graph.state import AgentState
from graph.nodes import (
    extract_intent_node,
    search_node,
    rerank_node,
    recommend_node,
    compare_node,
    summary_node,
    find_by_name_node,
)

# Khởi tạo một StateGraph mới, nói cho nó biết cấu trúc state là AgentState
graph = StateGraph(AgentState)

# Thêm từng hàm vào graph như một "trạm xử lý" có tên
graph.add_node("extract_intent", extract_intent_node)
graph.add_node("search", search_node)
graph.add_node("rerank", rerank_node)
graph.add_node("recommend", recommend_node)
graph.add_node("compare", compare_node)
graph.add_node("summary", summary_node)
graph.add_node("find_by_name", find_by_name_node)

# Thêm các cạnh kết nối cố định (ví dụ: sau khi extract_intent luôn là search)
graph.set_entry_point("extract_intent")

# Thêm các cạnh kết nối
graph.add_edge("extract_intent", "search")
graph.add_edge("search", "rerank")

# Thêm cạnh điều kiện: rẽ nhánh dựa trên kết quả của `question_type`
graph.add_conditional_edges("rerank", lambda s: s["question_type"], {
    "recommendation": "recommend",
    "comparison": "compare",
    "summary": "summary",
    "find_by_name": "find_by_name"
})

# Đặt các điểm mà luồng xử lý có thể kết thúc
graph.set_finish_point("recommend")
graph.set_finish_point("compare")
graph.set_finish_point("summary")
graph.set_finish_point("find_by_name")

# "Biên dịch" tất cả các cài đặt trên thành một ứng dụng `app` có thể chạy được
app = graph.compile()