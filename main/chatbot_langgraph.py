
# Import app đã được xây dựng và các hàm cần thiết
from graph.builder import app
from utils import save_chat_history_to_qdrant

if __name__ == "__main__":
    session = {"chat_history": []}
    print("Chatbot đã sẵn sàng. Nhập 'exit' để thoát.")
    while True:
        query = input("\n👤 Bạn cần gì? > ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        
        # Cập nhật session và gọi app
        session["query"] = query
        session = app.invoke(session)
        
        # Lưu lại lịch sử chat
        save_chat_history_to_qdrant(session, user_id="demo_user")

        # In kết quả
        print("\nGợi ý từ hệ thống:")
        print(session.get("recommendations", "[Không có gợi ý]"))
        
        #In lịch sử (tùy chọn để debug)
        print("\nLịch sử hội thoại:")
        for turn in session.get("chat_history", []):
            print("-", turn)
        print("-" * 50)