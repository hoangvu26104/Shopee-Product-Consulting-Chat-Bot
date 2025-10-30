# graph/nodes.py

import json
import re
import time
from fuzzywuzzy import fuzz
from langchain_core.messages import HumanMessage, AIMessage
from qdrant_client.http import models

# Import từ các file khác trong project
import config
from utils import (
    get_llm, get_embedding_model, get_qdrant_client,
    retrieve_related_chats, base_prompt, generate_prompt_context, fuzzy_match
)
from graph.state import AgentState

# ============================ NODES ============================

def extract_intent_node(state: AgentState) -> AgentState:
    """Node đầu tiên: Phân tích câu hỏi của người dùng để hiểu họ muốn gì."""
    # Lấy lịch sử chat và query hiện tại
    # Gọi hàm retrieve_related_chats để lấy các hội thoại cũ liên quan làm ngữ cảnh
    # Tạo một prompt phức tạp yêu cầu LLM trả về một cấu trúc JSON
    # Gọi LLM và phân tích kết quả JSON
    # Cập nhật state với user_intent và question_type
    chat_history = state.get("chat_history", [])
    messages = []
    for msg in chat_history:
        if msg.startswith("Người dùng:"):
            messages.append(HumanMessage(content=msg.replace("Người dùng:", "").strip()))
        elif msg.startswith("Hệ thống:"):
            messages.append(AIMessage(content=msg.replace("Hệ thống:", "").strip()))

    query_text = state["query"].lower()
    related_chats = retrieve_related_chats(state["query"])
    related_context = "\n\n".join([
        f"📌 Lịch sử trước:\nNgười dùng: {c['query']}\nHệ thống: {c['response']}"
        for c in related_chats if c.get("query") and c.get("response")
    ])

    messages.append(HumanMessage(content=f"""
    Bạn là trợ lý thương mại điện tử. Dưới đây là các đoạn hội thoại liên quan trước đó:
    {related_context}

    Hãy phân tích mục đích người dùng và chỉ TRẢ VỀ KẾT QUẢ DƯỚI DẠNG JSON **duy nhất**, bằng tiếng Việt. ❌ Không viết bất kỳ giải thích, mô tả, hay chú thích nào khác.

    JSON bắt buộc phải chứa đủ các trường sau:
    - "loại_sản_phẩm": (ví dụ: "ốp điện thoại", "tai nghe", "chuột", hoặc null nếu không rõ)
    - "tên_sản_phẩm": (tên cụ thể, ví dụ: "Elough Air31", "Forter V181", hoặc null nếu không có)
    - "chất_liệu": (ví dụ: "silicon", hoặc null nếu không rõ)
    - "giá_tối_đa": (ví dụ: 500000, hoặc null nếu không rõ)
    - "tính_năng": (danh sách từ khóa, ví dụ: ["chống sốc", "bluetooth"])
    - "loại_câu_hỏi": chỉ chọn 1 trong: "recommendation", "comparison", "summary", "find_by_name"
    - "số_lượng_sản_phẩm": (số nguyên, ví dụ: 1 hoặc 3)

    **Hướng dẫn bổ sung**:
    - Nếu câu hỏi chứa tên cụ thể (ví dụ: "Elough Air31", "iPhone 15") và có dạng "Có sản phẩm ... không?", đặt tên đó vào "tên_sản_phẩm" và chọn "loại_câu_hỏi" là "find_by_name".
    - Xác định "loại_sản_phẩm" dựa trên câu hỏi hoặc tên sản phẩm:
      - Nếu chứa "ốp", "case", đặt "loại_sản_phẩm" là "ốp điện thoại".
      - Nếu chứa "tai nghe", "headphone", đặt "loại_sản_phẩm" là "tai nghe".
      - Nếu chứa "chuột", "mouse", đặt "loại_sản_phẩm" là "chuột máy tính".
    - Nếu câu hỏi chứa "so sánh" và tham chiếu "sản phẩm trên", "các sản phẩm đó", lấy "loại_sản_phẩm" và "số_lượng_sản_phẩm" từ câu hỏi trước đó trong lịch sử hội thoại. Đặt "tên_sản_phẩm" là null và "loại_câu_hỏi" là "comparison".
    - "tính_năng" chỉ bao gồm đặc điểm sản phẩm (ví dụ: "chống sốc", "bluetooth"). Không đưa thông tin như bảo hành hoặc giá vào "tính_năng".
    - Nếu không rõ, để các trường tương ứng là null.

    Câu hỏi người dùng: \"{query_text}\"
    """))

    llm = get_llm()
    response = llm.invoke(messages)

    match = re.search(r"\{.*\}", response.content, re.DOTALL)
    if not match:
        raise ValueError("Không tìm thấy JSON hợp lệ:\n" + response.content)

    try:
        vi_intent = json.loads(match.group(0))
        prev_intent = state.get("user_intent", {})  # Sử dụng dictionary rỗng làm mặc định
        vi_intent["loại_sản_phẩm"] = vi_intent.get("loại_sản_phẩm") or prev_intent.get("product_type", None)
        vi_intent["tên_sản_phẩm"] = vi_intent.get("tên_sản_phẩm") or prev_intent.get("product_name", None)
        vi_intent["chất_liệu"] = vi_intent.get("chất_liệu") or prev_intent.get("material", None)
        vi_intent["giá_tối_đa"] = vi_intent.get("giá_tối_đa") or prev_intent.get("price_max", None)
        vi_intent["tính_năng"] = list(set((vi_intent.get("tính_năng") or []) + (prev_intent.get("features") or [])))

        qtype = vi_intent.get("loại_câu_hỏi", "summary")
        if qtype not in ["recommendation", "comparison", "summary", "find_by_name"]:
            qtype = "summary"   
        prev_num_products = prev_intent.get("num_products", 1)
        vi_intent["số_lượng_sản_phẩm"] = vi_intent.get("số_lượng_sản_phẩm") or prev_num_products
        state["user_intent"] = {
            "product_type": vi_intent.get("loại_sản_phẩm"),
            "product_name": vi_intent.get("tên_sản_phẩm"),
            "material": vi_intent.get("chất_liệu"),
            "price_max": vi_intent.get("giá_tối_đa"),
            "features": vi_intent.get("tính_năng"),
            "question_type": qtype,
            "num_products": vi_intent["số_lượng_sản_phẩm"]
        }
        state["question_type"] = qtype
        state["chat_history"] = chat_history + [
            f"Người dùng: {state['query']}",
            f"Hệ thống: {response.content}"
        ]
        return state
    except json.JSONDecodeError:
        raise ValueError("JSON parse error:\n" + match.group(0))



def search_node(state: AgentState) -> AgentState:
    """Node thứ hai: Tìm kiếm trong Qdrant dựa trên câu hỏi."""
    # Lấy câu hỏi từ state['query']
    # Chuyển câu hỏi thành vector bằng model embedding
    # Gọi Qdrant để tìm 100 sản phẩm có vector gần nhất
    # Lưu kết quả vào state["results"]
    start = time.time()

    embedding_model = get_embedding_model()
    query_vector = embedding_model.embed_query(state['query'])
    client = get_qdrant_client()
    results = client.search(
        collection_name=config.COLLECTION_NAME,
        query_vector=query_vector,
        limit=100,
        with_payload=True
    )
    state["results"] = [r.payload for r in results]
    print(f"[⏱ search_node] took {time.time() - start:.2f}s")
    return state


def rerank_node(state: AgentState) -> AgentState:
    start = time.time()
    intent = state.get("user_intent", {}) or {}
    results = state["results"]
    product_name_query = intent.get("product_name")
    if product_name_query:
        # Lọc ra tất cả các kết quả có chứa tên sản phẩm
        exact_matches = [
            r for r in results 
            if product_name_query.lower() in r.get("name_product", "").lower()
        ]
        # Nếu tìm thấy kết quả khớp chính xác, chỉ sử dụng chúng
        if exact_matches:
            results = exact_matches
    # keywords = PRODUCT_KEYWORDS.get(intent.get("product_type"), DEFAULT_KEYWORDS)

    ranked = []
    for r in results:
        text = f"{r.get('name_product', '')} {r.get('describe', '')} {r.get('detail', '')}".lower()
        score = 0

        # Kiểm tra product_type
        if intent.get("product_type") and intent["product_type"] in text:
            score += 1

        # Kiểm tra material
        if intent.get("material"):
            material = intent["material"]
            if isinstance(material, str) and material in text:
                score += 1
            elif isinstance(material, list) and any(m in text for m in material):
                score += 1

        # ✅ Xử lý features nested list
        features = intent.get("features")
        if not isinstance(features, list):
            features = []

        flattened_features = []
        for f in features:
            if isinstance(f, list):
                flattened_features.extend(f)
            elif isinstance(f, str):
                flattened_features.append(f)
        if any(f.lower() in text for f in flattened_features):
            score += 1

        # Kiểm tra giá
        try:
            if intent.get("price_max") and float(r.get("price", 9999999)) <= intent["price_max"]:
                score += 1
        except:
            pass

        r["score"] = score
        ranked.append(r)

    ranked.sort(key=lambda x: x.get("score", 0), reverse=True)
    state["results"] = ranked[:10]
    print(f"[⏱ rerank_node] took {time.time() - start:.2f}s")
    return state



def generate_prompt_context(results, product_type):
    keywords = config.PRODUCT_KEYWORDS.get(product_type, config.DEFAULT_KEYWORDS)
    context_lines = []

    for idx, r in enumerate(results):
        product_name = r.get("name_product", f"Sản phẩm {idx+1}")
        comment_text = r.get("comment", "") or r.get("describe", "") or r.get("detail", "") or ""
        if not comment_text.strip():
            continue

        added = False
        for kw in keywords:
            if kw in comment_text.lower():
                context_lines.append(
                    f"Tên: {product_name}\nBình luận: {comment_text.strip()}\n"
                )
                added = True
                break
        # Fallback: nếu không khớp từ khóa mà index nhỏ → vẫn lấy
        if not added and idx < 3:
            context_lines.append(
                f"Tên: {product_name}\nBình luận: {comment_text.strip()}\n"
            )

    return "\n".join(context_lines)

def contains_partial_keywords(product_type: str, text: str) -> bool:
    if not product_type or not text:
        return False
    return all(word in text.lower() for word in product_type.lower().split())

def base_prompt(task: str, context: str, query: str):
    if not context.strip():
        return None
    return f"""
Bạn là trợ lý AI tư vấn sản phẩm. Trả lời hoàn toàn bằng tiếng Việt.

Người dùng hỏi: \"{query}\"

Dưới đây là các bình luận thật:
{context}

Nhiệm vụ:
{task}
"""


def find_by_name_node(state: AgentState) -> AgentState:
    start = time.time()  # Ghi lại thời gian bắt đầu để đo hiệu suất

    # Lấy intent từ state (nội dung yêu cầu của người dùng)
    intent = state.get("user_intent") or {}

    # Lấy tên sản phẩm cần tìm (product_name), chuyển thành chữ thường và loại bỏ khoảng trắng
    product_name = (intent.get("product_name") or "").lower().strip()
    # Lấy loại sản phẩm (nếu có), cũng chuẩn hóa tương tự
    product_type = (intent.get("product_type") or "").lower().strip()

    # Nếu không có tên sản phẩm thì phản hồi yêu cầu nhập lại
    if not product_name:
        state["recommendations"] = "Vui lòng cung cấp tên sản phẩm cụ thể để tìm kiếm."
        state["chat_history"].append(f"Hệ thống: {state['recommendations']}")
        print(f"[⏱ find_by_name_node] took {time.time() - start:.2f}s")
        return state

    # Kết nối tới Qdrant để truy xuất dữ liệu
    client = get_qdrant_client()

    # Duyệt toàn bộ 1000 sản phẩm từ collection để tìm sản phẩm trùng tên
    all_records, _ = client.scroll(
        collection_name=config.COLLECTION_NAME,
        limit=1000,  # Tăng cơ hội tìm thấy sản phẩm
        with_payload=True
    )

    matches = []  # Danh sách sản phẩm trùng khớp

    for r in all_records:
        # Lấy tên sản phẩm trong DB và chuẩn hóa
        name_in_db = (r.payload.get("name_product") or "").lower().strip()

        # Nếu tên chứa product_name hoặc độ tương đồng cao thì thêm vào kết quả
        if (product_name in name_in_db or fuzz.partial_ratio(product_name, name_in_db) > 85):
            matches.append(r.payload)

    # Nếu không tìm thấy sản phẩm nào khớp
    if not matches:
        state["recommendations"] = f"Tôi không tìm thấy sản phẩm '{product_name}' trong các kết quả liên quan."
        state["chat_history"].append(f"Hệ thống: {state['recommendations']}")
        print(f"[⏱ find_by_name_node] took {time.time() - start:.2f}s")
        return state

    # Sắp xếp danh sách khớp theo mức độ tương đồng (cao → thấp)
    matches.sort(
        key=lambda x: fuzz.ratio(product_name, x.get("name_product", "").lower()),
        reverse=True
    )

    # Lấy sản phẩm phù hợp nhất
    top_product = matches[0]

    # Tạo context để gửi vào prompt cho LLM
    context = f"Tên: {top_product.get('name_product', 'Sản phẩm')}\nBình luận: {top_product.get('comment', '') or top_product.get('describe', '') or top_product.get('detail', '')}"

    # Tạo prompt với yêu cầu rõ ràng: không bịa, chỉ dùng dữ liệu hiện có
    prompt = base_prompt(
        task="""\
- Xác nhận có sản phẩm theo yêu cầu.
- Cung cấp thông tin chi tiết (tên, giá, bình luận nếu có).
- Không thêm thông tin ngoài dữ liệu.
""",
        context=context,
        query=state["query"]
    )

    # Nếu không tạo được prompt (có thể thiếu thông tin)
    if not prompt:
        state["recommendations"] = f"Tìm thấy sản phẩm '{product_name}', nhưng không có thông tin chi tiết."
    else:
        # Gọi LLM để sinh câu trả lời dựa trên prompt
        llm = get_llm()
        response = llm.invoke(prompt)
        state["recommendations"] = response.content.strip()  # Lấy nội dung trả lời

        # Lấy các thông tin bổ sung từ sản phẩm
        name = top_product.get("name_product", "Sản phẩm")
        link = top_product.get("link_product")
        price = top_product.get("price")
        image_url = top_product.get("link_image")

        # Xử lý định dạng giá
        try:
            price_display = f"{float(price):,.0f}đ" if price else "Không rõ"
        except:
            price_display = str(price) or "Không rõ"

        # Tạo phần hiển thị chi tiết
        details_string = "\n\n🔗 **Link sản phẩm:**"
        details_string += f"\n- [{name}]({link}) — 💰 {price_display}" if link else f"\n- {name} — 💰 {price_display}"

        # Nếu có hình ảnh thì thêm đường link xem ảnh
        if image_url:
            details_string += f"\n  Hình ảnh: [Xem ảnh]({image_url})"

        # Ghép phần chi tiết vào kết quả trả lời
        state["recommendations"] += details_string

        # Cập nhật danh sách sản phẩm đã được đề cập gần đây (giữ 2 sản phẩm cuối cùng)
        last_recs = state.get("last_recommended_products", [])
        new_recs = (last_recs + [top_product])[-2:]
        state["last_recommended_products"] = new_recs

    # Lưu lại câu trả lời vào lịch sử hội thoại
    state["chat_history"].append(f"Hệ thống: {state['recommendations']}")
    print(f"[⏱ find_by_name_node] took {time.time() - start:.2f}s")
    return state


   
def recommend_node(state: AgentState) -> AgentState:
    """Node xử lý yêu cầu 'gợi ý sản phẩm'."""

    start = time.time()  # Bắt đầu đếm thời gian để log thời gian thực thi

    # Trích xuất thông tin từ intent
    intent = state.get("user_intent") or {}
    num_products = intent.get("num_products", 2)  # Số lượng sản phẩm muốn gợi ý
    product_type = (intent.get("product_type") or "").lower()  # Loại sản phẩm

    # Lọc danh sách sản phẩm phù hợp từ `state["results"]`
    filtered = [
        r for r in state["results"]
        if fuzzy_match(
            product_type,
            f"{r.get('name_product', '')} {r.get('describe', '') or ''} {r.get('detail', '') or ''}"
        )
    ]

    # Nếu không có sản phẩm phù hợp
    if not filtered:
        state["recommendations"] = (
            f"Xin lỗi, tôi không tìm thấy sản phẩm nào phù hợp với '{product_type}'. "
            "Vui lòng kiểm tra lại từ khóa hoặc thêm thông tin chi tiết hơn."
        )
        state["chat_history"].append(f"Hệ thống: {state['recommendations']}")
        print(f"[⏱ recommend_node] took {time.time() - start:.2f}s")
        return state

    # Xác định nếu có yêu cầu sắp xếp theo giá
    features = intent.get("features") or []
    sort_by_price = None
    if "giá rẻ nhất" in [f.lower() for f in features]:
        sort_by_price = "asc"
    elif "giá cao nhất" in [f.lower() for f in features]:
        sort_by_price = "desc"

    # Sắp xếp theo giá nếu có yêu cầu
    if sort_by_price:
        def safe_price(p):
            try:
                return float(p.get("price", 9999999 if sort_by_price == "asc" else 0))
            except:
                return 9999999 if sort_by_price == "asc" else 0

        sorted_products = sorted(filtered, key=safe_price, reverse=(sort_by_price == "desc"))
    else:
        sorted_products = filtered  # Không cần sắp xếp, giữ nguyên thứ tự

    # Loại bỏ sản phẩm trùng tên
    unique_products = []
    seen_names = set()
    for p in sorted_products:
        name = (p.get("name_product") or "").strip().lower()
        if name and name not in seen_names:
            p["__type"] = product_type  # Gán loại sản phẩm vào để sử dụng về sau
            unique_products.append(p)
            seen_names.add(name)

    # Chọn ra số lượng sản phẩm cần gợi ý
    top_products = unique_products[:num_products]
    if not top_products:
        state["recommendations"] = "Không có sản phẩm đủ thông tin để tư vấn."
        state["chat_history"].append(f"Hệ thống: {state['recommendations']}")
        print(f"[⏱ recommend_node] took {time.time() - start:.2f}s")
        return state

    # Lưu lại danh sách sản phẩm được gợi ý để có thể so sánh sau
    state["last_recommended_products"] = top_products

    # Tạo context chứa các sản phẩm để đưa vào prompt
    context = generate_prompt_context(top_products, product_type)

    # Tạo prompt yêu cầu LLM viết tư vấn
    prompt = base_prompt(
        task="""\
- Trả lời hoàn toàn bằng tiếng Việt.
- Tóm tắt cảm nhận chung của khách hàng về sản phẩm (bao gồm cả giá nếu có).
- Gợi ý các sản phẩm phù hợp từ danh sách trên (đúng số lượng yêu cầu).
- Nêu rõ ưu nhược điểm.
- Không cần so sánh.
- Không bịa tên hay thêm thông tin ngoài dữ liệu.
""",
        context=context,
        query=state["query"]
    )

    # Nếu không tạo được prompt, trả lời lỗi
    if not prompt:
        state["recommendations"] = "Xin lỗi, không đủ thông tin để tư vấn."
        state["chat_history"].append(f"Hệ thống: {state['recommendations']}")
        print(f"[⏱ recommend_node] took {time.time() - start:.2f}s")
        return state

    # Gọi LLM để sinh nội dung tư vấn
    llm = get_llm()
    response = llm.invoke(prompt)
    reply = response.content.strip()

    # Thêm link và giá vào phần trả lời
    reply += "\n\n🔗 **Link sản phẩm:**"
    for product in top_products:
        name = product.get("name_product", "Sản phẩm").strip()
        link = product.get("link_product", "").strip()
        price = product.get("price")
        image_url = product.get("link_image", "").strip()

        # Format giá
        try:
            price_display = f"{float(price):,.0f}đ" if price else "Không rõ"
        except:
            price_display = str(price) or "Không rõ"

        reply += f"\n- [{name}]({link}) — 💰 {price_display}"
        # Nếu muốn hiển thị hình ảnh:
        # if image_url:
        #     reply += f"\n  Hình ảnh: [Xem ảnh]({image_url})"

    # Ghi vào state kết quả cuối cùng
    state["recommendations"] = reply
    state["chat_history"].append(f"Hệ thống: {reply}")
    print(f"[⏱ recommend_node] took {time.time() - start:.2f}s")
    return state



def keyword_in_text(keyword: str, text: str) -> bool:
    words = keyword.lower().split()
    return all(w in text.lower() for w in words)
def fuzzy_match(product_type: str, text: str) -> bool:
    if not product_type or not text:
        return False
    words = product_type.lower().split()
    matched = sum(1 for w in words if w in text.lower())
    return matched >= len(words) - 1  # cho phép sai 1 từ


def compare_node(state: AgentState) -> AgentState:
    start = time.time()  # Ghi lại thời gian bắt đầu xử lý

    # Lấy thông tin từ intent
    intent = state.get("user_intent") or {}
    num_products = intent.get("num_products", 2)  # Số lượng sản phẩm cần so sánh (thường là 2)

    # Lấy loại sản phẩm (nếu có)
    product_type = (intent.get("product_type") or "").lower()

    # Lấy danh sách sản phẩm gần đây đã được gợi ý hoặc tìm ra
    previous = state.get("last_recommended_products") or []

    
    top_results = previous  # Dùng toàn bộ danh sách để so sánh
    # =========================================================

    # Nếu không có đủ 2 sản phẩm, từ chối so sánh
    if len(top_results) < 2:
        state["recommendations"] = (
            "Không đủ sản phẩm để so sánh. Vui lòng yêu cầu gợi ý hoặc tìm ít nhất 2 sản phẩm trước."
        )
        state["chat_history"].append(f"Hệ thống: {state['recommendations']}")
        print(f"[⏱ compare_node] took {time.time() - start:.2f}s")
        return state

    if not product_type and top_results:
        pass  # Giữ đơn giản, không cần suy ra

    # Tạo context chứa thông tin các sản phẩm cần so sánh
    context = generate_prompt_context(top_results, product_type)

    # Tạo prompt gửi tới LLM để yêu cầu viết so sánh
    prompt = base_prompt(
        task="""\
- So sánh các sản phẩm được cung cấp trong phần Bình luận bên dưới.
- Gọi tên sản phẩm đúng theo phần "Tên: ...".
- Phân tích ưu nhược điểm của từng sản phẩm một cách rõ ràng.
- Đưa ra kết luận nên chọn sản phẩm nào cho nhu cầu nào.
- Không bịa tên hay thêm dữ kiện không có.
""",
        context=context,
        query=state["query"]
    )

    # Nếu không tạo được prompt (thiếu dữ liệu)
    if not prompt:
        state["recommendations"] = "Không có đủ dữ liệu để so sánh."
        state["chat_history"].append(f"Hệ thống: {state['recommendations']}")
        print(f"[⏱ compare_node] took {time.time() - start:.2f}s")
        return state

    # Gọi LLM để sinh nội dung so sánh
    llm = get_llm()
    response = llm.invoke(prompt)
    reply = response.content.strip()  # Nội dung phản hồi

    # Thêm phần danh sách link sản phẩm được so sánh
    reply += "\n\n🔗 **Link các sản phẩm được so sánh:**"
    for product in top_results:
        name = product.get("name_product", "Sản phẩm").strip()
        link = product.get("link_product", "").strip()
        price = product.get("price")
        image_url = product.get("link_image", "").strip()

        # Hiển thị giá
        try:
            price_display = f"{float(price):,.0f}đ" if price else "Không rõ"
        except:
            price_display = str(price) or "Không rõ"

        reply += f"\n- [{name}]({link}) — 💰 {price_display}"
        # Nếu cần hiển thị ảnh:
        # if image_url:
        #     reply += f"\n  Hình ảnh: [Xem ảnh]({image_url})"

    # Lưu kết quả vào state
    state["recommendations"] = reply
    state["chat_history"].append(f"Hệ thống: {reply}")
    print(f"[⏱ compare_node] took {time.time() - start:.2f}s")
    return state



def summary_node(state: AgentState) -> AgentState:
    # Lấy intent từ người dùng
    intent = state.get("user_intent", {})

    # Lấy danh sách kết quả sản phẩm đã tìm được
    results = state.get("results", [])

    # Lấy loại sản phẩm và tên sản phẩm cụ thể nếu có
    product_type = intent.get("product_type", "")
    product_name_query = intent.get("product_name")

    # --- Bước 1: Tạo bản tóm tắt bằng AI từ comment ---
    # Dựa trên danh sách kết quả và loại sản phẩm đã có
    context = generate_prompt_context(results, product_type)

    # Tạo prompt yêu cầu AI tóm tắt cảm nhận khách hàng
    prompt = base_prompt("""\
- Tóm tắt cảm nhận người dùng.
- Chỉ sử dụng thông tin có trong comment.
- Giữ giọng văn tự nhiên, khách quan.
- Không bịa tên hay thêm dữ kiện không có.
""", context, state["query"])

    # Nếu không có dữ liệu để tạo prompt thì trả về luôn
    if not prompt:
        state["recommendations"] = "Không có dữ liệu đủ rõ để tóm tắt."
        return state

    # Gọi LLM để tạo bản tóm tắt
    llm = get_llm()
    response = llm.invoke(prompt)
    summary_text = response.content.strip()

    top_product = None
    if product_name_query and results:
        best_match = max(
            results, 
            key=lambda r: fuzz.ratio(product_name_query.lower(), r.get("name_product", "").lower()),
            default=None
        )

        # Nếu độ tương đồng đủ cao thì chấp nhận sản phẩm đó
        if best_match and fuzz.ratio(product_name_query.lower(), best_match.get("name_product", "").lower()) > 75:
            top_product = best_match

            # Trích xuất các thông tin chi tiết
            name = top_product.get("name_product", "Sản phẩm")
            link = top_product.get("link_product")
            price = top_product.get("price")
            image_url = top_product.get("link_image")

            # Format giá để hiển thị
            try:
                price_display = f"{float(price):,.0f}đ" if price else "Không rõ"
            except (ValueError, TypeError):
                price_display = str(price) or "Không rõ"

            # Ghép thông tin chi tiết thành chuỗi
            details_string = f"\n\n🔗 **Thông tin sản phẩm:**"
            details_string += f"\n- [{name}]({link}) — 💰 {price_display}" if link else f"\n- {name} — 💰 {price_display}"
            if image_url:
                details_string += f"\n  Hình ảnh: [Xem ảnh]({image_url})"

            # Gắn phần chi tiết vào bản tóm tắt
            summary_text += details_string

            # --- Bước 3: Cập nhật "trí nhớ" để có thể so sánh sau ---
            last_recs = state.get("last_recommended_products", [])
            new_recs = (last_recs + [top_product])[-2:]  # Giữ tối đa 2 sản phẩm gần nhất
            state["last_recommended_products"] = new_recs

    # --- Bước 4: Ghi kết quả vào state ---
    state["recommendations"] = summary_text
    state["chat_history"].append(f"Hệ thống: {summary_text}")
    
    return state


