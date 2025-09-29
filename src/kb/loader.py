from requests.packages import package

# package Build_LLM_RS
import json
from typing import List, Dict, Any

    # ==============================================================================
    # PHẦN 1: CÁC HÀM CỐT LÕI (Copy từ bộ kit gốc)
    # Đây là những hàm anh cần "mổ xẻ" và hiểu rõ.
    # ==============================================================================

def load_products(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Hàm này đọc một file .jsonl (mỗi dòng là một đối tượng JSON).
    Nó trả về một danh sách (List) các sản phẩm, mỗi sản phẩm là một từ điển (Dict).
    """
    print(f"--- Bắt đầu đọc file sản phẩm từ: {jsonl_path} ---")
    items = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Loại bỏ các khoảng trắng thừa ở đầu và cuối dòng
            line = line.strip()
            # Bỏ qua các dòng trống
            if not line:
                continue
            # Chuyển đổi chuỗi JSON thành đối tượng từ điển Python
            items.append(json.loads(line))
    print(f"Đã đọc thành công {len(items)} sản phẩm.\n")
    return items


def product_text_repr(prod: Dict[str, Any]) -> str:
    """
    Đây là hàm quan trọng nhất!
    Nó nhận vào một sản phẩm (dạng từ điển) và "ghép nối" các thông tin
    quan trọng lại thành một chuỗi văn bản duy nhất.
    """
    # Lấy mô tả cơ bản từ trường "text"
    base = prod.get("text", "")
    # Tạo một danh sách chứa các thông tin meta (dữ liệu có cấu trúc)
    # Hàm .get(key, default_value) rất hữu ích để tránh lỗi nếu một trường không tồn tại.
    meta = [
        f"Loại sản phẩm là {prod.get('type', 'không rõ')}.",
        f"Độ tuổi áp dụng từ {prod.get('age_min', '?')} đến {prod.get('age_max', '?')}.",
        f"Các khu vực được bảo hiểm bao gồm {', '.join(prod.get('regions', []))}.",
    ]

    # Ghép chuỗi `base` và các chuỗi trong `meta` lại với nhau.
    # Dùng " \n " để xuống dòng, giúp mô hình có thể phân tách thông tin tốt hơn.
    return base + " \nThông tin bổ sung: " + " ".join(meta)


    # ==============================================================================
    # PHẦN 2: BÀI TẬP THỰC HÀNH
    # Đây là phần anh sẽ chạy để xem kết quả và thử nghiệm.
    # ==============================================================================

if __name__ == "__main__":
    # Đường dẫn đến file dữ liệu mẫu
    product_file_path = r'C:\Users\QUIN\Desktop\Build_LLM_RS\data\products_examples.jsonl'

    # 1. Gọi hàm để load tất cả sản phẩm từ file
    all_products = load_products(product_file_path)
    # print(len(all_products))
    # In ra sản phẩm đầu tiên để xem cấu trúc gốc của nó
    print("--- Cấu trúc dữ liệu gốc của sản phẩm đầu tiên: ---")
    # `json.dumps` giúp in từ điển ra màn hình một cách đẹp mắt
    print(json.dumps(all_products[0], indent=2, ensure_ascii=False))
    print("\n" + "=" * 50 + "\n")

    # 2. Dùng vòng lặp để xử lý từng sản phẩm và in ra kết quả
    print("--- Kết quả biểu diễn sản phẩm dưới dạng văn bản: ---")
    for i, product_data in enumerate(all_products):
        # Với mỗi sản phẩm, gọi hàm để tạo ra chuỗi văn bản đại diện
        text_representation = product_text_repr(product_data)

        print(f"SẢN PHẨM #{i + 1} (ID: {product_data.get('product_id')})")
        print("-" * 20)
        print(text_representation)
        print("\n")

    # ==============================================================================
    # PHẦN 3: GỢI Ý THỬ NGHIỆM THÊM
    # Anh có thể bỏ comment các dòng dưới đây để thực hành.
    # ==============================================================================

    # print("--- Thử nghiệm với sản phẩm tự tạo: ---")
    # my_new_product = {
    #     "product_id": "KID-CARE-2025",
    #     "type": "health_for_kids",
    #     "age_min": 0,
    #     "age_max": 17,
    #     "regions": ["VN"],
    #     "text": "Bảo hiểm sức khỏe cho bé yêu, bảo vệ toàn diện từ những năm tháng đầu đời."
    #     # Trường "benefits" và "exclusions" bị thiếu, hàm get() sẽ xử lý được.
    # }
    # new_text_repr = product_text_repr(my_new_product)
    # print(new_text_repr)

