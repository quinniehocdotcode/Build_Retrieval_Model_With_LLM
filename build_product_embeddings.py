import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

def load_product(jsonl_path:str)->List[Dict[str, Any]]:
     items = []
     with open(jsonl_path,'r',encoding='utf8') as f:
         for line in f:
             line = line.strip()
             if not line: continue
             items.append(json.loads(line))
     return items
def product_text_repr(prod: Dict[str, Any]) -> str:
    base = prod.get("text", "")
    meta = [
        f"Loại sản phẩm là {prod.get('type', 'không rõ')}.",
        f"Độ tuổi áp dụng từ {prod.get('age_min', '?')} đến {prod.get('age_max', '?')}.",
        f"Các khu vực được bảo hiểm bao gồm {', '.join(prod.get('regions', []))}.",
    ]
    return base + " \nThông tin bổ sung: " + " ".join(meta)

def create_and_save_product_embeddings(in_jsonl:str, out_npy:str, out_ids:str, encoder_name:str):
    print("")
    products = load_product(in_jsonl)
    product_texts = [product_text_repr(p) for p in products]
    product_ids = [p['product_id'] for p in products]

    model = SentenceTransformer(encoder_name)
    product_embeddings = model.encode(
        product_texts,
        normalize_embeddings=True,
        show_progress_bar=True,  # <--- ĐÃ SỬA LẠI ĐÚNG
        convert_to_numpy=True
    )

    np.save(out_npy, product_embeddings)
    with open(out_ids, 'w', encoding='utf-8') as f:
        json.dump(product_ids, f, ensure_ascii=False, indent=2)
if __name__ == "__main__":
    # Định nghĩa các đường dẫn và tên mô hình
    # Anh có thể thay đổi các giá trị này để thử nghiệm
    INPUT_JSONL_PATH = 'data/products_examples.jsonl'
    OUTPUT_NPY_PATH = 'embeddings/products.npy'
    OUTPUT_IDS_PATH = 'embeddings/product_ids.json'
    ENCODER_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2' # Đây là model trong kit gốc

    # Gọi hàm chính để thực hiện nhiệm vụ
    create_and_save_product_embeddings(
        in_jsonl=INPUT_JSONL_PATH,
        out_npy=OUTPUT_NPY_PATH,
        out_ids=OUTPUT_IDS_PATH,
        encoder_name=ENCODER_MODEL_NAME
    )

    # Bài tập thêm: Đọc lại file vừa lưu để kiểm tra
    print("\n--- KIỂM TRA LẠI FILE ĐÃ LƯU ---")
    # Đọc file .npy
    loaded_embeddings = np.load(OUTPUT_NPY_PATH)
    print(f"Đọc lại file .npy, kích thước: {loaded_embeddings.shape}")
    # Đọc file .json
    with open(OUTPUT_IDS_PATH, 'r') as f:
        loaded_ids = json.load(f)
    print(f"Đọc lại file .json, có {len(loaded_ids)} IDs.")
    print(f"ID của sản phẩm đầu tiên là: {loaded_ids[0]}")
    print("Vector embedding của sản phẩm đầu tiên là (chỉ in 5 giá trị đầu):")
    print(loaded_embeddings[0, :5])