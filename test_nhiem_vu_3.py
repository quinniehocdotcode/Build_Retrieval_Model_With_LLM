import numpy as np
import os
import faiss  # Thư viện FAISS


# ==============================================================================
# PHẦN 1: LOGIC CHÍNH CỦA NHIỆM VỤ 3
# ==============================================================================

def create_faiss_index(embedding_path: str, output_path: str):
    """
    Hàm chính thực hiện toàn bộ Nhiệm vụ 3.
    Đọc các vector embedding và xây dựng chỉ mục FAISS để tìm kiếm nhanh.
    """
    print("--- BẮT ĐẦU NHIỆM VỤ 3: XÂY DỰNG CHỈ MỤC TÌM KIẾM ---")

    # 1. Kiểm tra xem file embedding có tồn tại không
    if not os.path.exists(embedding_path):
        print(f"[LỖI] Không tìm thấy file embedding tại: {embedding_path}")
        print("Vui lòng chạy Nhiệm vụ 2 để tạo file này trước.")
        return

    # 2. Tải ma trận embedding từ file .npy
    print(f"\n[Bước 1/4] Đang tải các vector embedding từ: {embedding_path}...")
    product_embeddings = np.load(embedding_path)
    # FAISS yêu cầu dữ liệu đầu vào phải là float32
    product_embeddings = product_embeddings.astype('float32')
    print("Tải embedding thành công!")
    print(f"Kích thước ma trận: {product_embeddings.shape}")

    # 3. Khởi tạo chỉ mục FAISS
    print("\n[Bước 2/4] Đang khởi tạo chỉ mục FAISS...")
    # Lấy số chiều của vector (ví dụ: 384 với model all-MiniLM-L6-v2)
    d = product_embeddings.shape[1]

    # `IndexFlatIP`:
    # - `Index`: Đây là một chỉ mục FAISS.
    # - `Flat`: Nghĩa là tìm kiếm chính xác (brute-force), không có xấp xỉ. Với số lượng sản phẩm nhỏ, cách này vừa nhanh vừa chính xác 100%.
    # - `IP`: (Inner Product) - Phép đo tương đồng là Tích vô hướng. Vì các vector đã được chuẩn hóa (normalize) ở Nhiệm vụ 2,
    #          tích vô hướng sẽ tương đương với độ tương đồng cosine.
    index = faiss.IndexFlatIP(d)
    print(f"Đã tạo chỉ mục IndexFlatIP với {d} chiều.")

    # 4. Thêm các vector vào chỉ mục
    print("\n[Bước 3/4] Đang thêm các vector vào chỉ mục...")
    index.add(product_embeddings)
    print(f"Thêm thành công {index.ntotal} vector vào chỉ mục.")

    # 5. Lưu chỉ mục ra file
    print("\n[Bước 4/4] Đang lưu chỉ mục ra file...")
    # Tạo thư mục `index/` nếu chưa có
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    faiss.write_index(index, output_path)
    print(f"-> Đã lưu chỉ mục vào: {output_path}")

    print("\n--- HOÀN THÀNH NHIỆM VỤ 3! ---")


# ==============================================================================
# PHẦN 2: BÀI TẬP THỰC HÀNH
# ==============================================================================

if __name__ == "__main__":
    # Định nghĩa các đường dẫn
    EMBEDDING_NPY_PATH = 'embeddings/products.npy'
    OUTPUT_FAISS_PATH = 'index/products.faiss'

    # Gọi hàm chính để thực hiện nhiệm vụ
    create_faiss_index(
        embedding_path=EMBEDDING_NPY_PATH,
        output_path=OUTPUT_FAISS_PATH
    )

    # Bài tập thêm: Thử tìm kiếm với chỉ mục vừa tạo
    print("\n--- KIỂM TRA TÌM KIẾM TRÊN CHỈ MỤC VỪA TẠO ---")
    if os.path.exists(OUTPUT_FAISS_PATH):
        # Tải lại chỉ mục
        index = faiss.read_index(OUTPUT_FAISS_PATH)
        # Tải lại embeddings để lấy vector truy vấn
        embeddings = np.load(EMBEDDING_NPY_PATH).astype('float32')

        # Lấy vector của sản phẩm thứ 2 (index=1) làm vector truy vấn
        query_vector = embeddings[1:2]  # Lấy dạng [1, D] thay vì [D,]
        k = 3  # Tìm 3 sản phẩm gần nhất

        print(f"\nĐang tìm {k} sản phẩm gần nhất với sản phẩm thứ 2...")
        distances, indices = index.search(query_vector, k)

        print(f"Kết quả indices (vị trí các sản phẩm gần nhất): {indices}")
        print(f"Kết quả distances (điểm tương đồng): {distances}")
        print("Lưu ý: Sản phẩm gần nhất với chính nó phải là chính nó (vị trí 1).")
