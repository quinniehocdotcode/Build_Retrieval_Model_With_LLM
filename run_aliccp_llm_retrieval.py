# run_aliccp_llm_retrieval_with_progressbar.py
# Yêu cầu: pip install pandas numpy tqdm faiss-cpu sentence-transformers torch

import os, json
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, InputExample, losses, datasets
import torch

# Kích hoạt tqdm cho các hàm apply của pandas
tqdm.pandas(desc="Processing Prompts")


# ----------------- Prompt builders -----------------
def build_user_prompt(row: pd.Series) -> str:
    # ... (giữ nguyên như code gốc)
    return (
        "User Profile:\n"
        f"- Mã khách hàng: {row.get('user_id', 'unk')}\n"
        f"- Tuổi: {row.get('age', 'unk')}\n"
        f"- Giới tính: {row.get('gender', 'unk')}\n"
        f"- Tình trạng hôn nhân: {row.get('marital_status', 'unk')}\n"
        f"- Số con: {row.get('num_children', 'unk')}\n"
        f"- Nơi ở: {row.get('location', 'unk')}\n"
        f"- Nghề nghiệp: {row.get('job_title', 'unk')}\n"
        f"- Thu nhập/tháng: {row.get('income_monthly', 'unk')}\n"
        "Goal: find the most suitable insurance product for this profile."
    )


def build_item_prompt(row: pd.Series) -> str:
    # ... (giữ nguyên như code gốc)
    return (
        "Insurance Product:\n"
        f"- Mã sản phẩm: {row.get('product_id', 'unk')}\n"
        f"- Tên sản phẩm: {row.get('product_name', 'unk')}\n"
        f"- Mô tả: {row.get('description', 'unk')}\n"
        "Goal: match to users who would benefit most."
    )


# ----------------- Core utils -----------------
def encode_texts(texts, model, batch_size=256, normalize=True):
    # Hàm encode gốc đã có show_progress_bar=True, chúng ta giữ nguyên
    print(f"🚀 Bắt đầu mã hóa {len(texts)} văn bản...")
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=normalize
    )
    print("✅ Mã hóa hoàn tất.")
    return embeddings


def build_faiss_index(item_embs: np.ndarray):
    d = item_embs.shape[1]
    print(f"🛠️ Bắt đầu xây dựng FAISS index với {item_embs.shape[0]} vectors (chiều: {d})...")
    index = faiss.IndexFlatIP(d)
    index.add(item_embs.astype(np.float32))
    print("✅ Xây dựng FAISS index hoàn tất.")
    return index


def search_topk(index, q_embs: np.ndarray, topk=10):
    print(f"🔍 Bắt đầu tìm kiếm top {topk} cho {q_embs.shape[0]} truy vấn...")
    D, I = index.search(q_embs.astype(np.float32), topk)
    print("✅ Tìm kiếm hoàn tất.")
    return D, I


# ----------------- Build training pairs -----------------
def build_training_pairs(users: pd.DataFrame, items: pd.DataFrame):
    """Build InputExample list for training: each (user_prompt, purchased_product_prompt)."""
    item_dict = dict(zip(items["product_id"], items["prompt"]))
    examples = []
    miss = 0
    # Thêm tqdm vào vòng lặp để theo dõi tiến trình
    print("🔄 Bắt đầu tạo cặp dữ liệu huấn luyện...")
    for _, row in tqdm(users.iterrows(), total=users.shape[0], desc="Creating Training Pairs"):
        pid = row.get("purchased_product_id")
        if pd.isna(pid) or pid not in item_dict:
            miss += 1
            continue
        u_prompt = row["prompt"]
        i_prompt = item_dict[pid]
        examples.append(InputExample(texts=[u_prompt, i_prompt]))
    print(f"✅ Tạo {len(examples)} training pairs (bỏ qua {miss} user không có product match).")
    return examples


# ----------------- Train function -----------------
from torch.utils.data import DataLoader  # <-- THÊM DÒNG NÀY


def train_two_tower(model, examples, out_dir, epochs=3, batch_size=32, lr=2e-5):
    # THAY ĐỔI QUAN TRỌNG: Sử dụng DataLoader tiêu chuẩn của PyTorch
    # thay vì NoDuplicatesDataLoader
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)

    train_loss = losses.MultipleNegativesRankingLoss(model)

    print("🏋️ Bắt đầu quá trình fine-tuning mô hình (sử dụng DataLoader tiêu chuẩn)...")

    # Hàm fit vẫn giữ nguyên, nhưng giờ nó nhận DataLoader tiêu chuẩn
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=int(len(train_dataloader) * 0.1),
        optimizer_params={'lr': lr},
        show_progress_bar=True
    )

    model.save(out_dir)
    print(f"✅ Model đã fine-tune xong và lưu tại: {out_dir}")
    return model


# ----------------- Main pipeline -----------------
def run_pipeline(
        train_skeleton: str,
        train_common: str,
        out_dir: str = "artifacts_insurance_rs",
        base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        topk: int = 10,
        do_train: bool = False,
        epochs: int = 3,
        batch_size: int = 32,
        lr: float = 2e-5,
):
    os.makedirs(out_dir, exist_ok=True)

    print("--- Bước 1: Đọc và chuẩn bị dữ liệu ---")
    users = pd.read_csv(train_skeleton)
    items = pd.read_csv(train_common)
    items = items[["product_id", "product_name", "description"]].drop_duplicates("product_id").reset_index(drop=True)

    print("\n--- Bước 2: Tạo prompts từ dữ liệu ---")
    # Sử dụng .progress_apply thay vì .apply để có thanh tiến trình
    users["prompt"] = users.progress_apply(build_user_prompt, axis=1)
    items["prompt"] = items.progress_apply(build_item_prompt, axis=1)

    print("\n--- Bước 3: Tải mô hình Sentence Transformer ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị: {device}")
    model = SentenceTransformer(base_model, device=device)

    if do_train:
        print("\n--- Bước 4: Huấn luyện (Fine-tuning) mô hình ---")
        train_pairs = build_training_pairs(users, items)
        model = train_two_tower(model, train_pairs, out_dir=os.path.join(out_dir, "finetuned_model"),
                                epochs=epochs, batch_size=batch_size, lr=lr)
    else:
        print("\n--- Bước 4: Bỏ qua huấn luyện ---")

    print("\n--- Bước 5: Mã hóa văn bản thành embeddings ---")
    item_embs = encode_texts(items["prompt"].tolist(), model, batch_size=batch_size)
    user_embs = encode_texts(users["prompt"].tolist(), model, batch_size=batch_size)

    print("\n--- Bước 6: Xây dựng Index và Tìm kiếm ---")
    index = build_faiss_index(item_embs)
    D, I = search_topk(index, user_embs, topk=topk)
    item_ids = items["product_id"].tolist()
    item_names = items["product_name"].tolist()

    print("\n--- Bước 7: Tổng hợp và lưu kết quả ---")
    recs = []
    # Thêm tqdm vào vòng lặp cuối cùng
    user_ids = users["user_id"].tolist()
    for i in tqdm(range(len(user_ids)), desc="Generating Recommendations"):
        uid = user_ids[i]
        scores = D[i]
        idxs = I[i]
        recs.append({
            "user_id": uid,
            "rec_product_names": " | ".join(item_names[j] for j in idxs),
            "scores": ",".join(f"{s:.4f}" for s in scores)
        })

    recs_df = pd.DataFrame(recs)
    output_path = os.path.join(out_dir, "recommendations.csv")
    recs_df.to_csv(output_path, index=False)

    print(f"\n✅ Hoàn tất! Kết quả gợi ý đã được lưu tại: {output_path}")
    return recs_df


if __name__ == "__main__":
    # Demo chạy: bạn có thể chạy trực tiếp file để test
    run_pipeline(
        train_skeleton="data/fake_insurance/raw_customer_and_purchase_data.csv",
        train_common="data/fake_insurance/Train_insurance_products_description.csv",
        out_dir="artifacts_insurance_rs",
        do_train=True,
        epochs=10,
        batch_size=128,  # Tăng batch size để huấn luyện và encode nhanh hơn nếu có GPU tốt
        lr=2e-5,
        topk=5
    )
