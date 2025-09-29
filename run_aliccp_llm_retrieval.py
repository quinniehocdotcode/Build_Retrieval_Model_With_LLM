# run_aliccp_llm_retrieval.py
# Requirements:
#   pip install pandas numpy tqdm faiss-cpu sentence-transformers torch

import os, math, json, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer, InputExample, losses, datasets, evaluation
import torch

# ----------------- Prompt builders -----------------
def build_user_prompt(row: pd.Series) -> str:
    return (
        "User Profile:\n"
        f"- user_id: {row.get('user_id', 'unk')}\n"
        f"- HistCategories: {row.get('user_item_categories','unk')}\n"
        f"- HistShops: {row.get('user_item_shops','unk')}\n"
        f"- HistBrands: {row.get('user_item_brands','unk')}\n"
        f"- HistIntentions: {row.get('user_item_intentions','unk')}\n"
        "Preference: retrieve suitable items given user historical patterns."
    )

def build_item_prompt(row: pd.Series) -> str:
    return (
        "Item Metadata:\n"
        f"- item_id: {row.get('item_id','unk')}\n"
        f"- Category: {row.get('item_category','unk')}\n"
        f"- Shop: {row.get('item_shop','unk')}\n"
        f"- Brand: {row.get('item_brand','unk')}\n"
        "Goal: match relevance for likely click."
    )

# ----------------- Core utils -----------------
def encode_texts(texts, model, batch_size=512, normalize=True):
    return model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=normalize
    )

def recall_at_k(gt_items, pred_items, k=10):
    topk = pred_items[:k]
    hit = len(set(topk) & set(gt_items))
    denom = min(len(gt_items), k) if len(gt_items) else 1
    return hit / denom

def ndcg_at_k(gt_items, pred_items, k=10):
    dcg = 0.0
    for i, it in enumerate(pred_items[:k], start=1):
        if it in gt_items:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gt_items), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0

def build_faiss_index(item_embs: np.ndarray):
    d = item_embs.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine nếu đã normalize
    index.add(item_embs.astype(np.float32))
    return index

def search_topk(index, q_embs: np.ndarray, topk=10):
    D, I = index.search(q_embs.astype(np.float32), topk)
    return D, I

# ----------------- Schema/merge helpers -----------------
RENAME_MAP_GENERIC = {
    "clk": "click", "label": "click", "is_click": "click",
    "item": "item_id", "item_sku_id": "item_id",
}
ITEM_META_RENAME = {
    "cate_id": "item_category", "category_id": "item_category", "cat_id": "item_category",
    "seller_id": "item_shop", "shop_id": "item_shop",
    "brand_id": "item_brand", "brand": "item_brand",
}

def _rename_common(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in RENAME_MAP_GENERIC.items() if k in df.columns})
    df = df.rename(columns={k: v for k, v in ITEM_META_RENAME.items() if k in df.columns})
    return df

def merge_skeleton_common(skeleton_csv: str, common_csv: str) -> pd.DataFrame:
    sk = pd.read_csv(skeleton_csv)
    fe = pd.read_csv(common_csv)
    sk = _rename_common(sk)
    fe = _rename_common(fe)

    if "user_id" in fe.columns:
        on_keys = ["user_id", "item_id"]
        for k in on_keys:
            if k not in sk.columns:
                raise ValueError(f"{skeleton_csv} thiếu '{k}'")
            if k not in fe.columns:
                raise ValueError(f"{common_csv} thiếu '{k}'")
    else:
        on_keys = ["item_id"]
        if "item_id" not in fe.columns:
            raise ValueError(f"{common_csv} thiếu 'item_id'")

    df = sk.merge(fe, on=on_keys, how="left")

    for col in ["item_category", "item_shop", "item_brand"]:
        if col not in df.columns:
            df[col] = "unk"

    # ✅ Fix cột click bị mất hoặc bị rename
    click_cols = [c for c in df.columns if c.startswith("click")]
    if "click" not in df.columns:
        if len(click_cols) > 0:
            df["click"] = df[click_cols[0]]
        else:
            raise ValueError("Không tìm thấy bất kỳ cột click/clk/label nào sau merge")

    df["click"] = df["click"].astype(int)

    for col in ["user_id", "item_id"]:
        if col not in df.columns:
            raise ValueError(f"Thiếu cột {col} sau merge.")
    return df

# ----------------- Build user aggregates -----------------
def build_user_aggregates(df: pd.DataFrame, agg_mode: str = "topfreq", topn: int = 5,
                          time_col: str | None = None) -> pd.DataFrame:
    g = df.groupby("user_id", sort=False)

    if agg_mode == "lastn":
        if time_col is None or time_col not in df.columns:
            raise ValueError("agg_mode='lastn' cần time_col (vd: 'time','timestamp','time_stamp').")
        df = df.sort_values([ "user_id", time_col ])
        def lastn_join(s):
            s = s.dropna().astype(str).tail(topn)
            return ",".join(s.tolist()) if len(s) else "unk"
        user_hist = pd.DataFrame({
            "user_item_categories": g["item_category"].apply(lambda s: lastn_join(s)),
            "user_item_shops":     g["item_shop"].apply(lambda s: lastn_join(s)),
            "user_item_brands":    g["item_brand"].apply(lambda s: lastn_join(s)),
        }).reset_index()
    else:
        def top_tokens(s, n=topn):
            vc = s.dropna().astype(str).value_counts()
            return ",".join(vc.index[:n]) if not vc.empty else "unk"
        user_hist = pd.DataFrame({
            "user_item_categories": g["item_category"].apply(lambda s: top_tokens(s, topn)),
            "user_item_shops":     g["item_shop"].apply(lambda s: top_tokens(s, topn)),
            "user_item_brands":    g["item_brand"].apply(lambda s: top_tokens(s, topn)),
        }).reset_index()

    intent = g["click"].apply(lambda s: "click" if s.sum() > 0 else "view").to_frame("user_item_intentions").reset_index()
    user_hist = user_hist.merge(intent, on="user_id", how="left")
    return user_hist

# ----------------- Build training pairs for two-tower -----------------
def build_training_pairs(train_df: pd.DataFrame, agg_mode="topfreq", topn=5, time_col=None):
    # user aggregates từ TRAIN để tránh leakage
    user_hist = build_user_aggregates(train_df, agg_mode=agg_mode, topn=topn, time_col=time_col)
    items = train_df[["item_id","item_category","item_shop","item_brand"]].drop_duplicates("item_id")

    items["prompt"] = items.apply(build_item_prompt, axis=1)
    user_hist["prompt"] = user_hist.apply(build_user_prompt, axis=1)

    pos = train_df[train_df["click"] == 1][["user_id","item_id"]].drop_duplicates()

    uid2prompt = dict(zip(user_hist["user_id"], user_hist["prompt"]))
    iid2prompt = dict(zip(items["item_id"], items["prompt"]))

    ex = []
    miss_uid, miss_iid = 0, 0
    for u, i in pos.itertuples(index=False):
        up = uid2prompt.get(u); ip = iid2prompt.get(i)
        if up is None: miss_uid += 1; continue
        if ip is None: miss_iid += 1; continue
        ex.append(InputExample(texts=[up, ip]))
    return ex, user_hist, items, {"miss_uid": miss_uid, "miss_iid": miss_iid}

# ----------------- Train two-tower (shared encoder) -----------------
def train_two_tower(
    base_model: str,
    train_examples: list,
    epochs: int = 2,
    batch_size: int = 64,
    lr: float = 2e-5,
    out_model_dir: str = "model_finetuned",
    eval_examples: list | None = None
):
    from torch.utils.data import DataLoader
    from sentence_transformers import SentenceTransformer, losses, SentencesDataset

    model = SentenceTransformer(base_model)

    # ⚡ Chuyển sang DataLoader thông thường để kiểm soát rõ hơn
    train_dataset = SentencesDataset(train_examples, model)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    loss_fn = losses.MultipleNegativesRankingLoss(model)

    print(f"🧠 Training with {len(train_examples)} pairs, batch={batch_size}, epochs={epochs}")

    model.fit(
        train_objectives=[(train_loader, loss_fn)],
        epochs=epochs,
        warmup_steps=int(0.1 * len(train_loader)),
        optimizer_params={'lr': lr},
        show_progress_bar=True,
        use_amp=False,   # Tắt AMP trên CPU cho an toàn
        output_path=out_model_dir
    )

    print(f"✅ Fine-tune done, model saved at {out_model_dir}")
    return model


# ----------------- Full pipeline -----------------
def run_pipeline(
    train_skeleton: str,
    train_common: str,
    valid_skeleton: str,
    valid_common: str,
    out_dir: str = "artifacts_aliccp_llm",
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    topk: int = 10,
    agg_mode: str = "topfreq",
    agg_topn: int = 5,
    time_col: str | None = None,
    do_train: bool = False,
    epochs: int = 2,
    batch_size: int = 256,
    lr: float = 2e-5,
):
    os.makedirs(out_dir, exist_ok=True)

    print(">> Merging train...")
    train = merge_skeleton_common(train_skeleton, train_common)
    print(">> Merging valid...")
    valid = merge_skeleton_common(valid_skeleton, valid_common)

    # Catalog (train+valid)
    items_catalog = pd.concat([
        train[["item_id","item_category","item_shop","item_brand"]],
        valid[["item_id","item_category","item_shop","item_brand"]],
    ], axis=0).drop_duplicates("item_id").reset_index(drop=True)

    # Build VALID user features & GT
    print(f">> Building user aggregates on VALID: mode={agg_mode}, topn={agg_topn}")
    users_valid = build_user_aggregates(valid, agg_mode=agg_mode, topn=agg_topn, time_col=time_col)
    inter_valid = valid[valid["click"] == 1][["user_id","item_id"]]
    gt = inter_valid.groupby("user_id")["item_id"].apply(set).to_dict()

    # Build prompts for VALID and items
    items_catalog = items_catalog.copy()
    users_valid = users_valid.copy()
    items_catalog["prompt"] = items_catalog.apply(build_item_prompt, axis=1)
    users_valid["prompt"] = users_valid.apply(build_user_prompt, axis=1)

    # --------- TRAIN (optional) ----------
    model_dir = os.path.join(out_dir, "model_finetuned")
    if do_train:
        print(">> Building training pairs...")
        train_examples, user_hist_train, items_train, miss_info = build_training_pairs(
            train, agg_mode=agg_mode, topn=agg_topn, time_col=time_col
        )
        print(f">> Train pairs: {len(train_examples)} (miss_uid={miss_info['miss_uid']}, miss_iid={miss_info['miss_iid']})")

        # Tiny eval split for sanity (optional)
        eval_examples = None
        if len(train_examples) > 1000:
            eval_examples = train_examples[:1000]

        print(">> Fine-tuning two-tower (shared encoder)...")
        model = train_two_tower(
            base_model=base_model,
            train_examples=train_examples,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            out_model_dir=model_dir,
            eval_examples=eval_examples
        )
    else:
        print(f">> Skipping training. Loading base model: {base_model}")
        model = SentenceTransformer(base_model)

    # --------- ENCODE & RETRIEVE ----------
    print(">> Encoding items...")
    item_embs = encode_texts(items_catalog["prompt"].tolist(), model, batch_size=1024, normalize=True)

    print(">> Encoding valid users...")
    user_embs = encode_texts(users_valid["prompt"].tolist(), model, batch_size=1024, normalize=True)

    print(">> FAISS search...")
    index = build_faiss_index(item_embs)
    _, I = search_topk(index, user_embs, topk=topk)

    item_ids = items_catalog["item_id"].tolist()
    preds_per_user = [[item_ids[j] for j in row] for row in I]
    uids = users_valid["user_id"].tolist()

    print(">> Computing metrics...")
    recalls, ndcgs = [], []
    for uid, preds in zip(uids, preds_per_user):
        recalls.append(recall_at_k(gt.get(uid, set()), preds, k=topk))
        ndcgs.append(ndcg_at_k(gt.get(uid, set()), preds, k=topk))

    metrics = {
        "dataset": "Ali-CCP CSV (merged skeleton+common)",
        "n_items": int(len(items_catalog)),
        "n_valid_users": int(len(uids)),
        "topk": topk,
        "agg_mode": agg_mode,
        "agg_topn": agg_topn,
        "trained": bool(do_train),
        "epochs": epochs if do_train else 0,
        "batch_size": batch_size if do_train else 0,
        "lr": lr if do_train else 0.0,
        "Recall@K_mean": float(np.mean(recalls) if recalls else 0.0),
        "NDCG@K_mean": float(np.mean(ndcgs) if ndcgs else 0.0),
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # --------- SAVE ARTIFACTS ----------
    print(f">> Saving artifacts to: {out_dir}")
    np.save(os.path.join(out_dir, "item_embs.npy"), item_embs.astype(np.float32))
    np.save(os.path.join(out_dir, "user_embs_valid.npy"), user_embs.astype(np.float32))
    items_catalog[["item_id","prompt"]].to_csv(os.path.join(out_dir, "items_prompt.csv"), index=False)
    users_valid[["user_id","user_item_categories","user_item_shops","user_item_brands",
                 "user_item_intentions","prompt"]].to_csv(os.path.join(out_dir, "valid_users_prompt.csv"), index=False)
    faiss.write_index(index, os.path.join(out_dir, "faiss_ip.index"))
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # save model path (base or finetuned) for reproducibility
    with open(os.path.join(out_dir, "model_path.txt"), "w") as f:
        f.write(model_dir if do_train else base_model)

    return metrics

# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-skeleton", required=True)
    ap.add_argument("--train-common",   required=True)
    ap.add_argument("--valid-skeleton", required=True)
    ap.add_argument("--valid-common",   required=True)
    ap.add_argument("--out-dir", default="artifacts_aliccp_llm")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--agg-mode", choices=["topfreq", "lastn"], default="topfreq")
    ap.add_argument("--agg-topn", type=int, default=5)
    ap.add_argument("--time-col", default=None,
                    help="Tên cột thời gian nếu dùng --agg-mode lastn (vd: time, timestamp, time_stamp)")

    # train flags
    ap.add_argument("--train", action="store_true", help="Bật fine-tune two-tower (contrastive).")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        train_skeleton=args.train_skeleton,
        train_common=args.train_common,
        valid_skeleton=args.valid_skeleton,
        valid_common=args.valid_common,
        out_dir=args.out_dir,
        base_model=args.model,
        topk=args.topk,
        agg_mode=args.agg_mode,
        agg_topn=args.agg_topn,
        time_col=args.time_col,
        do_train=args.train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


# Zero-shot (không huấn luyện):
# python run_aliccp_llm_retrieval.py \
#   --train-skeleton sample_train/sample_skeleton_train.csv \
#   --train-common   sample_train/common_features_train.csv \
#   --valid-skeleton sample_test/sample_skeleton_test.csv \
#   --valid-common   sample_test/common_features_test.csv \
#   --out-dir artifacts_zero_shot \
#   --model sentence-transformers/all-MiniLM-L6-v2 \
#   --topk 10 --agg-mode topfreq --agg-topn 5


# Fine-tune two-tower (contrastive, in-batch negatives):
# python run_aliccp_llm_retrieval.py --train-skeleton demo_data/sample_train/sample_skeleton_train.csv --train-common   demo_data/sample_train/common_features_train.csv --valid-skeleton demo_data/sample_test/sample_skeleton_test.csv  --valid-common   demo_data/sample_test/common_features_test.csv   --out-dir artifacts_trained   --model sentence-transformers/all-MiniLM-L6-v2   --topk 10 --agg-mode topfreq --agg-topn 5    --train --epochs 2 --batch-size 256 --lr 2e-5


