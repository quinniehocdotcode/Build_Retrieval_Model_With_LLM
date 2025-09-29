# File: src/train_user_encoder.py
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from accelerate import Accelerator

# Import các module đã tạo
from src.models.tab_user_encoder import TabUserEncoder, NeedHead
from src.data.dataset import DemoUserDataset

# ==============================================================================
# ĐẶT TẤT CẢ CODE THỰC THI VÀO BÊN TRONG KHỐI NÀY
# ==============================================================================
if __name__ == '__main__':
    print("--- BẮT ĐẦU NHIỆM VỤ 4: HUẤN LUYỆN BỘ MÃ HÓA NGƯỜI DÙNG ---")

    # --- 1. Cấu hình ---
    EPOCHS = 10
    BATCH_SIZE = 128
    LEARNING_RATE = 1.5e-3
    MODEL_DIM = 256
    DROPOUT = 0.3
    SAVE_PATH = 'embeddings/user_encoder.pt'
    NUM_NEED_CLASSES = 6  # health, accident, term_life, critical_illness, property, travel

    # Sử dụng Accelerator để dễ dàng huấn luyện trên GPU và dùng mixed precision
    accelerator = Accelerator(mixed_precision='fp16')

    # --- 2. Chuẩn bị Dữ liệu ---
    print("\n[Bước 1/4] Đang tạo dữ liệu huấn luyện giả lập...")
    train_ds = DemoUserDataset(n=4000, d_need=NUM_NEED_CLASSES)
    val_ds = DemoUserDataset(n=800, d_need=NUM_NEED_CLASSES)

    # Chỗ này là nguyên nhân gây lỗi trên Windows nếu không có if __name__ == '__main__'
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print("Tạo dữ liệu thành công!")

    # --- 3. Khởi tạo Mô hình, Loss, Optimizer ---
    print("\n[Bước 2/4] Đang khởi tạo mô hình...")
    model = TabUserEncoder(cats_cardinals=[2, 64, 4], num_num=3, d=MODEL_DIM, dropout=DROPOUT)
    need_head = NeedHead(d=MODEL_DIM, k=NUM_NEED_CLASSES)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(need_head.parameters()), lr=LEARNING_RATE)
    print("Khởi tạo mô hình thành công!")

    # --- 4. Đưa mọi thứ cho Accelerator chuẩn bị ---
    model, need_head, optimizer, train_loader, val_loader = accelerator.prepare(
        model, need_head, optimizer, train_loader, val_loader
    )

    # --- 5. Vòng lặp Huấn luyện ---
    print("\n[Bước 3/4] Bắt đầu vòng lặp huấn luyện...")
    best_val_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        need_head.train()

        total_train_loss = 0
        for x_cat, x_num, y_need in train_loader:
            user_embedding = model(x_cat, x_num)
            need_logits = need_head(user_embedding)
            loss = F.binary_cross_entropy_with_logits(need_logits, y_need)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Vòng lặp Đánh giá (Validation) ---
        model.eval()
        need_head.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_cat, x_num, y_need in val_loader:
                user_embedding = model(x_cat, x_num)
                need_logits = need_head(user_embedding)
                loss = F.binary_cross_entropy_with_logits(need_logits, y_need)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        accelerator.print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_need_head = accelerator.unwrap_model(need_head)

            cpu_state = {
                'model': {k: v.cpu() for k, v in unwrapped_model.state_dict().items()},
                'need_head': {k: v.cpu() for k, v in unwrapped_need_head.state_dict().items()},
            }
            torch.save(cpu_state, SAVE_PATH)
            accelerator.print(f"-> Val loss cải thiện. Đã lưu mô hình tốt nhất vào: {SAVE_PATH}")

    print("\n[Bước 4/4] Huấn luyện hoàn tất!")
    print("\n--- HOÀN THÀNH NHIỆM VỤ 4! ---")
