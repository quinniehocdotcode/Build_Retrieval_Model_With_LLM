# File: src/models/tab_user_encoder.py
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class TabUserEncoder(nn.Module):
    """Mô hình chính để mã hóa dữ liệu người dùng dạng bảng."""

    def __init__(self, cats_cardinals: List[int], num_num: int, d: int = 256, dropout: float = 0.3):
        super().__init__()
        # Tạo một lớp Embedding cho mỗi trường categorical
        self.cat_embs = nn.ModuleList([nn.Embedding(c, 32) for c in cats_cardinals])

        # Một MLP nhỏ để xử lý các trường numerical
        self.num_proj = nn.Sequential(
            nn.LayerNorm(num_num),
            nn.Linear(num_num, 64),
            nn.ReLU(inplace=True)
        )

        # Kích thước đầu vào cho MLP cuối cùng
        in_dim = 32 * len(cats_cardinals) + 64

        # MLP cuối cùng để trộn thông tin và tạo ra embedding
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d, d)
        )
        self.out_norm = nn.LayerNorm(d)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        # x_cat: (Batch, Số_trường_categorical)
        # x_num: (Batch, Số_trường_numerical)

        # Lấy embedding cho từng trường categorical
        e = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)]
        # Xử lý các trường numerical
        e.append(self.num_proj(x_num))

        # Ghép nối tất cả các vector lại
        h = torch.cat(e, dim=-1)

        # Đưa qua MLP cuối cùng để tạo user embedding
        u = self.mlp(h)

        # Chuẩn hóa vector đầu ra
        return F.normalize(self.out_norm(u), dim=-1)


class NeedHead(nn.Module):
    """Đầu ra phụ (head) để dự đoán nhu cầu của người dùng."""

    def __init__(self, d: int, k: int):
        super().__init__()
        self.fc = nn.Linear(d, k)

    def forward(self, u: torch.Tensor):
        # Trả về logits cho bài toán phân loại đa nhãn
        return self.fc(u)

# ZHead và các hàm loss khác sẽ được thêm vào sau nếu cần
