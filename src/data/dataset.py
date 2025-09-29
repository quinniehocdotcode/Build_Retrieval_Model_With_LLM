# File: src/data/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np


class DemoUserDataset(Dataset):
    """
    Dataset giả lập để train nhanh khi chưa có logs.
    Tạo ra các cặp (hồ sơ người dùng, nhu cầu sản phẩm) dựa trên một vài quy tắc đơn giản.
    """

    def __init__(self, n=2000, cats_cardinals=(2, 64, 4), num_num=3, d_need=6):
        super().__init__()
        self.n = n

        # Tạo dữ liệu categorical giả lập
        self.Xc = np.column_stack([
            np.random.randint(0, cats_cardinals[0], size=n),  # gender
            np.random.randint(0, cats_cardinals[1], size=n),  # city
            np.random.randint(0, cats_cardinals[2], size=n),  # marital status
        ]).astype('int64')

        # Tạo dữ liệu numerical giả lập
        age = np.random.randint(18, 66, size=n)
        deps = np.random.randint(0, 4, size=n)  # dependents
        mob = np.random.randint(0, 11, size=n)  # mobility score
        self.Xn = np.stack([age, deps, mob], axis=1).astype('float32')

        # Tạo nhãn nhu cầu (y) dựa trên các quy tắc giả
        self.y_need = np.zeros((n, d_need), dtype='float32')
        for i in range(n):
            a, d = age[i], deps[i]
            if a <= 28 and d == 0: self.y_need[i, 1] = 1  # accident
            if 25 <= a <= 40 and d >= 1: self.y_need[i, 0] = 1; self.y_need[i, 2] = 1  # health, term_life
            if a >= 45: self.y_need[i, 3] = 1  # critical_illness
            if mob[i] >= 8: self.y_need[i, 5] = 1  # travel

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.Xc[idx]),
            torch.from_numpy(self.Xn[idx]),
            torch.from_numpy(self.y_need[idx]),
        )
