import pandas as pd
import numpy as np
import os

np.random.seed(42)

def make_demo_csv(path_skeleton, path_common, n_rows=10_000, n_users=200, n_items=500):
    os.makedirs(os.path.dirname(path_skeleton), exist_ok=True)

    users = [f"U{u:04d}" for u in range(1, n_users+1)]
    items = [f"I{i:04d}" for i in range(1, n_items+1)]
    cats  = [f"C{i:02d}" for i in range(1, 20)]
    shops = [f"S{i:02d}" for i in range(1, 30)]
    brands= [f"B{i:02d}" for i in range(1, 30)]

    data = []
    for _ in range(n_rows):
        u = np.random.choice(users)
        i = np.random.choice(items)
        c = np.random.choice(cats)
        s = np.random.choice(shops)
        b = np.random.choice(brands)
        click = np.random.choice([0, 1], p=[0.85, 0.15])  # 15% click
        data.append([u, i, click, c, s, b])

    df = pd.DataFrame(data, columns=[
        "user_id", "item_id", "click", "item_category", "item_shop", "item_brand"
    ])
    df.to_csv(path_skeleton, index=False)
    df.to_csv(path_common, index=False)
    print(f"✅ Created: {path_skeleton} ({len(df)}) & {path_common} ({len(df)})")

make_demo_csv("demo_data/sample_train/sample_skeleton_train.csv",
              "demo_data/sample_train/common_features_train.csv",
              n_rows=10_000)

make_demo_csv("demo_data/sample_test/sample_skeleton_test.csv",
              "demo_data/sample_test/common_features_test.csv",
              n_rows=10_000)
