import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Đang train trên: {device.upper()}")