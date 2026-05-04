import torch
from mamba_ssm import Mamba

# H100 check
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

# Mamba Parameters
batch, seqlen, dim = 2, 64, 128
model = Mamba(
    d_model=dim, 
    d_state=16, 
    d_conv=4, 
    expand=2
).to(device)

# Fake Input (Molecules sequence batch)
x = torch.randn(batch, seqlen, dim).to(device)
y = model(x)

print(f"Input Shape: {x.shape}")
print(f"Output Shape: {y.shape}")
if y.shape == x.shape:
    print("\n" + "!"*20)
    print("MAMBA IS ALIVE ON H100!")
    print("!"*20)