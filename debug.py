import torch

x = torch.randn(5, 5, device="cuda", dtype=torch.float16)
out = torch.empty(5, dtype=torch.float, device="cuda")

with torch.amp.autocast("cuda", torch.float16):
    torch.sum(x, dim=0, out=out)

print(out.dtype)
print(out)
print(x.sum(dim=0))

x += torch.randn(5, 5, device="cuda")
