"""CPU check of dl_arch.forward multi-step (2026-11-18): 1-step path identical to the pre-change code
(scripts/_dl_arch_1step_ref.py), k=1 of a 2-step student == 1-step, 2-step finite, gradient reaches the last step."""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as N, _dl_arch_1step_ref as O
torch.manual_seed(0)
u = N.load_unet(Path(__file__).resolve().parent.parent / "weights/student/dl2/unet").eval()
te = torch.load(Path(__file__).resolve().parent.parent / "weights/student/dl2/task_emb.pt")
z = torch.randn(1, 4, 64, 64); e = te["emb"][0:1].float(); m = te["mask"][0:1]
with torch.no_grad():
    a = O.forward(u, z, e, m); b = N.forward(u, z, e, m); c = N.forward(u, z, e, m, sigmas=[1.0, 0.5], k=1)
    d = N.forward(u, z, e, m, sigmas=[1.0, 0.5])
print("1-step new vs old max diff", (a - b).abs().max().item(), "| k=1 of 2-step vs old", (a - c).abs().max().item())
print("2-step vs 1-step mean abs diff", round((d - a).abs().mean().item(), 4), "finite", torch.isfinite(d).all().item())
for p in u.parameters():
    p.requires_grad_(False)
w = [p for n, p in u.named_parameters() if "conv_out" in n]
for p in w:
    p.requires_grad_(True)
N.forward(u, z, e, m, sigmas=[1.0, 0.5], k=2).float().pow(2).mean().backward()
print("grad on last step ok:", all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in w))
print("TEST_DONE")
