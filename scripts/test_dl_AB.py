"""Exactness tests for the idea-B shared trunk (dl_arch.install_shared_trunk) and the idea-A wide latent (train_dl --wide-latent)."""
import sys, torch
sys.path.insert(0, "scripts")
import dl_arch as DLA
te = torch.load("weights/student/dl6g2/task_emb.pt"); emb = te["emb"].cuda(); mask = te["mask"].cuda()
torch.manual_seed(0)
z = torch.randn(2, 4, 64, 64, device="cuda")
u = DLA.load_unet("weights/student/dl6g2/unet").cuda().eval()
k = 10
with torch.no_grad():
    ref = DLA.forward(u, z, emb[[k, k]], mask[[k, k]])
    n = DLA.install_shared_trunk(u, emb[k:k + 1], mask[k:k + 1]); u.cuda()
    got = DLA.forward(u, z, emb[[k, k]], mask[[k, k]])
    print("B: wrapped modules", n, "| shared = own token: max diff", (got - ref).abs().max().item())
    u.shared_emb.data = emb[0:1].clone(); u.shared_mask.data = mask[0:1].float().clone()
    feats = {}
    u.mid_block.register_forward_hook(lambda m, i, o: feats.__setitem__("mid", o.detach().clone()))
    a = DLA.forward(u, z, emb[[3, 3]], mask[[3, 3]]); fa = feats["mid"]
    b = DLA.forward(u, z, emb[[12, 12]], mask[[12, 12]]); fb = feats["mid"]
    print("B: mid-block output task3 vs task12 max diff", (fa - fb).abs().max().item(), "| final output diff", (a - b).abs().max().item())
