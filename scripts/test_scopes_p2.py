"""P2 / P2c scope check (2026-11-20): module counts, shared = own token reproduces the model exactly, and with a fixed shared
context the features entering the first option-reading module are identical across options (= truly shared trunk)."""
import sys, torch
sys.path.insert(0, "scripts")
import dl_arch as DLA
te = torch.load("weights/student/dl6g2/task_emb.pt"); emb = te["emb"].cuda(); mask = te["mask"].cuda()
torch.manual_seed(0)
z = torch.randn(2, 4, 64, 64, device="cuda")
for scope, probe in (("down_mid_up0", "up_blocks.1"), ("only_up0a2", "up_blocks.0.attentions.2")):
    u = DLA.load_unet("weights/student/dl6g2/unet").cuda().eval()
    k = 10
    with torch.no_grad():
        ref = DLA.forward(u, z, emb[[k, k]], mask[[k, k]])
        n = DLA.install_shared_trunk(u, emb[k:k + 1], mask[k:k + 1], scope=scope); u.cuda()
        got = DLA.forward(u, z, emb[[k, k]], mask[[k, k]])
        u.shared_emb.data = emb[0:1].clone(); u.shared_mask.data = mask[0:1].float().clone()
        f = {}
        u.get_submodule(probe).register_forward_pre_hook(lambda m, a, kw: f.__setitem__("x", (a[0] if a else kw["hidden_states"]).detach().clone()), with_kwargs=True)
        a = DLA.forward(u, z, emb[[3, 3]], mask[[3, 3]]); fa = f["x"]
        b = DLA.forward(u, z, emb[[12, 12]], mask[[12, 12]]); fb = f["x"]
    print(f"{scope}: shared modules {n} (option-reading {36 - n}) | shared=own token max diff {(got - ref).abs().max().item():.2e} | "
          f"input of {probe}: task3 vs task12 max diff {(fa - fb).abs().max().item():.2e} | output diff {(a - b).abs().max().item():.3f}")
