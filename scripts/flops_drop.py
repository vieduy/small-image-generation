"""#5: UNet GFLOPs per image and per 22-option set for hhB0 with/without removed cross-attention (one 512 px diptych)."""
import sys, json, torch
sys.path.insert(0, "scripts")
import dl_arch as DLA
from torch.utils.flop_counter import FlopCounterMode
u = DLA.load_unet("weights/student/hhB0/unet").cuda().eval()
te = torch.load("weights/student/hhB0/task_emb.pt"); e, m = te["emb"][:1].cuda(), te["mask"][:1].cuda()
z = torch.zeros(1, 4, DLA.LAT, DLA.LAT, device="cuda")
def gf():
    marks = {}
    fc = FlopCounterMode(display=False)
    h = u.up_blocks[0].register_forward_pre_hook(lambda *_: marks.__setitem__("trunk", fc.get_total_flops()))
    with torch.no_grad(), fc:
        DLA.forward(u, z, e, m)
    h.remove()
    return fc.get_total_flops() / 1e9, marks["trunk"] / 1e9
full, trunk = gf()
n = DLA.install_drop_attn2(u, sys.argv[1].split(","))
full2, trunk2 = gf()
r = {"groups": sys.argv[1], "modules": n, "hhB0": {"image": round(full, 1), "trunk": round(trunk, 1), "head": round(full - trunk, 1),
     "set22": round(trunk + 22 * (full - trunk), 1)},
     "dropped": {"image": round(full2, 1), "trunk": round(trunk2, 1), "head": round(full2 - trunk2, 1), "set22": round(trunk2 + 22 * (full2 - trunk2), 1)}}
r["saving_image_pct"] = round(100 * (1 - full2 / full), 1); r["saving_set_pct"] = round(100 * (1 - r["dropped"]["set22"] / r["hhB0"]["set22"]), 1)
print(json.dumps(r)); open(f"results/flops_drop_{sys.argv[1].replace(',', '_')}.json", "w").write(json.dumps(r, indent=1))
