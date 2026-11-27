"""Why are tS1/tS2 less cartoon3d than dl6g2? (2026-11-25, user question). Compare the two TRAINING TARGETS for the same rows:
klein target (latents_dl_manifest_v2.pt zt, what dl6g2 learned) vs teacher tB1 target (teacher_tB1_s8.pt zt, what tS1/tS2 learned),
both decoded with the same TinyVAE, scored with eval_quality's style metric (DINOv2 / CLIP-L cosine to mean of styles/cartoon3d refs).
Random 600 avatar rows (seed 0), split synthetic pools (c/f/r) vs FFHQ (q). Out: results/style_targets_klein_vs_tB1.json,
grids/style_targets_klein_vs_tB1.jpg (photo | klein | teacher)."""
import json, random, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA, metrics as M
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/pairs/cartoon3d"
C = torch.load(D / "latents_dl_manifest_v2.pt", mmap=True)
TL = torch.load(D / "teacher_tB1_s8.pt", mmap=True)
assert TL["key"] == C["key"]
av = [i for i, k in enumerate(C["key"]) if k[1] == "avatar"]
idx = sorted(random.Random(0).sample(av, 600))
vae = DLA.load_vae(torch.float32).to("cuda").eval()
def dec(Z):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 50):
            y = DLA.decode(vae, Z[idx[i:i + 50]].to("cuda").float()).clamp(-1, 1)
            out += [Image.fromarray(a) for a in ((y + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()]
    return out
K, T = dec(C["zt"]), dec(TL["zt"])
emb = M.Embedder()
rd, rc = emb([Image.open(p) for p in sorted((ROOT / "styles/cartoon3d").glob("ref_*.png"))])
rdm, rcm = [torch.nn.functional.normalize(x.mean(0), dim=0) for x in (rd, rc)]
def score(imgs):
    d, c = zip(*[emb(imgs[i:i + 50]) for i in range(0, len(imgs), 50)])
    return (torch.cat(d) @ rdm).cpu().numpy(), (torch.cat(c) @ rcm).cpu().numpy()
kd, kc = score(K); td, tc = score(T)
q = np.array([C["key"][i][0].startswith("q") for i in idx])
res = {}
for name, sel in (("synthetic_cfr", ~q), ("ffhq_q", q), ("all", np.ones_like(q))):
    res[name] = {"n": int(sel.sum()), "klein_style_clip": round(float(kc[sel].mean()), 3), "teacher_style_clip": round(float(tc[sel].mean()), 3),
                 "klein_style_dino": round(float(kd[sel].mean()), 3), "teacher_style_dino": round(float(td[sel].mean()), 3),
                 "teacher_lower_clip_frac": round(float((tc[sel] < kc[sel]).mean()), 3)}
(ROOT / "results/style_targets_klein_vs_tB1.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
src = lambda pid: ROOT / "data/ffhq_norm" / f"{pid}.png" if pid.startswith("q") else D / "inputs" / f"{pid}.png"
order = np.argsort(tc - kc)  # biggest teacher drop first
pick = [j for j in order if not q[j]][:5] + [j for j in order if q[j]][:1] + [j for j in order if not q[j]][len(order) // 3:][:2]
c = 192
g = Image.new("RGB", (3 * c, len(pick) * (c + 16) + 20), "white"); d = ImageDraw.Draw(g)
d.text((4, 4), "TRAINING TARGETS, same row: photo | klein target (dl6g2 learned) | teacher tB1 target (tS1/tS2 learned)", fill="black")
for r, j in enumerate(pick):
    y = 20 + r * (c + 16); pid = C["key"][idx[j]][0]
    for k, (im, cap) in enumerate(((Image.open(src(pid)), f"photo {pid}"), (K[j], f"klein  style {kc[j]:.2f}"), (T[j], f"teacher  style {tc[j]:.2f}"))):
        g.paste(im.convert("RGB").resize((c, c)), (k * c, y)); d.text((k * c + 3, y + c + 2), cap, fill="black")
g.save(ROOT / "grids/style_targets_klein_vs_tB1.jpg", quality=90)
