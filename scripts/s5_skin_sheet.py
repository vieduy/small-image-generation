"""Sheet of real-photo teacher pairs vs the skin filter (2026-11-25): 400 random photos of the s5 pool -> 40 across input skin
tone (darkest first) -> teacher tB1 avatar (8 steps, CFG 3/4, phone fixed noise) -> photo | output with skin dL* (red = dL > 10,
fails the filter). Out: grids/s5_skin_filter_sheet.jpg"""
import random, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA, metrics as M, teacher_dl as TD
from eval_e1 import to_t
from s5_real_pairs import photo_list
ROOT, DEV = TD.ROOT, "cuda"
fid = M.FaceID()
ph, _ = photo_list()
cand = random.Random(7).sample(ph, 400)
sk = sorted((s, p) for p in cand if (s := fid.skin_L(Image.open(p))) is not None)
pick = [sk[int(i)] for i in np.linspace(0, len(sk) - 1, 40)]
unet, T = TD.load_teacher("tB1"); unet = unet.to(DEV).eval()
k = T["tasks"].index("avatar")
vae = DLA.load_vae(torch.float16).to(DEV).eval()
with torch.inference_mode():
    x = torch.stack([to_t(Image.open(p)) for _, p in pick]).to(DEV, torch.float16)
    outs = []
    for i in range(0, len(pick), 20):
        z = TD.sample(unet, DLA.encode(vae, x[i:i + 20]), T["emb"][k:k + 1].to(DEV, torch.float16).expand(len(x[i:i + 20]), -1, -1),
                      T["mask"][k:k + 1].to(DEV).expand(len(x[i:i + 20]), -1), T["null_emb"].to(DEV, torch.float16), T["null_mask"].to(DEV),
                      3.0, 4.0, steps=8, noise=DLA.fixed_noise(DEV, torch.float32))
        outs += [Image.fromarray(((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()) for y in DLA.decode(vae, z.half())]
c, cols = 160, 10
g = Image.new("RGB", (cols * c, 22 + 4 * (2 * c + 24)), "white"); d = ImageDraw.Draw(g)
d.text((4, 4), "Real photos (darkest skin first). Top: photo + skin L*; bottom: teacher avatar + dL* (RED = dL > 10, fails the skin filter)", fill="black")
nf = 0
for i, ((si, p), o) in enumerate(zip(pick, outs)):
    r, cc = divmod(i, cols); y0 = 22 + r * (2 * c + 24)
    so = fid.skin_L(o); dl = None if so is None else so - si
    g.paste(Image.open(p).convert("RGB").resize((c, c)), (cc * c, y0)); g.paste(o.resize((c, c)), (cc * c, y0 + c))
    bad = dl is None or dl > 10; nf += bad
    d.rectangle([cc * c, y0 + 2 * c, cc * c + c - 1, y0 + 2 * c + 22], fill=(200, 30, 30) if bad else (40, 140, 60))
    d.text((cc * c + 4, y0 + 2 * c + 5), f"L {si:.0f}  dL {'n/a' if dl is None else f'{dl:+.0f}'}", fill="white")
g.save(ROOT / "grids/s5_skin_filter_sheet.jpg", quality=88); print("fail", nf, "/", len(pick))
