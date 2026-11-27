"""Picture for scripts/s7_guidance_sweep.py (its own sheet only showed the qualified settings): 5 example rows x key settings."""
import sys, json, torch
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA, teacher_dl as TD
from s7_guidance_sweep import sample_gs, EXAMPLE_IDS, tex
ROOT = TD.ROOT; D = ROOT / "data/pairs/cartoon3d"
R = json.loads((ROOT / "results/teacher/tB1_guidance_sweep.json").read_text())["grid"]
C = torch.load(D / "latents_dl_manifest_v2.pt", mmap=True)
idx = [next(i for i, k in enumerate(C["key"]) if k[0] == p and k[1] == "avatar") for p in EXAMPLE_IDS]
unet, T = TD.load_teacher("tB1"); unet = unet.to("cuda").eval()
vae = DLA.load_vae(torch.float16).to("cuda").eval(); noise = DLA.fixed_noise("cuda", torch.float32)
z = C["zi"][idx].to("cuda", torch.float16); k = torch.full((len(idx),), T["tasks"].index("avatar"), device="cuda")
e, m = T["emb"].to("cuda", torch.float16)[k], T["mask"].to("cuda")[k]
ne, nm = T["null_emb"].to("cuda", torch.float16), T["null_mask"].to("cuda")
cols = ["base_img1", "base_img3_txt4", "int_img2.0_k2", "int_img2.0_k5", "apg_img3", "apg_img4_k4"]
pil = lambda y: [Image.fromarray(a) for a in ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()]
out = {}
with torch.inference_mode():
    K = pil(DLA.decode(vae, C["zt"][idx].to("cuda", torch.float16)))
    for c in cols:
        r = R[c]; out[c] = pil(DLA.decode(vae, sample_gs(unet, z, e, m, ne, nm, r["g_img"], r["k_img"], r["g_txt"], r["apg"], 8, noise).half()))
s = 200; g = Image.new("RGB", ((2 + len(cols)) * s, 34 + len(idx) * (s + 16)), "white"); d = ImageDraw.Draw(g)
d.text((4, 4), "teacher tB1, 8 steps, fixed noise. Header: sweep means (style / tex / likeness / expr) on 100+50 tune avatars; klein style 0.774 tex 14.6 id 0.298", fill="black")
for j, c in enumerate(cols):
    r = R[c]; d.text(((2 + j) * s + 3, 18), f"{c}: {r['style_clip_syn']:.2f}/{r['tex_syn']:.0f}/{r['likeness_syn']:.2f}/{r['expr_all']:.2f}", fill="black")
for i, p in enumerate(EXAMPLE_IDS):
    y = 34 + i * (s + 16)
    cells = [(Image.open(D / "inputs" / f"{p}.png"), "photo"), (K[i], f"klein tex {tex(K[i]):.0f}")] + [(out[c][i], f"{c} tex {tex(out[c][i]):.0f}") for c in cols]
    for j, (im, cap) in enumerate(cells):
        g.paste(im.convert("RGB").resize((s, s)), (j * s, y)); d.text((j * s + 3, y + s + 2), cap, fill="black")
g.save(ROOT / "grids/teacher/tB1_guidance_sweep_examples.jpg", quality=90); print("[sheet] ok", g.size)
