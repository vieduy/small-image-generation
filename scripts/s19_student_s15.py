"""Render a student on the s15 ghost-test inputs (2026-11-27, tS6): same 100 eval faces x thumbs / wave / ok / think, base + 4 small
variants + mirror, phone path (fixed noise, decg2) -> outputs/s15_ghost/<student>/<variant>/<face>_<opt>.png, for scripts/s18_owl_hands.py
+ s18_ghost_rate.py.   python scripts/s19_student_s15.py tS6"""
import sys
from pathlib import Path
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
from eval_e1 import to_t  # noqa: E402
from s15_ghost_test import OPTS, OUT, VARIANTS, pil, variant  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sname = sys.argv[1]
faces = sorted((ROOT / "data/faces").glob("*.png"))[:100]
names = [p.stem for p in faces]
jobs = [(i, o) for i in range(len(names)) for o in OPTS]
enc = DLA.load_vae(torch.float16).to("cuda").eval()
dec = DLA.load_vae(torch.float16, ROOT / "weights/decoder/decg2").to("cuda").eval()
w = ROOT / "weights/student" / sname
su = DLA.load_unet(w / "unet").to("cuda", torch.float16).eval()
te = torch.load(w / "task_emb.pt")
se, sm, tasks = te["emb"].to("cuda", torch.float16), te["mask"].to("cuda"), te["tasks"]
ks = torch.tensor([tasks.index(f"expr_{o}") for _, o in jobs], device="cuda")
ji = torch.tensor([i for i, _ in jobs], device="cuda")
with torch.inference_mode():
    for vn, v in VARIANTS.items():
        xs = torch.stack([to_t(variant(Image.open(p), v)) for p in faces]).to("cuda", torch.float16)
        Z = torch.cat([DLA.encode(enc, xs[i:i + 50]) for i in range(0, len(xs), 50)])
        d = OUT / sname / vn
        d.mkdir(parents=True, exist_ok=True)
        for s in range(0, len(jobs), 32):
            ims = pil(DLA.decode(dec, DLA.forward(su, Z[ji[s:s + 32]], se[ks[s:s + 32]], sm[ks[s:s + 32]])))
            for (i, o), im in zip(jobs[s:s + 32], ims):
                im.save(d / f"{names[i]}_{o}.png")
        print(f"[s19r] {sname} {vn} done", flush=True)
