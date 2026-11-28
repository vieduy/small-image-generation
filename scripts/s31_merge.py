"""Merge best-of-N shards (scripts/s31_best_of_n.py) -> teacher_tB1_s8_img3.0_k1_bon.pt (all rows; chosen gesture latents replaced) +
_bon_drop.json (= consistency drop list + rows with an undecided view) + _bon.json (per-option counts) + grids/s31_bon_examples.jpg.
Exit 1 (gate) if more than 25% of the best-of-N rows are dropped."""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/pairs/cartoon3d"
NAME = "teacher_tB1_s8_img3.0_k1"
TP = torch.load(D / f"{NAME}.pt")
zt, ztf = TP["zt"].clone(), TP["zt_f"].clone()
keys = [tuple(k) for k in TP["key"]]
cnt = defaultdict(lambda: defaultdict(int))
drop, ex = [], []
for f in sorted((D / "bon_tB1").glob("shard_*.pt")):
    S = torch.load(f)
    for m, i in enumerate(S["rows"]):
        o = keys[i][1][5:]
        ch = S["choice"][m].tolist()
        cnt[o]["rows"] += 1
        for v, c in enumerate(ch):
            cnt[o][f"{'plain' if v == 0 else 'flip'}_{'kept' if c == 0 else 'replaced' if c > 0 else 'undecided'}"] += 1
            if c > 0:
                (zt if v == 0 else ztf)[i] = S["new"][(i, v)].to(zt.dtype)
                if len(ex) < 8 and v == 0:
                    ex.append((i, S["stats"][m][0][0], S["stats"][m][0][c]))
        if min(ch) < 0:
            cnt[o]["dropped"] += 1
            drop.append([keys[i][0], keys[i][1]])
n = sum(v["rows"] for v in cnt.values())
rate = len(drop) / max(n, 1)
torch.save({"key": TP["key"], "zt": zt, "zt_f": ztf, "setting": dict(TP.get("setting", {})) | {"best_of_n": "s31"}, "noise_seed": TP.get("noise_seed")},
           D / f"{NAME}_bon.pt")
base = json.loads((D / f"{NAME}_consistency_drop.json").read_text())
have = {tuple(x) for x in base}
(D / f"{NAME}_bon_drop.json").write_text(json.dumps(base + [x for x in drop if tuple(x) not in have]))
out = {"rows": n, "dropped": len(drop), "drop_rate": round(rate, 3), "per_option": {k: dict(v) for k, v in cnt.items()}}
(D / f"{NAME}_bon.json").write_text(json.dumps(out, indent=1))
print("[s31m] " + json.dumps(out), flush=True)
vae = DLA.load_vae(torch.float16).to("cuda").eval()
c = 220
g = Image.new("RGB", (4 * c, 20 + 4 * (c + 14)), "white")
d = ImageDraw.Draw(g)
d.text((4, 4), "best-of-N: original target | chosen replacement (plain view); caption = hands / faint / CLIP p(correct)", fill="black")
with torch.inference_mode():
    for k, (i, s0, s1) in enumerate(ex):
        a = ((DLA.decode(vae, torch.stack([TP["zt"][i], zt[i]]).to("cuda", torch.float16)).float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        x0, y = (k % 2) * 2 * c, 20 + (k // 2) * (c + 14)
        for j, (im, s) in enumerate(zip(a, (s0, s1))):
            g.paste(Image.fromarray(im).resize((c, c)), (x0 + j * c, y))
            d.text((x0 + j * c + 3, y + c + 1), f"{keys[i][1][5:]} {s['n']}h {'faint' if s['faint'] else ''} p {s['clip']:.2f}", fill="black")
g.save(ROOT / "grids/s31_bon_examples.jpg", quality=90)
sys.exit(0 if rate <= 0.25 else 1)
