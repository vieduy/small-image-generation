"""Paper (idea B part 2, 2026-11-19): how exact is dropping the photo half of the diptych in the per-option head?

dl_arch.install_half_head(unet, margin): from up_blocks.1 on (no self-attention) only the output half (+ margin columns of the
photo half) is computed. Two error sources: (1) 3x3 convs near the cut see zero padding instead of photo-half features,
(2) GroupNorm statistics are taken over the kept columns instead of the full diptych. This script separates them:
  local  = what would ship (GroupNorm over the kept columns)
  gnfull = GroupNorm statistics injected from the full-width run -> only conv-padding error is left (-> 0 once margin >= RF)
vs the unchanged model, fp32, on synthetic faces x all options. Also FLOPs: full UNet, shared trunk, per-option head, and
per-set (22 options) speedup over 22 full runs.
  source env.sh; CUDA_VISIBLE_DEVICES=1 $VENV/bin/python scripts/halfhead_check.py --tag dlB --decoder decg2
Out: results/halfhead_check_<tag>.json, grids/halfhead_check_<tag>.jpg
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda"


def gn_modules(unet, start=1):
    mods = [m for b in list(unet.up_blocks)[start:] for m in b.modules() if isinstance(m, torch.nn.GroupNorm)]
    return mods + ([unet.conv_norm_out] if unet.conv_norm_out is not None else [])


class GNTap:
    """record: store per-sample/group mean+var of each GroupNorm input. inject: normalise with the stored statistics."""

    def __init__(self, mods):
        self.mods, self.stats, self.mode = mods, {}, None
        for m in mods:
            m.forward = types.MethodType(self._fwd, m)

    def _fwd(self, m, x):
        if self.mode is None:
            return torch.nn.functional.group_norm(x, m.num_groups, m.weight, m.bias, m.eps)
        B, C = x.shape[:2]
        xg = x.float().reshape(B, m.num_groups, -1)
        if self.mode == "record":
            self.stats[id(m)] = (xg.mean(-1, keepdim=True), xg.var(-1, unbiased=False, keepdim=True))
        mu, var = self.stats[id(m)]
        y = ((xg - mu) / torch.sqrt(var + m.eps)).reshape(x.shape).to(x.dtype)
        return y * m.weight.view(1, C, 1, 1) + m.bias.view(1, C, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dlB")
    ap.add_argument("--decoder", default="decg2")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--margins", default="0,1,2,4,6,8,12,16")
    a = ap.parse_args()
    t0 = time.time()
    w = ROOT / "weights/student" / a.tag
    te = torch.load(w / "task_emb.pt")
    tasks = te["tasks"]
    emb, mask = te["emb"].to(DEV), te["mask"].to(DEV)
    unet = DLA.load_unet(w / "unet").to(DEV).eval()
    assert not hasattr(unet, "_half_head"), "start from a full-width student"
    venc = DLA.load_vae(torch.float32).to(DEV).eval()
    vdec = DLA.load_vae(torch.float32, ROOT / "weights/decoder" / a.decoder).to(DEV).eval()
    tap = GNTap(gn_modules(unet))
    print(f"[hh] {a.tag}: {len(tap.mods)} GroupNorms in the head part (up_blocks.1, up_blocks.2, conv_norm_out); "
          f"shared trunk: {getattr(unet, '_shared_scope', 'none')}", flush=True)

    # FLOPs: whole UNet, and the part before up_blocks.0 / before up_blocks.1
    from torch.utils.flop_counter import FlopCounterMode
    z0 = torch.zeros(1, 4, DLA.LAT, DLA.LAT, device=DEV)
    marks = {}

    def flops(margin=None):
        if margin is not None:
            DLA.install_half_head(unet, margin)
        fc = FlopCounterMode(display=False)
        hs = [unet.up_blocks[i].register_forward_pre_hook(lambda *_, i=i: marks.__setitem__(f"before_up{i}", fc.get_total_flops()))
              for i in range(len(unet.up_blocks))]
        with torch.no_grad(), fc:
            DLA.forward(unet, z0, emb[:1], mask[:1])
        for h in hs:
            h.remove()
        if margin is not None:
            DLA.remove_half_head(unet)
        return fc.get_total_flops(), dict(marks)

    full_f, mk = flops()
    trunk_f = mk["before_up0"] if getattr(unet, "_shared_scope", None) == "down_mid" else None
    if getattr(unet, "_shared_scope", None) == "down":  # mid block is per option: trunk ends at the mid block's start
        h = unet.mid_block.register_forward_pre_hook(lambda *_: marks.__setitem__("before_mid", fc_.get_total_flops()))
        fc_ = FlopCounterMode(display=False)
        with torch.no_grad(), fc_:
            DLA.forward(unet, z0, emb[:1], mask[:1])
        h.remove()
        trunk_f = marks["before_mid"]
    trunk_f = trunk_f or 0.0

    def per_set(head_f, n=22):
        return round(n * full_f / (trunk_f + n * head_f), 3)
    fl = {"unet_gflops": round(full_f / 1e9, 1), "trunk_gflops": round(trunk_f / 1e9, 1),
          "head_gflops_full_width": round((full_f - trunk_f) / 1e9, 1), "speedup22_trunk_only": per_set(full_f - trunk_f)}
    print(f"[hh] FLOPs {fl}", flush=True)

    faces = sorted((ROOT / "data/faces").glob("*.png"))[:100]
    faces = [faces[int(i)] for i in np.linspace(0, len(faces) - 1, a.n)]
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
    res = {"flops": fl, "margins": {}}
    sheet = []
    with torch.no_grad():
        zs = []
        for f in faces:
            x = torch.from_numpy(np.array(Image.open(f).convert("RGB").resize((512, 512)))).permute(2, 0, 1)[None].float().to(DEV) / 127.5 - 1
            zs.append(DLA.encode(venc, x))
        refs = []
        for z in zs:  # full model; record the GroupNorm statistics of the full-width head
            tap.mode = "record"
            l0 = DLA.forward(unet, z.expand(len(tasks), -1, -1, -1), emb, mask)
            refs.append((l0, dict(tap.stats), DLA.decode(vdec, l0).clamp(-1, 1)))
        tap.mode = None
        for m in [int(v) for v in a.margins.split(",")]:
            hf, _ = flops(m)
            row = {"head_gflops": round((hf - trunk_f) / 1e9, 1), "speedup22": per_set(hf - trunk_f)}
            DLA.install_half_head(unet, m)
            for kind in ("local", "gnfull"):
                lat_max, psnr, lps, worst = [], [], [], (1e9, None)
                for i, (z, (l0, st, y0)) in enumerate(zip(zs, refs)):
                    tap.mode, tap.stats = (None, {}) if kind == "local" else ("inject", st)
                    if kind == "gnfull":
                        tap.stats = {k: (mu, var) for k, (mu, var) in st.items()}
                    l1 = DLA.forward(unet, z.expand(len(tasks), -1, -1, -1), emb, mask)
                    y1 = DLA.decode(vdec, l1).clamp(-1, 1)
                    lat_max.append(float((l1 - l0).abs().max()))
                    mse = ((y1 - y0) * 127.5).pow(2).mean((1, 2, 3)).clamp_min(1e-10)
                    ps = (10 * torch.log10(255 ** 2 / mse)).tolist()
                    psnr += ps
                    lps += lp(y1, y0).flatten().tolist()
                    j = int(np.argmin(ps))
                    if ps[j] < worst[0]:
                        worst = (ps[j], (faces[i].stem, tasks[j]))
                    if kind == "local" and i == 0 and m in (0, 4, 16):
                        k_ = [0, 1, tasks.index("expr_thumbs")]
                        sheet.append(torch.cat([torch.cat(list(y0[k_]), 2), torch.cat(list(y1[k_]), 2),
                                                torch.cat(list((y1[k_] - y0[k_]).abs().mul(4) - 1), 2)], 1))
                row[kind] = {"latent_maxabs": round(max(lat_max), 5), "psnr_mean": round(float(np.mean(psnr)), 2),
                             "psnr_min": round(float(np.min(psnr)), 2), "lpips_mean": round(float(np.mean(lps)), 5),
                             "lpips_max": round(float(np.max(lps)), 5), "worst": worst[1]}
            tap.mode = None
            DLA.remove_half_head(unet)
            res["margins"][m] = row
            print(f"[hh] margin {m:2d}: {row}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    res["meta"] = {"tag": a.tag, "decoder": a.decoder, "faces": [f.stem for f in faces], "n_images_per_margin": len(faces) * len(tasks),
                   "dtype": "fp32", "minutes": round((time.time() - t0) / 60, 1)}
    (ROOT / f"results/halfhead_check_{a.tag}.json").write_text(json.dumps(res, indent=1))
    g = ((torch.cat(sheet, 1).clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(g).resize((g.shape[1] // 2, g.shape[0] // 2)).save(ROOT / f"grids/halfhead_check_{a.tag}.jpg", quality=88)
    print(f"[hh] done -> results/halfhead_check_{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()
