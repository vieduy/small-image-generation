"""Best-of-N clean teacher pairs for one-handed gestures (2026-11-27, user "go best-of-N clean teacher pairs for gestures + retrain with DMD2").
Rows: cartoon3d teacher pairs teacher_tB1_s8_img3.0_k1.pt, options thumbs / wave / ok / think / facepalm (one hand each; shrug / thanks are
two-handed and left as is). Per row and view (plain photo -> zt, flipped photo -> zt_f): candidate c0 = the existing target (phone noise,
seed 1234) + c1..c3 = the teacher at the SAME setting (8 steps, img 3 on step 1, txt 2.5) with noise seeds 1235 / 1236 / 1237.
Scores per candidate: OWLv2 hands (s18 rules: score >= 0.12, nested boxes merged) -> count, faint (any hand opacity < 0.15), top hand score
and side (box centre x vs 256); CLIP-L probability of the correct option among the 20 expression captions (= eval_e1 / s6 prompts).
Selection rule (fixed before running): clean = exactly 1 hand, not faint, top score >= 0.2; SIDE-CONSISTENT = same side as c0's single hand
(if c0 has none / several, any side). Pick the clean, side-consistent candidate with the highest CLIP probability (c0 wins ties).
A view with no such candidate is undecided; a row with an undecided view is dropped (s31_merge.py; gate: <= 25% of these rows).
Side-consistency matters: ghost hands come from an unstable hand side (results/s18_ghost_rate.json), so replacements keep c0's side.
  CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/s31_best_of_n.py --shard 0 --nshards 2
Out: data/pairs/cartoon3d/bon_tB1/shard_<k>.pt {rows, choice [n,2], new latents of chosen non-c0 candidates, stats per candidate}
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.ops import nms

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
import metrics as M  # noqa: E402
import teacher_dl as TD  # noqa: E402
from s7_guidance_sweep import sample_gs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/pairs/cartoon3d"
NAME = "teacher_tB1_s8_img3.0_k1"
OPTS = ["thumbs", "wave", "ok", "think", "facepalm"]
QUERIES = ["a hand", "a human hand", "a cartoon hand"]


def main():
    from transformers import CLIPModel, CLIPProcessor, Owlv2ForObjectDetection, Owlv2Processor
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, default=2)
    ap.add_argument("--seeds", default="1235,1236,1237")
    a = ap.parse_args()
    t0 = time.time()
    TP = torch.load(D / f"{NAME}.pt", mmap=True)
    C = torch.load(D / "latents_dl_manifest_v2.pt", mmap=True)
    FL = torch.load(D / "latents_dl_manifest_v2_flip.pt", mmap=True)
    keys = [tuple(k) for k in TP["key"]]
    assert keys == [tuple(k) for k in C["key"]]
    rows = [i for i, k in enumerate(keys) if k[1] in [f"expr_{o}" for o in OPTS]][a.shard::a.nshards]
    print(f"[s31] shard {a.shard}: {len(rows)} rows x 2 views x {1 + len(a.seeds.split(','))} candidates", flush=True)
    unet, T = TD.load_teacher("tB1")
    unet = unet.to("cuda").eval()
    emb_, mask_ = T["emb"].to("cuda", torch.float16), T["mask"].to("cuda")
    ne, nm = T["null_emb"].to("cuda", torch.float16), T["null_mask"].to("cuda")
    vae = DLA.load_vae(torch.float16).to("cuda").eval()
    noises = [DLA.fixed_noise("cuda", torch.float32, seed=int(s)) for s in a.seeds.split(",")]
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", torch_dtype=torch.float16).to("cuda").eval()
    EX = yaml.safe_load((ROOT / "configs/sticker_expressions.yaml").read_text())
    ids = [e["id"] for e in EX]
    cp = M._snap("openai/clip-vit-large-patch14")
    cm, pr = CLIPModel.from_pretrained(str(cp), dtype=torch.float16).to("cuda").eval(), CLIPProcessor.from_pretrained(str(cp))
    with torch.inference_mode():
        ti = pr(text=[f"a cartoon sticker of a person, {e['phrase']}" for e in EX], return_tensors="pt", padding=True).to("cuda")
        tf = cm.text_projection(cm.text_model(**ti).pooler_output).float()
        tf = tf / tf.norm(dim=-1, keepdim=True)

    def pil(y):
        return [Image.fromarray(x) for x in ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()]

    def score(ims, opt_idx):
        inp = proc(text=[QUERIES] * len(ims), images=ims, return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].half()
        with torch.inference_mode():
            res = proc.post_process_grounded_object_detection(owl(**inp), threshold=0.12, target_sizes=[(im.height, im.width) for im in ims])
            pv = pr.image_processor(images=ims, return_tensors="pt")["pixel_values"].to("cuda", torch.float16)
            f = cm.visual_projection(cm.vision_model(pixel_values=pv).pooler_output).float()
            f = f / f.norm(dim=-1, keepdim=True)
            pc = (100 * f @ tf.T).softmax(-1)
        out = []
        for n, (r, im) in enumerate(zip(res, ims)):
            b, s = r["boxes"].float(), r["scores"].float()
            keep = sorted(nms(b, s, 0.3).tolist(), key=lambda k: -float(s[k])) if len(s) else []
            hs = []
            for k in keep:
                x0, y0, x1, y1 = b[k].tolist()
                ar = (x1 - x0) * (y1 - y0)
                if any(max(0, min(x1, X1) - max(x0, X0)) * max(0, min(y1, Y1) - max(y0, Y0)) >= 0.6 * min(ar, (X1 - X0) * (Y1 - Y0))
                       for X0, Y0, X1, Y1, _ in hs):
                    continue
                hs.append((x0, y0, x1, y1, float(s[k])))
            arr = np.asarray(im).astype(np.float32)
            faint = False
            for x0, y0, x1, y1, _ in hs:
                pt = arr[int(max(y0, 0)):int(min(y1, 512)), int(max(x0, 0)):int(min(x1, 512))]
                if pt.size and np.abs(255 - pt).mean() / 255 < 0.15:
                    faint = True
            side = ("L" if (hs[0][0] + hs[0][2]) / 2 < 256 else "R") if hs else "-"
            out.append({"n": len(hs), "faint": faint, "top": hs[0][4] if hs else 0.0, "side": side, "clip": float(pc[n, opt_idx[n]])})
        return out

    n = len(rows)
    nc = 1 + len(noises)
    choice = np.zeros((n, 2), np.int8)
    stats = [[None] * 2 for _ in range(n)]
    newlat = {}
    for j in range(0, n, 32):
        idx = rows[j:j + 32]
        k_t = torch.tensor([T["tasks"].index(keys[i][1]) for i in idx], device="cuda")
        oi = [ids.index(keys[i][1][5:]) for i in idx]
        for v, (Zsrc, Zt) in enumerate(((C["zi"], TP["zt"]), (FL["zi_f"], TP["zt_f"]))):
            lat = [Zt[idx].to("cuda", torch.float16)]
            with torch.inference_mode():
                for nz in noises:
                    lat.append(sample_gs(unet, Zsrc[idx].to("cuda", torch.float16), emb_[k_t], mask_[k_t], ne, nm, 3.0, 1, 2.5, False, 8, nz).half())
                S = [score(pil(DLA.decode(vae, z)), oi) for z in lat]
            for m, i in enumerate(idx):
                cand = [S[c][m] for c in range(nc)]
                s0 = cand[0]["side"] if cand[0]["n"] == 1 else None
                ok = [c for c in range(nc) if cand[c]["n"] == 1 and not cand[c]["faint"] and cand[c]["top"] >= 0.2
                      and (s0 is None or cand[c]["side"] == s0)]
                pick = max(ok, key=lambda c: (cand[c]["clip"], c == 0)) if ok else -1
                choice[j + m, v] = pick
                stats[j + m][v] = cand
                if pick > 0:
                    newlat[(i, v)] = lat[pick][m].cpu()
        if (j // 32) % 10 == 0:
            el = time.time() - t0
            print(f"[s31] {j + len(idx)}/{n} rows {el:.0f}s (~{el / (j + len(idx)) * (n - j - len(idx)) / 60:.0f} min left)", flush=True)
    out = D / "bon_tB1"
    out.mkdir(exist_ok=True)
    torch.save({"rows": rows, "keys": [keys[i] for i in rows], "choice": torch.from_numpy(choice), "stats": stats, "new": newlat,
                "seeds": a.seeds}, out / f"shard_{a.shard}.pt")
    c_ = choice.reshape(-1)
    print(f"[s31] done {(time.time() - t0) / 60:.1f} min; kept c0 {np.mean(c_ == 0):.3f}, replaced {np.mean(c_ > 0):.3f}, "
          f"undecided {np.mean(c_ < 0):.3f}", flush=True)


if __name__ == "__main__":
    main()
