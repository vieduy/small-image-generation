"""Proposed quality filter for the stage-3 synthetic-face teacher pairs (2026-11-25): score a random sample with the klein keep-rule
checks and show pass / fail examples per rule. No training, no data changed.
Rules (same code paths as scripts/gen_pairs_v2.py score / skincolor):
  R1 identity : SFace rank of the photo among itself + 99 neighbours of the same source pool (sorted ids, gen_pairs order) vs the
                output; pass if rank <= 5 (avatar, sticker) or <= 10 (expressions).
  R2 cutout   : stickers/expressions only. metrics.sticker_postprocess(BiRefNet_lite alpha): pass_sticker (fg 5-90%, main
                component >= 0.9, WebP <= 100 KB).
  R3 skin     : d_chroma = skin chroma(out) - chroma(photo) on the central face box (CIELAB); pass if <= 17.
  R4 option   : expressions only (new). CLIP ViT-L/14, 20 phrases "a cartoon sticker of a person, {phrase}": the requested
                expression must be in the top 3.
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/s3_filter_examples.py --n 600
Out: results/s3_filter_sample.json (pass rates), grids/s3_filter_examples.jpg
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
import metrics as M  # noqa: E402
from gen_pairs_v2 import _skin_lab  # noqa: E402

ROOT, DEV = Path(__file__).resolve().parent.parent, "cuda"
D = ROOT / "data/pairs/cartoon3d"


def src_path(i):
    return ROOT / "data/ffhq_norm" / f"{i}.png" if i.startswith("q") else D / "inputs" / f"{i}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    C = torch.load(D / "latents_dl_manifest_v2.pt", mmap=True)
    TL = torch.load(D / "teacher_tB1_s8.pt", mmap=True)
    assert TL["key"] == C["key"]
    rng = random.Random(1)
    idx = sorted(rng.sample(range(len(C["key"])), a.n))
    vae = DLA.load_vae(torch.float32).to(DEV).eval()
    od = ROOT / "outputs/s3_filter_sample"
    od.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i in range(0, len(idx), 32):
            j = idx[i:i + 32]
            for k, y in zip(j, DLA.decode(vae, TL["zt"][j].to(DEV).float()).clamp(-1, 1)):
                Image.fromarray(((y + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).save(od / f"{k}.png")
    fid, mat = M.FaceID(), M.Matter()
    pools = {}
    feat = {}

    def f_in(i):
        if i not in feat:
            f = fid.feat(Image.open(src_path(i)))
            feat[i] = None if f is None else f.ravel() / np.linalg.norm(f)
        return feat[i]

    def id_rank(own, fo):
        p = own[0]
        if p not in pools:
            pools[p] = sorted(x.stem for x in ((ROOT / "data/ffhq_norm").glob("q*.png") if p == "q" else (D / "inputs").glob(f"{p}*.png")))
        pool = pools[p]
        k = pool.index(own)
        gal = [own] + [pool[(k + 1 + m) % len(pool)] for m in range(99)]
        G = [(g, f_in(g)) for g in gal]
        G = [x for x in G if x[1] is not None]
        if not G or G[0][0] != own:
            return None
        sims = np.stack([x[1] for x in G]) @ (fo.ravel() / np.linalg.norm(fo))
        return int((sims > sims[0]).sum()) + 1

    EX = yaml.safe_load((ROOT / "configs/sticker_expressions.yaml").read_text())
    ids = [e["id"] for e in EX]
    from transformers import CLIPModel, CLIPProcessor
    cp = M._snap("openai/clip-vit-large-patch14")
    cm, pr = CLIPModel.from_pretrained(str(cp), dtype=torch.float16).to(DEV).eval(), CLIPProcessor.from_pretrained(str(cp))
    with torch.no_grad():
        ti = pr(text=[f"a cartoon sticker of a person, {e['phrase']}" for e in EX], return_tensors="pt", padding=True).to(DEV)
        tf = cm.text_projection(cm.text_model(**ti).pooler_output).float()
        tf = tf / tf.norm(dim=-1, keepdim=True)
    rows = []
    for k in idx:
        pid, task, _ = C["key"][k]
        o = Image.open(od / f"{k}.png").convert("RGB")
        r = {"k": k, "id": pid, "task": task}
        fo = fid.feat(o)
        rk = None if fo is None else id_rank(pid, fo)
        r["id_rank"] = rk
        r["R1"] = rk is not None and rk <= (10 if task.startswith("expr_") else 5)
        if task != "avatar":
            _, info = M.sticker_postprocess(o, mat(o))
            r["R2"], r["fg_frac"], r["main_comp"] = bool(info["pass_sticker"]), info.get("fg_frac"), info.get("main_comp")
        si, so = _skin_lab(src_path(pid)), _skin_lab(od / f"{k}.png")
        r["d_chroma"] = None if (si is None or so is None) else round(so[3] - si[3], 1)
        r["R3"] = r["d_chroma"] is None or r["d_chroma"] <= 17
        if task.startswith("expr_"):
            with torch.no_grad():
                pv = pr.image_processor(images=[o], return_tensors="pt")["pixel_values"].to(DEV, torch.float16)
                f = cm.visual_projection(cm.vision_model(pixel_values=pv).pooler_output).float()
                s = (f / f.norm(dim=-1, keepdim=True) @ tf.T)[0]
            rank = int((s > s[ids.index(task[5:])]).sum()) + 1
            r["clip_rank"], r["clip_top"] = rank, ids[int(s.argmax())]
            r["R4"] = rank <= 3
        r["keep"] = all(r.get(x, True) for x in ("R1", "R2", "R3", "R4"))
        rows.append(r)
    S = {"n": len(rows)}
    for R in ("R1", "R2", "R3", "R4", "keep"):
        v = [x[R] for x in rows if R in x]
        S[R] = {"pass_rate": round(float(np.mean(v)), 3), "n": len(v)}
    for t in ("avatar", "sticker", "expr"):
        v = [x["keep"] for x in rows if x["task"].startswith(t)]
        S[f"keep_{t}"] = round(float(np.mean(v)), 3) if v else None
    (ROOT / "results/s3_filter_sample.json").write_text(json.dumps({"summary": S, "rows": rows}, indent=1))
    print("[s3f]", json.dumps(S), flush=True)

    # sheet: per rule, 4 pass + 4 fail (photo | teacher output, caption)
    def cap(r, R):
        return {"R1": f"id rank {r['id_rank']}", "R2": f"fg {r.get('fg_frac')} comp {r.get('main_comp')}",
                "R3": f"d_chroma {r['d_chroma']}", "R4": f"{r['task'][5:]}: rank {r.get('clip_rank')} (top: {r.get('clip_top')})"}[R]
    c = 128
    title = {"R1": "R1 identity (rank <= 5 / <= 10 expr)", "R2": "R2 sticker cut-out", "R3": "R3 skin colour (d_chroma <= 17)",
             "R4": "R4 expression in CLIP top-3"}
    g = Image.new("RGB", (8 * 2 * c, 4 * (c + 40)), "white")
    d = ImageDraw.Draw(g)
    for ri, R in enumerate(("R1", "R2", "R3", "R4")):
        y = ri * (c + 40)
        d.text((4, y + 2), title[R] + "  |  left 4: PASS  |  right 4: FAIL", fill="black")
        P = [r for r in rows if r.get(R) is True][:4]
        Fl = [r for r in rows if r.get(R) is False][:4]
        for j, r in enumerate(P + [None] * (4 - len(P)) + Fl):
            if r is None:
                continue
            x = j * 2 * c
            g.paste(Image.open(src_path(r["id"])).convert("RGB").resize((c, c)), (x, y + 16))
            g.paste(Image.open(od / f"{r['k']}.png").convert("RGB").resize((c, c)), (x + c, y + 16))
            d.rectangle([x, y + 16 + c, x + 2 * c - 2, y + 16 + c + 20], fill=(40, 140, 60) if j < 4 else (200, 30, 30))
            d.text((x + 3, y + 16 + c + 4), cap(r, R)[:40], fill="white")
    g.save(ROOT / "grids/s3_filter_examples.jpg", quality=88)


if __name__ == "__main__":
    main()
