"""Fixed hand side for one-handed gesture pairs (2026-11-27, user "yes go with the fixed side fix tS6").
Why (results/s18_ghost_rate.json): the teacher's hand side flips under 8 px input shifts in ~30% of cases, and tS4/tS4o draw extra / ghost
hands 24x more often exactly there (11.5% vs 0.5%). A deterministic side makes the target learnable.
Rows: expr_thumbs, expr_wave, expr_ok of the cartoon3d teacher pairs (teacher_tB1_s8_img3.0_k1.pt: zt = teacher(photo), zt_f =
teacher(mirrored photo), both from the fixed phone noise). Candidates per row (no new teacher sampling):
  plain view  : A  = dec(zt)          or  B  = mirror(dec(zt_f))   (the mirrored photo's output, flipped back)
  flipped view: A' = dec(zt_f)        or  B' = mirror(dec(zt))
Hand side = OWLv2 (as scripts/s18_owl_hands.py: queries "a hand" / "a human hand" / "a cartoon hand", score >= 0.12, nested boxes merged),
exactly one hand -> L / R by box centre vs x = 256; else ambiguous. side(B) = opposite(side(A')), side(B') = opposite(side(A)).
Canonical side per option = the side the plain teacher targets use most often (fixed by this rule before looking at the counts).
Each view takes the candidate whose single hand is on the canonical side (A first); a swapped candidate is re-encoded with the TinyVAE
encoder (enc(mirror(dec(z)))). v2 (same day, after v1 dropped 61% of the gesture rows): views still without one are re-sampled
from the teacher with noise seeds 1235-1237 (first canonical output used). A row where EITHER view cannot be made canonical (no single hand on the canonical side in either candidate) is dropped
(added to the drop list): a half-canonical row would keep the coin flip.
Out: data/pairs/cartoon3d/teacher_tB1_s8_img3.0_k1_side.pt (all rows; only gesture rows changed), _side_drop.json (= consistency drop
list + undecidable gesture rows), _side.json (counts), grids/s19_fixed_side_examples.jpg
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/s19_fixed_side.py
"""
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision.ops import nms

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
import teacher_dl as TD  # noqa: E402
from s7_guidance_sweep import sample_gs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/pairs/cartoon3d"
NAME = "teacher_tB1_s8_img3.0_k1"
OPTS = ["expr_thumbs", "expr_wave", "expr_ok"]
QUERIES = ["a hand", "a human hand", "a cartoon hand"]


def main():
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    t0 = time.time()
    TP = torch.load(D / f"{NAME}.pt")
    keys = [tuple(k) for k in TP["key"]]
    zt, ztf = TP["zt"].clone(), TP["zt_f"].clone()
    rows = [i for i, k in enumerate(keys) if k[1] in OPTS]
    print(f"[s19] {len(rows)} gesture rows", flush=True)
    vae = DLA.load_vae(torch.float16).to("cuda").eval()
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", torch_dtype=torch.float16).to("cuda").eval()

    def pil(y):
        return [Image.fromarray(a) for a in ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()]

    def side_of(ims):
        inp = proc(text=[QUERIES] * len(ims), images=ims, return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].half()
        with torch.inference_mode():
            out = owl(**inp)
        res = proc.post_process_grounded_object_detection(out, threshold=0.12, target_sizes=[(im.height, im.width) for im in ims])
        sides = []
        for r in res:
            b, s = r["boxes"].float(), r["scores"].float()
            keep = nms(b, s, 0.3).tolist() if len(s) else []
            hs = []
            for k in sorted(keep, key=lambda k: -float(s[k])):
                x0, y0, x1, y1 = b[k].tolist()
                a = (x1 - x0) * (y1 - y0)
                if any(max(0, min(x1, X1) - max(x0, X0)) * max(0, min(y1, Y1) - max(y0, Y0)) >= 0.6 * min(a, (X1 - X0) * (Y1 - Y0))
                       for X0, Y0, X1, Y1 in hs):
                    continue
                hs.append((x0, y0, x1, y1))
            sides.append(("L" if (hs[0][0] + hs[0][2]) / 2 < 256 else "R") if len(hs) == 1 else "?")
        return sides

    sA, sAf, imA, imAf = {}, {}, {}, {}
    for j in range(0, len(rows), 32):
        idx = rows[j:j + 32]
        with torch.inference_mode():
            a = pil(DLA.decode(vae, zt[idx].to("cuda", torch.float16)))
            af = pil(DLA.decode(vae, ztf[idx].to("cuda", torch.float16)))
        for i, s1, s2, x, xf in zip(idx, side_of(a), side_of(af), a, af):
            sA[i], sAf[i] = s1, s2
            imA[i], imAf[i] = x, xf
        if (j // 32) % 20 == 0:
            print(f"[s19] {j + len(idx)}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    opp = {"L": "R", "R": "L", "?": "?"}
    canon = {}
    for o in OPTS:
        ss = [sA[i] for i in rows if keys[i][1] == o]
        canon[o] = "L" if ss.count("L") >= ss.count("R") else "R"
    stats = {o: {"rows": 0, "plain_kept": 0, "plain_swapped": 0, "plain_undecided": 0, "flip_kept": 0, "flip_swapped": 0,
                 "flip_undecided": 0, "dropped": 0, "canonical_side": canon[o],
                 "plain_side_counts": {s: [sA[i] for i in rows if keys[i][1] == o].count(s) for s in "LR?"}} for o in OPTS}
    drop, examples = [], []
    swap_p, swap_f = [], []
    und, resampled = {"plain": [], "flip": []}, []
    for i in rows:
        o, c = keys[i][1], canon[keys[i][1]]
        st = stats[o]
        st["rows"] += 1
        if sA[i] == c:
            st["plain_kept"] += 1
        elif opp[sAf[i]] == c:
            st["plain_swapped"] += 1
            swap_p.append(i)
        else:
            und["plain"].append(i)
        if sAf[i] == c:
            st["flip_kept"] += 1
        elif opp[sA[i]] == c:
            st["flip_swapped"] += 1
            swap_f.append(i)
        else:
            und["flip"].append(i)
    # 2026-11-27 v2: views still without a canonical hand -> resample the teacher (same photo / view, same setting) with other noise
    # seeds; the first output with exactly one hand on the canonical side is used as is (no re-encode). Seeds 1235-1237.
    unet, T = TD.load_teacher("tB1")
    unet = unet.to("cuda").eval()
    emb_, mask_ = T["emb"].to("cuda", torch.float16), T["mask"].to("cuda")
    ne, nm = T["null_emb"].to("cuda", torch.float16), T["null_mask"].to("cuda")
    C = torch.load(D / "latents_dl_manifest_v2.pt", mmap=True)
    FLc = torch.load(D / "latents_dl_manifest_v2_flip.pt", mmap=True)
    assert [tuple(k) for k in C["key"]] == keys
    for seed in (1235, 1236, 1237):
        noise = DLA.fixed_noise("cuda", torch.float32, seed=seed)
        for view, Zsrc, dst in (("plain", C["zi"], zt), ("flip", FLc["zi_f"], ztf)):
            todo, left = und[view], []
            for j in range(0, len(todo), 48):
                idx = todo[j:j + 48]
                k = torch.tensor([T["tasks"].index(keys[i][1]) for i in idx], device="cuda")
                with torch.inference_mode():
                    z = sample_gs(unet, Zsrc[idx].to("cuda", torch.float16), emb_[k], mask_[k], ne, nm, 3.0, 1, 2.5, False, 8, noise).half()
                    ims = pil(DLA.decode(vae, z))
                for n, (i, sd) in enumerate(zip(idx, side_of(ims))):
                    if sd == canon[keys[i][1]]:
                        dst[i] = z[n].cpu().to(dst.dtype)
                        stats[keys[i][1]][f"{view}_resampled"] = stats[keys[i][1]].get(f"{view}_resampled", 0) + 1
                        resampled.append((i, view, imA[i] if view == "plain" else imAf[i], ims[n]))
                    else:
                        left.append(i)
            und[view] = left
        print(f"[s19] seed {seed}: still undecided plain {len(und['plain'])} flip {len(und['flip'])} ({time.time() - t0:.0f}s)", flush=True)
    del unet
    torch.cuda.empty_cache()
    bad = set(und["plain"]) | set(und["flip"])  # a half-canonical row would keep the coin flip -> drop the row
    for i in sorted(bad):
        st = stats[keys[i][1]]
        st["plain_undecided"] += int(i in und["plain"])
        st["flip_undecided"] += int(i in und["flip"])
        st["dropped"] += 1
        drop.append([keys[i][0], keys[i][1]])
    # re-encode swapped candidates
    with torch.inference_mode():
        for lst, src, dst in ((swap_p, imAf, zt), (swap_f, imA, ztf)):
            for j in range(0, len(lst), 32):
                idx = lst[j:j + 32]
                x = torch.stack([torch.from_numpy(__import__("numpy").asarray(ImageOps.mirror(src[i]))).permute(2, 0, 1).float() / 127.5 - 1
                                 for i in idx]).to("cuda", torch.float16)
                dst[idx] = DLA.encode(vae, x).to(dst.dtype).cpu()
    examples = [(i, "plain") for i in swap_p[:4]] + [(i, "flip") for i in swap_f[:2]]
    base_drop = json.loads((D / f"{NAME}_consistency_drop.json").read_text())
    have = {tuple(x) for x in base_drop}
    all_drop = base_drop + [x for x in drop if tuple(x) not in have]
    torch.save({"key": TP["key"], "zt": zt, "zt_f": ztf, "setting": dict(TP.get("setting", {})) | {"fixed_side": canon},
                "noise_seed": TP.get("noise_seed")}, D / f"{NAME}_side.pt")
    (D / f"{NAME}_side_drop.json").write_text(json.dumps(all_drop))
    out = {"canonical_side": canon, "per_option": stats, "gesture_rows_dropped": len(drop), "drop_list_total": len(all_drop),
           "minutes": round((time.time() - t0) / 60, 1)}
    (D / f"{NAME}_side.json").write_text(json.dumps(out, indent=1))
    # example sheet: original target | new target, for swapped rows
    c = 220
    g = Image.new("RGB", (4 * c, 20 + ((len(examples) + 1) // 2) * (c + 14)), "white")
    dr = ImageDraw.Draw(g)
    dr.text((4, 4), f"fixed side {canon}: per row original teacher target | new (swapped) target", fill="black")
    with torch.inference_mode():
        for n, (i, view) in enumerate(examples):
            old = imA[i] if view == "plain" else imAf[i]
            new = pil(DLA.decode(vae, (zt if view == "plain" else ztf)[i:i + 1].to("cuda", torch.float16)))[0]
            x0, y = (n % 2) * 2 * c, 20 + (n // 2) * (c + 14)
            g.paste(old.resize((c, c)), (x0, y)); g.paste(new.resize((c, c)), (x0 + c, y))
            dr.text((x0 + 3, y + c + 1), f"{keys[i][0]} {keys[i][1][5:]} {view}: before | after", fill="black")
    g.save(ROOT / "grids/s19_fixed_side_examples.jpg", quality=90)
    g2 = Image.new("RGB", (4 * c, 20 + 4 * (c + 14)), "white")
    dr = ImageDraw.Draw(g2)
    dr.text((4, 4), f"fixed side {canon}: resampled rows, original teacher target | new target (other noise seed)", fill="black")
    for n, (i, view, old, new) in enumerate(resampled[::max(1, len(resampled) // 8)][:8]):
        x0, y = (n % 2) * 2 * c, 20 + (n // 2) * (c + 14)
        g2.paste(old.resize((c, c)), (x0, y)); g2.paste(new.resize((c, c)), (x0 + c, y))
        dr.text((x0 + 3, y + c + 1), f"{keys[i][0]} {keys[i][1][5:]} {view}: before | after", fill="black")
    g2.save(ROOT / "grids/s19_fixed_side_resampled.jpg", quality=90)
    print("[s19] " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
