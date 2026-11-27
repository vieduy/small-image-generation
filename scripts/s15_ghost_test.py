"""Ghost-hand test (2026-11-26, user "yes run the test on dev-duyhv").
Hypothesis: the one-step student draws a faded hand on BOTH sides where the teacher's (fixed-noise) hand side is UNSTABLE, i.e. flips
under tiny input changes, so the student cannot learn it and hedges.
Faces: data/faces[:100] (synthetic eval faces, in no training set). One-handed options: thumbs, wave, ok, think.
Inputs: base (512 resize) + 4 small variants (resize to 528, crop 512 at offsets (0,8), (16,8), (8,0), (8,16): ~3% zoom + 8 px shifts)
+ mirror (reported separately: does the layout follow the face or the noise?).
Teacher tB1 at the tS4 pair setting (img 3 on step 1 of 8, txt 2.5, the phone's FIXED noise = what the students were trained to copy).
Students (phone path, fixed noise, decg2): dl6g2, tS4, tS4o on base + variants.
Hand locator (CPU, no model): YuNet face box; skin colour = median Lab of the central face; skin mask = foreground pixels with Lab (a,b)
distance < 18 and |L - L_face| < 40, outside the face box expanded (x -15%..+15% w, y -25%..+25% h) and outside a central neck band
(|x - face centre| < 0.45 w below the face); components >= 600 px after a 3x3 opening; side = component centroid left / right of the face
centre. Checked by eye on grids/s15_locator_check.jpg BEFORE the numbers are used.
Definitions: teacher UNSTABLE (face, option) = the set of hand sides on any of the 4 small variants differs from the base;
student GHOST = hands on BOTH sides on the base input (one-handed option).
Decision rule (fixed before running), pooled over tS4 + tS4o:
  CONFIRMED if ghost rate on teacher-unstable cases >= 2x the rate on teacher-stable cases AND Fisher exact p < 0.05;
  REJECTED if the ratio < 1.5 or p >= 0.2; otherwise INCONCLUSIVE.
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/s15_ghost_test.py
Out: results/s15_ghost_test.json, outputs/s15_ghost/<model>/<variant>/<face>_<opt>.png, grids/s15_locator_check.jpg, grids/s15_ghost_cases.jpg
"""
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
import metrics as M  # noqa: E402
import teacher_dl as TD  # noqa: E402
from eval_e1 import to_t  # noqa: E402
from s7_guidance_sweep import sample_gs  # noqa: E402

ROOT, DEV = TD.ROOT, "cuda"
OPTS = ["thumbs", "wave", "ok", "think"]
VARIANTS = {"base": None, "v1": (0, 8), "v2": (16, 8), "v3": (8, 0), "v4": (8, 16), "mirror": "mirror"}
STUDENTS = ["dl6g2", "tS4", "tS4o"]
OUT = ROOT / "outputs/s15_ghost"


def variant(img, v):
    img = img.convert("RGB").resize((512, 512))
    if v is None:
        return img
    if v == "mirror":
        return ImageOps.mirror(img)
    big = img.resize((528, 528), Image.LANCZOS)
    return big.crop((v[0], v[1], v[0] + 512, v[1] + 512))


def pil(y):
    return [Image.fromarray(a) for a in ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()]


class HandLocator:
    def __init__(self):
        self.det = M.FaceID().det

    def __call__(self, img):
        a = np.asarray(img.convert("RGB"))
        H, W = a.shape[:2]
        bgr = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        self.det.setInputSize((W, H))
        _, f = self.det.detect(bgr)
        if f is None or len(f) == 0:
            return None
        x, y, w, h = [float(v) for v in max(f, key=lambda r: r[2] * r[3])[:4]]
        lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32)
        cx0, cx1, cy0, cy1 = int(x + .3 * w), int(x + .7 * w), int(y + .45 * h), int(y + .75 * h)
        ref = np.median(lab[max(cy0, 0):cy1, max(cx0, 0):cx1].reshape(-1, 3), 0)
        fg = a.min(2) < 235
        d_ab = np.sqrt(((lab[..., 1:] - ref[1:]) ** 2).sum(-1))
        skin = fg & (d_ab < 18) & (np.abs(lab[..., 0] - ref[0]) < 40)
        yy, xx = np.mgrid[0:H, 0:W]
        fcx = x + w / 2
        face_zone = (xx > x - .15 * w) & (xx < x + 1.15 * w) & (yy > y - .25 * h) & (yy < y + 1.25 * h)
        neck = (np.abs(xx - fcx) < .45 * w) & (yy > y + h)
        m = (skin & ~face_zone & ~neck).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lbl, st, cen = cv2.connectedComponentsWithStats(m, 8)
        comps = [(int(st[i, cv2.CC_STAT_AREA]), float(cen[i][0]), float(cen[i][1])) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 600]
        sides = sorted({"L" if c[1] < fcx else "R" for c in comps})
        return {"sides": "".join(sides) or "-", "comps": comps, "face": [x, y, w, h], "mask": m}


def main():
    t0 = time.time()
    random.seed(0)
    faces = sorted((ROOT / "data/faces").glob("*.png"))[:100]
    photos = {p.stem: Image.open(p) for p in faces}
    vae = DLA.load_vae(torch.float16).to(DEV).eval()
    noise = DLA.fixed_noise(DEV, torch.float32)
    Z = {}
    with torch.inference_mode():
        for vn, v in VARIANTS.items():
            xs = torch.stack([to_t(variant(photos[s], v)) for s in photos]).to(DEV, torch.float16)
            Z[vn] = torch.cat([DLA.encode(vae, xs[i:i + 50]) for i in range(0, len(xs), 50)])
    names = list(photos)
    jobs = [(i, o) for i in range(len(names)) for o in OPTS]

    def save(model, vn, imgs):
        d = OUT / model / vn
        d.mkdir(parents=True, exist_ok=True)
        for (i, o), im in zip(jobs, imgs):
            im.save(d / f"{names[i]}_{o}.png")

    # teacher (fixed noise)
    unet, T = TD.load_teacher("tB1")
    unet = unet.to(DEV).eval()
    emb_, mask_ = T["emb"].to(DEV, torch.float16), T["mask"].to(DEV)
    ne, nm = T["null_emb"].to(DEV, torch.float16), T["null_mask"].to(DEV)
    kt = torch.tensor([T["tasks"].index(f"expr_{o}") for _, o in jobs], device=DEV)
    ji = torch.tensor([i for i, _ in jobs], device=DEV)
    for vn in VARIANTS:
        out = []
        with torch.inference_mode():
            for s in range(0, len(jobs), 48):
                z = Z[vn][ji[s:s + 48]]
                k = kt[s:s + 48]
                out += pil(DLA.decode(vae, sample_gs(unet, z, emb_[k], mask_[k], ne, nm, 3.0, 1, 2.5, False, 8, noise).half()))
        save("teacher", vn, out)
        print(f"[s15] teacher {vn} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    del unet
    torch.cuda.empty_cache()
    # students (phone path, fixed noise, decg2)
    dec = DLA.load_vae(torch.float16, ROOT / "weights/decoder/decg2").to(DEV).eval()
    for sname in STUDENTS:
        w = ROOT / "weights/student" / sname
        su = DLA.load_unet(w / "unet").to(DEV, torch.float16).eval()
        te = torch.load(w / "task_emb.pt")
        se, sm, tasks = te["emb"].to(DEV, torch.float16), te["mask"].to(DEV), te["tasks"]
        ks = torch.tensor([tasks.index(f"expr_{o}") for _, o in jobs], device=DEV)
        for vn in VARIANTS:
            out = []
            with torch.inference_mode():
                for s in range(0, len(jobs), 32):
                    out += pil(DLA.decode(dec, DLA.forward(su, Z[vn][ji[s:s + 32]], se[ks[s:s + 32]], sm[ks[s:s + 32]])))
            save(sname, vn, out)
        del su
        torch.cuda.empty_cache()
        print(f"[s15] {sname} done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    # locate hands
    HL = HandLocator()
    side = {}
    for model in ["teacher"] + STUDENTS:
        for vn in VARIANTS:
            for i, o in jobs:
                im = Image.open(OUT / model / vn / f"{names[i]}_{o}.png")
                if vn == "mirror":
                    im = ImageOps.mirror(im)  # back to the base orientation
                r = HL(im)
                side[(model, vn, names[i], o)] = None if r is None else r["sides"]
    print(f"[s15] hands located ({(time.time() - t0) / 60:.1f} min)", flush=True)

    # locator check sheet: 40 random images with mask overlay (teacher + students, base)
    pick = random.sample([(m, names[i], o) for m in ["teacher"] + STUDENTS for i, o in jobs], 40)
    c = 200
    g = Image.new("RGB", (8 * c, 5 * (c + 14) + 20), "white")
    d = ImageDraw.Draw(g)
    d.text((4, 4), "Hand locator check: red = skin components counted as hands, box = face; caption = sides found", fill="black")
    for n_, (m, f, o) in enumerate(pick):
        im = Image.open(OUT / m / "base" / f"{f}_{o}.png").convert("RGB")
        r = HL(im)
        a = np.asarray(im).copy()
        if r is not None:
            a[r["mask"] > 0] = (0.4 * a[r["mask"] > 0] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
        t = Image.fromarray(a)
        if r is not None:
            x, y, w, h = r["face"]
            ImageDraw.Draw(t).rectangle([x, y, x + w, y + h], outline=(0, 0, 255), width=3)
        px, py = (n_ % 8) * c, 20 + (n_ // 8) * (c + 14)
        g.paste(t.resize((c, c)), (px, py))
        d.text((px + 3, py + c + 1), f"{m} {f[-3:]} {o}: {None if r is None else r['sides']}", fill="black")
    g.save(ROOT / "grids/s15_locator_check.jpg", quality=88)

    # statistics
    res = {"n_cases": len(jobs), "teacher": {}, "students": {}}
    small = ["v1", "v2", "v3", "v4"]
    unstable = {}
    for i, o in jobs:
        b = side[("teacher", "base", names[i], o)]
        if b is None:
            continue
        unstable[(names[i], o)] = any(side[("teacher", v, names[i], o)] != b for v in small)
    tb = [side[("teacher", "base", names[i], o)] for i, o in jobs]
    res["teacher"] = {"base_sides": {s: tb.count(s) for s in sorted(set(map(str, tb)))},
                      "unstable_rate": round(float(np.mean(list(unstable.values()))), 3), "n": len(unstable),
                      "mirror_follows_face": round(float(np.mean([side[("teacher", "mirror", names[i], o)] == side[("teacher", "base", names[i], o)]
                                                                  for i, o in jobs if side[("teacher", "base", names[i], o)] is not None])), 3)}
    from scipy.stats import fisher_exact
    pooled = np.zeros((2, 2), int)  # rows: unstable / stable; cols: ghost / no ghost
    for sname in STUDENTS:
        tab = np.zeros((2, 2), int)
        sb = [side[(sname, "base", names[i], o)] for i, o in jobs]
        for (f, o), u in unstable.items():
            sv = side[(sname, "base", f, o)]
            if sv is None:
                continue
            tab[0 if u else 1, 0 if sv == "LR" else 1] += 1
        ghost_u = tab[0, 0] / max(tab[0].sum(), 1)
        ghost_s = tab[1, 0] / max(tab[1].sum(), 1)
        res["students"][sname] = {"base_sides": {s: sb.count(s) for s in sorted(set(map(str, sb)))},
                                  "ghost_rate": round(float(tab[:, 0].sum() / max(tab.sum(), 1)), 3),
                                  "ghost_rate_teacher_unstable": round(float(ghost_u), 3), "ghost_rate_teacher_stable": round(float(ghost_s), 3),
                                  "table_unstable_stable_x_ghost_noghost": tab.tolist(),
                                  "student_unstable_rate": round(float(np.mean([any(side[(sname, v, names[i], o)] != side[(sname, "base", names[i], o)]
                                                                                    for v in small) for i, o in jobs])), 3)}
        if sname in ("tS4", "tS4o"):
            pooled += tab
    gu, gs = pooled[0, 0] / max(pooled[0].sum(), 1), pooled[1, 0] / max(pooled[1].sum(), 1)
    ratio = gu / gs if gs > 0 else float("inf")
    p = fisher_exact(pooled)[1]
    verdict = "CONFIRMED" if (ratio >= 2 and p < 0.05) else ("REJECTED" if (ratio < 1.5 or p >= 0.2) else "INCONCLUSIVE")
    res["pooled_tS4_tS4o"] = {"table": pooled.tolist(), "ghost_unstable": round(float(gu), 3), "ghost_stable": round(float(gs), 3),
                              "ratio": round(float(ratio), 2), "fisher_p": round(float(p), 4), "verdict": verdict}
    res["minutes"] = round((time.time() - t0) / 60, 1)
    (ROOT / "results/s15_ghost_test.json").write_text(json.dumps(res, indent=1))
    print("[s15] " + json.dumps(res), flush=True)

    # example sheet: student ghost cases with teacher base + variants
    ex = [(f, o) for (f, o), u in unstable.items() if side[("tS4", "base", f, o)] == "LR"][:8]
    cols = [("teacher", "base"), ("teacher", "v1"), ("teacher", "v2"), ("teacher", "v3"), ("teacher", "v4"), ("tS4", "base"), ("tS4o", "base"), ("dl6g2", "base")]
    g = Image.new("RGB", (len(cols) * c, 20 + max(len(ex), 1) * (c + 14)), "white")
    d = ImageDraw.Draw(g)
    d.text((4, 4), "tS4 ghost cases (hands both sides): teacher base + 4 small input variants | tS4 | tS4o | dl6g2 (sides found)", fill="black")
    for r_, (f, o) in enumerate(ex):
        for j, (m, vn) in enumerate(cols):
            px, py = j * c, 20 + r_ * (c + 14)
            g.paste(Image.open(OUT / m / vn / f"{f}_{o}.png").convert("RGB").resize((c, c)), (px, py))
            d.text((px + 3, py + c + 1), f"{m} {vn} {o}: {side[(m, vn, f, o)]}", fill="black")
    g.save(ROOT / "grids/s15_ghost_cases.jpg", quality=88)
    print("[s15] done", res["minutes"], "min", flush=True)


if __name__ == "__main__":
    main()
