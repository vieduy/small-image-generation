"""Pose-conditioning go / no-go (2026-11-27, user "yes go with the 2k step check"): does the student FOLLOW the pose map?
Student = weights/student/<tag> (train_dl_pose.py, ~2k steps from tS4). 50 eval faces (data/faces[:50], in no training set) x thumbs / wave / ok,
each rendered with (a) the option's TEMPLATE map (s23 pose_templates: median keypoints of the most common hand pattern), (b) the MIRRORED
template (map flipped left-right and L/R hand channels swapped -> the hand on the other image side), (c) an all-zero map (= no pose; reference).
Hand side = OWLv2 hand boxes (s18 rules: score >= 0.12, nested boxes merged); side of the highest-scoring hand (box centre x vs 256).
Expected side = the template hand's side (keypoint centre of its visible hand); mirrored = the opposite.
Rule (fixed before running): CONTINUE (12k-step tSP1) if the hand is on the expected side in >= 70% of (a) + (b) cases with a detected hand;
otherwise STOP and switch to a stronger injection (pose features at several UNet levels).
Also reported: extra-hand (>= 2 hands) and no-hand rates for (a), (b), (c).
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/s24_swap_test.py tSP0
Out: results/s24_swap_<tag>.json, grids/s24_swap_<tag>.jpg
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
import pose_student as PS  # noqa: E402
from eval_e1 import to_t  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OPTS = ["thumbs", "wave", "ok"]
QUERIES = ["a hand", "a human hand", "a cartoon hand"]
LH0, RH0 = 91, 112


def main():
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    tag = sys.argv[1]
    tfile = sys.argv[2] if len(sys.argv) > 2 else "pose_templates_teacher_tB1_s8_img3.0_k1.pt"  # 2026-11-27: tSP0b uses the dense templates
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    T = torch.load(ROOT / "data/pairs/cartoon3d" / tfile)
    unet, te = PS.load_pose_student_any(ROOT / "weights/student" / tag)
    emb, mask, tasks = te["emb"].to("cuda", torch.float16), te["mask"].to("cuda"), te["tasks"]
    enc = DLA.load_vae(torch.float16).to("cuda").eval()
    dec = DLA.load_vae(torch.float16, ROOT / "weights/decoder/decg2").to("cuda").eval()
    faces = sorted((ROOT / "data/faces").glob("*.png"))[:50]
    with torch.inference_mode():
        Z = DLA.encode(enc, torch.stack([to_t(Image.open(p)) for p in faces]).to("cuda", torch.float16))
    ad_ = getattr(unet, "_pose_adapter", None)
    nch = ad_.e0[0].in_channels if ad_ is not None else 3
    maps, expect = {}, {}
    for o in OPTS:
        m = torch.tensor(np.array(T[o]["map"], np.uint8)).float() / 255 * scale
        kp, sc = np.array(T[o]["kp"]), np.array(T[o]["score"])
        h0 = LH0 if sc[LH0:LH0 + 21].mean() >= sc[RH0:RH0 + 21].mean() else RH0
        side = "L" if kp[h0:h0 + 21, 0].mean() < 256 else "R"
        mir = m.flip(-1)[[1, 0, 2]]
        if nch == 4:  # tSP0c: map-active flag (1 with a map, 0 for "none")
            one = torch.ones(1, *m.shape[1:])
            m, mir = torch.cat([m, one]), torch.cat([mir, one])
        maps[(o, "template")], maps[(o, "mirrored")], maps[(o, "none")] = m, mir, torch.zeros_like(m)
        expect[(o, "template")], expect[(o, "mirrored")] = side, {"L": "R", "R": "L"}[side]
    imgs = {}
    with torch.inference_mode():
        for (o, kind), m in maps.items():
            k = tasks.index(f"expr_{o}")
            PS.set_pose(unet, PS.diptych(m[None].expand(len(faces), -1, -1, -1).to("cuda", torch.float16)))
            y = DLA.decode(dec, DLA.forward(unet, Z, emb[k:k + 1].expand(len(faces), -1, -1), mask[k:k + 1].expand(len(faces), -1)))
            imgs[(o, kind)] = [Image.fromarray(a) for a in ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()]
    del unet
    torch.cuda.empty_cache()
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", torch_dtype=torch.float16).to("cuda").eval()

    def hands(ims):
        out = []
        for j in range(0, len(ims), 25):
            b_ = ims[j:j + 25]
            inp = proc(text=[QUERIES] * len(b_), images=b_, return_tensors="pt").to("cuda")
            inp["pixel_values"] = inp["pixel_values"].half()
            with torch.inference_mode():
                r_ = proc.post_process_grounded_object_detection(owl(**inp), threshold=0.12, target_sizes=[(im.height, im.width) for im in b_])
            for r in r_:
                b, s = r["boxes"].float(), r["scores"].float()
                keep = sorted(nms(b, s, 0.3).tolist(), key=lambda k: -float(s[k])) if len(s) else []
                hs = []
                for k in keep:
                    x0, y0, x1, y1 = b[k].tolist()
                    a = (x1 - x0) * (y1 - y0)
                    if any(max(0, min(x1, X1) - max(x0, X0)) * max(0, min(y1, Y1) - max(y0, Y0)) >= 0.6 * min(a, (X1 - X0) * (Y1 - Y0))
                           for X0, Y0, X1, Y1 in hs):
                        continue
                    hs.append((x0, y0, x1, y1))
                out.append(hs)
        return out

    res, follow = {}, []
    for (o, kind), ims in imgs.items():
        H = hands(ims)
        sides = [("L" if (h[0][0] + h[0][2]) / 2 < 256 else "R") if h else "-" for h in H]
        r = {"extra_hand": round(float(np.mean([len(h) >= 2 for h in H])), 3), "no_hand": round(float(np.mean([len(h) == 0 for h in H])), 3),
             "side_counts": {x: sides.count(x) for x in "LR-"}}
        if kind != "none":
            ok = [s_ == expect[(o, kind)] for s_ in sides if s_ != "-"]
            r["expected_side"], r["follow_rate"] = expect[(o, kind)], round(float(np.mean(ok)), 3) if ok else None
            follow += ok
        res[f"{o}/{kind}"] = r
    fr = float(np.mean(follow)) if follow else 0.0
    ex_map = float(np.mean([res[f"{o}/{k}"]["extra_hand"] for o in OPTS for k in ("template", "mirrored")]))
    ex_none = float(np.mean([res[f"{o}/none"]["extra_hand"] for o in OPTS]))
    verdict = "CONTINUE" if fr >= 0.70 else "STOP"
    verdict2 = "CONTINUE" if (fr >= 0.70 and ex_map <= ex_none) else "STOP"  # tSP0c rule: the map must not add hands
    out = {"student": tag, "follow_rate_template_and_mirrored": round(fr, 3), "n": len(follow), "verdict": verdict,
           "extra_hand_with_map": round(ex_map, 3), "extra_hand_no_map": round(ex_none, 3), "verdict_with_extra_hand_rule": verdict2,
           "per_case": res}
    (ROOT / f"results/s24_swap_{tag}.json").write_text(json.dumps(out, indent=1))
    print("[s24] " + json.dumps(out), flush=True)
    c = 180
    g = Image.new("RGB", (9 * c, 20 + 6 * (c + 14)), "white")
    d = ImageDraw.Draw(g)
    d.text((4, 4), f"{tag}: per option, rows = template map | mirrored map | no map; 6 faces each (+ map in the first column)", fill="black")
    r_ = 0
    for o in OPTS:
        for kind in ("template", "mirrored"):
            y0 = 20 + r_ * (c + 14)
            mp = np.moveaxis((maps[(o, kind)][:3].numpy() / scale * 255).clip(0, 255).astype(np.uint8), 0, -1)
            g.paste(Image.fromarray(mp).resize((c, c), Image.NEAREST), (0, y0))
            d.text((3, y0 + c + 1), f"{o} {kind}", fill="black")
            for j in range(6):
                g.paste(imgs[(o, kind)][j].resize((c, c)), ((j + 1) * c, y0))
            for j in range(2):
                g.paste(imgs[(o, "none")][j].resize((c, c)), ((j + 7) * c, y0))
            r_ += 1
    g.save(ROOT / f"grids/s24_swap_{tag}.jpg", quality=88)


if __name__ == "__main__":
    main()
