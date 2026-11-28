"""Ghost-hand metric, step 1 (2026-11-27, user "yes start the ghost metric first"): hand detection with OWLv2 (open-vocabulary
detector, google/owlv2-base-patch16-ensemble, run locally; public weights, no image leaves the machine) on the images already made by
scripts/s15_ghost_test.py (outputs/s15_ghost/<model>/<variant>/<face>_<opt>.png: teacher tB1 at the tS4 pair setting with the fixed
phone noise, dl6g2 / tS4 / tS4o; 100 eval faces x thumbs / wave / ok / think; base + 4 small input variants + mirror).
Queries "a hand", "a human hand", "a cartoon hand"; per image: boxes with score >= 0.05 after class-agnostic NMS (IoU 0.3), plus each
box's opacity proxy = mean |pixel - white| inside the box (a faded / see-through hand is close to the background) and its edge energy.
Step 1 writes detections + an eye-check sheet (grids/s18_owl_check.jpg: known ghost cases first, then random); the thresholds that
turn detections into a ghost rate are chosen from that sheet BEFORE any rate is computed (step 2, s18_ghost_rate.py).
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/s18_owl_hands.py
Out: results/s18_owl_detections.json, grids/s18_owl_check.jpg
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs/s15_ghost"
QUERIES = ["a hand", "a human hand", "a cartoon hand"]
KNOWN = [("tS4o", "face_002", "wave"), ("tS4", "face_034", "thumbs"), ("tS4o", "face_034", "thumbs"), ("tS4", "face_002", "ok"),
         ("dl6g2", "face_097", "thumbs"), ("dl6g2", "face_034", "thumbs"), ("teacher", "face_002", "wave"), ("teacher", "face_034", "thumbs")]


def main():
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    t0 = time.time()
    name = "google/owlv2-base-patch16-ensemble"
    proc = Owlv2Processor.from_pretrained(name)
    model = Owlv2ForObjectDetection.from_pretrained(name, torch_dtype=torch.float16).to("cuda").eval()
    models = [x for x in sys.argv[1:] if not x.startswith("--")]  # 2026-11-27: optional model filter (parallel runs)
    outf = ROOT / (f"results/s18_owl_detections_{'_'.join(models)}.json" if models else "results/s18_owl_detections.json")
    files = sorted(f for f in SRC.glob("*/*/*.png") if not models or f.relative_to(SRC).parts[0] in models)
    print(f"[s18] {len(files)} images, model loaded {time.time() - t0:.0f}s", flush=True)
    det = {}
    bs = 32
    for i in range(0, len(files), bs):
        ims = [Image.open(f).convert("RGB") for f in files[i:i + bs]]
        inp = proc(text=[QUERIES] * len(ims), images=ims, return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].half()
        with torch.inference_mode():
            out = model(**inp)
        # OWLv2 pads to a square; our images are square, so target size = original size
        res = proc.post_process_grounded_object_detection(out, threshold=0.05, target_sizes=torch.tensor([[im.height, im.width] for im in ims]).cuda())
        for f, im, r in zip(files[i:i + bs], ims, res):
            b, s = r["boxes"].float(), r["scores"].float()
            keep = nms(b, s, 0.3) if len(s) else torch.zeros(0, dtype=torch.long)
            a = np.asarray(im).astype(np.float32)
            boxes = []
            for k in keep.tolist():
                x0, y0, x1, y1 = [int(round(v)) for v in b[k].tolist()]
                x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, im.width), min(y1, im.height)
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                patch = a[y0:y1, x0:x1]
                g = patch.mean(2)
                edge = float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())
                boxes.append({"box": [x0, y0, x1, y1], "score": round(float(s[k]), 3),
                              "opacity": round(float(np.abs(255 - patch).mean() / 255), 3), "edge": round(edge, 2)})
            det[str(f.relative_to(SRC))] = boxes
        if (i // bs) % 50 == 0:
            print(f"[s18] {i + len(ims)}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    outf.write_text(json.dumps(det))
    if models:  # filtered run: no eye-check sheet
        print(f"[s18] done {time.time() - t0:.0f}s -> {outf}", flush=True)
        return
    # eye-check sheet: known ghost cases (base) first, then random base images; boxes coloured by score
    random.seed(0)
    keys = [f"{m}/base/{f}_{o}.png" for m, f, o in KNOWN]
    rest = [k for k in det if "/base/" in k and k not in keys]
    keys += random.sample(rest, 40)
    c = 240
    g = Image.new("RGB", (8 * c, 20 + 6 * (c + 28)), "white")
    d = ImageDraw.Draw(g)
    d.text((4, 4), "OWLv2 hand boxes (score >= 0.05): red >= 0.3, orange 0.15-0.3, yellow < 0.15; label = score / opacity. First 8 = known ghost cases + teacher refs",
           fill="black")
    for n, k in enumerate(keys[:48]):
        im = Image.open(SRC / k).convert("RGB")
        dr = ImageDraw.Draw(im)
        for bx in det[k]:
            col = (255, 0, 0) if bx["score"] >= 0.3 else ((255, 140, 0) if bx["score"] >= 0.15 else (230, 200, 0))
            dr.rectangle(bx["box"], outline=col, width=4)
            dr.text((bx["box"][0] + 4, bx["box"][1] + 4), f"{bx['score']:.2f}/{bx['opacity']:.2f}", fill=col)
        px, py = (n % 8) * c, 20 + (n // 8) * (c + 28)
        g.paste(im.resize((c, c)), (px, py))
        d.text((px + 3, py + c + 2), k.replace("/base/", " ").replace(".png", ""), fill="black")
    g.save(ROOT / "grids/s18_owl_check.jpg", quality=90)
    print(f"[s18] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
