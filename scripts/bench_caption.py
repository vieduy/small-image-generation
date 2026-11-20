"""Describer speed check (2026-11-18): sequential caption() vs caption_batch() on 64 FFHQ photos; prints s/img and field agreement."""
import sys, time, json
sys.path.insert(0, "/mnt/project/kchat/duyhv/i2i-sticker-avatar/scripts")
from pathlib import Path
from PIL import Image
import caption_photos as CP
ps = sorted(Path("/mnt/project/kchat/duyhv/i2i-sticker-avatar/data/ffhq").glob("*.png"))[1000:1064]
ims = [Image.open(p).convert("RGB").resize((448, 448)) for p in ps]
proc, model = CP.load()
CP.caption(proc, model, ims[0])  # warm-up
t = time.time(); seq = [CP.caption(proc, model, im) for im in ims]; ts = time.time() - t
res = {}
for bs in (16, 32, 64):
    t = time.time(); bat = CP.caption_batch(proc, model, ims, bs); res[bs] = (time.time() - t, bat)
print(f"sequential: {ts / len(ims):.3f} s/img")
for bs, (tb, bat) in res.items():
    agree = {k: sum(a[k] == b[k] for a, b in zip(seq, bat)) / len(ims) for k in ("gender", "age", "child", "bald", "skin_tone", "hair", "clothing")}
    parsed = sum(bool(b["gender"]) for b in bat) / len(ims)
    print(f"batch {bs}: {tb / len(ims):.3f} s/img ({ts / tb:.1f}x), json ok {parsed:.2f}, agreement with sequential {agree}")
