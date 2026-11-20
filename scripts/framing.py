"""Shared input normaliser (product + eval + demo): put any photo into the training framing.
The small student only saw synthetic head-and-shoulders inputs whose YuNet face box (measured on the 512 crop) has
height ~0.56 of the side (p10-p90 0.47-0.67) and top at ~0.18. A real arm's-length selfie cropped naively had 0.84 and
the student output became soft and generic; reframed to 0.56 it became clean (README, real-photo demo).
If the photo is too tight, the square is padded with a blurred copy of the photo (no hard border).
"""
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

TARGET_FACE, TARGET_TOP = 0.56, 0.18
FACE_RANGE, TOP_RANGE = (0.45, 0.62), (0.08, 0.28)   # accepted as-is; upper bound kept tight: 0.84 (selfie) gave soft output, 0.56 clean
PERSON_SCORE = 0.87  # YuNet score gate: eval people >= 0.889, animal false positives <= 0.852 (thin margin, 430 imgs)


def normalize_face_crop(img, fid, face_frac=TARGET_FACE, top=TARGET_TOP, size=512, iters=6):
    """Returns (512 RGB image, info or None). info None = no confident human face (pet/object): plain centre square.
    A photo whose face is already inside the training range is only square-cropped (no zoom, no padding)."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    # detect on a <=640 px copy: YuNet scores/boxes on 3-4k phone photos are unreliable (low scores, odd box sizes)
    k = min(1.0, 640 / max(w, h))
    small = img.resize((max(1, int(w * k)), max(1, int(h * k))), Image.LANCZOS)
    a = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2BGR)
    fid.det.setInputSize(small.size); _, fs = fid.det.detect(a)
    if fs is not None and len(fs):
        f0 = max(fs, key=lambda r: r[2] * r[3]).copy()
        f0[:14] /= k  # box + landmarks back to full-res coordinates (score untouched)
    if fs is None or len(fs) == 0 or float(f0[-1]) < PERSON_SCORE:   # no confident human face: plain centre square
        s = min(w, h)
        return img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.LANCZOS), None
    x, y, fw, fh = f0[:4]
    blur = None

    def crop_at(side, cx, ty):
        nonlocal blur
        side = int(side); x0, y0 = int(cx - side / 2), int(ty)
        pad = max(0, -x0, -y0, x0 + side - w, y0 + side - h)
        if pad:
            big = img.resize((w + 2 * pad, h + 2 * pad)).filter(ImageFilter.GaussianBlur(40))
            big.paste(img, (pad, pad))
            return big.crop((x0 + pad, y0 + pad, x0 + pad + side, y0 + pad + side)).resize((size, size), Image.LANCZOS), pad
        return img.crop((x0, y0, x0 + side, y0 + side)).resize((size, size), Image.LANCZOS), 0

    def box(im):
        b = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        fid.det.setInputSize((size, size)); _, ff = fid.det.detect(b)
        if ff is None or len(ff) == 0:
            return None
        f = max(ff, key=lambda r: r[2] * r[3])
        return f[1] / size, f[3] / size, (f[0] + f[2] / 2) / size

    # 1) natural square (no zoom): keep it if the face is already in the training range
    side0 = min(w, h)
    cx0 = min(max(x + fw / 2, side0 / 2), w - side0 / 2)
    ty0 = min(max(y - top * side0 * 1.0 + 0 * fh, 0), h - side0)
    im, _ = crop_at(side0, cx0, ty0)
    b = box(im)
    if b is not None and FACE_RANGE[0] <= b[1] <= FACE_RANGE[1] and TOP_RANGE[0] <= b[0] <= TOP_RANGE[1]:
        return im, {"face_box_512": b, "padded": False, "reframed": False, "score": float(f0[-1])}
    # 2) out of range (e.g. arm's-length selfie): iterate zoom/shift to the training median, pad if needed
    side, cx, ty = float(min(w, h)), x + fw / 2, y - 0.3 * fh
    for _ in range(iters):
        im, _ = crop_at(side, cx, ty)
        b = box(im)
        if b is None:
            break
        t, fr, xc = b
        side *= fr / face_frac
        cx += (xc - 0.5) * side
        ty += (t - top) * side
    im, pad = crop_at(side, cx, ty)
    return im, {"face_box_512": box(im), "padded": bool(pad), "reframed": True, "score": float(f0[-1])}
