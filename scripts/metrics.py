"""Quality metrics for avatar/sticker outputs (evaluation only; models listed in LICENSES.md).

Identity  : OpenCV Zoo YuNet detector + SFace recognizer (Apache-2.0). Cosine of aligned-face features input vs output.
            SFace's same-identity threshold (OpenCV docs) is cosine >= 0.363. Stylized faces may not be detected:
            detection failures are reported as a rate, and counted as identity failures in pass rates.
Style     : DINOv2-base CLS and CLIP ViT-L/14 image embeddings. style_* = cosine(output, mean of style refs).
            refcopy_* = max cosine to any single ref (high = copying a reference instead of stylizing the input).
Sticker   : BiRefNet_lite (MIT) matte -> binary alpha -> keep largest component -> 8 px white outline (dilation)
            -> fit into 512x512 with margin -> WebP (alpha) quality search <= 100 KB.
            pass = subject found (5%..90% of area) & main component >= 90% of fg & webp <= 100 KB.
            raw_bg_white = share of near-white pixels in the raw output background (1.0 = model already drew a clean white bg).
"""
import io
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
HUB = Path(os.environ.get("HF_HOME", ROOT / "hf_cache")) / "hub"
DEV = "cuda"
SFACE_THR = 0.363


def _snap(repo):
    d = HUB / f"models--{repo.replace('/', '--')}" / "snapshots"
    return next(d.iterdir())


class FaceID:
    def __init__(self):
        self.det = cv2.FaceDetectorYN.create(str(_snap("opencv/face_detection_yunet") / "face_detection_yunet_2023mar.onnx"),
                                             "", (320, 320), 0.6, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(str(_snap("opencv/face_recognition_sface") / "face_recognition_sface_2021dec.onnx"), "")

    def feat(self, img: Image.Image):
        a = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        h, w = a.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(a)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda r: r[2] * r[3])  # largest face
        return self.rec.feature(self.rec.alignCrop(a, f))

    def skin_L(self, img: Image.Image):
        """Median L* (CIELAB, 0-100) of the central 40% of the largest detected face box (cheeks/nose), or None.
        Skin-tone preservation = skin_L(output) - skin_L(input); ~0 is good, large + means the face was lightened."""
        a = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        h, w = a.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(a)
        if faces is None or len(faces) == 0:
            return None
        x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])[:4]
        cx, cy = x + fw / 2, y + fh * 0.55
        x0, x1 = int(max(0, cx - 0.2 * fw)), int(min(w, cx + 0.2 * fw))
        y0, y1 = int(max(0, cy - 0.2 * fh)), int(min(h, cy + 0.2 * fh))
        if x1 <= x0 or y1 <= y0:
            return None
        lab = cv2.cvtColor(a[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)
        return float(np.median(lab[:, :, 0])) * 100 / 255

    def cos(self, fa, fb):
        if fa is None or fb is None:
            return None
        return float(self.rec.match(fa, fb, cv2.FaceRecognizerSF_FR_COSINE))


class Embedder:
    def __init__(self):
        from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPVisionModelWithProjection
        self.dp = AutoImageProcessor.from_pretrained(str(_snap("facebook/dinov2-base")))
        self.dm = AutoModel.from_pretrained(str(_snap("facebook/dinov2-base")), torch_dtype=torch.float16).to(DEV).eval()
        cp = _snap("openai/clip-vit-large-patch14")
        self.cp = CLIPImageProcessor.from_pretrained(str(cp))
        self.cm = CLIPVisionModelWithProjection.from_pretrained(str(cp), torch_dtype=torch.float16).to(DEV).eval()

    @torch.inference_mode()
    def __call__(self, imgs):
        imgs = [i.convert("RGB") for i in imgs]
        d = self.dm(**{k: v.to(DEV, torch.float16) for k, v in self.dp(imgs, return_tensors="pt").items()}).last_hidden_state[:, 0]
        c = self.cm(**{k: v.to(DEV, torch.float16) for k, v in self.cp(imgs, return_tensors="pt").items()}).image_embeds
        return F.normalize(d.float(), dim=-1), F.normalize(c.float(), dim=-1)


class Matter:
    def __init__(self):
        from transformers import AutoModelForImageSegmentation
        self.m = AutoModelForImageSegmentation.from_pretrained(str(_snap("ZhengPeng7/BiRefNet_lite")), trust_remote_code=True)
        self.m = self.m.to(DEV).eval().half()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

    @torch.inference_mode()
    def batch(self, imgs, size=1024):
        """Batched matte (same math as __call__); returns a list of HxW float arrays in [0,1]."""
        x = torch.stack([torch.from_numpy(np.array(i.convert("RGB").resize((size, size)))).permute(2, 0, 1) for i in imgs]).to(DEV).float() / 255
        x = ((x - self.mean) / self.std).half()
        p = self.m(x)[-1].sigmoid().float()
        return [F.interpolate(p[k:k + 1], size=im.size[::-1], mode="bilinear")[0, 0].clamp(0, 1).cpu().numpy() for k, im in enumerate(imgs)]

    @torch.inference_mode()
    def __call__(self, img: Image.Image, size=1024):
        x = torch.from_numpy(np.array(img.convert("RGB").resize((size, size)))).to(DEV).permute(2, 0, 1)[None].float() / 255
        x = ((x - self.mean) / self.std).half()
        p = self.m(x)[-1].sigmoid().float()
        p = F.interpolate(p, size=img.size[::-1], mode="bilinear")[0, 0]
        return p.clamp(0, 1).cpu().numpy()


def sticker_postprocess(img: Image.Image, alpha: np.ndarray, outline_px=8, out_size=512, max_kb=100, webp_method=6):
    """Returns (webp_bytes or None, info dict). Deterministic, phone-friendly (no model besides the matte)."""
    a = (alpha > 0.5).astype(np.uint8)
    fg_frac = float(a.mean())
    info = {"fg_frac": round(fg_frac, 4)}
    n, lab, stats, _ = cv2.connectedComponentsWithStats(a, connectivity=8)
    if n <= 1:
        info.update(main_comp=0.0, webp_kb=None, pass_sticker=False, fail="no_subject")
        return None, info
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    info["main_comp"] = round(float(areas.max() / areas.sum()), 4)
    raw = np.array(img.convert("RGB"))
    bg = a == 0
    info["raw_bg_white"] = round(float((raw[bg].min(axis=1) > 235).mean()) if bg.any() else 0.0, 4)
    keep = (lab == k).astype(np.uint8) if info["main_comp"] < 0.9 else a  # drop specks only if they matter
    # crop to subject, add margin for the outline, fit into out_size
    ys, xs = np.where(keep)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgba = np.dstack([raw, keep * 255]).astype(np.uint8)[y0:y1, x0:x1]
    m = outline_px * 2
    side = max(y1 - y0, x1 - x0)
    scale = (out_size - 2 * m) / side
    sub = Image.fromarray(rgba).resize((max(1, int((x1 - x0) * scale)), max(1, int((y1 - y0) * scale))), Image.LANCZOS)
    canvas = Image.new("RGBA", (out_size, out_size), (0, 0, 0, 0))
    canvas.paste(sub, ((out_size - sub.width) // 2, (out_size - sub.height) // 2), sub)
    al = (np.array(canvas)[:, :, 3] > 127).astype(np.uint8)
    ring = cv2.dilate(al, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outline_px + 1, 2 * outline_px + 1)))
    base = np.zeros((out_size, out_size, 4), np.uint8)
    base[ring > 0] = (255, 255, 255, 255)
    out = Image.alpha_composite(Image.fromarray(base), canvas)
    data = None
    for q in (90, 80, 70, 60, 50, 40):
        b = io.BytesIO()
        out.save(b, format="WEBP", quality=q, method=webp_method)
        if b.tell() <= max_kb * 1024:
            data, info["webp_q"] = b.getvalue(), q
            break
    info["webp_kb"] = round(len(data) / 1024, 1) if data else None
    ok = 0.05 <= fg_frac <= 0.9 and info["main_comp"] >= 0.9 and data is not None
    info["pass_sticker"] = bool(ok)
    if not ok:
        info["fail"] = "fg_frac" if not 0.05 <= fg_frac <= 0.9 else ("fragmented" if info["main_comp"] < 0.9 else "size")
    return data, info


class SubjectCheck:
    """Non-person sticker fidelity (humanization bug, README deviation 8).
    human_face      : YuNet finds a face in the output (proxy; cartoon animal faces can also trigger it).
    clip_subject_ok : CLIP ViT-L/14 prefers 'a cartoon sticker of a {subject}' over 'a cartoon sticker of a person's face'."""
    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        cp = _snap("openai/clip-vit-large-patch14")
        self.m = CLIPModel.from_pretrained(str(cp), dtype=torch.float16).to(DEV).eval()
        self.p = CLIPProcessor.from_pretrained(str(cp))
        self.fid = FaceID()

    @torch.inference_mode()
    def batch(self, imgs, subjects):
        out = []
        for img, s in zip(imgs, subjects):  # CLIP pairs differ per image (own subject text); face check is cheap
            out.append(self(img, s))
        return out

    @torch.inference_mode()
    def __call__(self, img, subject):
        img = img.convert("RGB")
        r = {"human_face": self.fid.feat(img) is not None}
        if subject:
            x = self.p(text=[f"a cartoon sticker of a {subject}", "a cartoon sticker of a person's face"], images=img,
                       return_tensors="pt", padding=True).to(DEV)
            lg = self.m(**{k: (v.half() if v.dtype == torch.float32 else v) for k, v in x.items()}).logits_per_image[0]
            r["clip_subject_ok"] = bool(lg[0] > lg[1])
        return r


class FaceCache:
    """Persistent per-image cache of (SFace feature, skin L*) so re-runs only run YuNet/SFace on new or changed files.
    One detection per image (feat + skin from the same face box). Key = (path relative to root, mtime_ns, size).
    OpenCV DNN runs on CPU here (CUDA target unsupported), which made full re-scoring of 16k pairs take ~20 min."""
    def __init__(self, fid, root, name="face_cache.pkl"):
        import pickle
        self.fid, self.root, self.f, self.pickle = fid, Path(root), Path(root) / name, pickle
        self.d = pickle.loads(self.f.read_bytes()) if self.f.exists() else {}
        self.hits = self.miss = 0

    def get(self, p):
        p = Path(p)
        stt = p.stat()
        k = (str(p.relative_to(self.root)), stt.st_mtime_ns, stt.st_size)
        if k in self.d:
            self.hits += 1
            return self.d[k]
        self.miss += 1
        a = cv2.cvtColor(np.array(Image.open(p).convert("RGB")), cv2.COLOR_RGB2BGR)
        h, w = a.shape[:2]
        self.fid.det.setInputSize((w, h))
        _, faces = self.fid.det.detect(a)
        if faces is None or len(faces) == 0:
            v = (None, None)
        else:
            f = max(faces, key=lambda r: r[2] * r[3])
            feat = self.fid.rec.feature(self.fid.rec.alignCrop(a, f))
            x, y, fw, fh = f[:4]
            cx, cy = x + fw / 2, y + fh * 0.55
            x0, x1 = int(max(0, cx - 0.2 * fw)), int(min(w, cx + 0.2 * fw))
            y0, y1 = int(max(0, cy - 0.2 * fh)), int(min(h, cy + 0.2 * fh))
            L = None if x1 <= x0 or y1 <= y0 else float(np.median(cv2.cvtColor(a[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)[:, :, 0])) * 100 / 255
            v = (feat, L)
        self.d[k] = v
        return v

    def save(self):
        tmp = self.f.with_suffix(".tmp")
        tmp.write_bytes(self.pickle.dumps(self.d))
        tmp.replace(self.f)


class AttrCheck:
    """Verify that a generation-prompt 'extra' (beard, glasses, cap, ...) is actually visible in the photo, so the
    per-person teacher prompt does not inject attributes the generator failed to draw (README: face_041 beard bug).
    CLIP ViT-L/14 zero-shot, positive vs specific negative caption. Expressions are not checked (returns None)."""
    NEG = {"with a beard": "a clean-shaven {g}", "with a mustache": "a clean-shaven {g}", "wearing glasses": "a {g} without glasses",
           "wearing sunglasses on the head": "a {g} with nothing on the head", "with freckles": "a {g} with clear skin",
           "wearing earrings": "a {g} without earrings", "wearing a cap": "a {g} with a bare head", "wearing a beanie": "a {g} with a bare head",
           "wearing headphones around the neck": "a {g} without headphones", "with a nose ring": "a {g} without a nose piercing"}

    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        cp = _snap("openai/clip-vit-large-patch14")
        self.m = CLIPModel.from_pretrained(str(cp), dtype=torch.float16).to(DEV).eval()
        self.p = CLIPProcessor.from_pretrained(str(cp))

    @torch.inference_mode()
    def __call__(self, img, gender, extra):
        if not extra or extra not in self.NEG:
            return None
        g = {"woman": "woman", "man": "man"}.get(gender, "person")
        x = self.p(text=[f"a photo of a {g} {extra}", "a photo of " + self.NEG[extra].format(g=g)], images=img.convert("RGB"),
                   return_tensors="pt", padding=True).to(DEV)
        lg = self.m(**{k: (v.half() if v.dtype == torch.float32 else v) for k, v in x.items()}).logits_per_image[0]
        return bool(lg[0] > lg[1])
