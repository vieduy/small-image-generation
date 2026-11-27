"""Attribute agreement + face fine detail (2026-11-26, stage-2 follow-up sweep A; user "yes do A first").
Why: SFace likeness cannot tell "kept hair colour / beard / headwear / age" apart from "copied photo texture" (README 2026-11-26
sweep: APG and img-3 score highest on SFace with near-photo outputs). This measures what we want kept, independent of texture.
attr: CLIP ViT-L/14 zero-shot, same domain-neutral captions for photo and output, argmax per attribute group; agreement = output
      label == photo label, averaged over the groups below. Labels on photos are CLIP's (not ground truth): use as a relative
      score between settings / models on the same photos, with klein as reference.
tex:  mean |Laplacian| of the central face box at 256 px (as in the stage-2 deep dive); tex_face: same on a detected face box.
"""
import numpy as np
import torch
from PIL import Image

import metrics as M

GROUPS = {
    "hair": ["black hair", "brown hair", "blonde hair", "red hair", "grey or white hair",
             "brightly dyed pink, blue or purple hair", "a shaved or bald head"],
    "age": ["a young adult in their twenties", "a middle-aged adult in their forties or fifties", "an elderly person in their seventies"],
    "beard": ["a man with a beard or mustache", "a person with a clean-shaven face"],
    "headwear": ["a person wearing a hat, cap or beanie", "a person with a bare head"],
    "glasses": ["a person wearing glasses", "a person without glasses"],
}


def tex(img):
    g = np.asarray(img.convert("L").resize((256, 256)), dtype=np.float32)[70:190, 70:190]
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(np.abs(lap).mean())


def tex_box(img, box):
    """Same measure on a face box (x, y, w, h in the image's own pixels), resized to 128 px."""
    x, y, w, h = [int(round(v)) for v in box]
    W, H = img.size
    c = img.convert("L").crop((max(x, 0), max(y, 0), min(x + w, W), min(y + h, H))).resize((128, 128))
    g = np.asarray(c, dtype=np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(np.abs(lap).mean())


class Attr:
    def __init__(self, dev="cuda", cm=None, pr=None):
        from transformers import CLIPModel, CLIPProcessor
        if cm is None:
            cp = M._snap("openai/clip-vit-large-patch14")
            cm, pr = CLIPModel.from_pretrained(str(cp), dtype=torch.float16).to(dev).eval(), CLIPProcessor.from_pretrained(str(cp))
        self.cm, self.pr, self.dev = cm, pr, dev
        self.tf = {}
        with torch.inference_mode():
            for g, caps in GROUPS.items():
                ti = pr(text=[f"a picture of {c}" for c in caps], return_tensors="pt", padding=True).to(dev)
                t = cm.text_projection(cm.text_model(**ti).pooler_output).float()
                self.tf[g] = t / t.norm(dim=-1, keepdim=True)

    @torch.inference_mode()
    def labels(self, imgs, bs=50):
        """-> {group: np.array of label indices}"""
        fs = []
        for i in range(0, len(imgs), bs):
            pv = self.pr.image_processor(images=[im.convert("RGB") for im in imgs[i:i + bs]], return_tensors="pt")["pixel_values"]
            f = self.cm.visual_projection(self.cm.vision_model(pixel_values=pv.to(self.dev, torch.float16)).pooler_output).float()
            fs.append(f / f.norm(dim=-1, keepdim=True))
        f = torch.cat(fs)
        return {g: (f @ t.T).argmax(1).cpu().numpy() for g, t in self.tf.items()}

    @staticmethod
    def agree(lab_photo, lab_out):
        """-> {group: agreement rate, 'attr': mean over groups}"""
        r = {g: float(np.mean(lab_photo[g] == lab_out[g])) for g in GROUPS}
        r["attr"] = float(np.mean(list(r.values())))
        return r
