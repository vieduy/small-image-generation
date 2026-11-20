"""Local photo attributes for teacher prompts (2026-11-18). Real photos (FFHQ, CelebA-derived r*) have no generator prompt,
so the old teacher prompt only said "a person with <tone> skin". This asks the Qwen3-VL-2B that ships inside DreamLite-mobile
(local weights, no external API) for structured attributes, so every photo gets the same detailed "keep exactly" prompt.
  $VENV/bin/python scripts/caption_photos.py OUT.jsonl IMG [IMG ...]      (or --list file_with_paths)
Each line: {"path", "gender", "age", "hair", "facial_hair", "glasses", "headwear", "clothing", "expression", "raw"}
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402

Q = ("Describe the main person in this photo for an artist. Answer with JSON only, no other text, with these keys:\n"
     '"gender": "man" or "woman" (or "person" if unclear),\n'
     '"age": one of "early twenties", "late twenties", "thirties", "forties", "fifties", "sixties", "seventies or older",\n'
     '"hair": hairstyle, length and colour, e.g. "short curly black hair" (or "bald"),\n'
     '"facial_hair": e.g. "full grey beard", or "none",\n'
     '"glasses": e.g. "thin black glasses", or "none",\n'
     '"headwear": e.g. "red baseball cap", or "none",\n'
     '"clothing": visible clothing with colours, e.g. "navy t-shirt", or "not visible",\n'
     '"expression": e.g. "broad smile", "neutral", "slight smile",\n'
     '"child": "yes" if the person is a baby, child or teenager (under 18), else "no",\n'
     '"bald": "yes" if the head is bald or shaved, else "no",\n'
     '"skin_tone": one of "very fair", "fair", "light", "medium", "tan", "brown", "dark brown", "very dark".')
KEYS = ("gender", "age", "hair", "facial_hair", "glasses", "headwear", "clothing", "expression", "child", "bald", "skin_tone")


def load():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(DLA.DL_REPO, revision=DLA.DL_REV, allow_patterns=["text_encoder/*", "processor/*", "tokenizer/*"]))
    proc = AutoProcessor.from_pretrained(root / "processor")
    model = Qwen3VLForConditionalGeneration.from_pretrained(root / "text_encoder", dtype=torch.bfloat16).to("cuda").eval()
    return proc, model


def caption(proc, model, img):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": Q}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=220, do_sample=False)
    txt = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    return {k: str(d.get(k, "")).strip() for k in KEYS} | {"raw": txt.strip()}


def _parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    return {k: str(d.get(k, "")).strip() for k in KEYS} | {"raw": txt.strip()}


def caption_batch(proc, model, imgs, bs=32):
    """Batched greedy decoding (left padding): same answers as caption() up to padding/bf16 noise, ~10x faster."""
    proc.tokenizer.padding_side = "left"
    out = []
    for i in range(0, len(imgs), bs):
        msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": Q}]}] for im in imgs[i:i + bs]]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
                                       padding=True).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**inp, max_new_tokens=220, do_sample=False)
        out += [_parse(t) for t in proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("imgs", nargs="*")
    ap.add_argument("--list")
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    paths = a.imgs + ([l.strip() for l in open(a.list) if l.strip()] if a.list else [])
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["path"] for l in open(a.out)}
    proc, model = load()
    todo = [p for p in paths if p not in done]
    with open(a.out, "a") as f:
        for i in range(0, len(todo), a.bs):
            ch = todo[i:i + a.bs]
            for p, r in zip(ch, caption_batch(proc, model, [Image.open(p).convert("RGB").resize((448, 448)) for p in ch], a.bs)):
                f.write(json.dumps({"path": p} | r) + "\n")
            f.flush()
            print(f"[caption] {min(i + a.bs, len(todo))}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
