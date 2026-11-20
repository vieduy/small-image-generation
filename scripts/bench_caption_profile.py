"""Describer profile (2026-11-18): where a batch of 32 spends its time (processor vs generate), attention impl, tokens."""
import sys, time
sys.path.insert(0, "/mnt/project/kchat/duyhv/i2i-sticker-avatar/scripts")
from pathlib import Path
import torch
from PIL import Image
import caption_photos as CP
ps = sorted(Path("/mnt/project/kchat/duyhv/i2i-sticker-avatar/data/ffhq").glob("*.png"))[2000:2032]
ims = [Image.open(p).convert("RGB").resize((448, 448)) for p in ps]
proc, model = CP.load()
print("attn impl:", model.config._attn_implementation, "| processor:", type(proc.image_processor).__name__)
proc.tokenizer.padding_side = "left"
msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": CP.Q}]}] for im in ims]
for rep in range(2):
    t = time.time()
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True).to("cuda")
    torch.cuda.synchronize(); tp = time.time() - t
    t = time.time()
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=220, do_sample=False)
    torch.cuda.synchronize(); tg = time.time() - t
    new = out.shape[1] - inp["input_ids"].shape[1]
    print(f"rep{rep}: processor {tp:.2f}s, generate {tg:.2f}s, prompt len {inp['input_ids'].shape[1]}, new tokens {new}, {tg / new * 1000:.0f} ms/step")
