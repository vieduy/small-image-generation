"""Side-by-side on dl1's own sample rows: photo | klein | SDXS b1 | DreamLite dl1 (latest samples sheet) | SDXS c3_dmd3 checkpoint."""
import json, sys
from pathlib import Path
import torch
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).parent))
from train_student import DEV, TASKS, alphas_cumprod, expr_tasks, forward, to_t  # noqa: E402
from diffusers import AutoencoderTiny, UNet2DConditionModel  # noqa: E402
from peft import PeftModel  # noqa: E402
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/pairs/cartoon3d"
tasks = TASKS + expr_tasks()
rows = [json.loads(l) for l in open(D / "manifest_round4.jsonl")]
rows = [r for r in rows if r["keep"] and r["task"] in tasks]
val = []
for k, t in enumerate(tasks):
    val += [r for r in rows if r["task"] == t][:32 if k < 2 else 4]
G = ("think", "thumbs", "wave", "ok", "facepalm", "shrug", "thanks")
pick = val[:3] + val[32:35] + [[r for r in val if r["task"] == f"expr_{g}"][0] for g in G]
dl_sheet = Image.open(sys.argv[1])  # dl1 samples_XXXXX.jpg: rows of (photo | klein | student) at 256
tae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to(DEV).eval()
ac = alphas_cumprod().to(DEV)
def load(tag):
    w = ROOT / "weights/student" / tag; cfg = json.loads((w / "config.json").read_text())
    if cfg.get("lora_rank", 0) > 0:
        u = PeftModel.from_pretrained(UNet2DConditionModel.from_pretrained(ROOT / "weights/student" / cfg["init_from"] / "unet"), w / "unet_lora").merge_and_unload()
    else:
        u = UNet2DConditionModel.from_pretrained(w / "unet")
    return u.to(DEV, torch.float16).eval(), torch.load(w / "task_emb.pt")["emb"].to(DEV, torch.float16), cfg
models = [load("b1"), load(sys.argv[2])]
S = 256
out = Image.new("RGB", (S * 5, 22 + S * len(pick)), "white"); d = ImageDraw.Draw(out)
for j, l in enumerate(["photo", "klein target", "SDXS b1 (regression)", "DreamLite dl1 (regression)", f"SDXS + DMD ({sys.argv[2]})"]):
    d.text((j * S + 4, 5), l, fill="black")
for i, r in enumerate(pick):
    y = 22 + i * S
    out.paste(dl_sheet.crop((0, i * S, 2 * S, (i + 1) * S)), (0, y))            # photo | klein (from the dl1 sheet)
    out.paste(dl_sheet.crop((2 * S, i * S, 3 * S, (i + 1) * S)), (3 * S, y))    # DreamLite student
    x = to_t(Image.open(D / "inputs" / f"{r['id']}.png"))[None].to(DEV, torch.float16)
    k = tasks.index(r["task"])
    with torch.inference_mode():
        z = tae.encode(x).latents
        for col, (u, emb, cfg) in zip((2, 4), models):
            yimg = tae.decode(forward(u, z, emb[k:k + 1], cfg["param"], cfg["T"], ac)).sample[0]
            im = Image.fromarray(((yimg.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy())
            out.paste(im.resize((S, S), Image.LANCZOS), (col * S, y))
    d.text((4, y + 4), r["task"].replace("expr_", ""), fill="white")
out.save(ROOT / "grids/cmp_dl1_vs_sdxs.jpg", quality=88); print("wrote grids/cmp_dl1_vs_sdxs.jpg")
