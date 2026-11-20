"""Export a trained student to Core ML (.mlpackage, ML Program) for iPhone / iOS Simulator.
One model, two tasks: inputs  image (512x512 RGB, CVPixelBuffer/CGImage) + task (Int32[1]: 0=avatar, 1=sticker)
                      output  stylized (512x512 RGB image)
Graph: [0,255] -> [-1,1] -> TAESD encoder -> 1-step UNet (task tokens baked in, no text encoder) -> TAESD decoder -> [0,255].
Not included (do these in the app with Apple Vision, no extra model download):
  framing: VNDetectFaceRectanglesRequest + the crop rule of scripts/framing.py (face height ~0.56 of the square, top ~0.18)
  sticker cutout: VNGenerateForegroundInstanceMaskRequest (iOS 17+) -> dilate 8 px white outline -> WebP/PNG
Validation: Core ML cannot run on Linux, so we ship reference input/output PNGs (PyTorch fp32 of the same traced graph)
for an on-device comparison.
  source env.sh; CUDA_VISIBLE_DEVICES= $VENV/bin/python scripts/export_coreml.py --tag s5d
"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
import torch
from diffusers import AutoencoderTiny, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
from train_student import alphas_cumprod  # noqa: E402
ROOT = Path(__file__).resolve().parent.parent


class Student(torch.nn.Module):
    def __init__(self, unet, tae, emb, param, T, ac):
        super().__init__()
        self.unet, self.enc, self.dec = unet, tae.encoder, tae.decoder
        self.register_buffer("emb", emb)                      # [2, 77, 1024] task tokens
        self.register_buffer("ids", torch.arange(emb.shape[0]).float())
        self.T = int(T)                                       # python int: traced as a constant (no aten::Int on tensors)
        self.param, self.a = param, float(ac[T])

    def forward(self, image, task):                           # image in [-1,1] (Core ML ImageType scale/bias)
        sel = (task.float().reshape(1) == self.ids).float()   # one-hot without integer casts
        e = (sel.reshape(-1, 1, 1) * self.emb).sum(0, keepdim=True)
        z = self.enc(image)
        o = self.unet(z, self.T, encoder_hidden_states=e).sample
        if self.param != "direct":
            o = (z - (1 - self.a) ** 0.5 * o) / self.a ** 0.5
        return ((self.dec(o).clamp(-1, 1) + 1) * 127.5)      # [0,255] for an ImageType output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--int8", action="store_true", help="also write an 8-bit weight-quantized variant (quality unverified)")
    a = ap.parse_args()
    import coremltools as ct
    w = ROOT / "weights/student" / a.tag
    cfg = json.loads((w / "config.json").read_text())
    if cfg.get("lora_rank", 0) > 0:
        from peft import PeftModel
        unet = PeftModel.from_pretrained(UNet2DConditionModel.from_pretrained(ROOT / "weights/student" / cfg["init_from"] / "unet"),
                                         w / "unet_lora").merge_and_unload()
    else:
        unet = UNet2DConditionModel.from_pretrained(w / "unet")
    unet.set_attn_processor(AttnProcessor())                  # plain matmul/softmax attention: simplest op set for Core ML
    tae = AutoencoderTiny.from_pretrained("madebyollin/taesd")
    te = torch.load(w / "task_emb.pt")
    emb = te["emb"].float()
    tasks = te.get("tasks") or torch.load(ROOT / "weights/student" / cfg["init_from"] / "task_emb.pt")["tasks"]
    task_doc = ", ".join(f"{i}={t.replace('expr_', '')}" for i, t in enumerate(tasks))
    m = Student(unet.eval(), tae.eval(), emb, cfg["param"], cfg["T"], alphas_cumprod()).eval()
    od = w / "coreml"; od.mkdir(exist_ok=True)
    x = torch.rand(1, 3, 512, 512) * 2 - 1
    tk = torch.tensor([0], dtype=torch.int32)
    with torch.no_grad():  # torch.export (static shapes): torch 2.11 jit.trace leaves 1-D size tensors that coremltools 9 cannot cast
        tr = torch.export.export(m, (x, tk)).run_decompositions({})
    print("[coreml] exported", flush=True)
    ml = ct.convert(tr, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS16, compute_precision=ct.precision.FLOAT16,
                    inputs=[ct.ImageType(name="image", shape=(1, 3, 512, 512), scale=1 / 127.5, bias=[-1, -1, -1],
                                         color_layout=ct.colorlayout.RGB),
                            ct.TensorType(name="task", shape=(1,), dtype=np.int32)],
                    outputs=[ct.ImageType(name="stylized", color_layout=ct.colorlayout.RGB)])
    ml.short_description = (f"1-step text-free image-to-image student '{a.tag}' (style {cfg['style']}): task {task_doc}. "
                            "Input: face-framed 512x512 RGB photo. Base: SDXS-512-0.9 UNet (CreativeML OpenRAIL++-M) + TAESD (MIT).")
    ml.author = "i2i-sticker-avatar study"
    ml.license = "Derivative of SDXS-512-0.9 (CreativeML OpenRAIL++-M use restrictions apply) + TAESD (MIT)"
    ml.input_description["image"] = "512x512 RGB, face framed like scripts/framing.py"
    ml.input_description["task"] = f"Int32 menu option: {task_doc}"
    ml.output_description["stylized"] = "512x512 RGB stylized image (sticker: cut out with Vision + add 8px white outline in-app)"
    p = od / f"Student_{a.tag}_fp16.mlpackage"
    if p.exists():
        shutil.rmtree(p)
    ml.save(str(p))
    size = lambda d: round(sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file()) / 2**20, 1)
    info = {"tag": a.tag, "style": cfg["style"], "fp16_mlpackage": str(p.relative_to(ROOT)), "fp16_MB": size(p),
            "min_ios": 16, "inputs": {"image": "RGB 512x512", "task": f"Int32[1] {task_doc}"}, "output": "stylized RGB 512x512",
            "tasks": {i: t for i, t in enumerate(tasks)}}
    if a.int8:
        import coremltools.optimize.coreml as cto
        q = cto.linear_quantize_weights(ml, config=cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric")))
        p8 = od / f"Student_{a.tag}_w8.mlpackage"
        if p8.exists():
            shutil.rmtree(p8)
        q.save(str(p8))
        info |= {"w8_mlpackage": str(p8.relative_to(ROOT)), "w8_MB": size(p8), "w8_note": "8-bit weights; quality NOT verified (Core ML cannot run on Linux)"}
    # reference I/O from the same torch graph (fp32), for on-device comparison
    rd = od / "reference"; rd.mkdir(exist_ok=True)
    faces = sorted((ROOT / "data/faces").glob("*.png"))[:3] + sorted((ROOT / "data/subjects").glob("*.png"))[:1]
    for f in faces:
        im = Image.open(f).convert("RGB").resize((512, 512))
        im.save(rd / f"in_{f.stem}.png")
        xi = torch.from_numpy(np.array(im)).permute(2, 0, 1)[None].float() / 127.5 - 1
        ref_tasks = [t for t in ("avatar", "sticker", "expr_thumbs", "expr_wave", "expr_love") if t in tasks]
        for task in ref_tasks if f.stem.startswith("face") else ("avatar", "sticker"):
            k = tasks.index(task)
            with torch.no_grad():
                y = m(xi, torch.tensor([k], dtype=torch.int32))[0]
            Image.fromarray(y.round().byte().permute(1, 2, 0).numpy()).save(rd / f"out_{f.stem}_{task}.png")
    (od / "export_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
