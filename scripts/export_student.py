"""Mobile-path conversion check for a trained student: one ONNX graph per task, photo -> stylized image.
Graph: x [1,3,512,512] in [-1,1] -> TAESD encoder -> UNet (task tokens baked in as a constant) -> TAESD decoder -> y [-1,1].
No text encoder. Checks: export succeeds, ONNX Runtime CPU output matches PyTorch fp32, file size, CPU latency.
CPU numbers below are from this x86 server CPU, NOT phone numbers.

  source env.sh; CUDA_VISIBLE_DEVICES= $VENV/bin/python scripts/export_student.py --tag s1d
Out: weights/student/<tag>/onnx/student_<task>.onnx (+ .data), results/student_<tag>_export.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from diffusers import AutoencoderTiny, UNet2DConditionModel
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
from train_student import SDXS, TASKS, alphas_cumprod, to_t  # noqa: E402
ROOT = Path(__file__).resolve().parent.parent


class Student(torch.nn.Module):
    def __init__(self, unet, tae, emb, param, T, ac):
        super().__init__()
        self.unet, self.enc, self.dec = unet, tae.encoder, tae.decoder
        self.register_buffer("emb", emb[None])
        self.register_buffer("t", torch.tensor([T]))
        self.param, self.a = param, float(ac[T])

    def forward(self, x):
        z = self.enc(x)                                     # EncoderTiny maps [-1,1] -> [0,1] internally
        o = self.unet(z, self.t, encoder_hidden_states=self.emb).sample
        if self.param != "direct":
            o = (z - (1 - self.a) ** 0.5 * o) / self.a ** 0.5
        return self.dec(o)                                  # DecoderTiny returns [-1,1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    w = ROOT / "weights/student" / a.tag
    cfg = json.loads((w / "config.json").read_text())
    unet = UNet2DConditionModel.from_pretrained(w / "unet").eval()
    tae = AutoencoderTiny.from_pretrained("madebyollin/taesd").eval()
    emb = torch.load(w / "task_emb.pt")["emb"].float()
    ac = alphas_cumprod()
    od = w / "onnx"; od.mkdir(exist_ok=True)
    x = to_t(Image.open(sorted((ROOT / "data/faces").glob("*.png"))[0]))[None]
    res = {"tag": a.tag, "note": "x86 server CPU timings, NOT phone numbers", "torch": torch.__version__}
    import onnxruntime as ort
    for k, task in enumerate(TASKS):
        m = Student(unet, tae, emb[k], cfg["param"], cfg["T"], ac).eval()
        with torch.inference_mode():
            ref = m(x)
            # sanity: same as the diffusers path used in eval (tae.encode -> unet -> tae.decode)
            z = tae.encode(x).latents
            o = unet(z, torch.tensor([cfg["T"]]), encoder_hidden_states=emb[k][None]).sample
            if cfg["param"] != "direct":
                o = (z - (1 - ac[cfg["T"]]).sqrt() * o) / ac[cfg["T"]].sqrt()
            ref2 = tae.decode(o).sample
        r = {"wrapper_vs_diffusers_maxabs": float((ref - ref2).abs().max())}
        f = od / f"student_{task}.onnx"
        t0 = time.time()
        try:
            torch.onnx.export(m, (x,), str(f), input_names=["image"], output_names=["out"], opset_version=18, dynamo=True,
                              external_data=True)
            r["export"] = "ok"
        except Exception as e:  # report honestly
            r["export"] = f"FAILED: {type(e).__name__}: {str(e)[:300]}"
            res[task] = r
            continue
        r["export_s"] = round(time.time() - t0, 1)
        r["onnx_MB_fp32"] = round(sum(p.stat().st_size for p in od.glob(f"student_{task}.onnx*")) / 2**20, 1)
        so = ort.SessionOptions(); so.intra_op_num_threads = 8
        s = ort.InferenceSession(str(f), so, providers=["CPUExecutionProvider"])
        y = s.run(None, {"image": x.numpy()})[0]
        r["ort_vs_torch_maxabs"] = float(np.abs(y - ref.numpy()).max())
        r["ort_vs_torch_meanabs"] = float(np.abs(y - ref.numpy()).mean())
        ts = []
        for _ in range(5):
            t0 = time.time(); s.run(None, {"image": x.numpy()}); ts.append(time.time() - t0)
        r["ort_cpu_8threads_s_median"] = round(float(np.median(ts)), 3)
        Image.fromarray(((np.clip(y[0], -1, 1) + 1) * 127.5).round().astype(np.uint8).transpose(1, 2, 0)).save(od / f"ort_{task}.png")
        res[task] = r
        print(task, json.dumps(r), flush=True)
    (ROOT / f"results/student_{a.tag}_export.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
