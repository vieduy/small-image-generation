"""Does the 8-bit Core ML build give the same images as fp16? (Core ML cannot run on Linux, so we rebuild its weights in PyTorch.)

Reads the real int8 weights + per-channel scales from the w8 .mlpackage (constexpr_affine_dequantize ops), puts the dequantized
values into the same export graph (scripts/export_coreml_dl.py: DLStudent + PlainAttn), and compares on synthetic faces:
  ref  = fp32 weights, fp32 compute (the trained model)
  fp16 = fp16-rounded weights (what the fp16 .mlpackage stores), fp16 compute (Core ML's compute precision)
  w8   = int8-dequantized weights from the w8 .mlpackage, fp16 compute
Weights the w8 build left unquantized (< 2048 elements) are fp16 in both builds.
  source env.sh; CUDA_VISIBLE_DEVICES=3 $VENV/bin/python scripts/w8_check.py --tag dl6g2 --decoder decg2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
from export_coreml_dl import DLStudent, PlainAttn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def w8_weights(pkg):
    import coremltools as ct
    from coremltools.converters.mil.frontend.milproto.load import load as mload
    m = ct.models.MLModel(str(pkg), skip_model_load=True)
    spec = m.get_spec()
    prog = mload(spec, spec.specificationVersion, file_weights_dir=m.weights_dir)
    out = {}
    for f in prog.functions.values():
        for op in f.operations:
            if op.op_type != "constexpr_affine_dequantize":
                continue
            q = op.quantized_data.val.astype(np.float32)
            zp, sc, ax = op.zero_point.val.astype(np.float32), op.scale.val.astype(np.float32), int(op.axis.val)
            shp = [1] * q.ndim
            shp[ax] = -1
            deq = (q - zp.reshape(shp)) * sc.reshape(shp)
            out[op.name.replace("_to_fp16_quantized", "")] = torch.from_numpy(deq.astype(np.float16))
    return out


def mil_name(k, kind):  # torch.export naming: parameter p_<path>, buffer b_<path>
    return ("p_" if kind == "p" else "b_") + k.replace(".", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dl6g2")
    ap.add_argument("--decoder", default="decg2")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--variant", default="w8", help="package suffix: w8 or w8e16 (scripts/requant_w8.py)")
    ap.add_argument("--fp32-compute", action="store_true", help="run fp16/w8 weights in fp32 (isolates weight rounding)")
    a = ap.parse_args()
    name = f"{a.tag}_{a.decoder}"
    pkg = ROOT / f"weights/student/{a.tag}/coreml_{a.decoder}/Student_{name}_{a.variant}.mlpackage"
    w = ROOT / "weights/student" / a.tag
    te = torch.load(w / "task_emb.pt")
    tasks = te["tasks"]
    unet = DLA.load_unet(w / "unet").float().eval()
    unet.set_attn_processor(PlainAttn())
    vae = DLA.load_vae(torch.float32, path=ROOT / "weights/decoder" / a.decoder).eval()
    ref = DLStudent(unet, vae, te["emb"].float(), te["mask"]).eval()
    sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    params = {k for k, _ in ref.named_parameters()}

    q = w8_weights(pkg)
    wq, hit, kinds = {}, set(), {"param": 0, "buffer": 0}
    for k, v in sd.items():
        n = mil_name(k, "p" if k in params else "b")
        if n in q:
            assert tuple(q[n].shape) == tuple(v.shape), (k, q[n].shape, v.shape)
            wq[k] = q[n].float()
            hit.add(n)
            kinds["param" if k in params else "buffer"] += 1
        else:
            wq[k] = v.half().float() if v.is_floating_point() else v
    miss = sorted(set(q) - hit)
    print(f"[w8] {len(q)} quantized tensors in the package, matched {len(hit)} ({kinds}); unmatched: {miss[:5]}", flush=True)
    assert not miss, "some quantized tensors did not map to torch weights"
    qerr = {k: float((wq[k] - sd[k]).abs().max() / sd[k].abs().max().clamp_min(1e-12)) for k in wq if k in {n for n in sd}
            and mil_name(k, "p" if k in params else "b") in hit}
    worst = sorted(qerr.items(), key=lambda x: -x[1])[:3]
    print("[w8] worst relative weight error:", [(k, round(e, 4)) for k, e in worst], flush=True)
    emb_err = (wq["emb"].half().float() - sd["emb"].half().float()).abs().mean() / sd["emb"].abs().mean()
    print(f"[w8] task embeddings quantized too: mean relative error {emb_err:.4f}", flush=True)

    import copy
    dev = "cuda"
    cdt = torch.float32 if a.fp32_compute else torch.float16
    m_ref = ref.to(dev)
    m16 = copy.deepcopy(ref)
    m16.load_state_dict({k: (v.half().float() if v.is_floating_point() else v) for k, v in sd.items()})
    m16 = m16.to(dev, cdt)
    m8 = copy.deepcopy(ref)
    m8.load_state_dict(wq)
    m8 = m8.to(dev, cdt)
    def fwd(self, image, task):  # DLStudent.forward with the one-hot in the model dtype (Core ML runs it all in fp16)
        dt = self.emb.dtype
        sel = (task.to(dt).reshape(1) == self.ids).to(dt)
        e = (sel.reshape(-1, 1, 1) * self.emb).sum(0, keepdim=True)
        msk = (sel.reshape(-1, 1) * self.mask).sum(0, keepdim=True)
        z = self.enc(image)
        xx = torch.cat([self.noise, z], dim=3)
        v = self.unet(xx, timestep=self.t, encoder_hidden_states=e, encoder_attention_mask=msk,
                      added_cond_kwargs={"time_ids": self.tid}, return_dict=False)[0][..., :DLA.LAT]
        return (self.dec(self.noise - v).clamp(-1, 1) + 1) * 127.5

    import types
    for mm in (m16, m8):  # fixed noise / timestep buffers keep their fp16 value; ids stay exact
        mm.ids.data = torch.arange(len(tasks), device=dev, dtype=cdt)
        mm.forward = types.MethodType(fwd, mm)

    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    faces = sorted((ROOT / "data/faces").glob("*.png"))[: a.n]
    run_tasks = [t for t in ("avatar", "sticker", "expr_happy", "expr_thumbs", "expr_wave", "expr_love", "expr_cool") if t in tasks]
    rows = {"fp16": [], "w8": [], "w8_vs_fp16": []}
    lap = {"ref": [], "fp16": [], "w8": []}
    od = ROOT / f"grids/w8_check_{name}" / a.variant
    od.mkdir(parents=True, exist_ok=True)
    sheet = []
    lapk = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=dev).view(1, 1, 3, 3)

    def lapvar(y):  # y [1,3,H,W] in 0..255
        g = y.mean(1, keepdim=True)
        return float(torch.nn.functional.conv2d(g, lapk).var())

    worst = [1e9, None, None, None]
    with torch.no_grad():
        for i, f in enumerate(faces):
            x = torch.from_numpy(np.array(Image.open(f).convert("RGB").resize((512, 512)))).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1
            for t in run_tasks:
                k = torch.tensor([tasks.index(t)], dtype=torch.int32, device=dev)
                yr = m_ref(x, k).clamp(0, 255).round()
                y16 = m16(x.to(cdt), k).float().clamp(0, 255).round()
                y8 = m8(x.to(cdt), k).float().clamp(0, 255).round()
                assert torch.isfinite(y16).all() and torch.isfinite(y8).all(), "non-finite output (fp16 overflow?)"
                for key, (u, v) in {"fp16": (y16, yr), "w8": (y8, yr), "w8_vs_fp16": (y8, y16)}.items():
                    mse = float(((u - v) ** 2).mean())
                    rows[key].append({"psnr": 10 * np.log10(255 ** 2 / max(mse, 1e-10)), "mad": float((u - v).abs().mean()),
                                      "max": float((u - v).abs().max()),
                                      "lpips": float(lp(u / 127.5 - 1, v / 127.5 - 1))})
                rows["w8"][-1] |= {"face": f.stem, "task": t}
                if rows["w8_vs_fp16"][-1]["psnr"] < worst[0]:
                    worst[:] = [rows["w8_vs_fp16"][-1]["psnr"], f.stem, t, torch.cat([y16[0], y8[0], (y8 - y16).abs()[0].mul(4).clamp(0, 255)], 2)]
                lap["ref"].append(lapvar(yr)); lap["fp16"].append(lapvar(y16)); lap["w8"].append(lapvar(y8))
                if i < 3 and t in ("avatar", "expr_thumbs"):
                    crop = lambda y: y[0, :, 112:368, 128:384]
                    sheet.append(torch.cat([crop(yr), crop(y16), crop(y8), (crop(y8) - crop(y16)).abs().mul(8).clamp(0, 255)], 2))
    summ = {k: {m: round(float(np.mean([r[m] for r in v])), 4) for m in ("psnr", "mad", "max", "lpips")} | {"worst_psnr": round(min(r["psnr"] for r in v), 2)}
            for k, v in rows.items()}
    summ["lapvar_mean"] = {k: round(float(np.mean(v)), 1) for k, v in lap.items()}
    summ |= {"n_images": len(rows["w8"]), "faces": len(faces), "tasks": run_tasks, "compute": str(cdt).replace("torch.", ""),
             "w8_tensors": len(q), "task_emb_rel_err": round(float(emb_err), 4)}
    summ["w8_psnr_by_task"] = {t: round(float(np.mean([r["psnr"] for r in rows["w8"] if r["task"] == t])), 2) for t in run_tasks}
    summ["w8_lpips_by_task"] = {t: round(float(np.mean([r["lpips"] for r in rows["w8"] if r["task"] == t])), 4) for t in run_tasks}
    ps = sorted(r["psnr"] for r in rows["w8"])
    summ["w8_psnr_p5_p50"] = [round(ps[len(ps) // 20], 2), round(ps[len(ps) // 2], 2)]
    summ["w8_worst"] = {"psnr_vs_fp16": round(worst[0], 2), "face": worst[1], "task": worst[2]}
    Image.fromarray(worst[3].permute(1, 2, 0).byte().cpu().numpy()).save(od / "worst_fp16_w8_diffx4.jpg", quality=92)
    g = torch.cat(sheet, 1).permute(1, 2, 0).byte().cpu().numpy()
    Image.fromarray(g).save(od / "zoom_ref_fp16_w8_diffx8.jpg", quality=92)
    out = ROOT / f"results/w8_check_{name}{'' if a.variant == 'w8' else '_' + a.variant}{'_fp32c' if a.fp32_compute else ''}.json"
    out.write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
