"""Re-quantize a DreamLite fp16 Core ML package to 8-bit weights, keeping chosen tensors (default: the task embeddings) in fp16.
w8_check.py found the per-task 8-bit scale on the [22, 418, 2048] task embeddings loses ~18% (mean relative error), the likely cause
of gesture drift in the first w8 build. Starts from the existing fp16 .mlpackage (no re-conversion).
  source env.sh; CUDA_VISIBLE_DEVICES= $VENV/bin/python scripts/requant_w8.py --tag dl6g2 --decoder decg2
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dl6g2")
    ap.add_argument("--decoder", default="decg2")
    ap.add_argument("--keep", default="b_emb_to_fp16", help="comma list of const op names left in fp16")
    ap.add_argument("--suffix", default="w8e16")
    a = ap.parse_args()
    import coremltools as ct
    import coremltools.optimize.coreml as cto
    name = f"{a.tag}_{a.decoder}"
    od = ROOT / f"weights/student/{a.tag}/coreml_{a.decoder}"
    src = od / f"Student_{name}_fp16.mlpackage"
    ml = ct.models.MLModel(str(src), skip_model_load=True)
    keep = [k for k in a.keep.split(",") if k]
    cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric"),
                                 op_name_configs={k: None for k in keep})
    q = cto.linear_quantize_weights(ml, config=cfg)
    p = od / f"Student_{name}_{a.suffix}.mlpackage"
    if p.exists():
        shutil.rmtree(p)
    q.save(str(p))
    mb = round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**20, 1)
    info = json.loads((od / "export_info.json").read_text())
    info |= {f"{a.suffix}_mlpackage": str(p.relative_to(ROOT)), f"{a.suffix}_MB": mb, f"{a.suffix}_kept_fp16": keep}
    (od / "export_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({"out": str(p), "MB": mb, "kept_fp16": keep}), flush=True)


if __name__ == "__main__":
    main()
