"""Plan B0 item 3 (2026-11-20): does the Sana-Sprint 0.6B transformer convert to Core ML? Conversion only (Linux cannot run it).

Same path as scripts/export_coreml_dl.py: torch.export -> coremltools mlprogram, fp16 compute, iOS16 target. Inputs: the
diptych latent [1, 32, 16, 32] (DC-AE f32c32 at 512 px, [noise | photo]) and a context [1, 300, 2304] standing in for learned
option tokens; timestep 1.0 and guidance fixed. Reports success, failing op if any, package size, op histogram, and the
max |diff| between the converted model's MIL-level fp16 weights path cannot be executed here, so numerical checks are skipped.
  source env.sh; $VENV/bin/python scripts/sana_coreml_test.py
Out: weights/sana_b0/SanaSprint06_diptych_fp16.mlpackage (test artifact), results/sana_coreml_test.json
"""
import json
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SNAP = Path("/mnt/project/kchat/duyhv/sana/hf_cache/hub/models--Efficient-Large-Model--Sana_Sprint_0.6B_1024px_diffusers/snapshots/aa76e7f4f4928f378716b6716a2130fba3caf5b1")


class Wrap(torch.nn.Module):
    def __init__(self, tr):
        super().__init__()
        self.tr = tr
        self.register_buffer("t", torch.tensor([1.0]))
        self.register_buffer("g", torch.tensor([4.5]))

    def forward(self, x, ctx):
        kw = {"guidance": self.g} if self.tr.config.guidance_embeds else {}
        return self.tr(x, encoder_hidden_states=ctx, timestep=self.t, return_dict=False, **kw)[0]


def main():
    import coremltools as ct
    from diffusers import SanaTransformer2DModel
    res = {"coremltools": ct.__version__, "torch": torch.__version__}
    t0 = time.time()
    tr = SanaTransformer2DModel.from_pretrained(SNAP / "transformer", torch_dtype=torch.float32).eval()
    m = Wrap(tr).eval()
    x, ctx = torch.randn(1, 32, 16, 32), torch.randn(1, 300, 2304)
    try:
        with torch.no_grad():
            ep = torch.export.export(m, (x, ctx)).run_decompositions({})
        res["torch_export"] = "ok"
        ml = ct.convert(ep, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS16, compute_precision=ct.precision.FLOAT16,
                        inputs=[ct.TensorType(name="latent", shape=x.shape), ct.TensorType(name="context", shape=ctx.shape)])
        out = ROOT / "weights/sana_b0/SanaSprint06_diptych_fp16.mlpackage"
        out.parent.mkdir(parents=True, exist_ok=True)
        ml.save(str(out))
        size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 2**20
        ops = Counter(op.type for f in ml.get_spec().mlProgram.functions.values() for b in f.block_specializations.values()
                      for op in b.operations)
        res |= {"coreml_convert": "ok", "package": str(out.relative_to(ROOT)), "size_MB": round(size, 1),
                "op_histogram_top": ops.most_common(15), "n_ops": sum(ops.values())}
    except Exception as e:
        res |= {"coreml_convert": "FAILED", "error": f"{type(e).__name__}: {str(e)[:800]}", "trace": traceback.format_exc()[-1500:]}
    res["minutes"] = round((time.time() - t0) / 60, 1)
    (ROOT / "results/sana_coreml_test.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "trace"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
