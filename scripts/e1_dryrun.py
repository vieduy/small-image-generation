"""E1 Phase 0 dry run (2026-11-20): build every planned and existing run, verify on the model itself which cross-attention
modules read the real option, and count self-attention layers executed after the first such module. Nothing is trained.

Per run:
  * option-reading modules: EMPIRICAL. Every attn2's innermost processor is wrapped to record the context it actually uses;
    the model runs twice with two different options (expr_happy, expr_thumbs); a module reads the option iff its context differs.
  * execution order: forward pre-hooks on every attn1 / attn2 record the real call order (named_modules() lists the mid block last).
  * self-attention after entry: number of attn1 calls after the first option-reading attn2 call.
  * trunk check: the hidden states entering the entry module, max |diff| across the two options (0 = everything before is shared).
  * trainable params: as train_dl.py trains them (whole UNet + option tokens + shared context when sharing).
New runs are built exactly like train_dl.py --shared-trunk --shared-scope <scope> (dl6g2 init, shared context = avatar token,
trainable). Existing runs are loaded from their saved weights (dl_arch.load_unet re-installs the saved shared context/scope).
GPU if available (else CPU), fp32, small latent (16x16 per panel; the call order does not depend on resolution).
  source env.sh; CUDA_VISIBLE_DEVICES="" $VENV/bin/python scripts/e1_dryrun.py   -> reports/e1_plan.md (+ reports/e1_plan.json)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
REC = {"init_from": "dl6g2", "steps": 6000, "lr": 2e-5, "emb_lr": 1e-3, "bs": 16, "schedule": "cosine, 200 warm-up steps",
       "data": "data/pairs/cartoon3d/manifest_v2.jsonl (expr), 53,786 train pairs (+ proper flip, tgt-cache)",
       "losses": "dl6g2 recipe: mse + LPIPS256 + lp512 1, eye 1, eye-LPIPS 0.25, DINOv2 critic 0.2 from step 200",
       "decoder_eval": "decg2"}
# name -> (scope or None = no sharing, seed, priority, note)
NEW = {
    "mid4": ("only_mid", 0, "P0", ""),
    "last10": ("up0a2_plus_up1", 0, "P0", "frA2's entry block (up0.attentions.2, 4 inputs) + all 6 up1 inputs"),
    "up0b1_4": ("only_up0a0", 0, "P1", "= up0start_4 (identical config, trained once)"),
    "up0b2_4": ("only_up0a1", 0, "P1", ""),
    "up1_4": ("only_up1a01", 0, "P1", "entry at 32x64 (resolution changes too)"),
    "last1": ("only_up0a2_tb0", 0, "P2", ""),
    "last2": ("only_up0a2_tb01", 0, "P2", ""),
    "up0start_12": ("only_up0", 0, "P2", ""),
    "nosharing_matched": (None, 0, "control", "dl6g2 + the same 6000 steps, nothing shared"),
    "frA2_s1": ("only_up0a2", 1, "seed", "second seed of frA2"),
    "dlB_s1": ("down_mid", 1, "seed", "second seed of dlB"),
    "mid4_s1": ("only_mid", 1, "seed", ""),
    "last10_s1": ("up0a2_plus_up1", 1, "seed", ""),
}
EXISTING = ["dl6g2", "dlBd", "dlB", "frA2", "frUp1"]


def build_new(scope):
    te = torch.load(ROOT / "weights/student/dl6g2/task_emb.pt")
    u = DLA.load_unet(ROOT / "weights/student/dl6g2/unet")
    if scope:  # = train_dl.py: install_shared_trunk(unet, emb0[0:1], mask[0:1], trainable=True, scope=...)
        DLA.install_shared_trunk(u, te["emb"][0:1].float(), te["mask"][0:1], trainable=True, scope=scope)
    return u.to(DEV).eval(), te


def build_existing(tag):
    w = ROOT / "weights/student" / tag
    return DLA.load_unet(w / "unet").to(DEV).eval(), torch.load(w / "task_emb.pt")


class Rec:
    def __init__(self, inner, store, name):
        self.inner, self.store, self.name = inner, store, name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        self.store[self.name] = encoder_hidden_states.detach().clone() if encoder_hidden_states is not None else None
        return self.inner(attn, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **kw)


def probe(u, te):
    tasks = te["tasks"]
    emb, mask = te["emb"].float().to(DEV), te["mask"].to(DEV)
    ctx, order, ent_in = {}, [], {}
    a2 = {n: m for n, m in u.named_modules() if n.endswith("attn2")}
    a1 = {n: m for n, m in u.named_modules() if n.endswith("attn1") and m is not None}
    for n, m in a2.items():
        p = m.processor
        if isinstance(p, DLA._SharedCtxProc):
            p.inner = Rec(p.inner, ctx, n)  # innermost: sees the context actually used
        else:
            m.set_processor(Rec(p, ctx, n))
    hooks = [m.register_forward_pre_hook(lambda mod, a, kw, n=n: order.append(n), with_kwargs=True) for n, m in {**a1, **a2}.items()]
    torch.manual_seed(0)
    z = torch.randn(1, 4, 16, 16).to(DEV)
    nz = torch.randn(1, 4, 16, 16).to(DEV)  # small fixed noise (the phone constant is 64x64)
    runs = {}
    with torch.no_grad():
        for t in ("expr_happy", "expr_thumbs"):
            order.clear()
            k = tasks.index(t)
            DLA.forward(u, z, emb[k:k + 1], mask[k:k + 1], noise=nz)
            runs[t] = ({n: (None if v is None else v.clone()) for n, v in ctx.items()}, list(order))
    for h in hooks:
        h.remove()
    (c1, o1), (c2, o2) = runs["expr_happy"], runs["expr_thumbs"]
    assert o1 == o2, "call order differs between options"
    reads = [n for n in a2 if c1[n] is not None and (c1[n].shape != c2[n].shape or (c1[n] - c2[n]).abs().max() > 0)]
    calls = o1
    first = next((i for i, n in enumerate(calls) if n in reads), None)
    entry = calls[first] if first is not None else None
    after = sum(1 for n in calls[first + 1:] if n.endswith("attn1")) if first is not None else None
    exec_reads = [n for n in calls if n in reads]
    # trunk check: input of the entry module across the two options
    trunk = None
    if entry:
        m = u.get_submodule(entry)
        h = m.register_forward_pre_hook(lambda mod, a, kw: ent_in.__setitem__(len(ent_in), (a[0] if a else kw["hidden_states"]).clone()),
                                        with_kwargs=True)
        with torch.no_grad():
            for t in ("expr_happy", "expr_thumbs"):
                k = tasks.index(t)
                DLA.forward(u, z, emb[k:k + 1], mask[k:k + 1], noise=nz)
        h.remove()
        trunk = float((ent_in[0] - ent_in[1]).abs().max())
    n_tr = sum(p.numel() for n, p in u.named_parameters() if n != "shared_emb") + emb.numel() + \
        (u.shared_emb.numel() if hasattr(u, "shared_emb") else 0)
    return {"reads": exec_reads, "n_reads": len(reads), "entry": entry, "self_attn_after": after,
            "n_attn1": len(a1), "n_attn2": len(a2), "calls": len(calls), "trunk_input_maxdiff": trunk,
            "trainable_params_M": round(n_tr / 1e6, 1), "scope_installed": getattr(u, "_shared_scope", None)}


def short(n):
    return (n.replace("up_blocks.", "up").replace("down_blocks.", "down").replace("mid_block", "mid").replace(".attentions.", ".a")
             .replace(".transformer_blocks.", ".tb").replace(".attn2", ""))


def main():
    t0 = time.time()
    try:
        git = subprocess.run(["git", "-c", f"safe.directory={ROOT}", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-c", f"safe.directory={ROOT}", "status", "--porcelain", "--", "scripts"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip())
    except Exception:
        git, dirty = "?", True
    res = {}
    for name in EXISTING:
        u, te = build_existing(name)
        cfg = json.loads((ROOT / "weights/student" / name / "config.json").read_text())
        res[name] = probe(u, te) | {"kind": "existing", "seed": cfg.get("seed"), "steps": cfg.get("steps"), "lr": cfg.get("lr"),
                                    "init_from": cfg.get("init_from"), "scope_cfg": cfg.get("shared_scope")}
        print(name, {k: v for k, v in res[name].items() if k != "reads"}, flush=True)
    for name, (scope, seed, pri, note) in NEW.items():
        u, te = build_new(scope)
        res[name] = probe(u, te) | {"kind": pri, "scope": scope, "seed": seed, "steps": REC["steps"], "lr": REC["lr"],
                                    "init_from": REC["init_from"], "note": note}
        print(name, {k: v for k, v in res[name].items() if k != "reads"}, flush=True)
    (ROOT / "reports/e1_plan.json").write_text(json.dumps({"git": git, "scripts_dirty": dirty, "recipe": REC, "runs": res}, indent=1))

    TRAIN_H, EVAL_H = 1.0, 0.25
    L = ["# E1 plan (Phase 0 dry run, nothing trained)", "",
         f"Generated by `scripts/e1_dryrun.py` on {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}, git {git}"
         f"{' + uncommitted changes in scripts/' if dirty else ''}. All counts below were measured on the built models "
         "(empirical option-reading test + recorded call order), not read from module lists.", "",
         "## Common recipe (= dlB, unchanged)", ""] + [f"- **{k}**: {v}" for k, v in REC.items()] + [
         "- Modules that do not read the option read the shared context (learned, initialised from the avatar token, "
         "trainable; = dlB's mechanism). They are neutralised, not removed.",
         "- Only differences between new runs: `--shared-scope` (which cross-attention modules read the option) and `--seed`.", "",
         "## Runs", "",
         "| run | kind | scope | seed | # option inputs | entry (first option module, by execution) | self-attn layers after entry | "
         "entry input identical across options (max diff) | trainable params | steps | est. GPU-h |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        new = r["kind"] != "existing"
        est = f"{TRAIN_H + EVAL_H:.2f}" if new else f"{EVAL_H:.2f} (re-eval)"
        if name == "dl6g2":
            est = f"{EVAL_H:.2f} (re-eval, reference)"
        td = "–" if r["trunk_input_maxdiff"] is None else f"{r['trunk_input_maxdiff']:.1e}"
        L.append(f"| {name} | {r['kind']} | {r.get('scope', r.get('scope_cfg')) or '—'} | {r['seed']} | {r['n_reads']} | "
                 f"{short(r['entry']) if r['entry'] else '—'} | {r['self_attn_after']} | {td} | {r['trainable_params_M']}M | "
                 f"{r['steps']} | {est} |")
    L += ["", "## Option-reading modules per run (in execution order)", ""]
    for name, r in res.items():
        L.append(f"- **{name}** ({r['n_reads']}): " + ", ".join(short(n) for n in r["reads"]))
    n_new = sum(1 for r in res.values() if r["kind"] != "existing")
    L += ["", "## Notes", "",
          f"- Model: {res['dl6g2']['n_attn2']} cross-attention modules, {res['dl6g2']['n_attn1']} self-attention layers, "
          f"{res['dl6g2']['calls']} attention calls per forward.",
          "- `up0b1_4` and `up0start_4` are the same configuration (4 inputs in up_blocks.0.attentions.0); trained once.",
          "- `last10` = frA2's entry block (up0.attentions.2, 4 inputs) + the 6 up_blocks.1 inputs (the option inputs are added "
          "after the last self-attention, per e1_status.md section 5).",
          "- `up1_4` also moves the entry to 32x64 (up_blocks.1); the other depth arms stay at 16x32.",
          "- Second seeds: frA2, dlB, mid4, last10 listed; more are added after results for any run whose outside-the-face "
          "accuracy is in 0.10-0.40.",
          f"- GPU-hours are estimates: training {TRAIN_H} h (dlB 60.5 min, frUp1 58.3 min alone on one H200) + evaluation "
          f"{EVAL_H} h (scripts/eval_e1.py; not yet timed). {n_new} new runs ~{n_new * (TRAIN_H + EVAL_H):.1f} GPU-h + 5 re-evals "
          f"~{5 * EVAL_H:.2f} GPU-h (+ conditional second seeds).",
          f"- Dry run took {(time.time() - t0) / 60:.1f} min ({DEV}, fp32)."]
    (ROOT / "reports/e1_plan.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
