"""s12 copy of s11_ckpt_report.py for tS4o (from DreamLite-mobile original). tS4o checkpoint report (2026-11-26, user "yes do #1, train tS4o longer with checkpoint evals").
Reads results/e1/<run>.json (eval_e1: expression accuracy, per-option, likeness), results/expr_eval_<run>_decg2.json (sticker id) and
weights/student/tS4o/train_log.jsonl (val loss), for runs tS4o_s2000 / _s6000 / _s12000 / tS4o (18000) + references (teacher of tS4o =
results/teacher/tB1_final_img3.0_txt2.5_s8_k1.json, tS1, tS3, dl6g2). Writes results/tS4o_checkpoints.json + prints a table, and
grids/tS4o_hands_ckpt.jpg: hand / prop options on 4 faces, teacher | dl6g2 | tS4o checkpoints (outputs/expr_eval_<run>_decg2).
Snapshots before 18000 are mid-schedule (cosine lr not annealed)."""
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
HAND = ["thumbs", "wave", "ok", "facepalm", "shrug", "thanks", "think"]
PROPS = ["sleepy", "night", "love", "party", "angry"]
RUNS = ["tS4o_s2000", "tS4o_s6000", "tS4o_s12000", "tS4o"]
REFS = {"teacher (img3 k1, 8 st)": "results/teacher/tB1_final_img3.0_txt2.5_s8_k1.json", "dl6g2": "results/e1/dl6g2.json",
        "tS1": "results/e1/tS1.json", "tS3": "results/e1/tS3.json"}


def load(name, f):
    p = ROOT / f
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    pe = d["per_edit"]
    r = {"expr": d["expr_acc"], "hands7": round(sum(pe[e] for e in HAND) / len(HAND), 3),
         "props5": round(sum(pe[e] for e in PROPS) / len(PROPS), 3), "ok": pe["ok"], "sleepy": pe["sleepy"], "night": pe["night"],
         "likeness": (d.get("avatar_likeness") or {}).get("id_cos_mean") if isinstance(d.get("avatar_likeness"), dict) else d.get("avatar_likeness")}
    x = ROOT / f"results/expr_eval_{name}_decg2.json"
    if x.exists():
        xe = json.loads(x.read_text())
        r["sticker_id_top5"] = xe.get("_mean", {}).get("id_top5")
    return r


def main():
    log = [json.loads(l) for l in open(ROOT / "weights/student/tS4o/train_log.jsonl")] if (ROOT / "weights/student/tS4o/train_log.jsonl").exists() else []
    val = {r["step"]: r.get("val_mse") for r in log if "val_mse" in r}
    res = {}
    for n, f in REFS.items():
        res[n] = load(n if n in ("dl6g2", "tS1", "tS3") else "_", f)
    for run in RUNS:
        r = load(run, f"results/e1/{run}.json")
        if r:
            st = int(run.split("_s")[1]) if "_s" in run else 18000
            r["step"] = st
            r["val_mse"] = val.get(st)
        res[run] = r
    (ROOT / "results/tS4o_checkpoints.json").write_text(json.dumps(res, indent=1))
    print("| model | expr all / face / outside | 7 hand options | 5 prop options | ok / sleepy / night | likeness | sticker id top-5 | val MSE |")
    print("|---|---|---|---|---|---|---|---|")
    for n, r in res.items():
        if not r:
            print(f"| {n} | missing |")
            continue
        e = r["expr"]
        print(f"| {n} | {e['all']:.3f} / {e['face_only']:.3f} / {e['outside']:.3f} | {r['hands7']:.3f} | {r['props5']:.3f} | "
              f"{r['ok']:.2f} / {r['sleepy']:.2f} / {r['night']:.2f} | {r['likeness'] if r['likeness'] is None else round(r['likeness'], 3)} | "
              f"{r.get('sticker_id_top5', '-')} | {r.get('val_mse', '-')} |")
    # hands sheet
    cols = [("teacher", "expr_eval_teacher_tB1_final_img3.0_txt2.5_s8_k1"), ("dl6g2", "expr_eval_dl6g2_decg2")] + \
           [(r, f"expr_eval_{r}_decg2") for r in RUNS]
    cols = [(l, d) for l, d in cols if (ROOT / "outputs" / d).exists()]
    faces, opts, c = ["face_002", "face_015", "face_034", "face_053"], ["wave", "thumbs", "ok", "sleepy", "night"], 150
    g = Image.new("RGB", (120 + len(opts) * len(faces) * c // 2, 20 + len(cols) * (c + 4) * 2), "white")
    d = ImageDraw.Draw(g)
    d.text((4, 4), "tS4o checkpoints vs teacher / dl6g2: " + ", ".join(opts) + " on " + ", ".join(faces), fill="black")
    cells = [(f, o) for f in faces for o in opts]
    half = (len(cells) + 1) // 2
    for blk in range(2):
        for i, (lab, dd) in enumerate(cols):
            y = 20 + (blk * len(cols) + i) * (c + 4)
            d.text((4, y + c // 2), lab, fill="black")
            for j, (f, o) in enumerate(cells[blk * half:(blk + 1) * half]):
                p = ROOT / "outputs" / dd / "cartoon3d" / f"{f}_{o}.png"
                if p.exists():
                    g.paste(Image.open(p).convert("RGB").resize((c, c)), (120 + j * c, y))
    g.save(ROOT / "grids/tS4o_hands_ckpt.jpg", quality=88)
    print("[s12] sheet grids/tS4o_hands_ckpt.jpg", flush=True)


if __name__ == "__main__":
    main()
