"""Summarize results/serving.jsonl -> results/summary.md + results.csv rows (one per config).

Capacity = open-loop (Poisson) RPS: highest served img/s with 0 errors and p95 <= SLO. Closed-loop CCU 1..8 shown as
the ceiling and for "max CCU within SLO" (largest CCU whose p95 <= SLO). GPU-s/img = 1 / closed-loop peak img/s
(whole GPU, this server only). Contamination: % of pmon samples where any OTHER pid had sm% > 0 during that config.
H100 PCIe server numbers - NOT phone numbers.

  $VENV/bin/python scripts/summarize.py [results/serving.jsonl]
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLOS = (1.0, 1.5, 2.0)
src = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results/serving.jsonl")
rows = [json.loads(l) for l in open(src) if l.strip()]
by = defaultdict(list)
for r in rows:
    by[r["label"]].append(r)


def vram(label):
    m = re.findall(r"peak_vram_gb=([\d.]+)", (ROOT / f"logs/server_{label}.log").read_text()) if (ROOT / f"logs/server_{label}.log").exists() else []
    return float(m[-1]) if m else None


def contamination(label):
    f = ROOT / f"logs/pmon_{label}.log"
    if not f.exists():
        return None
    pids = {}
    for b in (ROOT / "logs").glob("bench_part*.log"):
        pids.update({lab: pid for lab, pid in re.findall(r"\[(\S+)\] server pid (\d+)", b.read_text())})
    mine = pids.get(label)
    ts_all, ts_busy = set(), set()  # sample timestamps; busy = any foreign pid with sm% > 0
    for l in f.read_text().splitlines():
        p = l.split()
        if l.startswith("#") or len(p) < 5:
            continue
        ts_all.add(p[0])
        if p[2] != mine and p[4] not in ("-", "0"):
            ts_busy.add(p[0])
    return round(100 * len(ts_busy) / len(ts_all), 1) if ts_all else 0.0


out_rows, md = [], []
md.append("| config | model | steps | peak VRAM GB | CCU1 img/s (p50 s) | CCU8 img/s (p95 s) | closed-loop peak img/s | GPU-s/img |"
          " max CCU p95≤1.5s | RPS p95≤1s | RPS p95≤1.5s | RPS p95≤2s | foreign GPU activity |")
md.append("|" + "---|" * 13)
ccu_md = ["| config | " + " | ".join(f"CCU{n}" for n in range(1, 9)) + " |", "|" + "---|" * 9]
for label, rr in by.items():
    ccu = {r["target_ccu"]: r for r in rr if r["mode"] == "ccu"}
    rate = [r for r in rr if r["mode"] == "rate"]
    peak = max((r["throughput_ips"] for r in ccu.values() if r["errors"] == 0), default=0)
    usable = {}
    for s in SLOS:
        ok = [r for r in rate if r["errors"] == 0 and r["lat_p95"] is not None and r["lat_p95"] <= s]
        usable[s] = max(ok, key=lambda r: r["throughput_ips"]) if ok else None
    max_ccu = max((n for n, r in ccu.items() if r["errors"] == 0 and r["lat_p95"] <= 1.5), default=0)
    c1, c8 = ccu.get(1), ccu.get(8)
    rec = {
        "config": label, "model": rr[0].get("model"), "steps": rr[0].get("steps"), "peak_vram_gb": vram(label),
        "ccu1_ips": c1 and c1["throughput_ips"], "ccu1_p50_s": c1 and c1["lat_p50"], "ccu1_p95_s": c1 and c1["lat_p95"],
        "ccu8_ips": c8 and c8["throughput_ips"], "ccu8_p95_s": c8 and c8["lat_p95"], "ccu8_mean_batch": c8 and c8.get("mean_batch"),
        "closed_peak_ips": peak, "gpu_s_per_img": round(1 / peak, 4) if peak else None, "max_ccu_p95_1p5s": max_ccu,
        **{f"rps_p95_le_{s}s": (usable[s]["throughput_ips"] if usable[s] else 0) for s in SLOS},
        **{f"rps_p95_le_{s}s_ccu_avg_peak": (f"{usable[s]['ccu_avg']}/{usable[s]['ccu_peak']}" if usable[s] else "") for s in SLOS},
        "foreign_gpu_pct": contamination(label),
    }
    out_rows.append(rec)
    f = lambda x: "-" if x is None else x  # noqa: E731
    md.append(f"| {label} | {rec['model']} | {rec['steps']} | {f(rec['peak_vram_gb'])} | {f(rec['ccu1_ips'])} ({f(rec['ccu1_p50_s'])}) | "
              f"{f(rec['ccu8_ips'])} ({f(rec['ccu8_p95_s'])}) | {peak} | {f(rec['gpu_s_per_img'])} | {max_ccu} | "
              f"{rec['rps_p95_le_1.0s']} | {rec['rps_p95_le_1.5s']} | {rec['rps_p95_le_2.0s']} | {f(rec['foreign_gpu_pct'])}% |")
    ccu_md.append(f"| {label} | " + " | ".join(
        (f"{ccu[n]['throughput_ips']:.2f} / {ccu[n]['lat_p95']:.2f}s" if n in ccu and ccu[n]["lat_p95"] is not None else "-") for n in range(1, 9)) + " |")

rate_md = ["| config | offered RPS | served img/s | p50 s | p95 s | p99 s | CCU avg/peak | errors |", "|" + "---|" * 8]
for label, rr in by.items():
    for r in rr:
        if r["mode"] == "rate":
            rate_md.append(f"| {label} | {r['target_rps']} | {r['throughput_ips']} | {r['lat_p50']} | {r['lat_p95']} | {r['lat_p99']} | "
                           f"{r['ccu_avg']}/{r['ccu_peak']} | {r['errors']} |")

text = ("# Serving benchmark (H100 PCIe, GPU 0, 512x512 image-to-image, 50% avatar / 50% sticker, 3 styles)\n\n"
        "Server numbers on an H100, not phone numbers.\n\n## Summary\n\n" + "\n".join(md) +
        "\n\n## Closed loop: img/s / p95 latency at each CCU\n\n" + "\n".join(ccu_md) +
        "\n\n## Open loop (Poisson arrivals)\n\n" + "\n".join(rate_md) + "\n")
(ROOT / "results/summary.md").write_text(text)
with open(ROOT / "results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
    w.writeheader()
    w.writerows(out_rows)
print(text)
