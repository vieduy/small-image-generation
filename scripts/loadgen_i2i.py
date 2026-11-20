"""Load generator for scripts/i2i_server.py: closed-loop CCU sweep and open-loop Poisson RPS sweep.

  $VENV/bin/python scripts/loadgen_i2i.py --label sdturbo_b8 --mode ccu --ccu 1 2 3 4 5 6 7 8 --duration 30
  $VENV/bin/python scripts/loadgen_i2i.py --label sdturbo_b8 --mode rate --rates 2 4 6 8 --duration 60

Input photos are the synthetic faces/subjects in data/ (pre-encoded once). Task mix: --sticker-frac of requests are
stickers (faces+subjects), the rest avatars (faces). Styles rotate over styles/*. Reuses summary/CCU code from
~/serving/images-based/bench/loadgen.py (same jsonl schema: lat p50..p99, throughput_ips, ccu_avg/peak, queue/infer).
"""
import argparse
import asyncio
import base64
import json
import random
import sys
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, "/home/duyhoang/serving/images-based/bench")
from loadgen import summarize  # noqa: E402

STYLES = sorted(p.parent.name for p in (ROOT / "styles").glob("*/style.yaml"))


def load_inputs():
    enc = lambda p: base64.b64encode(p.read_bytes()).decode()  # noqa: E731
    faces = [enc(p) for p in sorted((ROOT / "data/faces").glob("*.png"))]
    subj = [enc(p) for p in sorted((ROOT / "data/subjects").glob("*.png"))]
    assert faces, "run scripts/gen_data.py first"
    return faces, subj


def body(args, i, faces, subj):
    rng = random.Random(args.seed * 1_000_003 + i)
    sticker = rng.random() < args.sticker_frac
    pool = faces + subj if sticker else faces
    return {"image": pool[rng.randrange(len(pool))], "task": "sticker" if sticker else "avatar",
            "style": STYLES[i % len(STYLES)], "seed": i, "response_format": args.response_format}


async def one(s, args, b, out):
    t = time.perf_counter()
    rec = {"t_send": t}
    try:
        async with s.post(args.url + "/v1/i2i", json=b, timeout=aiohttp.ClientTimeout(total=args.timeout)) as r:
            data = await r.read()
            rec["status"] = r.status
            if r.status == 200:
                rec["timing"] = json.loads(data).get("timing")
            else:
                rec["error"] = data[:120].decode(errors="replace")
    except Exception as e:  # noqa: BLE001
        rec["status"], rec["error"] = -1, type(e).__name__
    rec["t_done"] = time.perf_counter()
    rec["latency"] = rec["t_done"] - t
    out.append(rec)


async def run_ccu(args, n, faces, subj):
    out, ctr = [], iter(range(10**9))
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as s:
        t0 = time.perf_counter()

        async def user():
            while time.perf_counter() - t0 < args.duration:
                await one(s, args, body(args, next(ctr), faces, subj), out)

        await asyncio.gather(*(user() for _ in range(n)))
        t_end = time.perf_counter()
    return summarize(args, out, t0, t0 + args.duration, t_end, {"mode": "ccu", "target_rps": None, "target_ccu": n})


async def run_rate(args, rate, faces, subj):
    out, tasks, rng = [], [], random.Random(args.seed)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as s:
        t0, i = time.perf_counter(), 0
        while time.perf_counter() - t0 < args.duration:
            tasks.append(asyncio.create_task(one(s, args, body(args, i, faces, subj), out)))
            i += 1
            await asyncio.sleep(rng.expovariate(rate))
        send_end = time.perf_counter()
        await asyncio.gather(*tasks)
        t_end = time.perf_counter()
    return summarize(args, out, t0, send_end, t_end, {"mode": "rate", "target_rps": rate, "target_ccu": None})


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8200")
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--mode", default="ccu", choices=["ccu", "rate"])
    ap.add_argument("--ccu", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--rates", nargs="+", type=float, default=[1.0])
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--sticker-frac", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--response-format", default="b64_json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "serving.jsonl"))
    args = ap.parse_args()
    args.api, args.guidance, args.set_size = "i2i", None, 1  # fields expected by summarize()
    faces, subj = load_inputs()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for lv in (args.ccu if args.mode == "ccu" else args.rates):
        rec = await (run_ccu(args, lv, faces, subj) if args.mode == "ccu" else run_rate(args, lv, faces, subj))
        rec["sticker_frac"] = args.sticker_frac
        print(json.dumps(rec), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
