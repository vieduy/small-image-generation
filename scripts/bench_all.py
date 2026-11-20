"""Run the serving benchmark for each model config: CCU 1..8 closed loop, then Poisson RPS sweep at fractions of the
closed-loop ceiling. One server at a time on the GPU in CUDA_VISIBLE_DEVICES. GPU activity of OTHER processes is
sampled (nvidia-smi pmon) for the whole run and stored next to the results so contaminated windows can be flagged.

  source env.sh; $VENV/bin/python scripts/bench_all.py --configs sdxs_b1 sdxs_b8 ... [--ccu-dur 30 --rate-dur 60]
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = os.environ["VENV"] + "/bin/python"
CONFIGS = {  # label: env for i2i_server.py  (settings locked after smoke tests, see results/smoke.jsonl + grids/)
    "klein_text_b1": dict(MODEL="klein", MAX_BATCH=1, KLEIN_REF=0, PROMPTS_FAMILY="klein_text"),  # 4 steps; pipeline can't batch per-sample input images
    "sprint_b1": dict(MODEL="sana_sprint", MAX_BATCH=1, STRENGTH=0.6), "sprint_b8": dict(MODEL="sana_sprint", MAX_BATCH=8, STRENGTH=0.6),
    "sdturbo_b1": dict(MODEL="sdturbo", MAX_BATCH=1, STEPS=4, STRENGTH=0.75), "sdturbo_b8": dict(MODEL="sdturbo", MAX_BATCH=8, STEPS=4, STRENGTH=0.75),
    "ip2p_b1": dict(MODEL="ip2p", MAX_BATCH=1), "ip2p_b8": dict(MODEL="ip2p", MAX_BATCH=8),
    "sd15ipa_b1": dict(MODEL="sd15_ipa_lcm", MAX_BATCH=1), "sd15ipa_b8": dict(MODEL="sd15_ipa_lcm", MAX_BATCH=8),
    "sdxs_timingonly_b1": dict(MODEL="sdxs", MAX_BATCH=1, STRENGTH=0.75), "sdxs_timingonly_b8": dict(MODEL="sdxs", MAX_BATCH=8, STRENGTH=0.75),
}
RATE_FRACS = [0.3, 0.5, 0.7, 0.85, 1.0]


def health(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--ccu-dur", type=float, default=30)
    ap.add_argument("--ccu", nargs="+", default=["1", "2", "3", "4", "5", "6", "7", "8"])
    ap.add_argument("--rate-dur", type=float, default=60)
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--out", default=str(ROOT / "results" / "serving.jsonl"))
    a = ap.parse_args()
    logs = ROOT / "logs"
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    for label in a.configs:
        env = {**os.environ, **{k: str(v) for k, v in CONFIGS[label].items()}}
        slog = open(logs / f"server_{label}.log", "w")
        pmon = subprocess.Popen(["nvidia-smi", "pmon", "-i", gpu, "-s", "um", "-o", "T"],
                                stdout=open(logs / f"pmon_{label}.log", "w"), stderr=subprocess.STDOUT)
        srv = subprocess.Popen([PY, str(ROOT / "scripts/i2i_server.py"), "--port", str(a.port)], env=env,
                               stdout=slog, stderr=subprocess.STDOUT)
        print(f"[{label}] server pid {srv.pid} starting", flush=True)
        try:
            t = time.time()
            while not (h := health(a.port)) or not h.get("ok"):
                if srv.poll() is not None or time.time() - t > 900:
                    raise RuntimeError(f"server failed, see logs/server_{label}.log")
                time.sleep(3)
            print(f"[{label}] ready {time.time() - t:.0f}s {h}", flush=True)
            common = [PY, str(ROOT / "scripts/loadgen_i2i.py"), "--url", f"http://127.0.0.1:{a.port}", "--label", label,
                      "--model", CONFIGS[label]["MODEL"], "--steps", str(h["steps"]), "--out", a.out]
            res = subprocess.run(common + ["--mode", "ccu", "--duration", str(a.ccu_dur), "--ccu", *a.ccu], capture_output=True, text=True)
            ccu = [json.loads(l) for l in res.stdout.splitlines() if l.startswith("{")]
            for r in ccu:
                print(f"[{label}] ccu={r['target_ccu']} ips={r['throughput_ips']} p50={r['lat_p50']} p95={r['lat_p95']} "
                      f"batch={r.get('mean_batch')} err={r['errors']}", flush=True)
            peak = max((r["throughput_ips"] for r in ccu if r["errors"] == 0), default=0)
            if not peak:
                raise RuntimeError(f"no successful ccu level: {res.stderr[-500:]}")
            rates = [round(peak * f, 2) for f in RATE_FRACS]
            res = subprocess.run(common + ["--mode", "rate", "--duration", str(a.rate_dur), "--rates", *map(str, rates)],
                                 capture_output=True, text=True)
            for l in res.stdout.splitlines():
                if l.startswith("{"):
                    r = json.loads(l)
                    print(f"[{label}] rps={r['target_rps']} served={r['throughput_ips']} p50={r['lat_p50']} "
                          f"p95={r['lat_p95']} ccu={r['ccu_avg']}/{r['ccu_peak']} err={r['errors']}", flush=True)
            print(f"[{label}] final health {health(a.port)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{label}] FAILED: {e}", flush=True)
        finally:
            srv.terminate()
            try:
                srv.wait(60)
            except subprocess.TimeoutExpired:
                srv.kill()
            pmon.terminate()
            time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
