"""Find (and with --delete remove) unreadable/truncated PNGs in a folder, CPU pool. 2026-11-18: a killed framing pool left
truncated files in data/ffhq_norm.   $VENV/bin/python scripts/verify_pngs.py DIR [--delete]"""
import sys
from multiprocessing import Pool
from pathlib import Path
from PIL import Image

def bad(p):
    try:
        with Image.open(p) as im:
            im.load()
        return None
    except Exception as e:
        return f"{p} {type(e).__name__}"

if __name__ == "__main__":
    d = Path(sys.argv[1])
    ps = sorted(d.glob("*.png"))
    with Pool(128) as pool:
        res = [r for r in pool.map(bad, ps, chunksize=64) if r]
    for t in d.glob("*.tmp*.png"):
        res.append(f"{t} tmpfile")
    print(f"[verify] {len(ps)} files, {len(res)} bad", flush=True)
    for r in res:
        print(" ", r)
        if "--delete" in sys.argv:
            Path(r.split()[0]).unlink()
