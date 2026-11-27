"""Option-token cache for a style the student never trained on (2026-11-26, comic teacher tC1): weights/student/_dl_tokens_<style>_22.pt =
train_dl.init_tokens (Qwen3-VL edit-mode hidden states of each option's instruction, averaged over 16 synthetic photos; comic_line has no
inputs/ dir, so the photos are cartoon3d's inputs f00000..f01500 - same photo pool train_teacher_dl uses for its null tokens).
  python scripts/s13_tokens.py comic_line"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import train_dl as T  # noqa: E402
from train_student import TASKS, expr_tasks  # noqa: E402

style = sys.argv[1]
tasks = TASKS + expr_tasks()
out = T.ROOT / f"weights/student/_dl_tokens_{style}_{len(tasks)}.pt"
assert not out.exists(), f"{out} exists"
photos = [T.ROOT / "data/pairs/cartoon3d/inputs" / f"f{i:05d}.png" for i in range(0, 1600, 100)]
emb, mask = T.init_tokens(style, tasks, photos, out)
print(f"[s13] {out} emb {tuple(emb.shape)}", flush=True)
