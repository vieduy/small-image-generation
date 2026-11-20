"""How much does each student's training loss notice brown -> black irises? CPU only, 64 klein cartoon3d targets."""
import sys, numpy as np, torch, cv2, lpips, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, "scripts")
import dl_arch as DLA
from metrics import FaceID
from diffusers import AutoencoderTiny
torch.set_grad_enabled(False)
R = Path(".")
paths = sorted((R / "data/pairs/cartoon3d/avatar").glob("c*.png"))[:400:6][:64]
det = FaceID().det; det.setInputSize((512, 512))
orig, mod, masks = [], [], []
for p in paths:
    bgr = cv2.resize(cv2.imread(str(p)), (512, 512))
    _, f = det.detect(bgr)
    if f is None: continue
    f = max(f, key=lambda r: r[2] * r[3]); rad = 0.07 * f[2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    yy, xx = np.mgrid[:512, :512]
    near = np.zeros((512, 512), bool)
    for k in range(2):
        near |= (xx - f[4 + 2 * k]) ** 2 + (yy - f[5 + 2 * k]) ** 2 < rad ** 2
    iris = near & (hsv[..., 1] > 60) & (hsv[..., 2] < 200) & (hsv[..., 2] > 25)  # coloured, not white/black
    if iris.sum() < 30: continue
    m = bgr.copy(); m[iris] = (m[iris].astype(np.float32).mean(1, keepdims=True) * 0.3).astype(np.uint8)  # dark grey iris
    orig.append(bgr[..., ::-1].copy()); mod.append(m[..., ::-1].copy()); masks.append(iris)
print("images", len(orig), "mean iris px @512:", np.mean([m.sum() for m in masks]).round(0), "=", (100 * np.mean([m.mean() for m in masks])).round(2), "% of image")
cv2.imwrite("/tmp/claude-1021/-home-duyhoang-serving-images-based/e8b9cdf4-9bbd-434e-966e-9204d269efa0/scratchpad/iris_mod_example.jpg", np.concatenate([orig[0][..., ::-1], mod[0][..., ::-1]], 1)[150:300])
t = lambda L: torch.tensor(np.stack(L)).permute(0, 3, 1, 2).float() / 127.5 - 1
xo, xm = t(orig), t(mod)
lp = lpips.LPIPS(net="vgg", verbose=False).eval()
d = lambda s: F.interpolate(s, size=s.shape[-1] if s is None else 256, mode="bilinear", antialias=True)
l256 = lp(d(xo), d(xm)).mean().item()
l512 = lp(xo, xm).mean().item()
for name, vae in [("TAESD (SDXS/SD1.5)", AutoencoderTiny.from_pretrained("madebyollin/taesd")), ("DreamLite TinyVAE", DLA.load_vae())]:
    zo, zm = vae.encode(xo).latents, vae.encode(xm).latents
    dz = F.mse_loss(zo, zm).item(); var = zo.var().item()
    print(f"{name:20s} latent var {var:.3f} | latent MSE(brown,black) {dz:.5f} = {dz / var * 100:.3f}% of latent var")
print(f"LPIPS(brown,black) @256 {l256:.4f} | @512 {l512:.4f}")
