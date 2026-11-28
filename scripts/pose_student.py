"""Pose-conditioned DreamLite student helpers (2026-11-27). conv_in(x) + pconv(pose), pconv zero-init, pose on the canvas half."""
from pathlib import Path

import torch
import torch.nn as nn


class PoseConvIn(nn.Module):
    def __init__(self, orig, ch=3):
        super().__init__()
        self.orig = orig
        self.pconv = nn.Conv2d(ch, orig.out_channels, 3, padding=1)
        nn.init.zeros_(self.pconv.weight)
        nn.init.zeros_(self.pconv.bias)
        self.pconv.to(orig.weight.device, orig.weight.dtype)
        self.pose = None  # [B, ch, H, W] set before each forward (diptych layout: map on the canvas half)

    def forward(self, x):
        y = self.orig(x)
        if self.pose is not None:
            y = y + self.pconv(self.pose.to(y.dtype))
        return y


def install_pose_conv(unet, ch=3):
    if not isinstance(unet.conv_in, PoseConvIn):
        unet.conv_in = PoseConvIn(unet.conv_in, ch)
    return unet.conv_in


def save_unet_and_pose(unet, out):
    out = Path(out)
    pci = unet.conv_in
    unet.conv_in = pci.orig
    try:
        unet.save_pretrained(out / "unet")
    finally:
        unet.conv_in = pci
    torch.save(pci.pconv.state_dict(), out / "pose_conv.pt")


def load_pose_student(w, dtype=torch.float16, device="cuda"):
    """weights/student/<tag> -> (unet with pose conv, task_emb dict)."""
    import dl_arch as DLA
    w = Path(w)
    unet = DLA.load_unet(w / "unet", style_film=False)
    pci = install_pose_conv(unet)
    pci.pconv.load_state_dict(torch.load(w / "pose_conv.pt"))
    return unet.to(device, dtype).eval(), torch.load(w / "task_emb.pt")


def diptych(pm):
    """pm [B, 3, 64, 64] float 0..1 -> [B, 3, 64, 128] (canvas half)."""
    return torch.cat([pm, torch.zeros_like(pm)], -1)


# ---- tSP0b (2026-11-27): multi-level pose adapter (T2I-Adapter-lite) -------------------------------------------------------------
class PoseAdapter(nn.Module):
    """pose [B, 3, 64, 128] -> features added to the INPUTS of down_blocks 0 / 1 / 2 (256@64x128, 256@32x64, 512@16x32) through
    forward pre-hooks; the three output 1x1 convs are zero-initialised (step 0 = the init student exactly). ~0.5M params."""

    def __init__(self, ch=3, outs=(256, 256, 512)):
        super().__init__()
        self.e0 = nn.Sequential(nn.Conv2d(ch, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 64, 3, padding=1), nn.SiLU())
        self.e1 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(), nn.Conv2d(128, 128, 3, padding=1), nn.SiLU())
        self.e2 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.SiLU(), nn.Conv2d(256, 256, 3, padding=1), nn.SiLU())
        self.z = nn.ModuleList([nn.Conv2d(64, outs[0], 1), nn.Conv2d(128, outs[1], 1), nn.Conv2d(256, outs[2], 1)])
        for zc in self.z:
            nn.init.zeros_(zc.weight)
            nn.init.zeros_(zc.bias)
        self.pose = None
        self._feats = None

    def features(self, dtype):
        if self.pose is None:
            return None
        f0 = self.e0(self.pose.to(self.z[0].weight.dtype))
        f1 = self.e1(f0)
        f2 = self.e2(f1)
        return [self.z[0](f0).to(dtype), self.z[1](f1).to(dtype), self.z[2](f2).to(dtype)]


def install_pose_adapter(unet, ch=3):
    if getattr(unet, "_pose_adapter", None) is not None:
        return unet._pose_adapter
    ad = PoseAdapter(ch).to(next(unet.parameters()).device)
    object.__setattr__(unet, "_pose_adapter", ad)  # NOT registered as a submodule -> unet.save_pretrained stays a plain DreamLite UNet

    def pre(i):
        def hook(m, args, kwargs):
            if ad.pose is None:
                return None
            if i == 0:
                ad._feats = ad.features(kwargs["hidden_states"].dtype)
            h = kwargs["hidden_states"]
            kwargs["hidden_states"] = h + ad._feats[i]
            return args, kwargs
        return hook

    for i in range(3):
        unet.down_blocks[i].register_forward_pre_hook(pre(i), with_kwargs=True)
    return ad


def set_pose(unet, pose):
    """pose [B, 3, 64, 128] (diptych layout) or None, for either the conv-in (tSP0) or the adapter (tSP0b) student."""
    if getattr(unet, "_pose_adapter", None) is not None:
        unet._pose_adapter.pose = pose
    elif isinstance(unet.conv_in, PoseConvIn):
        unet.conv_in.pose = pose


def load_pose_student_any(w, dtype=torch.float16, device="cuda"):
    """conv-in (pose_conv.pt) or adapter (pose_adapter.pt) pose student."""
    w = Path(w)
    if (w / "pose_adapter.pt").exists():
        import dl_arch as DLA
        unet = DLA.load_unet(w / "unet", style_film=False)
        sd = torch.load(w / "pose_adapter.pt")
        ad = install_pose_adapter(unet, ch=sd["e0.0.weight"].shape[1])  # 3 (tSP0b) or 4 (tSP0c: + map-active flag)
        ad.load_state_dict(sd)
        unet = unet.to(device, dtype).eval()
        ad.to(device, dtype).eval()
        return unet, torch.load(w / "task_emb.pt")
    return load_pose_student(w, dtype, device)
