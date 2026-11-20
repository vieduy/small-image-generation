"""Add `subject_description` (no face/eye/hair/skin words) to each style.yaml: the person-oriented style description
humanized ~71% of non-person sticker targets (balls became heads, dogs got human faces), see README deviation 8."""
import yaml
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
SD = {"cartoon3d": "3D cartoon render, smooth rounded shapes, soft studio lighting, soft vibrant colors, clean simple details",
      "cartoon3d_b": "3D cartoon render, smooth rounded shapes, soft studio lighting, clean simple details",
      "chibi": "chibi anime style, cute simplified proportions, cel shading, pastel colors",
      "flat_vector": "flat vector illustration, bold clean outlines, flat colors, no gradients, minimal shading, geometric simplified shapes",
      "toy3d": "3D toy figure render, glossy vinyl material, soft studio lighting, rounded shapes, collectible designer toy look"}
for s, d in SD.items():
    p = ROOT / "styles" / s / "style.yaml"
    y = yaml.safe_load(p.read_text())
    y["subject_description"] = d
    p.write_text(yaml.safe_dump(y, sort_keys=False, width=1000))
    print(s, "|", y["description"])
