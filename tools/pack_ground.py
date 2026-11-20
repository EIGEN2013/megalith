"""Pack Poly Haven ground scans for the terrain shader: per layer two WebP images
  <layer>_c.webp   RGB colour, A height (for height-blended transitions between layers)
  <layer>_n.webp   RGB OpenGL normal, A roughness
Writes web/megalith/ground/. Usage: python tools/pack_ground.py [size]
"""
import json
import pathlib
import sys

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
PH = ROOT / "ph"
OUT = ROOT / "web" / "megalith" / "ground"
OUT.mkdir(parents=True, exist_ok=True)
SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
# layer: (scan, metres per tile)
LAYERS = {"meadow": ("sparse_grass", 3.0), "forest": ("forest_leaves_04", 3.0), "road": ("rocky_trail", 3.0),
          "rock": ("rock_face_03", 5.0), "sand": ("coast_sand_03", 4.0)}


def find(asset, key, ext):
    hits = sorted((PH / asset).glob(f"*_{key}_2k.{ext}"))
    return hits[0] if hits else None


def load(path, mode):
    return Image.open(path).convert(mode).resize((SIZE, SIZE), Image.LANCZOS)


meta = {}
for layer, (asset, tile) in LAYERS.items():
    diff = load(find(asset, "diff", "jpg"), "RGB")
    if layer == "meadow":       # the scan is half bare soil; grade it to living turf (the grass tufts do the rest)
        lin = (np.asarray(diff).astype(np.float64) / 255) ** 2.2
        L = lin @ np.array([0.2126, 0.7152, 0.0722])
        green = np.stack([L * 0.62, L * 1.28, L * 0.45], axis=-1)
        diff = Image.fromarray((np.clip((lin * 0.35 + green * 0.65) * 1.25, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8))
    disp = find(asset, "disp", "png")
    if disp:
        raw = Image.open(disp)
        a = np.asarray(raw).astype(np.float64)
        if a.ndim == 3:
            a = a[..., 0]
        a = (a - a.min()) / max(a.max() - a.min(), 1)            # 16-bit heights -> 0..1, stretched
        h = Image.fromarray((a * 254 + 1).astype(np.uint8)).resize((SIZE, SIZE), Image.LANCZOS)
    else:
        h = Image.new("L", (SIZE, SIZE), 128)
    nor = load(find(asset, "nor_gl", "jpg"), "RGB")
    arm = load(find(asset, "arm", "jpg"), "RGB")
    rough = arm.split()[1]
    c = Image.merge("RGBA", (*diff.split(), h))
    n = Image.merge("RGBA", (*nor.split(), rough))
    c.save(OUT / f"{layer}_c.webp", quality=88, method=4, exact=True)
    n.save(OUT / f"{layer}_n.webp", quality=82, method=4, exact=True)
    mean = np.asarray(diff, dtype=np.float64).reshape(-1, 3).mean(axis=0) / 255
    meta[layer] = {"tile": tile, "mean": [round(float(x), 3) for x in mean]}
    print(layer, asset, [round((OUT / f"{layer}_{k}.webp").stat().st_size / 1e6, 2) for k in "cn"], "MB")
(OUT / "ground.json").write_text(json.dumps(meta, indent=1))
