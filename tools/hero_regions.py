"""Find the garment regions on Robin's texture, so villagers can be dressed differently.

Each texel is labelled from two things: which bones move the surface it is painted on (found by
drawing every triangle of the mesh into texture space with its vertices' bone weights), and its
own colour (green hood, khaki cape, brown leather, grey mail, skin, dark hair).

Writes assets_src/npc/regions.npz (one float mask per region, texture-sized) and a preview,
assets_src/npc/regions.png.
Usage: python hero_regions.py
"""
import io
import pathlib
import sys

import numpy as np
from PIL import Image, ImageFilter

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from glb import Glb  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "assets_src" / "npc"
OUT.mkdir(parents=True, exist_ok=True)

g = Glb(str(ROOT / "live_src" / "hero.glb"))
J = g.json
prim = J["meshes"][0]["primitives"][0]
uv = g.read(prim["attributes"]["TEXCOORD_0"]).astype(np.float64)
pos = g.read(prim["attributes"]["POSITION"]).astype(np.float64)
jnt = g.read(prim["attributes"]["JOINTS_0"]).astype(np.int64)
wgt = g.read(prim["attributes"]["WEIGHTS_0"]).astype(np.float64)
idx = g.read(prim["indices"]).reshape(-1, 3)
names = [J["nodes"][j]["name"] for j in J["skins"][0]["joints"]]
img_index = J["textures"][J["materials"][prim["material"]]["pbrMetallicRoughness"]["baseColorTexture"]["index"]]["source"]
image = Image.open(io.BytesIO(g.view_bytes(J["images"][img_index]["bufferView"]))).convert("RGB")
W, H = image.size
rgb = np.asarray(image, dtype=np.float64) / 255

# bone groups -> one weight per vertex
GROUPS = {"head": ("Head", "neck", "head_end", "headfront"), "arms": ("Arm", "ForeArm", "Hand", "Shoulder"),
          "legs": ("UpLeg", "Leg", "Foot", "Toe"), "body": ("Hips", "Spine")}
group_w = {}
for gname, keys in GROUPS.items():
    sel = np.array([any(k in n for k in keys) for n in names])
    if gname == "arms":
        sel &= np.array(["Leg" not in n for n in names])
    group_w[gname] = (wgt * sel[jnt]).sum(axis=1)
# where on the body: behind the torso (the cape hangs there), height
behind = np.clip((-pos[:, 2] - 0.02) / 0.1, 0, 1)
height = pos[:, 1]
attrs = np.stack([group_w["head"], group_w["arms"], group_w["legs"], group_w["body"], behind, height], axis=1)

# rasterise every triangle into texture space (barycentric fill of its uv footprint)
field = np.zeros((H, W, attrs.shape[1]))
cover = np.zeros((H, W))
P = np.stack([uv[:, 0] * W, uv[:, 1] * H], axis=1)
for a, b, c in idx:
    pa, pb, pc = P[a], P[b], P[c]
    x0, x1 = int(max(0, np.floor(min(pa[0], pb[0], pc[0])))), int(min(W - 1, np.ceil(max(pa[0], pb[0], pc[0]))))
    y0, y1 = int(max(0, np.floor(min(pa[1], pb[1], pc[1])))), int(min(H - 1, np.ceil(max(pa[1], pb[1], pc[1]))))
    if x1 < x0 or y1 < y0:
        continue
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
    d = (pb[1] - pc[1]) * (pa[0] - pc[0]) + (pc[0] - pb[0]) * (pa[1] - pc[1])
    if abs(d) < 1e-9:
        continue
    l1 = ((pb[1] - pc[1]) * (xs - pc[0]) + (pc[0] - pb[0]) * (ys - pc[1])) / d
    l2 = ((pc[1] - pa[1]) * (xs - pc[0]) + (pa[0] - pc[0]) * (ys - pc[1])) / d
    l3 = 1 - l1 - l2
    inside = (l1 >= -0.02) & (l2 >= -0.02) & (l3 >= -0.02)
    if not inside.any():
        continue
    vals = l1[..., None] * attrs[a] + l2[..., None] * attrs[b] + l3[..., None] * attrs[c]
    yy, xx = (ys[inside] - 0.5).astype(int), (xs[inside] - 0.5).astype(int)
    field[yy, xx] = vals[inside]
    cover[yy, xx] = 1
# spread labels a little past the islands so filtering never meets an unlabelled texel
for _ in range(6):
    empty = cover == 0
    if not empty.any():
        break
    acc = np.zeros_like(field)
    cnt = np.zeros_like(cover)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        acc += np.roll(np.roll(field, dy, 0), dx, 1) * np.roll(np.roll(cover, dy, 0), dx, 1)[..., None]
        cnt += np.roll(np.roll(cover, dy, 0), dx, 1)
    grow = empty & (cnt > 0)
    field[grow] = acc[grow] / cnt[grow][:, None]
    cover[grow] = 1

head, arms, legs, body, back, ht = [field[..., k] for k in range(6)]
r, gch, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
mx, mn = rgb.max(axis=2), rgb.min(axis=2)
sat = (mx - mn) / np.maximum(mx, 1e-4)
val = mx


def ramp(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1)


green = ramp(gch - np.maximum(r, b), 0.005, 0.04) * ramp(sat, 0.12, 0.25)
grey = (1 - ramp(sat, 0.10, 0.22)) * ramp(b - r, -0.03, 0.02)
khaki = ramp(sat, 0.15, 0.3) * (1 - ramp(np.abs(r / np.maximum(gch, 1e-3) - 1.07), 0.08, 0.2)) * ramp(gch - b, 0.04, 0.1) * (1 - green)
leather = ramp(r / np.maximum(gch, 1e-3), 1.12, 1.25) * (1 - ramp(val, 0.42, 0.55)) * ramp(sat, 0.2, 0.3)
skin = ramp(r / np.maximum(gch, 1e-3), 1.1, 1.22) * ramp(val, 0.45, 0.6) * ramp(sat, 0.22, 0.35)

regions = {
    "hood": green * (1 - legs),
    "cape": khaki * (1 - head) * np.maximum(back, ramp(ht, 0.0, 0.3) * (1 - arms)) * (1 - green),
    "mail": grey * (1 - head) * (1 - legs * 0.7),
    "vest": leather * body * (1 - back) * (1 - head),
    "trousers": np.maximum(leather, khaki * (1 - back)) * legs * (1 - back),
    "skin": skin * np.maximum(head, arms),
    "hair": head * ramp(0.32, 0.30, 0.18) * (1 - ramp(val, 0.18, 0.33)) * (1 - skin),
}
# the cape also takes whatever khaki hangs low behind the legs
regions["cape"] = np.maximum(regions["cape"], khaki * back * (1 - head))
for k in regions:
    m = Image.fromarray((np.clip(regions[k], 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0))
    regions[k] = np.asarray(m, dtype=np.float64) / 255
np.savez_compressed(OUT / "regions.npz", **regions, base=rgb)

palette = {"hood": (0.1, 0.9, 0.2), "cape": (0.95, 0.75, 0.1), "mail": (0.4, 0.6, 1.0), "vest": (0.8, 0.2, 0.1),
           "trousers": (0.7, 0.3, 0.9), "skin": (1.0, 0.7, 0.6), "hair": (0.1, 0.1, 0.1)}
preview = rgb * 0.25
for k, col in palette.items():
    preview = preview * (1 - regions[k][..., None]) + np.array(col) * regions[k][..., None]
Image.fromarray((np.clip(np.concatenate([rgb, preview], axis=1), 0, 1) * 255).astype(np.uint8)).save(OUT / "regions.png")
for k, m in regions.items():
    print(f"{k:9s} {m.mean() * 100:5.1f}% of the texture")
