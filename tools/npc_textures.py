"""Dress the villagers: recoloured copies of Robin's texture, one per villager.

Robin's texture is baked from many views, so garments cannot be told apart by colour alone.
Each vertex is therefore given a garment from where it sits on the body and which bones move
it (cowl round the neck, cloak hanging behind and at the sides, front of the torso, sleeves,
legs, head). Those labels are drawn into texture space through the mesh's own triangles, and
each garment is recoloured there while keeping the painted folds and shading.

Writes assets_src/npc/<villager>.jpg and a labelled preview assets_src/npc/garments.png.
Usage: python npc_textures.py
"""
import io
import json
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


def ramp(x, lo, hi):
    return np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0, 1)


# ------------------------------------------------------------------ villagers
# Colours are the garment's average colour (sRGB). None leaves that garment as it is.
VILLAGERS = {
    "marta": {"cloak": (0.55, 0.13, 0.12), "cowl": (0.80, 0.62, 0.22), "torso": (0.36, 0.22, 0.14), "legs": (0.30, 0.20, 0.14), "sleeves": (0.78, 0.72, 0.60), "hair": None},
    "oswin": {"cloak": (0.12, 0.18, 0.36), "cowl": (0.40, 0.42, 0.46), "torso": (0.22, 0.16, 0.12), "legs": (0.18, 0.18, 0.22), "sleeves": None, "hair": (0.10, 0.08, 0.07)},
    "eldric": {"cloak": (0.32, 0.18, 0.40), "cowl": (0.78, 0.72, 0.58), "torso": (0.40, 0.34, 0.28), "legs": (0.36, 0.30, 0.24), "sleeves": (0.70, 0.66, 0.58), "hair": (0.80, 0.80, 0.78)},
    "toby": {"cloak": (0.30, 0.52, 0.72), "cowl": (0.90, 0.52, 0.16), "torso": (0.52, 0.38, 0.22), "legs": (0.40, 0.30, 0.20), "sleeves": (0.86, 0.82, 0.70), "hair": (0.60, 0.36, 0.14)},
    "bren": {"cloak": (0.24, 0.36, 0.38), "cowl": (0.42, 0.30, 0.18), "torso": (0.30, 0.26, 0.20), "legs": (0.24, 0.26, 0.30), "sleeves": (0.62, 0.60, 0.56), "hair": (0.20, 0.14, 0.10)},
    "lina": {"cloak": (0.18, 0.36, 0.20), "cowl": (0.86, 0.80, 0.64), "torso": (0.56, 0.36, 0.30), "legs": (0.40, 0.26, 0.24), "sleeves": (0.90, 0.86, 0.76), "hair": (0.48, 0.24, 0.10)},
}


def main():
    g = Glb(str(ROOT / "live_src" / "hero.glb"))
    J = g.json
    prim = J["meshes"][0]["primitives"][0]
    uv = g.read(prim["attributes"]["TEXCOORD_0"]).astype(np.float64)
    pos = g.read(prim["attributes"]["POSITION"]).astype(np.float64)
    jnt = g.read(prim["attributes"]["JOINTS_0"]).astype(np.int64)
    wgt = g.read(prim["attributes"]["WEIGHTS_0"]).astype(np.float64)
    idx = g.read(prim["indices"]).reshape(-1, 3)
    names = [J["nodes"][j]["name"] for j in J["skins"][0]["joints"]]
    tex_i = J["textures"][J["materials"][prim["material"]]["pbrMetallicRoughness"]["baseColorTexture"]["index"]]["source"]
    image = Image.open(io.BytesIO(g.view_bytes(J["images"][tex_i]["bufferView"]))).convert("RGB")
    W, H = image.size
    rgb = np.asarray(image, dtype=np.float64) / 255

    def bone_weight(*keys, exclude=()):
        sel = np.array([any(k in n for k in keys) and not any(e in n for e in exclude) for n in names])
        return (wgt * sel[jnt]).sum(axis=1)

    head = bone_weight("Head", "neck", "head_end", "headfront")
    arms = bone_weight("Arm", exclude=("Shoulder", "Hand"))
    legs = bone_weight("UpLeg", "Leg", "Foot", "Toe")
    hands = bone_weight("Hand", "ForeArm")
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    # Robin faces +Z. The cloak hangs behind and at the sides; the cowl wraps neck and shoulders.
    behind = ramp(-z, -0.02, 0.06)
    side = ramp(np.abs(x), 0.22, 0.3) * (1 - ramp(arms + hands, 0.2, 0.5))
    cowl = ramp(y, 1.30, 1.36) * (1 - ramp(y, 1.6, 1.66)) * (1 - ramp(head, 0.45, 0.7)) * (1 - ramp(arms, 0.35, 0.6))
    cloak = np.maximum(behind, side) * ramp(y, 0.12, 0.25) * (1 - ramp(y, 1.32, 1.38)) * (1 - ramp(head, 0.3, 0.6))
    cloak = cloak * (1 - cowl) * (1 - ramp(hands, 0.3, 0.6))
    sleeves = ramp(arms, 0.4, 0.7) * (1 - cloak) * (1 - cowl) * ramp(np.abs(x), 0.2, 0.26)
    torso = (1 - cloak) * (1 - cowl) * (1 - sleeves) * ramp(y, 0.8, 0.9) * (1 - ramp(y, 1.32, 1.38)) * (1 - ramp(head, 0.3, 0.6))
    leg = (1 - cloak) * ramp(legs, 0.35, 0.6) * (1 - ramp(y, 0.95, 1.0)) * (1 - torso)
    hd = ramp(head, 0.55, 0.8) * ramp(y, 1.5, 1.56)
    labels = np.stack([cloak, cowl, torso, leg, sleeves, hd], axis=1)
    order = ["cloak", "cowl", "torso", "legs", "sleeves", "head"]

    # draw the per-vertex labels into texture space
    field = np.zeros((H, W, labels.shape[1]))
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
        inside = (l1 >= -0.05) & (l2 >= -0.05) & (l3 >= -0.05)
        if not inside.any():
            continue
        vals = l1[..., None] * labels[a] + l2[..., None] * labels[b] + l3[..., None] * labels[c]
        yy, xx = (ys[inside] - 0.5).astype(int), (xs[inside] - 0.5).astype(int)
        field[yy, xx] = vals[inside]
        cover[yy, xx] = 1
    for _ in range(8):                     # grow labels past the islands' edges
        empty = cover == 0
        acc = np.zeros_like(field)
        cnt = np.zeros_like(cover)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            acc += np.roll(np.roll(field, dy, 0), dx, 1) * np.roll(np.roll(cover, dy, 0), dx, 1)[..., None]
            cnt += np.roll(np.roll(cover, dy, 0), dx, 1)
        grow = empty & (cnt > 0)
        field[grow] = acc[grow] / cnt[grow][:, None]
        cover[grow] = 1
    masks = {}
    for k, name in enumerate(order):
        m = Image.fromarray((np.clip(field[..., k], 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
        masks[name] = np.asarray(m, dtype=np.float64) / 255

    lum = rgb @ np.array([0.299, 0.587, 0.114])
    # hair: the dark parts of the head (skin is lighter and warmer)
    warm = rgb[..., 0] - rgb[..., 2]
    masks["hair"] = masks["head"] * (1 - ramp(lum, 0.2, 0.34)) * (1 - ramp(warm, 0.16, 0.24) * ramp(lum, 0.25, 0.4))

    def recolour(base, mask, target):
        """Move a garment's colour to `target` while keeping its folds and shading."""
        if target is None:
            return base
        sel = mask > 0.5
        mean_l = max(float(lum[sel].mean()) if sel.any() else 0.3, 0.05)
        t = np.array(target) * 0.66        # Robin's own cloth is dark; keep villagers in the same light
        t_l = float(t @ np.array([0.299, 0.587, 0.114]))
        shaded = t[None, None, :] * (lum / mean_l)[..., None] * (t_l / max(t_l, 1e-3))
        shaded = np.clip(shaded, 0, 1)
        return base * (1 - mask[..., None]) + shaded * mask[..., None]

    garments = ["cloak", "cowl", "torso", "legs", "sleeves", "hair"]
    for name, look in VILLAGERS.items():
        out = rgb.copy()
        for gm in garments:
            out = recolour(out, masks[gm], look.get(gm))
        Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(OUT / f"{name}.jpg", quality=88)
    palette = {"cloak": (0.95, 0.75, 0.1), "cowl": (0.1, 0.9, 0.2), "torso": (0.8, 0.2, 0.1), "legs": (0.7, 0.3, 0.9),
               "sleeves": (0.4, 0.6, 1.0), "hair": (0.1, 0.1, 0.1), "head": (1.0, 0.8, 0.7)}
    preview = rgb * 0.3
    for k, col in palette.items():
        m = masks[k] if k != "head" else masks["head"] * (1 - masks["hair"])
        preview = preview * (1 - m[..., None] * 0.85) + np.array(col) * m[..., None] * 0.85
    Image.fromarray((np.clip(np.concatenate([rgb, preview], axis=1), 0, 1) * 255).astype(np.uint8)).resize((W, H // 2)).save(OUT / "garments.png")
    print({k: round(float(m.mean()) * 100, 1) for k, m in masks.items()})
    (OUT / "villagers.json").write_text(json.dumps(VILLAGERS, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
