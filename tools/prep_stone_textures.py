"""Step 1 of the stone pipeline: turn each traced prop of the reference painting into textures.

For every stone this writes, into assets_src/tex/:
    <name>_color.jpg    the painting's own surface, sharpened, with fine rock grain added and the
                        colours pushed outward past the outline so no background bleeds in
    <name>_normal.jpg   a normal map: rock relief plus the runes cut into the surface
    <name>_emit.jpg     where the runes (or the door's light) glow
    <name>.json         the outline and how model coordinates map onto the texture, for Blender

Step 2 is tools/blender_stones.py, which builds the meshes.
Usage: python prep_stone_textures.py
"""
import json
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from isolate_assets import PROPS, SOURCE

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "assets_src" / "tex"

# size: texture long side. depth: front-to-back thickness as a fraction of the height.
STONES = {
    "trilithon": {"size": 1024, "depth": 0.20, "tris": 7000},
    "door_stone": {"size": 768, "depth": 0.22, "tris": 4000, "glow": True},
    "altar": {"size": 768, "depth": 1.25, "tris": 4000},
    "monolith": {"size": 768, "depth": 0.24, "tris": 3500, "runes": False},
    "rune_stone": {"size": 768, "depth": 0.34, "tris": 3500},
    "small_stone": {"size": 512, "depth": 0.36, "tris": 2500},
    "rock_spire": {"size": 768, "depth": 0.55, "tris": 3000, "runes": False, "rock": True},
    "rock_crag": {"size": 768, "depth": 0.55, "tris": 3000, "runes": False, "rock": True},
}
rng = np.random.default_rng(5)


def blur(array, radius):
    image = Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8))
    return np.asarray(image.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float64) / 255


def fbm(shape, cells, octaves=4):
    """Smooth fractal noise, 0..1, built from upscaled random grids."""
    h, w = shape
    total, amp, norm = np.zeros(shape), 1.0, 0.0
    for o in range(octaves):
        n = max(2, int(cells * 2 ** o))
        grid = Image.fromarray((rng.random((max(2, n * h // max(h, w)), max(2, n * w // max(h, w)))) * 255).astype(np.uint8))
        total += amp * np.asarray(grid.resize((w, h), Image.BICUBIC), dtype=np.float64) / 255
        norm += amp
        amp *= 0.5
    return total / norm


def build(name, polygons, spec, source):
    xs = [x for poly in polygons for x, _ in poly]
    ys = [y for poly in polygons for _, y in poly]
    bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
    bw, bh = bx1 - bx0, by1 - by0
    margin = 0.08 * max(bw, bh)
    cx0, cy0 = max(0, bx0 - margin), max(0, by0 - margin)
    cx1, cy1 = min(source.width, bx1 + margin), min(source.height, by1 + margin)
    scale = spec["size"] / max(cx1 - cx0, cy1 - cy0)
    w, h = round((cx1 - cx0) * scale), round((cy1 - cy0) * scale)
    crop = source.resize((w, h), Image.LANCZOS, box=(cx0, cy0, cx1, cy1))
    crop = crop.filter(ImageFilter.UnsharpMask(radius=4, percent=110, threshold=2))
    rgb = np.asarray(crop, dtype=np.float64) / 255

    mask_img = Image.new("L", (w * 2, h * 2), 0)
    draw = ImageDraw.Draw(mask_img)
    for poly in polygons:
        draw.polygon([((x - cx0) * scale * 2, (y - cy0) * scale * 2) for x, y in poly], fill=255)
    mask = np.asarray(mask_img.resize((w, h), Image.LANCZOS), dtype=np.float64) / 255
    inside = np.asarray(mask_img.filter(ImageFilter.MinFilter(9)).resize((w, h), Image.LANCZOS), dtype=np.float64) / 255

    # Push the stone's colours outward so texture filtering never reaches the painting's background.
    padded, have = rgb * inside[..., None], inside.copy()
    for radius in (2, 4, 8, 16, 32, 64, 128):
        spread = np.stack([blur(padded[..., c], radius) for c in range(3)], axis=-1)
        weight = blur(have, radius)
        fill = spread / np.maximum(weight, 1e-4)[..., None]
        take = (have < 0.02) & (weight > 0.01)
        padded[take] = fill[take]
        have[take] = 1.0
    colour = rgb * inside[..., None] + padded * (1 - inside[..., None])

    lum = colour @ np.array([0.299, 0.587, 0.114])
    local = lum - blur(lum, spec["size"] * 0.012)
    core = np.asarray(mask_img.filter(ImageFilter.MinFilter(2 * int(spec["size"] * 0.035) + 1)).resize((w, h), Image.LANCZOS), dtype=np.float64) / 255
    runes = np.clip((local - 0.055) / 0.11, 0, 1) * core * (1.0 if spec.get("runes", True) else 0.0)   # rim light is not a rune
    runes = runes * np.clip((lum - 0.30) / 0.2, 0, 1)                 # only strokes that are actually light
    glow = np.zeros_like(lum)
    if spec.get("glow"):
        glow = blur(np.clip((lum - 0.6) / 0.2, 0, 1) * (colour[..., 2] >= colour[..., 0] - 0.02) * inside, 3)

    grain = fbm((h, w), 90, 3)
    coarse = fbm((h, w), 9, 4)
    colour = np.clip(colour * (0.86 + 0.28 * grain)[..., None] * (0.92 + 0.16 * coarse)[..., None], 0, 1)

    # Relief: broad rock forms, fine grain, the painting's own light and dark, runes cut inward.
    relief = 0.9 * coarse + 0.35 * grain + 0.9 * (lum - blur(lum, spec["size"] * 0.03)) - 0.55 * blur(runes, 1.2)
    gy, gx = np.gradient(relief)
    strength = spec["size"] / 160
    normal = np.stack([-gx * strength, gy * strength, np.ones_like(gx)], axis=-1)     # +Y up, as glTF expects
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)

    emit = np.clip(colour * 1.5, 0, 1) * blur(runes, 0.8)[..., None] + np.array([0.65, 0.9, 1.0]) * glow[..., None]

    def save(array, suffix, quality, size=None):
        image = Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8))
        if size:
            image = image.resize(size, Image.LANCZOS)
        image.save(OUT / f"{name}_{suffix}.jpg", quality=quality)

    save(colour, "color", 86)
    save(normal * 0.5 + 0.5, "normal", 88, (max(2, w * 3 // 4), max(2, h * 3 // 4)))
    save(emit, "emit", 82, (max(2, w // 2), max(2, h // 2)))

    centre = (bx0 + bx1) / 2
    meta = {
        "name": name, "depth": spec["depth"], "tris": spec["tris"], "aspect": bw / bh, "rock": bool(spec.get("rock")),
        # model space: x across (centred), z up from the base, both in units of the prop's height
        "polygons": [[[(x - centre) / bh, (by1 - y) / bh] for x, y in poly] for poly in polygons],
        # texture lookup for a model point: u = u0 + x * ku, v (from the top) = v0 - z * kv
        "u0": (centre - cx0) / (cx1 - cx0), "ku": bh / (cx1 - cx0),
        "v0": (by1 - cy0) / (cy1 - cy0), "kv": bh / (cy1 - cy0),
        "rune_coverage": float(runes.mean()),
    }
    (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{name}: {w}x{h}px, runes cover {runes.mean() * 100:.1f}%")
    return colour, emit


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = Image.open(SOURCE).convert("RGB")
    tiles = []
    for name, spec in STONES.items():
        colour, emit = build(name, PROPS[name], spec, source)
        for array in (colour, emit):
            thumb = Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8))
            thumb.thumbnail((240, 240))
            tiles.append(thumb)
    sheet = Image.new("RGB", (8 * 250, 2 * 250), (30, 30, 30))
    for i, thumb in enumerate(tiles):
        sheet.paste(thumb, ((i // 2) * 250 + 5, (i % 2) * 250 + 5))
    sheet.save(OUT / "_sheet.png")


if __name__ == "__main__":
    main()
