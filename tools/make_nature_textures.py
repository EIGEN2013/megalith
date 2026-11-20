"""Paint the textures the forest needs: bark (colour + normal), a sheet of leaf clusters with
transparency, and a grass detail tile for the ground. Everything tiles.

Writes assets_src/tex/bark_color.jpg, bark_normal.jpg, leaves.png, ground.jpg
Usage: python make_nature_textures.py
"""
import math
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = pathlib.Path(__file__).resolve().parent.parent / "assets_src" / "tex"
rng = np.random.default_rng(23)


def noise(shape, rows, cols, octaves=4):
    """Fractal noise 0..1; rows/cols set how many blobs fit down and across at the first octave."""
    h, w = shape
    total, amp, norm = np.zeros(shape), 1.0, 0.0
    for o in range(octaves):
        grid = (rng.random((max(2, rows * 2 ** o), max(2, cols * 2 ** o))) * 255).astype(np.uint8)
        total += amp * np.asarray(Image.fromarray(grid).resize((w, h), Image.BICUBIC), dtype=np.float64) / 255
        norm += amp
        amp *= 0.5
    return total / norm


def mirror_tile(array):
    """Make any image tile by mirroring it into a 2x2 block."""
    top = np.concatenate([array, array[:, ::-1]], axis=1)
    return np.concatenate([top, top[::-1]], axis=0)


def normal_from_height(height, strength):
    gy, gx = np.gradient(height)
    n = np.stack([-gx * strength, gy * strength, np.ones_like(gx)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


def save(array, name, **kwargs):
    Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8)).save(OUT / name, **kwargs)


def bark():
    shape = (256, 256)
    ridges = noise(shape, 3, 34, 4)                      # long vertical fibres
    twist = noise(shape, 6, 6, 3)
    height = np.clip(ridges * 0.8 + twist * 0.35, 0, 1)
    cracks = np.clip((0.42 - ridges) / 0.12, 0, 1)       # dark furrows between the ridges
    base = np.array([0.30, 0.215, 0.165])
    light = np.array([0.50, 0.385, 0.29])
    colour = base + (light - base) * height[..., None]
    colour *= (1 - 0.6 * cracks)[..., None]
    sheen = np.clip((noise(shape, 3, 3, 3) - 0.55) / 0.2, 0, 1)[..., None]   # the painting's blue-violet shimmer
    hue = noise(shape, 5, 5, 2)[..., None]
    shimmer = np.array([0.20, 0.42, 0.55]) * (1 - hue) + np.array([0.42, 0.28, 0.60]) * hue
    colour = colour * (1 - 0.45 * sheen) + shimmer * 0.45 * sheen
    colour *= (0.9 + 0.2 * noise(shape, 60, 60, 2))[..., None]
    save(mirror_tile(colour), "bark_color.jpg", quality=86)
    save(mirror_tile(normal_from_height(height - 0.5 * cracks, 5.0)), "bark_normal.jpg", quality=88)


def ground():
    size = 256
    image = Image.new("RGB", (size * 2, size * 2), (196, 200, 188))
    draw = ImageDraw.Draw(image)
    for _ in range(5200):                                 # short blades of grass, drawn at 2x
        x, y = rng.uniform(0, size * 2, 2)
        length, lean = rng.uniform(5, 15), rng.uniform(-0.6, 0.6)
        shade = int(rng.uniform(150, 250))
        tint = (int(shade * rng.uniform(0.86, 1.0)), shade, int(shade * rng.uniform(0.78, 0.95)))
        draw.line([(x, y), (x + lean * length, y - length)], fill=tint, width=2)
    blades = np.asarray(image.resize((size, size), Image.LANCZOS), dtype=np.float64) / 255
    patchy = 0.82 + 0.3 * noise((size, size), 5, 5, 4)
    save(mirror_tile(blades * patchy[..., None]), "ground.jpg", quality=84)


def leaves():
    """Four leaf clusters in a 2x2 sheet: long, overlapping, iridescent leaves on clear background."""
    cell, ss = 256, 3
    sheet = Image.new("RGBA", (cell * 2 * ss, cell * 2 * ss), (0, 0, 0, 0))
    palettes = [
        [(120, 232, 214), (92, 190, 222), (176, 150, 240), (236, 150, 214)],     # teal into violet
        [(206, 160, 246), (150, 132, 236), (112, 206, 226), (240, 170, 206)],    # violet
        [(242, 150, 204), (224, 128, 226), (150, 150, 240), (250, 208, 150)],    # magenta, gold tips
        [(150, 226, 140), (110, 206, 170), (196, 232, 130), (120, 190, 220)],    # green
    ]
    for k, palette in enumerate(palettes):
        ox, oy = (k % 2) * cell * ss + cell * ss / 2, (k // 2) * cell * ss + cell * ss / 2
        draw = ImageDraw.Draw(sheet)
        count = 150
        for i in range(count):
            ring = 1 - i / count                          # outer leaves first, inner ones on top
            angle = rng.uniform(0, 2 * math.pi)
            dist = cell * ss * 0.40 * ring ** 0.7 * rng.uniform(0.75, 1.0)
            length = cell * ss * rng.uniform(0.13, 0.2)
            width = length * rng.uniform(0.2, 0.3)
            cx, cy = ox + math.cos(angle) * dist, oy + math.sin(angle) * dist
            lean = angle + rng.uniform(-0.5, 0.5)
            ax, ay = math.cos(lean), math.sin(lean)
            pts = []
            for t in np.linspace(0, 1, 9):               # one side out to the tip ...
                w = width * math.sin(t * math.pi) ** 0.8 * (1 - 0.35 * t)
                pts.append((cx + ax * (t - 0.5) * length - ay * w, cy + ay * (t - 0.5) * length + ax * w))
            for t in np.linspace(1, 0, 9):               # ... and back down the other
                w = width * math.sin(t * math.pi) ** 0.8 * (1 - 0.35 * t)
                pts.append((cx + ax * (t - 0.5) * length + ay * w, cy + ay * (t - 0.5) * length - ax * w))
            a, b = palette[int(rng.integers(0, 4))], palette[int(rng.integers(0, 4))]
            mix = rng.uniform(0, 1)
            shade = rng.uniform(0.72, 1.0) * (0.8 + 0.2 * (1 - ring))
            colour = tuple(int((a[c] * (1 - mix) + b[c] * mix) * shade) for c in range(3))
            draw.polygon(pts, fill=colour + (255,))
            rib = tuple(int(c * 0.72) for c in colour)
            draw.line([(cx - ax * length * 0.45, cy - ay * length * 0.45), (cx + ax * length * 0.45, cy + ay * length * 0.45)],
                      fill=rib + (255,), width=max(1, ss))
    sheet = sheet.resize((cell * 2, cell * 2), Image.LANCZOS)
    # Bleed leaf colour into the clear pixels so filtered edges do not go dark.
    rgba = np.asarray(sheet, dtype=np.float64) / 255
    alpha = rgba[..., 3]
    colour, have = rgba[..., :3] * alpha[..., None], alpha.copy()
    for radius in (2, 4, 8, 16):
        spread = np.stack([np.asarray(Image.fromarray((colour[..., c] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float64) / 255 for c in range(3)], axis=-1)
        weight = np.asarray(Image.fromarray((have * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float64) / 255
        take = (have < 0.05) & (weight > 0.01)
        colour[take] = (spread / np.maximum(weight, 1e-4)[..., None])[take]
        have[take] = 1.0
    solid = np.where(alpha[..., None] > 0.05, rgba[..., :3], colour)
    out = np.concatenate([solid, alpha[..., None]], axis=-1)
    Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8), "RGBA").save(OUT / "leaves.png", optimize=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    bark()
    ground()
    leaves()
    for name in ("bark_color.jpg", "bark_normal.jpg", "ground.jpg", "leaves.png"):
        print(name, (OUT / name).stat().st_size // 1024, "KB")
