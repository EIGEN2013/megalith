"""Give every villager their own face, made from Robin's painted face rather than painted over it.

Robin's face texture keeps its painting; each villager's version re-shapes and re-colours it:
eyes widened or narrowed and given a new iris colour, brows thinned / thickened / re-arched and
recoloured, lips made fuller and redder, stubble removed (women), darkened or greyed, the skin
re-toned, and age lines, freckles, a scar, a mole or a ruddy nose added in the same soft style.

Writes assets_src/npc/face_<villager>.jpg and a line-up assets_src/npc/faces.png.
Usage: python tools/npc_faces.py
"""
import io
import math
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from glb import Glb  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "assets_src" / "npc"
S = 512
EYES = [(185.0, 121.0), (320.0, 121.0)]
BROWS = [(188.0, 86.0), (325.0, 86.0)]
MOUTH = (252.0, 286.0)
NOSE = (252.0, 238.0)
yy, xx = np.mgrid[0:S, 0:S].astype(np.float64)

FACES = {
    "eldric": dict(tone=(1.04, 0.97, 0.93), iris=(0.42, 0.55, 0.66), eye=(1.0, 0.78), brow=dict(col=(0.88, 0.87, 0.84), thick=2, drop=5, arch=0),
                   stubble=("grey", 0.6), lips=(1.0, 1.0, 0.0), age=1.0, spots=True),
    "oswin": dict(tone=(0.86, 0.78, 0.72), iris=(0.24, 0.15, 0.08), eye=(0.95, 0.9), brow=dict(col=(0.08, 0.06, 0.05), thick=2, drop=-1, arch=0),
                  stubble=("dark", 1.0), lips=(0.95, 1.0, 0.0), age=0.35, scar=True),
    "marta": dict(tone=(1.04, 0.98, 0.96), iris=(0.28, 0.50, 0.26), eye=(1.08, 1.08), brow=dict(col=(0.28, 0.17, 0.10), thick=-1, drop=-3, arch=5),
                  stubble=("none", 1.0), lips=(1.25, 1.12, 0.65), age=0.3, lash=True, mole=True, blush=0.25),
    "lina": dict(tone=(1.10, 1.03, 1.0), iris=(0.30, 0.50, 0.80), eye=(1.14, 1.16), brow=dict(col=(0.50, 0.24, 0.10), thick=-1, drop=-4, arch=6),
                 stubble=("none", 1.0), lips=(1.3, 1.12, 0.75), age=0.0, lash=True, freckles=True, blush=0.35),
    "bren": dict(tone=(1.0, 0.86, 0.80), iris=(0.40, 0.36, 0.18), eye=(0.98, 0.84), brow=dict(col=(0.42, 0.26, 0.13), thick=3, drop=2, arch=0),
                 stubble=("red", 1.0), lips=(1.0, 1.0, 0.0), age=0.65, ruddy=True),
}


def load():
    g = Glb(str(ROOT / "live_src" / "hero.glb"))
    for im in g.json["images"]:
        if im.get("name") == "hero_face":
            return np.asarray(Image.open(io.BytesIO(g.view_bytes(im["bufferView"]))).convert("RGB"), dtype=np.float64) / 255
    raise SystemExit("hero_face image not found")


def to_img(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def blur(a, r):
    """Gaussian blur (edge-padded, via FFT)."""
    if a.ndim == 3:
        return np.stack([blur(a[..., c], r) for c in range(a.shape[-1])], axis=-1)
    pad = int(3 * r) + 2
    p = np.pad(a, pad, mode="edge")
    fy = np.fft.fftfreq(p.shape[0])[:, None]
    fx = np.fft.rfftfreq(p.shape[1])[None, :]
    g = np.exp(-2 * (math.pi * r) ** 2 * (fx ** 2 + fy ** 2))
    out = np.fft.irfft2(np.fft.rfft2(p) * g, s=p.shape)
    return out[pad:-pad, pad:-pad]


def ellipse(c, rx, ry, soft=4.0):
    d = np.sqrt(((xx - c[0]) / rx) ** 2 + ((yy - c[1]) / ry) ** 2)
    return np.clip((1 - d) * min(rx, ry) / soft, 0, 1)


def lum(a):
    return a @ np.array([0.3, 0.55, 0.15])


def sample(a, sx, sy):
    """Bilinear sample image a at float coords."""
    sx = np.clip(sx, 0, S - 1.001)
    sy = np.clip(sy, 0, S - 1.001)
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    tx, ty = (sx - x0)[..., None], (sy - y0)[..., None]
    if a.ndim == 2:
        a = a[..., None]
        squeeze = True
    else:
        squeeze = False
    r = (a[y0, x0] * (1 - tx) + a[y0, x0 + 1] * tx) * (1 - ty) + (a[y0 + 1, x0] * (1 - tx) + a[y0 + 1, x0 + 1] * tx) * ty
    return r[..., 0] if squeeze else r


def warp_scale(a, c, sx, sy, R, dy=0.0):
    """Scale the picture around c by (sx, sy) with a smooth falloff to nothing at radius R; shift by dy."""
    d = np.sqrt(((xx - c[0]) / R[0]) ** 2 + ((yy - c[1]) / R[1]) ** 2)
    k = np.clip(1 - d, 0, 1)
    k = k * k * (3 - 2 * k)
    fx = 1 + (sx - 1) * k
    fy = 1 + (sy - 1) * k
    return sample(a, c[0] + (xx - c[0]) / fx, c[1] + (yy - c[1]) / fy - dy * k)


def paint(name, spec, base):
    rng = np.random.default_rng(abs(hash(name)) % 1000)
    a = base.copy()
    L = lum(a)
    # ---- what is skin, and the face as clean skin (features removed) for repainting
    feat = np.zeros((S, S))
    for c in EYES:
        feat = np.maximum(feat, ellipse(c, 46, 24, 3))
    for c in BROWS:
        feat = np.maximum(feat, ellipse(c, 62, 20, 3))
    feat = np.maximum(feat, ellipse(MOUTH, 66, 22, 3))
    for dx in (-22, 22):
        feat = np.maximum(feat, ellipse((NOSE[0] + dx, NOSE[1]), 12, 9, 2))
    stub_region = ellipse((258, 380), 130, 90, 30) * np.clip((yy - 290) / 20, 0, 1)
    w = (1 - feat) * (1 - stub_region)
    clean = None
    for r in (60, 30, 14):                    # coarse to fine: each scale fills only where it has enough support
        wb = blur(w, r)
        est = blur(a * w[..., None], r) / np.maximum(wb, 1e-4)[..., None]
        k = np.clip((wb - 0.08) / 0.2, 0, 1)[..., None]
        clean = est if clean is None else clean * (1 - k) + est * k
    detail = a / np.maximum(blur(a, 3.0), 1e-3)                     # the painted fine grain, kept on repainted skin
    clean = np.clip(clean * (0.85 + 0.15 * np.clip(detail, 0.7, 1.3)), 0, 1)
    cleanL = lum(clean)

    # ---- stubble: remove, deepen, grey or redden
    stub = np.clip((cleanL - L) / 0.22, 0, 1) * stub_region
    kind, amt = spec["stubble"]
    if kind == "none":
        a = a * (1 - stub_region[..., None]) + clean * stub_region[..., None]
    elif kind == "dark":
        a = a * (1 - 0.35 * stub[..., None] * amt)
    elif kind == "grey":
        g = lum(a)[..., None] * np.array([1.0, 1.0, 1.02])
        a = a * (1 - stub[..., None] * amt) + (g * 0.5 + 0.5 * clean * np.array([1.15, 1.12, 1.1])) * stub[..., None] * amt
    elif kind == "red":
        a = a * (1 - stub[..., None] * 0.6) + clean * np.array([0.62, 0.38, 0.22]) * stub[..., None] * 0.6

    # ---- eyes: reshape and recolour the iris
    for c in EYES:
        a = warp_scale(a, c, spec["eye"][0], spec["eye"][1], (60, 40))
    for c in EYES:
        r = np.hypot(xx - c[0], yy - c[1])
        sat = a.max(axis=-1) - a.min(axis=-1)
        iris = (r < 17 * spec["eye"][0]) & (lum(a) < 0.62) & (sat > 0.08)
        iris = blur(iris.astype(float), 0.8)
        pupil = np.clip((0.12 - lum(a)) / 0.06, 0, 1)
        newc = np.array(spec["iris"]) * (0.55 + 1.4 * lum(a)[..., None])
        newc = newc * (1 - 0.35 * np.clip((r / 16)[..., None], 0, 1))                  # darker limbal ring
        m = (iris * (1 - pupil))[..., None]
        a = a * (1 - m) + newc * m
        if spec.get("lash"):                                                              # darker, longer upper lash line
            lash = ellipse((c[0], c[1] - 2), 37 * spec["eye"][0], 15 * spec["eye"][1], 2) - ellipse((c[0], c[1] + 1), 34 * spec["eye"][0], 13 * spec["eye"][1], 2)
            lash = np.clip(lash, 0, 1) * (yy < c[1] - 2)
            a = a * (1 - 0.7 * lash[..., None]) + 0.06 * lash[..., None]

    # ---- brows: lift them off, reshape, and lay them back in a new colour and weight
    b = spec["brow"]
    browm = np.zeros((S, S))
    for c in BROWS:
        reg = ellipse(c, 64, 18, 4)
        browm = np.maximum(browm, np.clip((cleanL - lum(a)) / 0.25, 0, 1) * reg)
    a = a * (1 - np.clip(browm * 1.4, 0, 1)[..., None]) + clean * np.clip(browm * 1.4, 0, 1)[..., None]
    mask = browm
    for c in BROWS:                                                                       # arch: lift the middle
        if b["arch"]:
            mask = warp_scale(mask, c, 1.0, 1.0, (50, 30), dy=-b["arch"] * 0)
    shift_y = np.zeros((S, S))
    for c, sgn in zip(BROWS, (-1, 1)):
        u = np.clip((xx - c[0]) / 60 * sgn, -1, 1)                                         # -1 inner .. +1 outer
        inside = ellipse(c, 70, 26, 8)
        shift_y += inside * (b["drop"] * np.clip(u, 0, 1) + b["arch"] * (1 - u * u) * -1 + (-b["arch"] * 0.3))
    mask = sample(mask, xx, yy - shift_y)
    mimg = to_img(mask)
    if b["thick"] > 0:
        for _ in range(b["thick"]):
            mimg = mimg.filter(ImageFilter.MaxFilter(3))
    elif b["thick"] < 0:
        for _ in range(-b["thick"]):
            mimg = mimg.filter(ImageFilter.MinFilter(3))
    mask = np.asarray(mimg.filter(ImageFilter.GaussianBlur(0.9)), dtype=np.float64) / 255
    hair = np.asarray(Image.fromarray((np.random.default_rng(5).random((S, S)) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6)), dtype=np.float64) / 255
    bc = np.array(b["col"]) * (0.9 + 0.2 * hair[..., None])
    m = np.clip(mask * 1.1, 0, 0.95)[..., None]
    a = a * (1 - m) + bc * m

    # ---- lips
    full, red, pink = spec["lips"]
    if full != 1.0:
        a = warp_scale(a, MOUTH, 1.04, full, (75, 34))
    if red != 1.0 or pink:
        lip = ellipse(MOUTH, 50, 14 * full, 5) * (1 - np.clip(np.abs(yy - MOUTH[1]) < 1.2, 0, 1) * 0)
        tintc = np.array([0.78, 0.36, 0.38]) * (0.6 + 0.6 * lum(a)[..., None])
        a = a * (1 - pink * 0.6 * lip[..., None]) + tintc * pink * 0.6 * lip[..., None]

    # ---- skin tone (leave the whites of the eyes alone)
    whites = np.zeros((S, S))
    for c in EYES:
        whites = np.maximum(whites, ellipse(c, 30 * spec["eye"][0], 12 * spec["eye"][1], 2) * np.clip((lum(a) - 0.55) / 0.15, 0, 1))
    tone = np.array(spec["tone"])
    a = a * (1 - whites[..., None]) * tone + a * whites[..., None]

    # ---- marks of a life: lines, freckles, spots, scar, mole, blush, ruddy nose
    layer = Image.new("L", (S * 2, S * 2), 0)
    d = ImageDraw.Draw(layer)
    age = spec.get("age", 0)
    if age:
        for k in range(3 if age > 0.6 else 1):                         # forehead
            y = 2 * (52 + k * 9)
            d.arc([2 * 175, y - 14, 2 * 335, y + 26], 200, 340, fill=int(255 * age), width=2)
        for c, sgn in zip(EYES, (-1, 1)):                              # crow's feet
            for k in range(3):
                x0, y0 = c[0] + sgn * 40, c[1] - 4 + k * 6
                d.line([2 * x0, 2 * y0, 2 * (x0 + sgn * 14), 2 * (y0 - 6 + k * 6)], fill=int(200 * age), width=2)
            d.arc([2 * (c[0] - 28), 2 * (c[1] + 6), 2 * (c[0] + 28), 2 * (c[1] + 26)], 20, 160, fill=int(160 * age), width=2)   # under-eye
        for sgn in (-1, 1):                                            # smile lines
            d.arc([2 * (252 + sgn * 20 - 40), 2 * 215, 2 * (252 + sgn * 20 + 40), 2 * 320], 90 - sgn * 60 - 20, 90 - sgn * 60 + 20, fill=int(230 * age), width=3)
    lines = np.asarray(layer.resize((S, S), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float64) / 255
    a = a * (1 - 0.28 * lines[..., None])
    if spec.get("spots"):
        for _ in range(10):
            c = (rng.uniform(150, 360), rng.uniform(40, 200))
            m = ellipse(c, rng.uniform(3, 6), rng.uniform(3, 5), 3)
            a = a * (1 - 0.18 * m[..., None])
    if spec.get("freckles"):
        for _ in range(90):
            c = (252 + rng.normal(0, 58), 178 + rng.normal(0, 20))
            m = ellipse(c, rng.uniform(1.4, 2.6), rng.uniform(1.4, 2.4), 1.2)
            a = a * (1 - m[..., None] * 0.3) + m[..., None] * 0.3 * np.array([0.58, 0.32, 0.18])
    if spec.get("blush"):
        for c in EYES:
            m = ellipse((c[0] + (8 if c[0] < 252 else -8), c[1] + 62), 36, 22, 30)
            a = a * (1 - spec["blush"] * m[..., None]) + spec["blush"] * m[..., None] * a * np.array([1.18, 0.86, 0.86])
    if spec.get("ruddy"):
        m = ellipse((252, 222), 30, 26, 22)
        a = a * (1 - 0.35 * m[..., None]) + 0.35 * m[..., None] * a * np.array([1.3, 0.8, 0.78])
        for c in EYES:
            m = ellipse((c[0], c[1] + 58), 30, 18, 24)
            a = a * (1 - 0.2 * m[..., None]) + 0.2 * m[..., None] * a * np.array([1.25, 0.85, 0.82])
    if spec.get("mole"):
        m = ellipse((MOUTH[0] + 58, MOUTH[1] - 26), 3.2, 3.0, 1.5)
        a = a * (1 - 0.8 * m[..., None]) + 0.8 * m[..., None] * np.array([0.25, 0.14, 0.1])
    if spec.get("scar"):
        sl = Image.new("L", (S * 2, S * 2), 0)
        ImageDraw.Draw(sl).line([2 * 150, 2 * 140, 2 * 172, 2 * 182, 2 * 190, 2 * 222], fill=255, width=7)
        m = np.asarray(sl.resize((S, S), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float64) / 255
        a = a * (1 - 0.6 * m[..., None]) + 0.6 * m[..., None] * a * np.array([1.25, 0.95, 0.95])
    out = to_img(a)
    out.save(OUT / f"face_{name}.jpg", quality=92)
    return out


def main():
    base = load()
    tiles = [to_img(base)]
    for name, spec in FACES.items():
        tiles.append(paint(name, spec, base))
    sheet = Image.new("RGB", (300 * len(tiles), 300))
    for i, t in enumerate(tiles):
        sheet.paste(t.crop((110, 20, 410, 440)).resize((214, 300)), (i * 300 + 43, 0))
    sheet.save(OUT / "faces.png")
    print("faces:", list(FACES))


if __name__ == "__main__":
    main()
