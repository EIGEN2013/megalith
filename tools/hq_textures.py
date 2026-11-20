"""Tileable surface-detail textures for every material in the game (numpy, seamless by construction).

Each material gets
  <name>_d.jpg   detail colour: multiplies the model's own vertex colours, so its mean is ~0.85 and it
                 carries only the pattern (mortar, grain, fibres, pebbles...), plus a slight hue
  <name>_n.jpg   tangent-space normal map (OpenGL, +Y up) from the same height field
Writes web/megalith/tex/ and a contact sheet assets_src/hq/tex_sheet.jpg.
Usage: python tools/hq_textures.py
"""
import json
import math
import pathlib

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "megalith" / "tex"
OUT.mkdir(parents=True, exist_ok=True)
R = 1024
yy, xx = np.meshgrid((np.arange(R) + 0.5) / R, (np.arange(R) + 0.5) / R, indexing="ij")


# ----------------------------------------------------------------------------- periodic noise
def vnoise(P, seed, x=None, y=None):
    x = xx if x is None else x
    y = yy if y is None else y
    g = np.random.default_rng(seed).random((P, P))
    fx, fy = x * P, y * P
    ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = fx - ix, fy - iy
    tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
    a = g[iy % P, ix % P]
    b = g[iy % P, (ix + 1) % P]
    c = g[(iy + 1) % P, ix % P]
    d = g[(iy + 1) % P, (ix + 1) % P]
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty


def fbm(P, oct, seed, x=None, y=None, gain=0.5):
    tot, amp, norm = 0, 1.0, 0
    for o in range(oct):
        tot = tot + amp * vnoise(P * 2 ** o, seed + o * 13, x, y)
        norm += amp
        amp *= gain
    return tot / norm


def worley(Px, Py, seed, x=None, y=None, jitter=0.9):
    """Periodic Worley on a Px x Py cell grid: F1, F2, cell id (0..1), nearest point offset."""
    x = xx if x is None else x
    y = yy if y is None else y
    rng = np.random.default_rng(seed)
    pts = rng.random((Py, Px, 2)) * jitter + (1 - jitter) / 2
    ids = rng.random((Py, Px))
    fx, fy = x * Px, y * Py
    ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
    f1 = np.full(x.shape, 9.0)
    f2 = np.full(x.shape, 9.0)
    cid = np.zeros(x.shape)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx, cy = ix + dx, iy + dy
            p = pts[cy % Py, cx % Px]
            d = np.hypot(cx + p[..., 0] - fx, cy + p[..., 1] - fy)
            closer = d < f1
            f2 = np.where(closer, f1, np.minimum(f2, d))
            cid = np.where(closer, ids[cy % Py, cx % Px], cid)
            f1 = np.where(closer, d, f1)
    return f1, f2, cid


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def save(name, h, col, strength, tile_m):
    """h: height (0..1, ~metres scaled by strength); col: RGB detail colour."""
    h = (h - h.min()) / (h.max() - h.min() + 1e-9)
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
    gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
    k = BUMP.get(name, 5.0)
    n = np.stack([-gx * k, gy * k, np.ones_like(h)], axis=-1)      # image rows go down; +Y up in the normal map
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    col = np.clip(col, 0, 1)
    Image.fromarray((col ** (1 / 2.2) * 255).astype(np.uint8)).save(OUT / f"{name}_d.jpg", quality=88)
    Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8)).save(OUT / f"{name}_n.jpg", quality=92)
    META[name] = {"tile": tile_m, "mean": float(col.mean())}
    print(f"{name}: mean {col.mean():.2f}")
    SHEET.append((name, col, n))


def tint(gray, hue=(1.0, 1.0, 1.0)):
    return gray[..., None] * np.array(hue)[None, None]


META, SHEET = {}, []
BUMP = {"ashlar": 16, "fieldstone": 12, "cobble": 12, "slate": 14, "thatch": 6, "planks": 14, "wood": 4, "plaster": 6,
        "rock": 8, "iron": 6, "cloth": 3, "grass": 5, "ground": 7, "smooth": 2}


# ----------------------------------------------------------------------------- materials
def ashlar():
    """Dressed stone blocks in running bond (castle walls). Tile 2.4 m: 6 rows."""
    rows = 6
    row = np.floor(yy * rows).astype(int)
    rng = np.random.default_rng(1)
    vy = yy * rows - row
    # per row, 3 or 4 blocks with random joints, offset each row
    h = np.zeros_like(xx)
    cid = np.zeros_like(xx)
    edge_d = np.ones_like(xx)
    for r in range(rows):
        n = rng.integers(3, 5)
        cuts = np.sort(rng.random(n - 1) * 0.8 + 0.1)
        cuts = np.concatenate([[0], cuts, [1]]) + rng.random() * 0.5
        sel = row == r
        u = xx[sel]
        k = np.searchsorted(cuts, u, side="right") - 1
        k2 = np.searchsorted(cuts, u + 1, side="right") - 1
        use_wrap = k < 0
        kk = np.where(use_wrap, k2, k)
        uu = np.where(use_wrap, u + 1, u)
        left = cuts[np.clip(kk, 0, len(cuts) - 1)]
        right = cuts[np.clip(kk + 1, 0, len(cuts) - 1)]
        du = np.minimum(uu - left, right - uu) * 2.4          # metres to the vertical joint
        dv = np.minimum(vy[sel], 1 - vy[sel]) * 2.4 / rows
        edge_d[sel] = np.minimum(du, dv)
        cid[sel] = ((kk * 7919 + r * 104729) % 997) / 997.0
    face = sstep(0.008, 0.045, edge_d)                         # bevelled block edge into a mortar joint
    chip = sstep(0.55, 0.75, fbm(16, 4, 5)) * sstep(0.06, 0.0, edge_d)
    grain = fbm(32, 5, 7)
    h = face * (0.8 + 0.2 * cid) - chip * 0.6 + 0.12 * grain + 0.05 * fbm(128, 3, 9)
    tone = 0.78 + 0.22 * cid + 0.12 * (fbm(8, 4, 11) - 0.5)
    g = np.where(face > 0.2, tone, 0.5) * (0.9 + 0.12 * grain)
    streak = sstep(0.55, 0.85, fbm(4, 3, 13, x=xx * 1, y=yy * 0.25 + xx * 0.05)) * 0.18
    g = g * (1 - streak)
    save("ashlar", h * 0.025, tint(g, (1.0, 0.98, 0.94)), 0.045, 2.4)


def fieldstone():
    """Dry-stone: irregular rounded stones with dark gaps (village wall, well). Tile 2 m."""
    f1, f2, cid = worley(7, 9, 21, jitter=0.95)
    e = f2 - f1
    dome = sstep(0.0, 0.32, e) ** 0.6
    h = dome * (0.7 + 0.3 * cid) + 0.15 * fbm(32, 5, 23) * dome
    g = np.where(e > 0.04, 0.72 + 0.28 * cid, 0.3) * (0.88 + 0.2 * fbm(64, 4, 25))
    lich = sstep(0.62, 0.7, fbm(16, 4, 27)) * (e > 0.05)
    col = tint(g, (1.0, 0.97, 0.92))
    col = col * (1 - lich[..., None] * 0.5) + lich[..., None] * 0.5 * np.array([0.85, 0.9, 0.7])
    save("fieldstone", h * 0.04, col, 0.04, 2.0)


def cobble():
    f1, f2, cid = worley(12, 12, 31, jitter=0.85)
    e = f2 - f1
    dome = sstep(0.0, 0.3, e) ** 0.5
    h = dome * (0.8 + 0.2 * cid) + 0.06 * fbm(64, 3, 33)
    g = np.where(e > 0.05, 0.72 + 0.26 * cid, 0.38) * (0.9 + 0.15 * fbm(32, 3, 35))
    moss = sstep(0.55, 0.7, fbm(8, 4, 37)) * (e < 0.08)
    col = tint(g, (1.0, 0.98, 0.95))
    col = col * (1 - moss[..., None] * 0.6) + moss[..., None] * 0.6 * np.array([0.55, 0.7, 0.35])
    save("cobble", h * 0.03, col, 0.035, 1.6)


def slate():
    """Overlapping slates in staggered rows (roofs). Rows run along u; v goes up the roof. Tile 1.6 m."""
    rows, per = 8, 6
    r = np.floor(yy * rows)
    vy = yy * rows - r
    off = (r % 2) * 0.5 / per
    u = (xx + off + np.random.default_rng(3).random(rows)[r.astype(int) % rows] * 0.02) * per
    k = np.floor(u)
    vu = u - k
    cid = ((k * 31 + r * 17) % 23) / 23.0
    slope = vy                                       # each slate rises toward its lower (exposed) edge
    edge = sstep(0.0, 0.06, np.minimum(vu, 1 - vu)) * sstep(0.0, 0.08, 1 - vy)
    chip = sstep(0.6, 0.8, fbm(32, 3, 41)) * sstep(0.12, 0.0, np.minimum(vu, 1 - vu))
    h = (0.35 + 0.65 * slope) * edge - chip * 0.3 + 0.08 * fbm(64, 4, 43)
    g = (0.75 + 0.25 * cid) * (0.85 + 0.25 * fbm(32, 4, 45)) * np.where(edge > 0.3, 1.0, 0.45)
    lich = sstep(0.68, 0.74, fbm(16, 4, 47))
    col = tint(g, (0.96, 0.98, 1.03))
    col = col * (1 - lich[..., None] * 0.5) + lich[..., None] * 0.5 * np.array([0.95, 0.9, 0.6])
    save("slate", h * 0.012, col, 0.03, 1.6)


def thatch():
    """Bundled straw: fine fibres running down the slope (v), layered courses. Tile 1.2 m."""
    fib = fbm(128, 2, 51, x=xx, y=yy * 0.04) * 0.6 + fbm(256, 2, 53, x=xx, y=yy * 0.05) * 0.4
    course = (yy * 5) % 1.0
    layer = sstep(0.0, 0.85, course)
    h = 0.55 * fib + 0.45 * layer + 0.1 * fbm(16, 3, 55)
    g = (0.55 + 0.55 * fib) * (0.7 + 0.3 * layer) * (0.85 + 0.25 * fbm(8, 3, 57))
    dark = sstep(0.55, 0.8, fbm(8, 4, 59)) * 0.35
    col = tint(g * (1 - dark), (1.02, 0.98, 0.88))
    save("thatch", h * 0.03, col, 0.05, 1.2)


def planks():
    """Boards running along u, with gaps, grain and knots. Tile 2 m (8 boards)."""
    n = 8
    b = np.floor(yy * n)
    vb = yy * n - b
    rng = np.random.default_rng(61)
    shift = rng.random(n)[b.astype(int)]
    joint_u = (xx + shift) % 1.0
    gap = sstep(0.0, 0.05, np.minimum(vb, 1 - vb)) * sstep(0.0, 0.006, np.minimum(joint_u, 1 - joint_u))
    warp = fbm(8, 3, 63) * 0.6
    rings = np.sin((vb * 6 + warp * 8 + b * 3.1) * math.pi * 2 + fbm(64, 2, 65) * 1.5) * 0.5 + 0.5
    grain = 0.6 * rings + 0.4 * fbm(256, 2, 67, x=xx * 0.08, y=yy)
    knots_f1, _, _ = worley(5, 9, 69)
    knot = sstep(0.12, 0.0, knots_f1) * (rng.random() > -1)
    h = gap * (0.85 + 0.15 * grain) - knot * 0.1
    tone = 0.8 + 0.2 * rng.random(n)[b.astype(int)]
    g = gap * tone * (0.78 + 0.25 * grain) * (1 - 0.45 * knot) + (1 - gap) * 0.25
    save("planks", h * 0.012, tint(g, (1.03, 0.98, 0.9)), 0.03, 2.0)


def woodgrain():
    """Plain wood grain for beams and furniture (grain along u). Tile 1.2 m."""
    warp = fbm(8, 3, 71)
    rings = np.sin((yy * 18 + warp * 4) * math.pi * 2) * 0.5 + 0.5
    fibre = fbm(512, 2, 73, x=xx * 0.03, y=yy)
    grain = 0.55 * rings + 0.45 * fibre
    h = 0.6 * fibre + 0.4 * rings
    g = 0.72 + 0.3 * grain
    save("wood", h * 0.004, tint(g, (1.02, 0.98, 0.92)), 0.03, 1.2)


def plaster():
    lumps = fbm(8, 5, 81)
    fine = fbm(128, 3, 83)
    f1, f2, _ = worley(6, 6, 85)
    crack = sstep(0.025, 0.0, f2 - f1) * sstep(0.6, 0.75, fbm(4, 3, 87))
    h = 0.6 * lumps + 0.25 * fine - 0.6 * crack
    g = (0.86 + 0.14 * lumps) * (0.94 + 0.08 * fine) * (1 - 0.35 * crack)
    stain = sstep(0.6, 0.85, fbm(4, 4, 89, x=xx, y=yy * 0.4)) * 0.15
    save("plaster", h * 0.01, tint(g * (1 - stain), (1.0, 0.98, 0.95)), 0.02, 2.0)


def rock():
    strata = np.sin((yy * 9 + fbm(4, 4, 91) * 2.5) * math.pi * 2) * 0.5 + 0.5
    ridges = 1 - np.abs(fbm(8, 5, 93) * 2 - 1)
    f1, f2, cid = worley(5, 5, 95)
    facets = cid * 0.4 + sstep(0.0, 0.2, f2 - f1) * 0.3
    h = 0.35 * strata + 0.35 * ridges + facets * 0.5
    g = (0.72 + 0.25 * h) * (0.9 + 0.15 * fbm(64, 3, 97))
    save("rock", h * 0.08, tint(g, (1.0, 0.98, 0.95)), 0.06, 4.0)


def iron():
    f1, _, cid = worley(10, 10, 101)
    dents = sstep(0.0, 0.5, f1) ** 2
    scratch = sstep(0.985, 1.0, vnoise(512, 103, x=xx, y=yy * 0.02 + xx * 0.3)) * 0.4
    h = 0.6 * dents + 0.1 * fbm(64, 3, 105) - scratch * 0.3
    g = (0.8 + 0.2 * cid) * (0.85 + 0.2 * fbm(16, 4, 107)) + scratch * 0.3
    rust = sstep(0.6, 0.78, fbm(8, 5, 109))
    col = tint(g, (1.0, 1.0, 1.0))
    col = col * (1 - rust[..., None] * 0.6) + rust[..., None] * 0.6 * np.array([0.9, 0.55, 0.32])
    save("iron", h * 0.004, col, 0.03, 0.6)


def cloth():
    n = 96
    wx = np.sin(xx * n * math.pi * 2) * 0.5 + 0.5
    wy = np.sin(yy * n * math.pi * 2) * 0.5 + 0.5
    checker = ((np.floor(xx * n) + np.floor(yy * n)) % 2)
    h = np.where(checker > 0, wx, wy) * 0.7 + 0.3 * fbm(64, 2, 111)
    g = 0.82 + 0.16 * h + 0.06 * (fbm(16, 3, 113) - 0.5)
    save("cloth", h * 0.0015, tint(g, (1, 1, 1)), 0.02, 0.4)


def grass():
    """Grassy ground detail: blades and clumps. Tile 2 m."""
    blades = fbm(256, 2, 121, x=xx + fbm(16, 2, 123) * 0.1, y=yy * 0.35) * 0.6 + fbm(512, 1, 125) * 0.4
    clump = fbm(16, 4, 127)
    h = 0.6 * blades + 0.4 * clump
    g = (0.65 + 0.45 * blades) * (0.8 + 0.3 * clump)
    dry = sstep(0.6, 0.8, fbm(8, 4, 129))
    col = tint(g, (1, 1, 1))
    col = col * (1 - dry[..., None] * 0.4) + dry[..., None] * 0.4 * g[..., None] * np.array([1.35, 1.15, 0.6])
    save("grass", h * 0.01, col, 0.04, 2.0)


def ground():
    """Bare earth with pebbles and twigs (dirt paths, village ground). Tile 2 m."""
    f1, f2, cid = worley(40, 40, 131, jitter=0.9)
    peb = sstep(0.32, 0.12, f1) * (cid > 0.55)
    soil = fbm(32, 5, 133)
    h = 0.5 * soil + 0.6 * peb * (0.6 + 0.4 * cid)
    g = (0.75 + 0.3 * soil) * (1 - 0.15 * fbm(8, 3, 135))
    g = np.where(peb > 0.3, 0.7 + 0.35 * cid, g)
    save("ground", h * 0.012, tint(g, (1.0, 0.98, 0.95)), 0.04, 2.0)


def generic(name, seed, tile, amp, hue=(1, 1, 1)):
    f = fbm(16, 5, seed)
    h = f + 0.3 * fbm(128, 3, seed + 1)
    g = 0.85 + 0.15 * (f - 0.5) * 2
    save(name, h * amp, tint(g, hue), 0.02, tile)


def main():
    ashlar()
    fieldstone()
    cobble()
    slate()
    thatch()
    planks()
    woodgrain()
    plaster()
    rock()
    iron()
    cloth()
    grass()
    ground()
    generic("smooth", 141, 1.0, 0.002)
    (OUT / "tex.json").write_text(json.dumps(META, indent=1))
    # contact sheet
    S = 192
    sheet = Image.new("RGB", (S * len(SHEET), S * 2))
    for i, (name, col, n) in enumerate(SHEET):
        sheet.paste(Image.fromarray((np.clip(col, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)).resize((S, S)), (i * S, 0))
        sheet.paste(Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8)).resize((S, S)), (i * S, S))
    sheet.save(ROOT / "assets_src" / "hq" / "tex_sheet.jpg", quality=85)


main()
