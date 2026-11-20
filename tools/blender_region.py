"""Model the countryside round the village and the village's new furniture in Blender:

  nature   oak, apple tree (with its fruit recorded for shaking), pine, round bush, berry bush,
           reeds, two boulders, apples and pears
  village  dry-stone wall segment and pillar, town gate, well, fence, hay bale, cart,
           lamp post, crate stack, bench, flower trough

Everything goes to assets_src/region/<name>.glb (glTF axes, metres, base on y=0, front +Z).
Run: blender --background --factory-startup --python tools/blender_region.py
"""
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blender_flora as F  # noqa: E402
import geo  # noqa: E402
from geo import C, box, cylinder, lathe, blob, tube, quad, rounded_box, grain, fbm, vnoise, sstep, unit, rot_y, rot_x, rot_z  # noqa: E402,F401

OUT = HERE.parent / "assets_src" / "region"
MATS = {
    "wood": {"rough": 0.82}, "stone": {"rough": 0.92}, "iron": {"rough": 0.5, "metal": 0.55}, "straw": {"rough": 0.95},
    "leaf": {"rough": 0.7, "double": True}, "plant": {"rough": 0.8, "double": True}, "bark": {"rough": 0.92},
    "fruit": {"rough": 0.45}, "cloth": {"rough": 0.95, "double": True}, "rock": {"rough": 0.9},
    "glow_lamp": {"rough": 0.4, "emissive": (1.0, 0.72, 0.34), "strength": 1.6}, "glass": {"rough": 0.15, "metal": 0.2},
}
rng = np.random.default_rng(77)
WOOD, WOOD_DARK, STONE, IRON = C(0.46, 0.32, 0.20), C(0.30, 0.20, 0.12), C(0.52, 0.50, 0.46), C(0.24, 0.24, 0.26)


def save(asset):
    asset.write(OUT / f"{asset.name}.glb", MATS)


def wood_col(base=WOOD, streak=(1, 6, 1)):
    return lambda v: grain(v, base, 0.16, 3.0, rng.uniform(0, 9), streak)


def stone_col(base=STONE, moss=0.0):
    seed = rng.uniform(0, 9)

    def fn(v):
        c = grain(v, base, 0.18, 2.5, seed)
        if moss:
            m = np.clip(sstep(0.2, 0.9, vnoise(v * 1.8, seed + 3)) * moss, 0, 1)
            c = c * (1 - m[:, None]) + C(0.30, 0.42, 0.18) * m[:, None]
        return c
    return fn


# ------------------------------------------------------------------ trees (re-using the forest's tree builder)
def flora_tree(name, seed, height, kind, palettes, trunk_r, leaves, fruit=None):
    F.CROWN_LOG = []
    F.OUT = OUT
    F.tree(name, seed, height, kind, palettes, trunk_r, leaves_per_pad=leaves, gold=0.0)
    pads = F.CROWN_LOG
    F.CROWN_LOG = None
    if fruit:
        # fruit hangs under the outer half of each crown pad; Blender (x, y, z) -> glTF (x, z, -y)
        spots = []
        r2 = np.random.default_rng(seed + 5)
        for centre, radius in pads:
            for _ in range(fruit):
                a, d = r2.uniform(0, 2 * math.pi), radius * math.sqrt(r2.uniform(0.25, 0.85))
                p = centre + np.array([math.cos(a) * d, math.sin(a) * d, -radius * 0.12 + r2.uniform(-0.05, 0.05)])
                spots.append([round(float(p[0]), 3), round(float(p[2]), 3), round(float(-p[1]), 3)])
        (OUT / f"{name}.fruit.json").write_text(json.dumps(spots), encoding="utf-8")
        print(f"FRUIT {name}: {len(spots)} spots")


def pine(name, seed, height):
    """A stylised fir: a tapering trunk under stacked jagged skirts of needles."""
    r = np.random.default_rng(seed)
    a = geo.Asset(name)
    cylinder(a, "bark", (0, -0.3, 0), (0, height * 0.92, 0), height * 0.045, height * 0.012, wood_col(C(0.30, 0.20, 0.13), (1, 4, 1)), sides=9)
    tiers = 7
    for t in range(tiers):
        f = t / tiers
        y0 = height * (0.18 + 0.75 * f)
        rad = height * 0.33 * (1 - f) ** 0.9 + 0.25
        depth = height * 0.17 * (1 - 0.4 * f)
        seg = 28
        prof = []
        for k in range(6):
            s = k / 5
            prof.append((rad * s, y0 + depth * (1 - s) ** 1.4 - depth * 0.25 * s))
        verts, faces = [], []
        for j, (rr, yy) in enumerate(prof):
            for i in range(seg):
                ang = i / seg * 2 * math.pi + t * 0.4
                jag = 1 + (0.18 if i % 2 else -0.06) * (j / 5) + r.normal(0, 0.03)
                droop = -0.12 * depth * (j / 5) * (1 if i % 2 else 0)
                verts.append((math.cos(ang) * rr * jag, yy + droop, math.sin(ang) * rr * jag))
        for j in range(5):
            for i in range(seg):
                p0, p1 = j * seg + i, j * seg + (i + 1) % seg
                faces += [(p0, p1, p1 + seg), (p0, p1 + seg, p0 + seg)]
        verts = np.array(verts)
        rel = np.hypot(verts[:, 0], verts[:, 2]) / max(rad, 1e-3)
        col = C(0.07, 0.20, 0.12)[None] + (C(0.22, 0.42, 0.20) - C(0.07, 0.20, 0.12))[None] * rel[:, None]
        col = col * (0.85 + 0.3 * vnoise(verts * 2.2, seed + t)[:, None])
        faces = geo._orient(verts, faces, None, np.array([0, y0 - depth, 0]))
        a.add("leaf", verts, col, faces)
        # underside, darker
        under = verts.copy()
        under[:, 1] -= depth * 0.18
        a.add("leaf", under, col * 0.55, faces[:, ::-1])
    save(a)


def bush(name, seed, size, palettes, berries=0):
    r = np.random.default_rng(seed)
    F.OUT = OUT
    asset = F.Asset(name)
    leaf = asset.part("leaf")
    for k in range(5):
        ang = k / 5 * 2 * math.pi + r.uniform(-0.4, 0.4)
        centre = np.array([math.cos(ang) * size * 0.35, math.sin(ang) * size * 0.35, size * r.uniform(0.35, 0.6)])
        F.crown(leaf, r, centre, size * r.uniform(0.45, 0.6), palettes, size * 0.22, int(110 * size), flat=0.9, gold=0.0, droop_edge=0.3)
    F.crown(leaf, r, np.array([0, 0, size * 0.75]), size * 0.55, palettes, size * 0.22, int(120 * size), flat=0.9, gold=0.0, droop_edge=0.3)
    if berries:
        b = asset.part("fruit")
        for _ in range(berries):
            ang, d = r.uniform(0, 2 * math.pi), size * r.uniform(0.3, 0.75)
            p = np.array([math.cos(ang) * d, math.sin(ang) * d, size * r.uniform(0.35, 0.95)])
            sv, sf = geo.icosphere(1)
            b.add(sv * 0.035 * size / 0.8 + p, F.srgb(0.75, 0.10, 0.18) * r.uniform(0.85, 1.1), sf)
    asset.save()


def reeds(name, seed):
    r = np.random.default_rng(seed)
    F.OUT = OUT
    asset = F.Asset(name)
    part = asset.part("plant")
    for k in range(26):
        ang, d = r.uniform(0, 2 * math.pi), r.uniform(0, 0.35)
        F.blade(part, r, np.array([math.cos(ang) * d, math.sin(ang) * d, 0]), r.uniform(0.9, 1.6), 0.015,
                F.srgb(0.18, 0.28, 0.10), F.srgb(0.62, 0.66, 0.32), lean=r.uniform(0.05, 0.35), segments=4)
    heads = asset.part("plant")
    for k in range(6):
        ang, d = r.uniform(0, 2 * math.pi), r.uniform(0, 0.3)
        base = np.array([math.cos(ang) * d, math.sin(ang) * d, 0])
        top = base + np.array([r.uniform(-0.08, 0.08), r.uniform(-0.08, 0.08), r.uniform(1.3, 1.7)])
        F.tube_line(heads, [base, (base + top) / 2, top], 0.01, F.srgb(0.36, 0.42, 0.18), 4)
        F.tube_line(heads, [top - np.array([0, 0, 0.02]), top + np.array([0, 0, 0.16])], 0.025, F.srgb(0.36, 0.22, 0.12), 6)
    asset.save()


def boulder(name, seed, size, moss=0.6):
    a = geo.Asset(name)
    s = rng.uniform(0, 50)

    def col(unitv, p):
        tone = vnoise(p * 1.2 / size, s)
        c = C(0.30, 0.30, 0.31)[None] + (C(0.56, 0.54, 0.50) - C(0.30, 0.30, 0.31))[None] * tone[:, None]
        g = np.clip(sstep(0.35, 0.85, unitv[:, 1]) * moss * (0.5 + vnoise(p * 2.5 / size, s + 4)), 0, 1)
        return c * (1 - g[:, None]) + C(0.30, 0.42, 0.18) * g[:, None]
    blob(a, "rock", (0, size * 0.32, 0), (size, size * 0.7, size * 0.85), col, level=3, noise=0.32, freq=1.3, seed=s, flat=True)
    save(a)


def fruit(name, colour, kind):
    a = geo.Asset(name)
    if kind == "apple":
        prof = [(0.0, 0.0), (0.028, 0.004), (0.042, 0.02), (0.046, 0.042), (0.040, 0.066), (0.022, 0.078), (0.008, 0.07), (0.0, 0.066)]
    else:   # pear
        prof = [(0.0, 0.0), (0.03, 0.005), (0.045, 0.03), (0.042, 0.055), (0.026, 0.08), (0.018, 0.1), (0.012, 0.112), (0.0, 0.112)]
    lathe(a, "fruit", prof, 16, lambda v: colour * (0.8 + 0.3 * sstep(0.0, 0.09, v[:, 1]))[:, None] * (0.92 + 0.12 * vnoise(v * 60, 2))[:, None])
    top = prof[-1][1]
    tube(a, "wood", [(0, top - 0.005, 0), (0.004, top + 0.02, 0.002)], 0.003, WOOD_DARK, sides=4)
    quad(a, "leaf", [(0.004, top + 0.012, 0), (0.03, top + 0.03, 0.006), (0.04, top + 0.025, 0.0), (0.014, top + 0.008, -0.004)], C(0.22, 0.48, 0.16))
    save(a)


# ------------------------------------------------------------------ village furniture
def wall_segment(name, length=2.4, height=1.05, depth=0.62, seed=1):
    """Dry-stone wall: courses of rough blocks, a row of cap stones, moss in the joints."""
    r = np.random.default_rng(seed)
    a = geo.Asset(name)
    y, course = 0.0, 0
    while y < height - 0.18:
        h = r.uniform(0.18, 0.27)
        x = -length / 2 + (0.15 if course % 2 else 0.0) - 0.15
        while x < length / 2:
            w = r.uniform(0.32, 0.6)
            x1 = min(x + w, length / 2)
            if x1 - x > 0.08:
                for side in (-1, 1):
                    d = depth / 2 * r.uniform(0.9, 1.0)
                    centre = ((x + x1) / 2, y + h / 2, side * d / 2)
                    tone = C(0.50, 0.48, 0.44) * r.uniform(0.75, 1.12)
                    box(a, "stone", centre, (x1 - x - 0.03, h - 0.025, d - 0.02), stone_col(tone, 0.25),
                        rot=rot_y(r.normal(0, 0.03)) @ rot_x(r.normal(0, 0.02)), jitter=0.04, seed=r.uniform(0, 9))
            x = x1
        y += h
        course += 1
    x = -length / 2
    while x < length / 2:          # cap stones laid on edge
        w = r.uniform(0.16, 0.24)
        box(a, "stone", (x + w / 2, y + 0.13, 0), (w - 0.02, 0.26, depth * 0.9), stone_col(C(0.46, 0.45, 0.42) * r.uniform(0.85, 1.1), 0.6),
            rot=rot_z(r.normal(0, 0.06)), jitter=0.03, seed=r.uniform(0, 9))
        x += w
    # moss and grass tufts tucked at the foot
    for k in range(10):
        p = (r.uniform(-length / 2, length / 2), 0.02, r.choice([-1, 1]) * depth * 0.52)
        blob(a, "plant", p, (0.12, 0.06, 0.1), C(0.24, 0.38, 0.14) * r.uniform(0.8, 1.2), level=1, noise=0.3, seed=r.uniform(0, 9))
    save(a)


def wall_pillar(name, seed=2):
    r = np.random.default_rng(seed)
    a = geo.Asset(name)
    y = 0.0
    while y < 1.35:
        h = r.uniform(0.22, 0.3)
        box(a, "stone", (0, y + h / 2, 0), (0.86, h - 0.02, 0.86), stone_col(C(0.52, 0.50, 0.46) * r.uniform(0.8, 1.1), 0.2),
            rot=rot_y(r.normal(0, 0.05)), jitter=0.04, seed=r.uniform(0, 9))
        y += h
    box(a, "stone", (0, y + 0.08, 0), (1.0, 0.16, 1.0), stone_col(C(0.48, 0.46, 0.43), 0.5), jitter=0.02)
    save(a)


def gate(name, width=4.2):
    """Town gate: two stone towers-of-blocks, a timber lintel with a shingle roof, lanterns, open doors."""
    a = geo.Asset(name)
    r = np.random.default_rng(5)
    for side in (-1, 1):
        x = side * (width / 2 + 0.6)
        y = 0.0
        while y < 3.0:
            h = r.uniform(0.3, 0.4)
            box(a, "stone", (x, y + h / 2, 0), (1.2, h - 0.02, 1.2), stone_col(C(0.55, 0.52, 0.47) * r.uniform(0.8, 1.1), 0.25),
                rot=rot_y(r.normal(0, 0.04)), jitter=0.05, seed=r.uniform(0, 9))
            y += h
        box(a, "stone", (x, y + 0.1, 0), (1.35, 0.2, 1.35), stone_col(C(0.5, 0.48, 0.44), 0.4))
        # lantern on an iron bracket, facing the road on both sides
        for fz in (-1, 1):
            cylinder(a, "iron", (x - side * 0.6, 2.4, fz * 0.62), (x - side * 0.6, 2.4, fz * 0.95), 0.025, colour=IRON, sides=6)
            box(a, "iron", (x - side * 0.6, 2.2, fz * 0.95), (0.2, 0.04, 0.2), IRON)
            box(a, "glow_lamp", (x - side * 0.6, 2.06, fz * 0.95), (0.13, 0.22, 0.13), C(1.0, 0.8, 0.45), shade=False)
            box(a, "iron", (x - side * 0.6, 1.93, fz * 0.95), (0.18, 0.04, 0.18), IRON)
    top = 3.5
    box(a, "wood", (0, top, 0), (width + 2.6, 0.32, 0.42), wood_col(WOOD_DARK, (6, 1, 1)))
    box(a, "wood", (0, top - 0.32, 0), (width + 1.8, 0.2, 0.3), wood_col(WOOD, (6, 1, 1)))
    for side in (-1, 1):            # braces
        cylinder(a, "wood", (side * width / 2, 2.5, 0), (side * (width / 2 - 0.7), top - 0.2, 0), 0.08, colour=WOOD_DARK, sides=6)
    # little roof of shingles
    for i in range(7):
        y = top + 0.3 + i * 0.09
        for zs in (-1, 1):
            box(a, "wood", (0, y, zs * (0.75 - i * 0.1)), (width + 3.0, 0.05, 0.22), wood_col(C(0.36, 0.24, 0.16), (8, 1, 1)),
                rot=rot_x(zs * 0.55))
    box(a, "wood", (0, top + 0.95, 0), (width + 3.1, 0.12, 0.14), wood_col(WOOD_DARK, (6, 1, 1)))
    # doors, swung open against the inside of the posts
    for side in (-1, 1):
        hinge = np.array([side * width / 2, 0, -0.3])
        with a.at(hinge, yaw=side * 1.35):
            for k in range(5):
                box(a, "wood", (-side * (0.2 + k * 0.38), 1.3, 0), (0.36, 2.5, 0.08), wood_col(C(0.42, 0.28, 0.17) * r.uniform(0.85, 1.1)))
            for yy in (0.5, 2.1):
                box(a, "iron", (-side * 0.95, yy, 0.05), (1.9, 0.08, 0.03), IRON)
    save(a)


def well(name):
    a = geo.Asset(name)
    r = np.random.default_rng(8)
    seg = 14
    for row in range(4):
        for i in range(seg):
            ang = (i + 0.5 * (row % 2)) / seg * 2 * math.pi
            c = np.array([math.cos(ang) * 0.95, 0.12 + row * 0.22, math.sin(ang) * 0.95])
            box(a, "stone", c, (0.42, 0.2, 0.32), stone_col(C(0.55, 0.52, 0.48) * r.uniform(0.8, 1.1), 0.3), rot=rot_y(-ang + math.pi / 2), jitter=0.03, seed=r.uniform(0, 9))
    lathe(a, "glass", [(0.8, 0.5), (0.0, 0.5)], 18, C(0.08, 0.18, 0.22))     # dark water
    for side in (-1, 1):
        box(a, "wood", (side * 1.05, 1.35, 0), (0.16, 2.1, 0.16), wood_col(WOOD_DARK))
    cylinder(a, "wood", (-1.1, 1.75, 0), (1.1, 1.75, 0), 0.07, colour=WOOD, sides=8)
    cylinder(a, "iron", (1.1, 1.75, 0), (1.3, 1.75, 0), 0.02, colour=IRON, sides=6)
    box(a, "iron", (1.3, 1.6, 0), (0.03, 0.3, 0.03), IRON)
    tube(a, "cloth", [(0, 1.7, 0.07), (0, 1.1, 0.0)], 0.012, C(0.7, 0.6, 0.42), sides=4, cap=False)
    lathe(a, "wood", [(0.12, 0.9), (0.15, 1.12), (0.0, 1.12)], 10, WOOD, centre=(0, 0, 0))
    for zs in (-1, 1):
        for i in range(8):
            box(a, "wood", (0, 2.45 + i * 0.07, zs * (0.75 - i * 0.1)), (2.6, 0.05, 0.22), wood_col(C(0.38, 0.25, 0.16), (8, 1, 1)), rot=rot_x(zs * 0.6))
    box(a, "wood", (0, 3.05, 0), (2.7, 0.1, 0.14), wood_col(WOOD_DARK))
    save(a)


def fence(name, length=2.5):
    a = geo.Asset(name)
    r = np.random.default_rng(9)
    for x in (-length / 2, length / 2):
        box(a, "wood", (x, 0.55, 0), (0.12, 1.1, 0.12), wood_col(WOOD_DARK), rot=rot_z(r.normal(0, 0.04)))
    for y in (0.4, 0.85):
        box(a, "wood", (0, y, 0.07), (length + 0.15, 0.1, 0.05), wood_col(C(0.48, 0.34, 0.22) * r.uniform(0.85, 1.1), (8, 1, 1)), rot=rot_z(r.normal(0, 0.02)))
    save(a)


def hay_bale(name):
    a = geo.Asset(name)
    s = rng.uniform(0, 9)
    rounded_box(a, "straw", (0, 0.4, 0), (1.2, 0.8, 0.8),
                lambda v: C(0.80, 0.66, 0.34) * (0.8 + 0.3 * vnoise(v * np.array([2, 40, 40]), s))[:, None], levels=2, lumpy=0.02, seed=s)
    for x in (-0.3, 0.3):
        lathe(a, "cloth", [(0.41, -0.41), (0.42, -0.38), (0.42, 0.38), (0.41, 0.41)], 12, C(0.42, 0.30, 0.18), centre=(x, 0.4, 0), rot=rot_z(math.pi / 2))
    save(a)


def cart(name):
    a = geo.Asset(name)
    for i in range(6):     # bed
        box(a, "wood", (0, 0.75, -0.75 + i * 0.3), (1.4, 0.06, 0.28), wood_col(C(0.48, 0.33, 0.2) * (0.9 + 0.04 * i), (6, 1, 1)))
    for side in (-1, 1):
        box(a, "wood", (side * 0.7, 0.98, 0), (0.06, 0.4, 1.8), wood_col(WOOD_DARK, (1, 1, 6)))
        box(a, "wood", (0, 0.98, side * 0.9), (1.4, 0.4, 0.06), wood_col(WOOD_DARK, (6, 1, 1)))
        # wheel: rim, hub, spokes
        cx = side * 0.82
        ang = np.linspace(0, 2 * math.pi, 25)
        rim = [(cx, 0.48 + math.sin(t) * 0.46, math.cos(t) * 0.46) for t in ang]
        tube(a, "wood", rim, 0.045, WOOD_DARK, sides=6, cap=False)
        cylinder(a, "wood", (cx - side * 0.08, 0.48, 0), (cx + side * 0.08, 0.48, 0), 0.08, colour=WOOD_DARK, sides=10)
        for k in range(8):
            t = k / 8 * 2 * math.pi
            cylinder(a, "wood", (cx, 0.48, 0), (cx, 0.48 + math.sin(t) * 0.44, math.cos(t) * 0.44), 0.022, colour=WOOD, sides=5, caps=False)
    cylinder(a, "wood", (-0.82, 0.48, 0), (0.82, 0.48, 0), 0.04, colour=WOOD_DARK, sides=6)
    for side in (-1, 1):
        cylinder(a, "wood", (side * 0.35, 0.78, 0.9), (side * 0.3, 0.45, 2.3), 0.05, colour=WOOD, sides=6)
    for i in range(3):     # a load of sacks
        rounded_box(a, "cloth", (-0.3 + i * 0.3, 1.0, -0.3 + (i % 2) * 0.4), (0.36, 0.3, 0.5), C(0.72, 0.62, 0.44), levels=2, lumpy=0.02, seed=i)
    save(a)


def lamp_post(name):
    a = geo.Asset(name)
    cylinder(a, "wood", (0, 0, 0), (0, 2.8, 0), 0.08, 0.06, colour=wood_col(WOOD_DARK), sides=8)
    box(a, "stone", (0, 0.15, 0), (0.36, 0.3, 0.36), stone_col(C(0.5, 0.48, 0.45), 0.3), jitter=0.03)
    cylinder(a, "iron", (0, 2.7, 0), (0.45, 2.7, 0), 0.02, colour=IRON, sides=6)
    box(a, "iron", (0.45, 2.55, 0), (0.24, 0.04, 0.24), IRON)
    box(a, "glow_lamp", (0.45, 2.38, 0), (0.15, 0.28, 0.15), C(1.0, 0.8, 0.45), shade=False)
    box(a, "iron", (0.45, 2.22, 0), (0.22, 0.04, 0.22), IRON)
    for dx, dz in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        cylinder(a, "iron", (0.45 + dx * 0.09, 2.22, dz * 0.09), (0.45 + dx * 0.09, 2.55, dz * 0.09), 0.012, colour=IRON, sides=4, caps=False)
    save(a)


def crates(name):
    a = geo.Asset(name)
    r = np.random.default_rng(11)

    def crate(centre, s, yaw):
        with a.at(centre, yaw=yaw):
            box(a, "wood", (0, s / 2, 0), (s * 0.94, s * 0.94, s * 0.94), wood_col(C(0.55, 0.40, 0.25) * r.uniform(0.85, 1.1)))
            for y in (0.04, 0.96):
                for z in (-1, 1):
                    box(a, "wood", (0, y * s, z * s * 0.48), (s, s * 0.1, s * 0.06), wood_col(WOOD_DARK, (6, 1, 1)))
                    box(a, "wood", (z * s * 0.48, y * s, 0), (s * 0.06, s * 0.1, s), wood_col(WOOD_DARK, (1, 1, 6)))
            for x in (-1, 1):
                for z in (-1, 1):
                    box(a, "wood", (x * s * 0.48, s / 2, z * s * 0.48), (s * 0.08, s, s * 0.08), wood_col(WOOD_DARK))
    crate((0, 0, 0), 0.8, 0.1)
    crate((0.85, 0, 0.1), 0.7, -0.2)
    crate((0.3, 0.8, 0.05), 0.6, 0.4)
    save(a)


def bench(name):
    a = geo.Asset(name)
    for i in range(3):
        box(a, "wood", (0, 0.46, -0.14 + i * 0.14), (1.6, 0.05, 0.12), wood_col(C(0.5, 0.35, 0.22), (8, 1, 1)))
    for x in (-0.65, 0.65):
        box(a, "wood", (x, 0.22, 0), (0.08, 0.44, 0.4), wood_col(WOOD_DARK))
    save(a)


def trough(name):
    a = geo.Asset(name)
    for side in (-1, 1):
        box(a, "wood", (0, 0.25, side * 0.25), (1.6, 0.4, 0.06), wood_col(C(0.45, 0.31, 0.2), (8, 1, 1)))
        box(a, "wood", (side * 0.8, 0.25, 0), (0.06, 0.4, 0.56), wood_col(C(0.45, 0.31, 0.2)))
    box(a, "wood", (0, 0.06, 0), (1.6, 0.06, 0.5), wood_col(WOOD_DARK))
    box(a, "rock", (0, 0.36, 0), (1.5, 0.08, 0.44), C(0.30, 0.22, 0.14))
    r = np.random.default_rng(4)
    for k in range(20):    # flowers in the trough
        p = (r.uniform(-0.7, 0.7), 0.42, r.uniform(-0.18, 0.18))
        tube(a, "plant", [p, (p[0] + r.normal(0, 0.03), p[1] + r.uniform(0.15, 0.3), p[2] + r.normal(0, 0.03))], 0.008, C(0.2, 0.42, 0.16), sides=4, cap=False)
        col = [C(0.95, 0.4, 0.5), C(0.98, 0.85, 0.3), C(0.7, 0.5, 0.95), C(0.98, 0.95, 0.9)][k % 4]
        blob(a, "plant", (p[0], p[1] + 0.3, p[2]), (0.05, 0.03, 0.05), col, level=1, noise=0.2, seed=k)
    save(a)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    F.clear_scene()
    P = F.LEAF_PALETTES
    if "--props-only" in sys.argv:
        return props()
    apple_leaf = [(F.srgb(0.10, 0.26, 0.10), F.srgb(0.40, 0.64, 0.24)), (F.srgb(0.14, 0.30, 0.10), F.srgb(0.50, 0.70, 0.26))]
    flora_tree("tree_oak", 501, 9.0, "umbrella", P["green"], 0.5, 80)
    flora_tree("tree_oak_b", 517, 7.5, "umbrella", P["green"] + [(F.srgb(0.18, 0.30, 0.10), F.srgb(0.62, 0.68, 0.28))], 0.44, 74)
    flora_tree("tree_apple", 523, 5.2, "umbrella", apple_leaf, 0.26, 60, fruit=4)
    flora_tree("tree_pear", 541, 5.6, "umbrella", apple_leaf, 0.27, 58, fruit=4)
    F.clear_scene()
    pine("tree_pine", 601, 11.0)
    pine("tree_pine_b", 613, 8.0)
    bush("bush_round", 621, 0.9, P["green"])
    bush("bush_berry", 633, 0.8, apple_leaf, berries=40)
    bush("bush_flower", 647, 0.75, [(F.srgb(0.40, 0.12, 0.30), F.srgb(0.95, 0.55, 0.78)), (F.srgb(0.12, 0.28, 0.12), F.srgb(0.42, 0.64, 0.30))])
    reeds("reeds", 655)
    boulder("boulder_a", 661, 1.6)
    boulder("boulder_b", 673, 2.6, moss=0.8)
    fruit("apple", C(0.82, 0.16, 0.12), "apple")
    fruit("pear", C(0.80, 0.72, 0.26), "pear")
    props()


def props():
    F.clear_scene()
    wall_segment("wall_seg", seed=1)
    wall_segment("wall_seg_b", seed=7)
    wall_pillar("wall_pillar")
    gate("town_gate")
    well("well")
    fence("fence")
    hay_bale("hay_bale")
    cart("cart")
    lamp_post("lamp_post")
    crates("crates")
    bench("bench")
    trough("flower_trough")


if __name__ == "__main__":
    main()
