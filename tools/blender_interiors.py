"""Model the rooms you can walk into: three village houses (a family home, the market shop, the
guards' lodge) and the castle's great hall, furnished piece by piece.

Room axes: glTF, floor at y = 0, the room centred on x = 0, the way out in the wall at +Z.
Writes assets_src/interiors/<room>.glb and interiors.json with each room's size, exit,
spawn point, furniture colliders and lights.
Run: blender --background --factory-startup --python tools/blender_interiors.py
"""
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geo  # noqa: E402
from geo import C, box, cylinder, lathe, blob, tube, quad, rounded_box, grain, vnoise, sstep, rot_y, rot_x, rot_z  # noqa: E402,F401

OUT = HERE.parent / "assets_src" / "interiors"
MATS = {"wood": {"rough": 0.8}, "plaster": {"rough": 0.95}, "stone": {"rough": 0.9}, "cloth": {"rough": 0.95, "double": True},
        "iron": {"rough": 0.45, "metal": 0.6}, "gold": {"rough": 0.3, "metal": 0.85}, "food": {"rough": 0.6}, "glass": {"rough": 0.1, "metal": 0.2},
        "ceramic": {"rough": 0.45}, "straw": {"rough": 0.95}, "plant": {"rough": 0.8, "double": True}, "paper": {"rough": 0.9},
        "glow_fire": {"rough": 0.5, "emissive": (1.0, 0.52, 0.16), "strength": 2.2},
        "glow_candle": {"rough": 0.5, "emissive": (1.0, 0.78, 0.42), "strength": 2.0},
        "glow_window": {"rough": 0.3, "emissive": (0.95, 0.85, 0.62), "strength": 1.3}}
rng = np.random.default_rng(31)
WOOD, WOOD_D, WOOD_L = C(0.50, 0.34, 0.21), C(0.30, 0.19, 0.11), C(0.62, 0.46, 0.30)
PLASTER = C(0.86, 0.80, 0.68)
IRON = C(0.22, 0.22, 0.24)


def wc(base=WOOD, streak=(1, 6, 1), amount=0.15):
    s = rng.uniform(0, 9)
    return lambda v: grain(v, base, amount, 3.5, s, streak)


class Room:
    def __init__(self, name, w, d, h):
        self.asset = geo.Asset(name)
        self.name, self.w, self.d, self.h = name, w, d, h
        self.colliders, self.lights, self.points = [], [], []

    def solid(self, x, z, hx, hz, yaw=0.0, top=1.0):
        self.colliders.append({"x": round(x, 3), "z": round(z, 3), "yaw": round(yaw, 4), "hx": round(hx, 3), "hz": round(hz, 3),
                               "y0": -0.5, "y1": round(top, 3), "a": 1})

    def light(self, x, y, z, colour, power, reach):
        self.lights.append({"x": x, "y": y, "z": z, "color": colour, "power": power, "reach": reach})


# ------------------------------------------------------------------ shells
def plank_floor(a, w, d, base=WOOD):
    x = -w / 2
    while x < w / 2:
        bw = rng.uniform(0.18, 0.26)
        z = -d / 2
        while z < d / 2:
            bl = rng.uniform(1.2, 2.6)
            z1 = min(z + bl, d / 2)
            box(a, "wood", (x + bw / 2, -0.03, (z + z1) / 2), (bw - 0.012, 0.06, z1 - z - 0.012), wc(base * rng.uniform(0.82, 1.1), (1, 1, 5)), shade=False)
            z = z1
        x += bw


def stone_floor(a, w, d, tile=0.9, base=C(0.52, 0.5, 0.47)):
    for i in range(int(w / tile) + 1):
        for j in range(int(d / tile) + 1):
            x0, z0 = -w / 2 + i * tile, -d / 2 + j * tile
            x1, z1 = min(x0 + tile, w / 2), min(z0 + tile, d / 2)
            if x1 - x0 < 0.05 or z1 - z0 < 0.05:
                continue
            box(a, "stone", ((x0 + x1) / 2, -0.04, (z0 + z1) / 2), (x1 - x0 - 0.03, 0.08, z1 - z0 - 0.03),
                lambda v, t=rng.uniform(0.8, 1.1): grain(v, base * t, 0.06, 3, 4), shade=False)


def timber_walls(room, door_w=1.3, windows=((-1, 0.0), (1, 0.0), (0, -1.5))):
    """Plaster walls in a timber frame, wood panelling below, beams overhead.
    windows: (wall, offset) with wall -1 = left (-x), 1 = right (+x), 0 = back (-z)."""
    a, w, d, h = room.asset, room.w, room.d, room.h
    t = 0.2
    walls = [((-w / 2, 0), (0, 1), d, rot_y(math.pi / 2), -1), ((w / 2, 0), (0, 1), d, rot_y(-math.pi / 2), 1), ((0, -d / 2), (1, 0), w, np.eye(3), 0)]
    for (cx, cz), along, length, rot, wall_id in walls:
        box(a, "plaster", (cx, h / 2, cz), (length if wall_id == 0 else t, h, t if wall_id == 0 else length),
            lambda v: grain(v, PLASTER, 0.05, 1.5, 2), shade=False)
    # front wall with the doorway
    for side in (-1, 1):
        seg = (w - door_w) / 2
        box(a, "plaster", (side * (door_w / 2 + seg / 2), h / 2, d / 2), (seg, h, t), lambda v: grain(v, PLASTER, 0.05, 1.5, 2), shade=False)
    box(a, "plaster", (0, (2.3 + h) / 2, d / 2), (door_w, h - 2.3, t), PLASTER, shade=False)
    # frame: posts, sills, braces, panelling
    for x in np.linspace(-w / 2 + 0.12, w / 2 - 0.12, int(w / 1.6) + 1):
        for z in (-d / 2 + 0.12, d / 2 - 0.12):
            if z > 0 and abs(x) < door_w / 2 + 0.15:
                continue
            box(a, "wood", (x, h / 2, z), (0.18, h, 0.1), wc(WOOD_D))
    for z in np.linspace(-d / 2 + 0.12, d / 2 - 0.12, int(d / 1.6) + 1):
        for x in (-w / 2 + 0.12, w / 2 - 0.12):
            box(a, "wood", (x, h / 2, z), (0.1, h, 0.18), wc(WOOD_D))
    for y in (0.95, h - 0.12):
        box(a, "wood", (0, y, -d / 2 + 0.12), (w, 0.14, 0.1), wc(WOOD_D, (6, 1, 1)))
        for x in (-w / 2 + 0.12, w / 2 - 0.12):
            box(a, "wood", (x, y, 0), (0.1, 0.14, d), wc(WOOD_D, (1, 1, 6)))
        for side in (-1, 1):
            box(a, "wood", (side * (door_w / 2 + (w - door_w) / 4), y, d / 2 - 0.12), ((w - door_w) / 2, 0.14, 0.1), wc(WOOD_D, (6, 1, 1)))
    # wainscot
    for (cx, cz, sx, sz) in ((0, -d / 2 + 0.13, w, 0.04), (-w / 2 + 0.13, 0, 0.04, d), (w / 2 - 0.13, 0, 0.04, d)):
        n = int((sx if sx > 0.1 else sz) / 0.22)
        for i in range(n):
            f = (i + 0.5) / n - 0.5
            box(a, "wood", (cx + f * sx if sx > 0.1 else cx, 0.47, cz + f * sz if sz > 0.1 else cz),
                (0.2 if sx > 0.1 else 0.05, 0.9, 0.05 if sx > 0.1 else 0.2), wc(WOOD * rng.uniform(0.85, 1.05)))
    # ceiling and beams
    box(a, "wood", (0, h + 0.05, 0), (w, 0.1, d), wc(WOOD_D * 0.9, (1, 1, 6)), shade=False)
    for z in np.linspace(-d / 2 + 0.6, d / 2 - 0.6, int(d / 1.4)):
        box(a, "wood", (0, h - 0.12, z), (w, 0.22, 0.2), wc(WOOD_D, (6, 1, 1)))
    box(a, "wood", (0, h - 0.3, 0), (0.24, 0.24, d), wc(WOOD_D, (1, 1, 6)))
    # doorway frame and an open door
    for side in (-1, 1):
        box(a, "wood", (side * (door_w / 2 + 0.06), 1.15, d / 2 - 0.05), (0.12, 2.3, 0.3), wc(WOOD_D))
    box(a, "wood", (0, 2.35, d / 2 - 0.05), (door_w + 0.24, 0.14, 0.3), wc(WOOD_D, (6, 1, 1)))
    box(a, "glow_window", (0, 1.15, d / 2 + 0.02), (door_w, 2.3, 0.02), C(0.95, 0.88, 0.70), shade=False)     # daylight in the doorway
    with a.at((door_w / 2, 0, d / 2 - 0.15), yaw=-1.3):
        for k in range(4):
            box(a, "wood", (-0.17 - k * 0.32, 1.1, 0), (0.3, 2.2, 0.06), wc(WOOD * rng.uniform(0.8, 1.0)))
    # windows: glowing panes with leading, sills and shutters
    for wall_id, off in windows:
        if wall_id == 0:
            centre, yaw = (off, 1.8, -d / 2 + 0.11), 0.0
        else:
            centre, yaw = (wall_id * (w / 2 - 0.11), 1.8, off), -wall_id * math.pi / 2
        with a.at(centre, yaw=yaw):
            box(a, "glow_window", (0, 0, 0), (0.9, 1.0, 0.03), C(0.95, 0.88, 0.70), shade=False)
            for k in (-0.3, 0.0, 0.3):
                box(a, "iron", (k, 0, 0.03), (0.02, 1.0, 0.02), IRON)
            for k in (-0.25, 0.25):
                box(a, "iron", (0, k, 0.03), (0.9, 0.02, 0.02), IRON)
            box(a, "wood", (0, -0.56, 0.1), (1.1, 0.08, 0.24), wc(WOOD_D, (6, 1, 1)))
            box(a, "wood", (0, 0.56, 0.06), (1.1, 0.1, 0.12), wc(WOOD_D, (6, 1, 1)))
            for s in (-1, 1):
                box(a, "wood", (s * 0.52, 0, 0.06), (0.1, 1.2, 0.12), wc(WOOD_D))
            # light falling in
        room.light(centre[0] * 0.8, 1.6, centre[2] * 0.8, [1.0, 0.92, 0.75], 0.9, 6.0)


# ------------------------------------------------------------------ furniture
def table(a, room, x, z, w=1.8, d=0.9, yaw=0.0, cloth=None):
    with a.at((x, 0, z), yaw=yaw):
        for i in range(int(w / 0.22)):
            box(a, "wood", (-w / 2 + (i + 0.5) * w / int(w / 0.22), 0.76, 0), (w / int(w / 0.22) - 0.01, 0.06, d), wc(WOOD_L * rng.uniform(0.85, 1.05), (1, 1, 5)))
        for sx in (-1, 1):
            for sz in (-1, 1):
                box(a, "wood", (sx * (w / 2 - 0.12), 0.37, sz * (d / 2 - 0.1)), (0.09, 0.74, 0.09), wc(WOOD_D))
            box(a, "wood", (sx * (w / 2 - 0.12), 0.18, 0), (0.06, 0.06, d - 0.2), wc(WOOD_D))
        box(a, "wood", (0, 0.18, 0), (w - 0.24, 0.06, 0.06), wc(WOOD_D, (6, 1, 1)))
        if cloth is not None:
            box(a, "cloth", (0, 0.795, 0), (w * 0.5, 0.012, d + 0.1), cloth, shade=False)
    room.solid(x, z, (w / 2 if abs(math.sin(yaw)) < 0.5 else d / 2), (d / 2 if abs(math.sin(yaw)) < 0.5 else w / 2), 0.0, 0.8)


def chair(a, x, z, yaw=0.0):
    with a.at((x, 0, z), yaw=yaw):
        box(a, "wood", (0, 0.45, 0), (0.44, 0.05, 0.42), wc(WOOD))
        for sx in (-1, 1):
            for sz in (-1, 1):
                box(a, "wood", (sx * 0.18, 0.22, sz * 0.17), (0.05, 0.44, 0.05), wc(WOOD_D))
            box(a, "wood", (sx * 0.18, 0.75, -0.19), (0.05, 0.6, 0.05), wc(WOOD_D))
        for y in (0.65, 0.85, 1.0):
            box(a, "wood", (0, y, -0.19), (0.4, 0.06, 0.04), wc(WOOD, (6, 1, 1)))


def stool(a, x, z):
    with a.at((x, 0, z)):
        lathe(a, "wood", [(0.18, 0.46), (0.19, 0.5), (0.0, 0.5)], 12, WOOD)
        for k in range(3):
            t = k / 3 * 2 * math.pi
            cylinder(a, "wood", (math.cos(t) * 0.12, 0.47, math.sin(t) * 0.12), (math.cos(t) * 0.18, 0, math.sin(t) * 0.18), 0.025, colour=WOOD_D, sides=5)


def bed(a, room, x, z, yaw=0.0, blanket=C(0.55, 0.16, 0.14)):
    with a.at((x, 0, z), yaw=yaw):
        box(a, "wood", (0, 0.3, 0), (1.0, 0.18, 2.0), wc(WOOD_D, (1, 1, 5)))
        for sz, hh in ((-1, 1.05), (1, 0.7)):
            box(a, "wood", (0, hh / 2, sz * 1.0), (1.1, hh, 0.08), wc(WOOD))
            for sx in (-1, 1):
                box(a, "wood", (sx * 0.52, hh / 2 + 0.05, sz * 1.0), (0.1, hh + 0.1, 0.1), wc(WOOD_D))
        rounded_box(a, "cloth", (0, 0.47, 0.1), (0.94, 0.18, 1.8), C(0.86, 0.82, 0.72), levels=2, lumpy=0.01, seed=1)
        rounded_box(a, "cloth", (0, 0.57, 0.3), (0.98, 0.1, 1.3), lambda v: blanket * (0.85 + 0.25 * (np.floor(v[:, 2] * 4 + v[:, 0] * 4) % 2))[:, None],
                    levels=2, lumpy=0.015, seed=2)
        rounded_box(a, "cloth", (0, 0.62, -0.7), (0.7, 0.14, 0.38), C(0.92, 0.9, 0.84), levels=2, lumpy=0.02, seed=3)
    room.solid(x, z, 0.55 if abs(math.sin(yaw)) < 0.5 else 1.05, 1.05 if abs(math.sin(yaw)) < 0.5 else 0.55, 0.0, 0.7)


def fireplace(a, room, x, z, yaw=0.0):
    with a.at((x, 0, z), yaw=yaw):
        for y in np.arange(0, 2.4, 0.3):
            for xx in np.arange(-1.0, 1.01, 0.4):
                if abs(xx) < 0.5 and y < 1.0:
                    continue
                box(a, "stone", (xx, y + 0.15, 0), (0.38, 0.28, 0.7), lambda v, t=rng.uniform(0.8, 1.1): grain(v, C(0.55, 0.52, 0.48) * t, 0.08, 3, 5), jitter=0.02)
        box(a, "stone", (0, 0.05, 0.3), (1.2, 0.1, 0.5), C(0.2, 0.18, 0.16))
        box(a, "wood", (0, 1.25, 0.33), (2.3, 0.12, 0.3), wc(WOOD_D, (6, 1, 1)))
        box(a, "stone", (0, 0.55, -0.15), (1.0, 1.0, 0.1), C(0.08, 0.06, 0.05))
        for k in range(3):
            cylinder(a, "wood", (-0.3, 0.14 + k * 0.05, 0.05 + k * 0.08), (0.3, 0.14 + k * 0.05, 0.05 - k * 0.04), 0.06, colour=WOOD_D, sides=6)
        for k in range(5):
            blob(a, "glow_fire", (rng.uniform(-0.25, 0.25), 0.3 + rng.uniform(0, 0.15), 0.05), (0.12, 0.28, 0.12), C(1.0, 0.6, 0.2), level=1, noise=0.4, seed=k)
        # chimney breast up to the ceiling
        box(a, "stone", (0, (2.4 + room.h) / 2, -0.1), (1.4, room.h - 2.4, 0.5), C(0.5, 0.48, 0.45))
        # mantel things: candles, a jug
        for cx in (-0.8, 0.8):
            cylinder(a, "plaster", (cx, 1.31, 0.35), (cx, 1.48, 0.35), 0.025, colour=C(0.95, 0.92, 0.82), sides=6)
            blob(a, "glow_candle", (cx, 1.52, 0.35), (0.015, 0.03, 0.015), C(1, 0.8, 0.4), level=1, noise=0.0)
        lathe(a, "ceramic", [(0.05, 1.31), (0.08, 1.38), (0.07, 1.48), (0.04, 1.52), (0.05, 1.56)], 10, C(0.62, 0.42, 0.28), centre=(0.3, 0, 0.35))
    # world-space light in front of the fire
    off = rot_y(yaw) @ np.array([0, 0, 0.6])
    room.light(x + off[0], 0.6, z + off[2], [1.0, 0.55, 0.22], 2.2, 7.0)
    room.solid(x, z, 1.1 if abs(math.sin(yaw)) < 0.5 else 0.4, 0.4 if abs(math.sin(yaw)) < 0.5 else 1.1, 0.0, 2.4)


def shelf(a, room, x, z, yaw=0.0, w=1.6, h=2.0, goods="pots"):
    with a.at((x, 0, z), yaw=yaw):
        for sx in (-1, 1):
            box(a, "wood", (sx * w / 2, h / 2, 0), (0.06, h, 0.36), wc(WOOD_D))
        box(a, "wood", (0, h / 2, -0.17), (w, h, 0.02), wc(WOOD_D * 0.8, (1, 6, 1)))
        for y in np.linspace(0.12, h - 0.05, 5):
            box(a, "wood", (0, y, 0), (w, 0.04, 0.36), wc(WOOD, (6, 1, 1)))
            if y > h - 0.1:
                continue
            xx = -w / 2 + 0.15
            while xx < w / 2 - 0.15:
                kind = goods if goods != "mixed" else ["pots", "books", "jars"][int(rng.integers(0, 3))]
                if kind == "books":
                    bw = rng.uniform(0.04, 0.08)
                    bh = rng.uniform(0.22, 0.32)
                    col = [C(0.5, 0.12, 0.1), C(0.15, 0.25, 0.45), C(0.2, 0.35, 0.18), C(0.55, 0.4, 0.2)][int(rng.integers(0, 4))]
                    box(a, "paper", (xx, y + 0.02 + bh / 2, 0.02), (bw, bh, 0.24), col * rng.uniform(0.8, 1.1), rot=rot_z(rng.normal(0, 0.05)))
                    xx += bw + 0.005
                elif kind == "jars":
                    hh = rng.uniform(0.15, 0.24)
                    lathe(a, "glass", [(0.06, y + 0.02), (0.065, y + 0.02 + hh * 0.8), (0.04, y + 0.02 + hh), (0.0, y + 0.02 + hh)], 10,
                          [C(0.45, 0.6, 0.35), C(0.7, 0.45, 0.2), C(0.6, 0.3, 0.45)][int(rng.integers(0, 3))], centre=(xx, 0, 0.02))
                    xx += 0.16
                else:
                    hh = rng.uniform(0.16, 0.28)
                    r = rng.uniform(0.06, 0.1)
                    lathe(a, "ceramic", [(r * 0.6, y + 0.02), (r, y + 0.02 + hh * 0.45), (r * 0.7, y + 0.02 + hh * 0.85), (r * 0.5, y + 0.02 + hh), (0.0, y + 0.02 + hh)], 12,
                          [C(0.66, 0.42, 0.26), C(0.42, 0.48, 0.55), C(0.8, 0.74, 0.6)][int(rng.integers(0, 3))] * rng.uniform(0.85, 1.1), centre=(xx, 0, 0.02))
                    xx += r * 2 + 0.05
    room.solid(x, z, w / 2 if abs(math.sin(yaw)) < 0.5 else 0.2, 0.2 if abs(math.sin(yaw)) < 0.5 else w / 2, 0.0, h)


def chest(a, room, x, z, yaw=0.0):
    with a.at((x, 0, z), yaw=yaw):
        box(a, "wood", (0, 0.25, 0), (0.9, 0.5, 0.5), wc(C(0.45, 0.28, 0.15)))
        lathe(a, "wood", [(0.25, 0.0), (0.25, 0.12), (0.17, 0.22), (0.0, 0.25)], 10, C(0.45, 0.28, 0.15), centre=(0, 0.5, 0), rot=rot_z(math.pi / 2) @ np.diag([1, 1, 1]))
        for xx in (-0.35, 0.35):
            box(a, "iron", (xx, 0.4, 0), (0.06, 0.82, 0.54), IRON)
        box(a, "gold", (0, 0.45, 0.26), (0.1, 0.12, 0.03), C(0.85, 0.68, 0.3))
    room.solid(x, z, 0.45, 0.3, yaw, 0.6)


def barrel(a, room, x, z, h=0.95, r=0.32):
    seg = 14
    prof = [(r * 0.85, 0.0), (r, h * 0.3), (r * 1.06, h * 0.5), (r, h * 0.7), (r * 0.85, h), (0.0, h)]
    lathe(a, "wood", prof, seg, lambda v: grain(v, WOOD * rng.uniform(0.85, 1.05), 0.2, 3, 7) * (0.85 + 0.2 * (np.floor(np.arctan2(v[:, 2] - z, v[:, 0] - x) / (2 * math.pi) * seg) % 2))[:, None],
          centre=(x, 0, z))
    for yy in (0.12, h * 0.5, h - 0.12):
        rr = r * (0.9 if yy < h * 0.2 or yy > h * 0.8 else 1.065)
        lathe(a, "iron", [(rr + 0.01, yy - 0.03), (rr + 0.012, yy + 0.03)], seg, IRON, centre=(x, 0, z))
    if room is not None:
        room.solid(x, z, r, r, 0.0, h)


def rug(a, x, z, w, d, col, border):
    box(a, "cloth", (x, 0.01, z), (w, 0.02, d), lambda v: col * (0.9 + 0.15 * vnoise(v * 6, 2))[:, None], shade=False)
    for (cx, cz, sx, sz) in ((x, z - d / 2 + 0.1, w, 0.12), (x, z + d / 2 - 0.1, w, 0.12), (x - w / 2 + 0.1, z, 0.12, d), (x + w / 2 - 0.1, z, 0.12, d)):
        box(a, "cloth", (cx, 0.022, cz), (sx, 0.01, sz), border, shade=False)


def candle(a, x, y, z):
    cylinder(a, "plaster", (x, y, z), (x, y + 0.14, z), 0.02, colour=C(0.95, 0.92, 0.8), sides=6)
    blob(a, "glow_candle", (x, y + 0.17, z), (0.012, 0.028, 0.012), C(1, 0.8, 0.4), level=1, noise=0.0)


def table_setting(a, x, z, w):
    for k in range(int(w / 0.5)):
        px = x - w / 2 + 0.3 + k * 0.5
        lathe(a, "ceramic", [(0.0, 0.79), (0.1, 0.795), (0.12, 0.81)], 12, C(0.86, 0.84, 0.78), centre=(px, 0, z + 0.25))
        lathe(a, "ceramic", [(0.0, 0.79), (0.035, 0.8), (0.04, 0.9), (0.045, 0.92)], 10, C(0.6, 0.4, 0.26), centre=(px + 0.12, 0, z + 0.05))
    blob(a, "food", (x, 0.86, z), (0.12, 0.07, 0.1), C(0.74, 0.5, 0.26), level=2, noise=0.15)       # a loaf
    for k in range(4):
        blob(a, "food", (x + 0.3 + k * 0.07, 0.83, z - 0.1), (0.035, 0.035, 0.035), [C(0.8, 0.2, 0.15), C(0.85, 0.75, 0.3)][k % 2], level=1, noise=0.1)


# ------------------------------------------------------------------ rooms
def home():
    room = Room("room_home", 8.0, 6.5, 3.2)
    a = room.asset
    plank_floor(a, room.w, room.d)
    timber_walls(room, windows=((-1, 0.6), (0, 1.5), (1, -1.6)))
    fireplace(a, room, 0.0, -room.d / 2 + 0.45, 0.0)
    table(a, room, 0.4, 0.4, 1.8, 0.9, cloth=C(0.85, 0.82, 0.72))
    table_setting(a, 0.4, 0.4, 1.8)
    for xx in (-0.3, 1.1):
        chair(a, xx, -0.25, 0.0)
        chair(a, xx, 1.05, math.pi)
    bed(a, room, -3.3, -1.6, 0.0)
    bed(a, room, 3.3, -1.6, 0.0, blanket=C(0.18, 0.32, 0.52))
    chest(a, room, -3.3, 0.2, 0.0)
    shelf(a, room, 3.75, 1.4, -math.pi / 2, 1.4, 1.9, "mixed")
    rug(a, 0.4, 0.4, 3.0, 2.0, C(0.55, 0.2, 0.15), C(0.85, 0.7, 0.35))
    barrel(a, room, -3.4, 2.4)
    stool(a, -2.5, 2.6)
    candle(a, 0.6, 0.8, 0.5)
    # hanging herbs and a pot rack
    for k in range(6):
        p = (-2.0 + k * 0.35, room.h - 0.35, -0.6)
        tube(a, "plant", [p, (p[0], p[1] - 0.45, p[2])], 0.04, C(0.35, 0.45, 0.2), sides=5)
    room.light(0.4, 2.6, 0.4, [1.0, 0.8, 0.5], 1.0, 8.0)
    return room


def shop():
    room = Room("room_shop", 9.0, 7.0, 3.4)
    a = room.asset
    plank_floor(a, room.w, room.d, WOOD_L * 0.8)
    timber_walls(room, windows=((-1, -1.0), (1, -1.0), (0, 0.0)))
    # counter across the room
    with a.at((0, 0, -0.6)):
        box(a, "wood", (0, 0.5, 0), (4.6, 1.0, 0.7), wc(WOOD_D))
        for i in range(10):
            box(a, "wood", (-2.1 + i * 0.47, 0.5, 0.36), (0.42, 0.9, 0.04), wc(WOOD * rng.uniform(0.85, 1.05)))
        box(a, "wood", (0, 1.03, 0), (4.8, 0.07, 0.8), wc(WOOD_L, (6, 1, 1)))
        # scales and goods on the counter
        cylinder(a, "gold", (1.4, 1.06, 0), (1.4, 1.45, 0), 0.02, colour=C(0.8, 0.62, 0.28), sides=6)
        box(a, "gold", (1.4, 1.45, 0), (0.6, 0.02, 0.02), C(0.8, 0.62, 0.28))
        for s in (-1, 1):
            lathe(a, "gold", [(0.0, 1.25), (0.1, 1.27), (0.12, 1.3)], 10, C(0.8, 0.62, 0.28), centre=(1.4 + s * 0.28, 0, 0))
        for k in range(7):
            blob(a, "food", (-1.6 + k * 0.12, 1.12, rng.uniform(-0.15, 0.15)), (0.045, 0.045, 0.045), [C(0.85, 0.15, 0.12), C(0.85, 0.72, 0.25)][k % 2], level=1, noise=0.1)
        lathe(a, "straw", [(0.25, 1.06), (0.3, 1.25), (0.32, 1.28)], 14, C(0.72, 0.56, 0.3), centre=(-1.0, 0, 0))
    room.solid(0, -0.6, 2.4, 0.42, 0.0, 1.1)
    for xx in (-3.2, -1.6, 0.0, 1.6, 3.2):
        shelf(a, room, xx, -room.d / 2 + 0.3, 0.0, 1.5, 2.4, "mixed" if xx else "jars")
    for k, (bx, bz) in enumerate(((-3.8, 1.8), (-3.2, 2.6), (3.6, 2.4), (3.9, 1.5))):
        barrel(a, room, bx, bz, 0.9 + 0.1 * (k % 2))
    # sacks of grain
    for k in range(5):
        rounded_box(a, "cloth", (2.6 + (k % 3) * 0.45, 0.25 + (k // 3) * 0.4, 1.0), (0.4, 0.5, 0.36), C(0.75, 0.64, 0.45), levels=2, lumpy=0.03, seed=k)
    room.solid(3.0, 1.0, 0.8, 0.3, 0.0, 0.9)
    # hanging herbs, garlic, a lantern
    for k in range(10):
        p = (-3.5 + k * 0.7, room.h - 0.3, 0.6)
        tube(a, "plant", [p, (p[0], p[1] - 0.5, p[2])], 0.05, [C(0.4, 0.5, 0.22), C(0.85, 0.82, 0.7), C(0.55, 0.3, 0.4)][k % 3], sides=5)
    rug(a, 0, 1.6, 3.2, 1.6, C(0.2, 0.32, 0.45), C(0.85, 0.75, 0.45))
    room.light(0.0, 2.8, 0.6, [1.0, 0.8, 0.5], 1.1, 9.0)
    return room


def lodge():
    room = Room("room_lodge", 8.5, 6.5, 3.2)
    a = room.asset
    stone_floor(a, room.w, room.d)
    timber_walls(room, windows=((-1, -0.5), (1, -0.5)))
    fireplace(a, room, -2.4, -room.d / 2 + 0.45, 0.0)
    for k in range(2):     # bunks
        x = 3.4
        z = -2.0 + k * 2.3
        bed(a, room, x, z, math.pi / 2, blanket=C(0.4, 0.38, 0.32))
    # weapon rack with swords and axes, an armour stand
    with a.at((1.0, 0, -room.d / 2 + 0.3)):
        box(a, "wood", (0, 1.0, 0), (2.0, 0.12, 0.12), wc(WOOD_D, (6, 1, 1)))
        box(a, "wood", (0, 0.2, 0.1), (2.0, 0.1, 0.25), wc(WOOD_D, (6, 1, 1)))
        for sx in (-1, 1):
            box(a, "wood", (sx * 0.95, 0.75, 0), (0.1, 1.5, 0.12), wc(WOOD_D))
        for k in range(5):
            xx = -0.75 + k * 0.38
            if k % 2:
                box(a, "iron", (xx, 1.05, 0.12), (0.06, 1.2, 0.012), C(0.75, 0.76, 0.78))
                box(a, "gold", (xx, 0.42, 0.12), (0.22, 0.04, 0.05), C(0.7, 0.55, 0.25))
            else:
                cylinder(a, "wood", (xx, 0.25, 0.12), (xx, 1.6, 0.12), 0.025, colour=WOOD, sides=6)
                box(a, "iron", (xx + 0.1, 1.45, 0.12), (0.22, 0.26, 0.02), C(0.6, 0.6, 0.62))
    room.solid(1.0, -room.d / 2 + 0.3, 1.05, 0.25, 0.0, 1.6)
    with a.at((-3.4, 0, 0.6)):
        cylinder(a, "wood", (0, 0, 0), (0, 1.5, 0), 0.04, colour=WOOD_D, sides=6)
        rounded_box(a, "iron", (0, 1.25, 0), (0.48, 0.6, 0.3), C(0.6, 0.6, 0.64), levels=2, seed=4)
        blob(a, "iron", (0, 1.72, 0), (0.14, 0.16, 0.14), C(0.55, 0.56, 0.6), level=2, noise=0.0)
    room.solid(-3.4, 0.6, 0.3, 0.25, 0.0, 1.8)
    table(a, room, -0.6, 0.9, 2.2, 1.0)
    for k in range(4):     # a map, mugs, a candle
        pass
    box(a, "paper", (-0.6, 0.795, 0.9), (1.0, 0.01, 0.7), lambda v: C(0.86, 0.78, 0.6) * (0.85 + 0.2 * vnoise(v * 8, 3))[:, None], shade=False)
    candle(a, -1.4, 0.79, 0.7)
    for xx in (-1.4, 0.2):
        stool(a, xx, 1.75)
        stool(a, xx, 0.05)
    barrel(a, room, 3.6, 2.6)
    barrel(a, room, 3.0, 2.8, 0.8, 0.28)
    rug(a, -0.6, 0.9, 3.2, 2.0, C(0.35, 0.18, 0.12), C(0.6, 0.48, 0.25))
    room.light(-0.6, 2.6, 0.9, [1.0, 0.82, 0.55], 1.0, 8.0)
    return room


def great_hall():
    room = Room("room_hall", 16.0, 30.0, 9.0)
    a = room.asset
    w, d, h = room.w, room.d, room.h
    stone_floor(a, w, d, 1.2, C(0.58, 0.55, 0.5))
    # walls of big ashlar
    for (cx, cz, sx, sz) in ((0, -d / 2, w, 0.8), (-w / 2, 0, 0.8, d), (w / 2, 0, 0.8, d)):
        n_row = int(h / 0.6)
        for r in range(n_row):
            along = sx if sx > 1 else sz
            u = -along / 2 - (0.6 if r % 2 else 0)
            while u < along / 2:
                bw = rng.uniform(0.9, 1.4)
                u0, u1 = max(u, -along / 2), min(u + bw, along / 2)
                if u1 - u0 > 0.2:
                    pos = (cx + (u0 + u1) / 2, r * 0.6 + 0.3, cz) if sx > 1 else (cx, r * 0.6 + 0.3, cz + (u0 + u1) / 2)
                    size = (u1 - u0 - 0.04, 0.56, 0.8) if sx > 1 else (0.8, 0.56, u1 - u0 - 0.04)
                    box(a, "stone", pos, size, lambda v, t=rng.uniform(0.82, 1.08): grain(v, C(0.6, 0.57, 0.52) * t, 0.06, 2, 3), shade=False)
                u += bw
    # entrance wall with a doorway at +Z
    for side in (-1, 1):
        box(a, "stone", (side * (w / 4 + 1.0), h / 2, d / 2), (w / 2 - 2.0, h, 0.8), C(0.58, 0.55, 0.5))
    box(a, "stone", (0, (3.6 + h) / 2, d / 2), (4.0, h - 3.6, 0.8), C(0.58, 0.55, 0.5))
    box(a, "glow_window", (0, 1.8, d / 2 + 0.3), (4.0, 3.6, 0.05), C(0.95, 0.88, 0.7), shade=False)
    # vaulted ceiling beams
    box(a, "wood", (0, h + 0.1, 0), (w, 0.2, d), wc(WOOD_D * 0.8, (1, 1, 6)), shade=False)
    for z in np.arange(-d / 2 + 2, d / 2, 3.0):
        box(a, "wood", (0, h - 0.4, z), (w, 0.5, 0.45), wc(WOOD_D, (6, 1, 1)))
        for side in (-1, 1):
            cylinder(a, "wood", (side * (w / 2 - 0.4), h - 2.5, z), (side * 2.0, h - 0.4, z), 0.16, colour=WOOD_D, sides=6)
    # two rows of pillars with arches
    for side in (-1, 1):
        x = side * 4.5
        for z in np.arange(-d / 2 + 4, d / 2 - 3, 4.0):
            lathe(a, "stone", [(0.75, 0.0), (0.75, 0.3), (0.55, 0.45), (0.5, 0.6), (0.5, h - 2.2), (0.62, h - 1.9), (0.8, h - 1.6), (0.8, h - 1.3), (0.0, h - 1.3)], 16,
                  lambda v: grain(v, C(0.64, 0.61, 0.56), 0.06, 2, 5), centre=(x, 0, z))
            room.solid(x, z, 0.6, 0.6, 0.0, h)
            # banner between pillars
            col = C(0.58, 0.1, 0.12) if int(z) % 8 == 0 else C(0.12, 0.2, 0.45)
            quad(a, "cloth", [(side * (w / 2 - 0.45), h - 2.2, z + 1.4), (side * (w / 2 - 0.45), h - 2.2, z + 2.6),
                              (side * (w / 2 - 0.45), h - 6.0, z + 2.6), (side * (w / 2 - 0.45), h - 6.0, z + 1.4)], col)
            quad(a, "gold", [(side * (w / 2 - 0.47), h - 3.4, z + 1.75), (side * (w / 2 - 0.47), h - 3.4, z + 2.25),
                             (side * (w / 2 - 0.47), h - 4.0, z + 2.25), (side * (w / 2 - 0.47), h - 4.0, z + 1.75)], C(0.9, 0.72, 0.3))
            # wall sconce
            box(a, "iron", (side * (w / 2 - 0.5), 3.0, z), (0.2, 0.3, 0.2), IRON)
            blob(a, "glow_fire", (side * (w / 2 - 0.5), 3.3, z), (0.12, 0.22, 0.12), C(1, 0.6, 0.2), level=1, noise=0.4, seed=z)
            room.light(side * (w / 2 - 1.2), 3.3, z, [1.0, 0.6, 0.28], 1.4, 7.0)
    # red carpet up the middle, to the dais and the throne
    rug(a, 0, 1.0, 3.0, d - 6.0, C(0.6, 0.08, 0.1), C(0.85, 0.66, 0.28))
    for s in range(3):
        box(a, "stone", (0, 0.15 + s * 0.3, -d / 2 + 3.2 - s * 0.6), (7.0 - s * 1.2, 0.3 * (s + 1) * 1.0, 1.0 + 0.0), C(0.62, 0.59, 0.54))
    dais_y = 0.9
    box(a, "stone", (0, dais_y / 2, -d / 2 + 1.6), (6.0, dais_y, 2.6), C(0.6, 0.57, 0.52))
    room.solid(0, -d / 2 + 2.4, 3.5, 2.2, 0.0, 0.9)
    with a.at((0, dais_y, -d / 2 + 1.4)):
        box(a, "gold", (0, 0.45, 0), (1.3, 0.9, 1.0), C(0.75, 0.58, 0.25))
        rounded_box(a, "cloth", (0, 0.95, 0.05), (1.0, 0.16, 0.8), C(0.55, 0.06, 0.1), levels=2, seed=1)
        box(a, "gold", (0, 1.9, -0.45), (1.3, 2.2, 0.2), C(0.75, 0.58, 0.25))
        rounded_box(a, "cloth", (0, 1.8, -0.32), (0.9, 1.5, 0.1), C(0.55, 0.06, 0.1), levels=2, seed=2)
        for side in (-1, 1):
            box(a, "gold", (side * 0.6, 1.25, 0.1), (0.15, 0.25, 0.9), C(0.8, 0.62, 0.28))
            blob(a, "gold", (side * 0.6, 3.05, -0.45), (0.12, 0.12, 0.12), C(0.9, 0.74, 0.32), level=1, noise=0.0)
        for k in range(5):     # crown-like crest
            box(a, "gold", (-0.5 + k * 0.25, 3.1 + (0.15 if k % 2 == 0 else 0), -0.45), (0.12, 0.3, 0.12), C(0.9, 0.74, 0.32))
    # long tables with benches either side of the carpet
    for side in (-1, 1):
        x = side * 2.6
        for z0 in (-4.0, 4.0):
            with a.at((x, 0, z0)):
                for i in range(6):
                    box(a, "wood", (0, 0.76, -3.0 + i + 0.5), (1.0, 0.06, 0.98), wc(WOOD * 0.9, (1, 1, 5)))
                for zz in (-2.7, 0.0, 2.7):
                    box(a, "wood", (0, 0.37, zz), (0.8, 0.74, 0.12), wc(WOOD_D))
                for bx in (-0.8, 0.8):
                    box(a, "wood", (bx, 0.42, 0), (0.35, 0.06, 5.8), wc(WOOD, (1, 1, 6)))
                    for zz in (-2.6, 2.6):
                        box(a, "wood", (bx, 0.2, zz), (0.3, 0.4, 0.08), wc(WOOD_D))
                for k in range(6):
                    lathe(a, "gold", [(0.0, 0.79), (0.035, 0.8), (0.03, 0.85), (0.05, 0.95)], 10, C(0.8, 0.64, 0.3), centre=(0.2 * (1 if k % 2 else -1), 0, -2.5 + k))
                candle(a, 0, 0.79, -1.0)
                candle(a, 0, 0.79, 1.0)
                blob(a, "food", (0, 0.86, 0.0), (0.22, 0.1, 0.16), C(0.62, 0.36, 0.18), level=2, noise=0.15)
            room.solid(x, z0, 1.1, 3.0, 0.0, 0.8)
    # chandeliers
    for z in (-8.0, 0.0, 8.0):
        y = h - 2.6
        tube(a, "iron", [(0, h, z), (0, y + 0.3, z)], 0.03, IRON, sides=5, cap=False)
        ring = [(math.cos(t) * 1.2, y, z + math.sin(t) * 1.2) for t in np.linspace(0, 2 * math.pi, 25)]
        tube(a, "iron", ring, 0.05, IRON, sides=6, cap=False)
        for k in range(8):
            t = k / 8 * 2 * math.pi
            candle(a, math.cos(t) * 1.2, y + 0.05, z + math.sin(t) * 1.2)
            cylinder(a, "iron", (0, y + 0.3, z), (math.cos(t) * 1.2, y, z + math.sin(t) * 1.2), 0.015, colour=IRON, sides=4, caps=False)
        room.light(0, y - 0.3, z, [1.0, 0.78, 0.45], 1.6, 12.0)
    # tall windows high on the side walls
    for side in (-1, 1):
        for z in np.arange(-d / 2 + 6, d / 2 - 3, 8.0):
            with a.at((side * (w / 2 - 0.42), 6.2, z), yaw=-side * math.pi / 2):
                box(a, "glow_window", (0, 0, 0), (1.4, 2.6, 0.05), C(0.85, 0.9, 1.0), shade=False)
                for k in (-0.35, 0.35):
                    box(a, "iron", (k, 0, 0.04), (0.03, 2.6, 0.03), IRON)
                for k in (-0.6, 0.0, 0.6):
                    box(a, "iron", (0, k, 0.04), (1.4, 0.03, 0.03), IRON)
    return room


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = {}
    for build in (home, shop, lodge, great_hall):
        room = build()
        room.asset.write(OUT / f"{room.name}.glb", MATS)
        info[room.name] = {"w": room.w, "d": room.d, "h": room.h, "exit": {"x": 0.0, "z": room.d / 2 - 0.6},
                           "spawn": {"x": 0.0, "z": room.d / 2 - 1.4, "heading": math.pi}, "colliders": room.colliders, "lights": room.lights}
    (OUT / "interiors.json").write_text(json.dumps(info, indent=1), encoding="utf-8")


main()
