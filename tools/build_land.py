"""The land around the village: height field, coastline, hills, roads, ground layers and where every
tree, rock and plant goes.

World axes: x east, z south, y up (metres). The village keeps its own terrain inside |x|, |z| <= 64;
outside it this generates
  heights   513 x 513 at 1 m over +-256 m (int16, 1 cm), equal to the village terrain inside the square
  far       129 x 129 at 20 m over +-1280 m, for the horizon hills
  splat     512 x 512 RGBA at 1 m: R meadow grass, G forest floor, B dirt road, A rock (sand comes from height)
  scatter   per prop: [x, y, z, yaw, scale] lists
Writes assets_src/hq/land.json and assets_src/hq/land_splat.png (+ a preview land_map.png).
Usage: python tools/build_land.py
"""
import base64
import json
import math
import pathlib
import sys

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import noise3 as N  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "assets_src" / "hq"
rng = np.random.default_rng(2024)

# ---------------------------------------------------------------- the village's own terrain
V = json.loads((ROOT / "raw" / "live" / "village.json").read_text(encoding="utf-8"))
VH = V["height"]
vh = np.frombuffer(base64.b64decode(VH["int16_le_base64"]), dtype="<i2").astype(np.float64).reshape(VH["size"], VH["size"]) * VH["unit"]
WATER = V["water_level"]


def village_h(x, z):
    fx = np.clip((x + VH["half"]) / VH["step"], 0, VH["size"] - 1.001)
    fz = np.clip((z + VH["half"]) / VH["step"], 0, VH["size"] - 1.001)
    ix, iz = np.floor(fx).astype(int), np.floor(fz).astype(int)
    tx, tz = fx - ix, fz - iz
    a = vh[iz, ix] * (1 - tx) + vh[iz, ix + 1] * tx
    b = vh[iz + 1, ix] * (1 - tx) + vh[iz + 1, ix + 1] * tx
    return a * (1 - tz) + b * tz


# ---------------------------------------------------------------- coast, roads
COAST = np.array([(-900, -330), (-420, -190), (-230, -135), (-130, -100), (-80, -78), (-62, -66),
                  (-40, -58), (20, -40), (58, 10), (60, 66), (78, 104), (112, 190), (150, 420), (220, 900)], dtype=float)
ROADS = [np.array([(0, 26), (2, 60), (-6, 100), (-30, 150), (-80, 190), (-150, 215)], dtype=float),     # south road into the hills
         np.array([(-30, 4), (-70, 0), (-110, -20), (-140, -55), (-150, -104)], dtype=float),           # west road to the bridge (castle)
         np.array([(-6, 100), (40, 130), (70, 170)], dtype=float)]                                       # a lane to the southern shore
CASTLE = {"x": -150.0, "z": -175.0, "island_r": 30.0, "top": 4.0}


def seg_dist(px, pz, poly):
    """Distance to a polyline, and the side (+1 right of travel, -1 left)."""
    best = np.full(px.shape, 1e9)
    side = np.ones(px.shape)
    for (ax, az), (bx, bz) in zip(poly[:-1], poly[1:]):
        dx, dz = bx - ax, bz - az
        L2 = dx * dx + dz * dz
        t = np.clip(((px - ax) * dx + (pz - az) * dz) / L2, 0, 1)
        qx, qz = ax + t * dx, az + t * dz
        d = np.hypot(px - qx, pz - qz)
        cr = dx * (pz - az) - dz * (px - ax)
        closer = d < best
        best = np.where(closer, d, best)
        side = np.where(closer, np.sign(cr + 1e-9), side)
    return best, side


def land_height(x, z):
    """Procedural heights for the land and sea outside the village."""
    d, side = seg_dist(x, z, COAST)
    s = d * side                        # inland distance (positive on land: south-west of the coast)
    p = np.stack([x * 0.01, z * 0.01, np.zeros_like(x)], axis=-1).reshape(-1, 3)
    f1 = N.fbm(p * 1.0, 5, 3).reshape(x.shape)
    f2 = N.fbm(p * 3.0, 4, 7).reshape(x.shape)
    ridge = N.ridged(p * 0.6, 4, 11).reshape(x.shape)
    west = np.clip((-x - 60) / 220, 0, 1)
    south = np.clip((z - 60) / 260, 0, 1)
    hills = 26 * west ** 1.3 * (0.55 + 0.45 * f1) + 9 * south * (0.5 + 0.5 * f2) + 4 * f2
    far = np.clip((np.hypot(x, z) - 280) / 500, 0, 1)
    hills = hills + far * (40 + 70 * ridge)
    coast = np.clip(s / 30, 0, 1)
    land = 0.5 + 2.2 * coast ** 0.7 + hills * np.clip((s - 4) / 60, 0, 1) ** 1.2
    sea = WATER - 0.6 - 9 * np.clip(-s / 70, 0, 1) - 0.8 * f2
    t = np.clip((s + 3) / 6, 0, 1)
    t = t * t * (3 - 2 * t)
    h = sea * (1 - t) + land * t
    # roads: shallow cuts that follow the ground
    for r in ROADS:
        rd, _ = seg_dist(x, z, r)
        h = h - 0.25 * np.clip(1 - rd / 3.5, 0, 1) * (h > 0.3)
    # castle island
    ci = np.hypot(x - CASTLE["x"], z - CASTLE["z"])
    isl = np.clip(1 - (ci - CASTLE["island_r"]) / 14, 0, 1)
    h = np.maximum(h, (CASTLE["top"] + 0.6 * f2 - 1.2) * isl ** 0.6 + h * (1 - isl ** 0.6) if True else h)
    return h, s


def blended_height(x, z):
    """Village terrain inside the square, procedural outside, blended over a 22 m band."""
    h, s = land_height(x, z)
    inside = (np.abs(x) <= 64) & (np.abs(z) <= 64)
    cx, cz = np.clip(x, -64, 64), np.clip(z, -64, 64)
    edge = village_h(cx, cz)
    dist = np.maximum(np.abs(x) - 64, np.abs(z) - 64)
    w = np.clip(dist / 22, 0, 1)
    w = w * w * (3 - 2 * w)
    out = edge * (1 - w) + h * w
    return np.where(inside, village_h(x, z), out), s


def main():
    # ---- near heights, 1 m
    g = np.arange(-256, 257, 1.0)
    X, Z = np.meshgrid(g, g)
    H, S = blended_height(X, Z)
    gy, gx = np.gradient(H, 1.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    # ---- far heights, 20 m
    gf = np.arange(-1280, 1281, 20.0)
    XF, ZF = np.meshgrid(gf, gf)
    HF, _ = blended_height(XF, ZF)

    # ---- ground layers (512 x 512 at 1 m, texel centres)
    c = np.arange(-255.5, 256, 1.0)
    CX, CZ = np.meshgrid(c, c)
    ch, cs = blended_height(CX, CZ)
    cgy, cgx = np.gradient(ch, 1.0)
    cslope = np.degrees(np.arctan(np.hypot(cgx, cgy)))
    p = np.stack([CX * 0.02, CZ * 0.02, np.zeros_like(CX)], axis=-1).reshape(-1, 3)
    forest_n = N.fbm(p, 4, 31).reshape(CX.shape)
    west = np.clip((-CX - 70) / 120, 0, 1)
    south = np.clip((CZ - 90) / 120, 0, 1)
    forest = np.clip((forest_n * 0.5 + 0.5) - 0.62 + 0.5 * west + 0.25 * south, 0, 1)
    forest = np.clip(forest * 2.2, 0, 1) * (cs > 10)
    vill = np.hypot(CX, CZ - 3) < 38
    forest[vill] = 0
    road = np.zeros_like(CX)
    for r in ROADS:
        rd, _ = seg_dist(CX, CZ, r)
        road = np.maximum(road, np.clip(1 - (rd - 1.2) / 1.6, 0, 1))
    rock = np.clip((cslope - 26) / 12, 0, 1) + 0.6 * np.clip((N.fbm(p * 3, 4, 41).reshape(CX.shape) - 0.35) * 3, 0, 1) * np.clip((cslope - 12) / 10, 0, 1)
    rock = np.clip(rock, 0, 1) * (cs > 2)
    grass = np.ones_like(CX)
    layers = np.stack([grass, forest, road, rock], axis=-1)
    layers[..., 0] = np.clip(1 - forest - road - rock, 0, 1)
    layers[..., 1] = np.clip(forest * (1 - road) * (1 - rock), 0, 1)
    layers[..., 2] = road * (1 - rock)
    tot = layers.sum(axis=-1, keepdims=True) + 1e-6
    layers = layers / tot
    # RGB only (meadow, forest, road; rock = the remainder): browsers lose colour wherever alpha is 0
    Image.fromarray((layers[..., :3] * 255).round().astype(np.uint8), "RGB").save(OUT / "land_splat.png")

    def at(arr, x, z):
        ix = np.clip(np.round(x + 255.5).astype(int), 0, 511)
        iz = np.clip(np.round(z + 255.5).astype(int), 0, 511)
        return arr[iz, ix]

    # ---- scatter
    scatter = {}

    def put(name, x, z, yaw=None, s=1.0):
        y = float(blended_height(np.array([x]), np.array([z]))[0][0])
        scatter.setdefault(name, []).append([round(x, 2), round(y, 2), round(z, 2), round(rng.uniform(0, 6.283) if yaw is None else yaw, 3), round(s, 3)])

    def ok(x, z, clear=40.0):
        if abs(x) > 250 or abs(z) > 250:
            return False
        if math.hypot(x, z - 3) < clear:
            return False
        if math.hypot(x - CASTLE["x"], z - CASTLE["z"]) < CASTLE["island_r"] + 10:
            return False
        sx = at(cs, np.array([x]), np.array([z]))[0]
        if sx < 6:
            return False
        rd = min(seg_dist(np.array([x]), np.array([z]), r)[0][0] for r in ROADS)
        return rd > 4.0

    # trees on a jittered 7 m grid, by forest density; firs uphill, oaks and ashes lower, lone oaks in meadows
    for gx_ in np.arange(-250, 250, 7.0):
        for gz_ in np.arange(-250, 250, 7.0):
            x, z = gx_ + rng.uniform(-3, 3), gz_ + rng.uniform(-3, 3)
            if not ok(x, z, 44):
                continue
            f = at(forest, np.array([x]), np.array([z]))[0]
            hh = float(blended_height(np.array([x]), np.array([z]))[0][0])
            sl = at(cslope, np.array([x]), np.array([z]))[0]
            if sl > 30:
                continue
            if rng.random() < f * 0.85:
                if hh > 13 or (x < -150 and rng.random() < 0.6):
                    put(rng.choice(["tree_fir", "tree_fir_b"]), x, z, s=rng.uniform(0.8, 1.15))
                else:
                    put(rng.choice(["tree_oak", "tree_oak_b", "tree_ash", "tree_ash"]), x, z, s=rng.uniform(0.8, 1.15))
            elif f < 0.05 and rng.random() < 0.035:
                put(rng.choice(["tree_oak", "tree_oak_b"]), x, z, s=rng.uniform(0.9, 1.2))
    # the castle island's rim: scanned cliff faces all round, except where the bridge lands
    for k in range(34):
        a = k / 34 * 2 * math.pi + rng.uniform(-0.05, 0.05)
        if abs(math.atan2(math.sin(a - math.pi / 2), math.cos(a - math.pi / 2))) < 0.22:
            continue
        r = CASTLE["island_r"] + rng.uniform(-1.5, 3.0)
        x, z = CASTLE["x"] + math.cos(a) * r, CASTLE["z"] + math.sin(a) * r
        put(rng.choice(["rock_face_01", "rock_face_02", "rock_face_01"]), x, z, yaw=-a + math.pi / 2 + rng.normal(0, 0.3), s=rng.uniform(1.6, 2.8))
        if rng.random() < 0.6:
            a2 = a + rng.uniform(-0.08, 0.08)
            r2 = r + rng.uniform(3, 7)
            put(rng.choice(["boulder_01", "namaqualand_boulder_02"]), CASTLE["x"] + math.cos(a2) * r2, CASTLE["z"] + math.sin(a2) * r2, s=rng.uniform(1.0, 2.2))
    # rocks: boulders in the open and on slopes, cliffs on steep ground, mossy sets in the woods
    rock_names = ["boulder_01", "namaqualand_boulder_02"] + [f"rock_moss_set_01_{i}" for i in range(6)] + [f"rock_moss_set_02_{i}" for i in range(7)]
    for _ in range(9000):
        x, z = rng.uniform(-250, 250), rng.uniform(-250, 250)
        if not ok(x, z, 36):
            continue
        sl = at(cslope, np.array([x]), np.array([z]))[0]
        f = at(forest, np.array([x]), np.array([z]))[0]
        u = rng.random()
        if sl > 28 and u < 0.25:
            put(rng.choice(["rock_face_01", "rock_face_02"]), x, z, s=rng.uniform(0.8, 1.6))
        elif u < 0.02 + 0.05 * f + 0.04 * (sl > 15):
            put(rng.choice(rock_names), x, z, s=rng.uniform(0.5, 1.3))
        elif u < 0.08 and sl < 20:
            put(rng.choice(["rock_07", "rock_09", "stone_01"]), x, z, s=rng.uniform(1.5, 4.0))
    # forest floor: stumps, logs, roots, fallen branches, ferns, shrubs, nettles
    floor = [("tree_stump_01", 0.004), ("tree_stump_02", 0.004), ("dead_tree_trunk_02", 0.003), ("root_cluster_01", 0.002),
             ("dry_branches_medium_01_0", 0.01), ("dry_branches_medium_01_1", 0.01), ("dry_branches_medium_01_2", 0.01)]
    plants = [(f"fern_02_{i}", 0.05) for i in range(4)] + [(f"shrub_02_{i}", 0.01) for i in range(4)] + [(f"shrub_03_{i}", 0.015) for i in range(4)] + \
             [("shrub_01", 0.004), ("shrub_04", 0.008)] + [(f"nettle_plant_{i}", 0.01) for i in range(6)]
    for _ in range(60000):
        x, z = rng.uniform(-250, 250), rng.uniform(-250, 250)
        if not ok(x, z, 36):
            continue
        f = at(forest, np.array([x]), np.array([z]))[0]
        sl = at(cslope, np.array([x]), np.array([z]))[0]
        if sl > 30:
            continue
        u = rng.random()
        acc = 0.0
        for name, p_ in floor + plants:
            acc += p_ * (f if name not in ("shrub_04",) else max(f, 0.3))
            if u < acc * 0.6:
                put(name, x, z, s=rng.uniform(0.8, 1.3))
                break
    # meadow flowers
    for _ in range(25000):
        x, z = rng.uniform(-250, 250), rng.uniform(-250, 250)
        if not ok(x, z, 34):
            continue
        g_ = at(layers[..., 0], np.array([x]), np.array([z]))[0]
        if rng.random() < 0.05 * g_:
            put(f"dandelion_01_{rng.integers(5)}", x, z, s=rng.uniform(0.8, 1.2))

    data = {
        "near": {"size": 513, "step": 1.0, "half": 256.0, "unit": 0.01,
                 "int16_le_base64": base64.b64encode(np.round(H / 0.01).astype("<i2").tobytes()).decode("ascii")},
        "far": {"size": len(gf), "step": 20.0, "half": 1280.0, "unit": 0.05,
                "int16_le_base64": base64.b64encode(np.round(HF / 0.05).astype("<i2").tobytes()).decode("ascii")},
        "splat": {"size": 512, "half": 256.0},
        "water": WATER, "castle": CASTLE, "roads": [r.tolist() for r in ROADS], "scatter": scatter,
    }
    (OUT / "land.json").write_text(json.dumps(data, separators=(",", ":")))
    # preview map
    img = np.zeros((512, 512, 3))
    img[..., 0] = layers[..., 2] * 0.8 + layers[..., 3] * 0.6
    img[..., 1] = layers[..., 0] * 0.7 + layers[..., 1] * 0.35 + layers[..., 3] * 0.6
    img[..., 2] = layers[..., 3] * 0.6
    img *= (0.6 + 0.4 * np.clip(ch / 30, 0, 1))[..., None]
    img[ch < WATER] = (0.1, 0.25, 0.5)
    for name, pts in scatter.items():
        col = (1, 1, 1) if name.startswith("tree") else (0.9, 0.5, 0.2)
        if not (name.startswith("tree") or "boulder" in name or "rock_face" in name):
            continue
        for x, y, z, _, _ in pts:
            ix, iz = int(x + 255.5), int(z + 255.5)
            if 0 <= ix < 512 and 0 <= iz < 512:
                img[iz, ix] = col
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((1024, 1024), Image.NEAREST).save(OUT / "land_map.png")
    print({k: len(v) for k, v in scatter.items()})
    print("heights", H.min(), H.max(), "json", round((OUT / "land.json").stat().st_size / 1e6, 2), "MB")


main()
