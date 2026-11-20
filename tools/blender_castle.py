"""Model the castle across the water: its island, the causeway bridge, the curtain wall with
round towers and a gatehouse, the courtyard and the keep - all masonry laid block by block.

Local axes: glTF, origin at the island's centre at sea level, the gate faces +Z (toward the
village). The island's walkable top is at y = TOP.

Writes assets_src/castle/
    island.glb  castle.glb  bridge.glb      models
    castle.json                             colliders, keep door, walkable floor height, gate
Run: blender --background --factory-startup --python tools/blender_castle.py
"""
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geo  # noqa: E402
from geo import C, box, cylinder, lathe, blob, tube, quad, grain, fbm, vnoise, sstep, unit, rot_y, rot_x, rot_z  # noqa: E402,F401

OUT = HERE.parent / "assets_src" / "castle"
MATS = {"stone": {"rough": 0.9}, "rock": {"rough": 0.95}, "slate": {"rough": 0.6}, "wood": {"rough": 0.82},
        "iron": {"rough": 0.45, "metal": 0.6}, "cloth": {"rough": 0.95, "double": True}, "grass": {"rough": 0.95},
        "glow_window": {"rough": 0.3, "emissive": (1.0, 0.70, 0.36), "strength": 1.4},
        "glow_lamp": {"rough": 0.4, "emissive": (1.0, 0.72, 0.34), "strength": 1.6}, "gold": {"rough": 0.35, "metal": 0.8}}
TOP = 4.0
ISLAND_R = 30.0
WALL_R = 19.0
WALL_H = 7.5
WALL_T = 2.2
TOWER_R = 3.4
TOWER_H = 12.5
rng = np.random.default_rng(1234)
colliders = []
STONE_A, STONE_B = C(0.66, 0.63, 0.57), C(0.50, 0.48, 0.45)


def save(asset):
    asset.write(OUT / f"{asset.name}.glb", MATS)


def collide(x, z, yaw, hx, hz, y0, y1):
    colliders.append({"x": round(float(x), 3), "z": round(float(z), 3), "yaw": round(float(yaw), 4), "hx": round(float(hx), 3),
                      "hz": round(float(hz), 3), "y0": round(float(y0), 3), "y1": round(float(y1), 3), "a": 1})


# ------------------------------------------------------------------ masonry
def block_colour(seed, base_a=STONE_A, base_b=STONE_B, moss=0.0, height=None):
    r = np.random.default_rng(seed)
    t = r.random()
    base = base_a * (1 - t) + base_b * t
    base = base * r.uniform(0.86, 1.08)

    def fn(v):
        c = grain(v, base, 0.07, 4.0, seed % 97)
        if moss and height is not None:
            low = 1 - sstep(height[0], height[1], v[:, 1])
            m = np.clip(low * moss * (0.4 + 0.8 * vnoise(v * 1.5, seed % 31)), 0, 1)
            c = c * (1 - m[:, None]) + C(0.30, 0.40, 0.20) * m[:, None]
        return c
    return fn


def masonry_flat(asset, start, end, y0, y1, thickness, outward, course=0.52, unit_len=0.95, moss=0.0, skip=None):
    """A straight wall of blocks between two ground points. Each block is its own slightly proud box,
    so mortar lines read as shadows. skip(u, y) -> True leaves a hole (doors, slits)."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = np.linalg.norm(end - start)
    d = (end - start) / length
    yaw = math.atan2(d[0], d[2]) - math.pi / 2
    rot = rot_y(yaw)
    n_course = max(1, int(round((y1 - y0) / course)))
    h = (y1 - y0) / n_course
    for row in range(n_course):
        y = y0 + row * h
        u = -(unit_len * 0.5 if row % 2 else 0.0)
        while u < length:
            w = unit_len * rng.uniform(0.75, 1.25)
            ua, ub = max(u, 0), min(u + w, length)
            if ub - ua > 0.15 and not (skip and skip((ua + ub) / 2, y + h / 2)):
                mid = start + d * (ua + ub) / 2
                proud = rng.uniform(-0.03, 0.04)
                seed = int(rng.integers(0, 1 << 30))
                box(asset, "stone", (mid[0], y + h / 2, mid[2]), (ub - ua - 0.05, h - 0.05, thickness + proud * 2), block_colour(seed, moss=moss, height=(y0, y0 + 2.5)),
                    rot=rot, jitter=0.025, seed=seed % 97, shade=False)
            u += w
    # a dark mortar core so gaps never show the sky
    mid = (start + end) / 2
    box(asset, "stone", (mid[0], (y0 + y1) / 2, mid[2]), (length, y1 - y0, thickness - 0.1), C(0.22, 0.21, 0.2), rot=rot, shade=False)
    return d, yaw


def crenellations(asset, start, end, y, thickness, outward_sign=1.0, merlon=0.9, gap=0.7, height=1.1):
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = np.linalg.norm(end - start)
    d = (end - start) / length
    yaw = math.atan2(d[0], d[2]) - math.pi / 2
    n = np.array([-d[2], 0, d[0]]) * outward_sign
    u = gap / 2
    while u + merlon <= length:
        mid = start + d * (u + merlon / 2) + n * (thickness * 0.3)
        seed = int(rng.integers(0, 1 << 30))
        box(asset, "stone", (mid[0], y + height / 2, mid[2]), (merlon, height, thickness * 0.42), block_colour(seed), rot=rot_y(yaw), jitter=0.02, seed=seed % 97)
        box(asset, "stone", (mid[0], y + height + 0.06, mid[2]), (merlon + 0.08, 0.12, thickness * 0.48), block_colour(seed + 1, C(0.6, 0.58, 0.54), C(0.5, 0.48, 0.45)),
            rot=rot_y(yaw), shade=False)
        u += merlon + gap
    # walkway slab
    mid = (start + end) / 2
    box(asset, "stone", (mid[0], y - 0.08, mid[2]), (length, 0.16, thickness), C(0.46, 0.44, 0.41), rot=rot_y(yaw), shade=False)


def round_tower(asset, centre, radius, y0, height, roof=True, banner_col=None, windows=True):
    cx, cz = centre
    course = 0.55
    n_course = int(round(height / course))
    circumference = 2 * math.pi * radius
    n_blocks = max(8, int(circumference / 1.0))
    for row in range(n_course):
        y = y0 + row * course
        off = 0.5 if row % 2 else 0.0
        for i in range(n_blocks):
            a0 = (i + off) / n_blocks * 2 * math.pi
            a1 = (i + 1 + off) / n_blocks * 2 * math.pi
            am = (a0 + a1) / 2
            if windows and row in (int(n_course * 0.45), int(n_course * 0.45) + 1, int(n_course * 0.75), int(n_course * 0.75) + 1) and i % (n_blocks // 4) == 0:
                continue      # arrow slits
            seed = int(rng.integers(0, 1 << 30))
            r_out = radius + rng.uniform(-0.02, 0.05)
            pos = (cx + math.cos(am) * (r_out - 0.35), y + course / 2, cz + math.sin(am) * (r_out - 0.35))
            box(asset, "stone", pos, (0.7, course - 0.05, 2 * r_out * math.sin((a1 - a0) / 2) - 0.05),
                block_colour(seed, moss=0.5, height=(y0, y0 + 3)), rot=rot_y(-am), jitter=0.02, seed=seed % 97, shade=False)
    cylinder(asset, "stone", (cx, y0, cz), (cx, y0 + height, cz), radius - 0.55, colour=C(0.2, 0.19, 0.18), sides=16, caps=True)
    if windows:   # glow in the slits
        for frac in (0.45, 0.75):
            for k in range(4):
                a = (k * (n_blocks // 4) + 0.5) / n_blocks * 2 * math.pi
                y = y0 + int(n_course * frac) * course + course
                box(asset, "glow_window", (cx + math.cos(a) * (radius - 0.5), y, cz + math.sin(a) * (radius - 0.5)), (0.3, course * 1.7, 0.18),
                    C(1.0, 0.75, 0.4), rot=rot_y(-a), shade=False)
    top = y0 + height
    # corbelled parapet
    lathe(asset, "stone", [(radius - 0.1, top - 0.6), (radius + 0.35, top - 0.1), (radius + 0.35, top + 0.1), (radius - 0.6, top + 0.1)], 24,
          lambda v: grain(v, C(0.56, 0.54, 0.5), 0.1, 3, 5), centre=(cx, 0, cz))
    n_m = max(6, int(2 * math.pi * radius / 1.5))
    for i in range(n_m):
        a = i / n_m * 2 * math.pi
        seed = int(rng.integers(0, 1 << 30))
        box(asset, "stone", (cx + math.cos(a) * (radius + 0.15), top + 0.65, cz + math.sin(a) * (radius + 0.15)), (0.45, 1.0, 2 * math.pi * radius / n_m * 0.55),
            block_colour(seed), rot=rot_y(-a), jitter=0.02, seed=seed % 97)
    if roof:
        rr = radius + 0.5
        prof = [(rr, top + 0.4), (rr * 0.82, top + 1.6), (rr * 0.5, top + 3.6), (rr * 0.2, top + 5.4), (0.05, top + 6.4)]

        def slate(v):
            rings = np.floor((v[:, 1] - top) / 0.28)
            tone = 0.85 + 0.25 * vnoise(np.stack([rings, np.arctan2(v[:, 2] - cz, v[:, 0] - cx) * 6, rings], axis=1), 3)
            return C(0.24, 0.27, 0.34)[None] * tone[:, None]
        lathe(asset, "slate", prof, 28, slate, centre=(cx, 0, cz))
        cylinder(asset, "gold", (cx, top + 6.3, cz), (cx, top + 7.3, cz), 0.05, colour=C(0.85, 0.68, 0.3), sides=6)
        blob(asset, "gold", (cx, top + 6.5, cz), (0.14, 0.14, 0.14), C(0.9, 0.72, 0.32), level=1, noise=0.0)
        if banner_col is not None:
            pts = [(cx, top + 7.2, cz), (cx + 1.3, top + 7.0, cz + 0.15), (cx, top + 6.6, cz)]
            asset.add("cloth", np.array(pts), banner_col, [(0, 1, 2)], normals=np.tile([0, 0, 1.0], (3, 1)))
    return top


def arch_window(asset, centre, yaw, width=1.0, height=1.8):
    with asset.at(centre, yaw=yaw):
        box(asset, "glow_window", (0, 0, 0), (width, height, 0.2), C(1.0, 0.76, 0.42), shade=False)
        for k in range(7):
            a = k / 6 * math.pi
            box(asset, "stone", (math.cos(a) * width * 0.6, height / 2 + math.sin(a) * width * 0.55, 0.05), (0.28, 0.24, 0.32),
                block_colour(int(rng.integers(0, 1 << 30)), C(0.62, 0.6, 0.55), C(0.55, 0.53, 0.5)), rot=rot_z(a - math.pi / 2))
        box(asset, "stone", (0, -height / 2 - 0.08, 0.08), (width + 0.4, 0.16, 0.4), C(0.6, 0.58, 0.54))
        box(asset, "iron", (0, 0, 0.11), (0.04, height, 0.03), C(0.2, 0.2, 0.22))
        box(asset, "iron", (0, height * 0.1, 0.11), (width, 0.04, 0.03), C(0.2, 0.2, 0.22))


# ------------------------------------------------------------------ the island
def island():
    a = geo.Asset("island")
    seg, rings = 72, 9
    verts, faces, cols = [], [], []
    base_noise = lambda ang, s: 1 + 0.12 * math.sin(ang * 3 + 1) + 0.08 * math.sin(ang * 7 + s) + 0.05 * math.sin(ang * 13)
    profile = [(-8.0, 1.25), (-2.5, 1.14), (-1.0, 1.08), (0.5, 1.03), (1.8, 1.0), (2.8, 0.98), (3.5, 0.985), (TOP - 0.15, 0.97), (TOP, 0.94)]
    for j, (y, rf) in enumerate(profile):
        for i in range(seg):
            ang = i / seg * 2 * math.pi
            r = ISLAND_R * rf * base_noise(ang, j * 0.0) * (1 + 0.06 * (vnoise(np.array([ang * 4, y * 0.8, 0.0]), 5) - 0.5))
            verts.append((math.cos(ang) * r, y + (0.3 * (vnoise(np.array([ang * 9, j, 1.0]), 2) - 0.5) if 0 < j < len(profile) - 1 else 0), math.sin(ang) * r))
    for j in range(len(profile) - 1):
        for i in range(seg):
            p0, p1 = j * seg + i, j * seg + (i + 1) % seg
            faces += [(p0, p1, p1 + seg), (p0, p1 + seg, p0 + seg)]
    verts = np.array(verts)
    faces = geo._orient(verts, faces, None, np.array([0, -2, 0]))
    tri = verts[faces]
    fn = unit(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
    flat = tri.reshape(-1, 3)
    seed = 7.0
    tone = vnoise(flat * 0.35, seed)
    c = C(0.36, 0.35, 0.34)[None] + (C(0.58, 0.55, 0.50) - C(0.36, 0.35, 0.34))[None] * tone[:, None]
    wet = 1 - sstep(-1.9, 0.2, flat[:, 1])
    c = c * (1 - 0.45 * wet[:, None])
    green = sstep(TOP - 1.0, TOP - 0.2, flat[:, 1]) * 0.7
    c = c * (1 - green[:, None]) + C(0.30, 0.42, 0.18) * green[:, None]
    a.add("rock", flat, c, np.arange(len(flat)).reshape(-1, 3), np.repeat(fn, 3, axis=0))
    # the top: a polar grass cap whose rim is exactly the cliff's top ring, with a worn path
    top_ring = verts[(len(profile) - 1) * seg:(len(profile)) * seg]
    rings_n = 14
    p = [np.array([0, TOP + 0.02, 0])]
    for k in range(1, rings_n + 1):
        f = k / rings_n
        for i in range(seg):
            q = top_ring[i] * f
            p.append(np.array([q[0], TOP + 0.02 + (0.0 if k == rings_n else 0.04 * vnoise(q * 0.5, 2)), q[2]]))
    p = np.array(p)
    fcs = [(0, 1 + (i + 1) % seg, 1 + i) for i in range(seg)]
    for k in range(rings_n - 1):
        for i in range(seg):
            a0, a1 = 1 + k * seg + i, 1 + k * seg + (i + 1) % seg
            fcs += [(a0, a1 + seg, a1), (a0, a0 + seg, a1 + seg)]
    fcs = geo._orient(p, fcs, np.tile([0, 1.0, 0], (len(p), 1)), None)
    col = C(0.30, 0.44, 0.17)[None] * (0.85 + 0.3 * vnoise(p * 0.4, 3)[:, None])
    path = (np.abs(p[:, 0]) < 2.2) & (p[:, 2] > 0)
    col[path] = C(0.48, 0.40, 0.28)
    a.add("grass", p, col, fcs, np.tile([0, 1.0, 0], (len(p), 1)))
    # rocks around the waterline
    for k in range(36):
        t = k / 36 * 2 * math.pi + rng.uniform(-0.1, 0.1)
        r = ISLAND_R * base_noise(t, 0) * rng.uniform(1.05, 1.25)
        s = rng.uniform(1.2, 3.2)
        blob(a, "rock", (math.cos(t) * r, -1.2, math.sin(t) * r), (s, s * 0.8, s * 0.9), lambda u, pp: grain(pp, C(0.42, 0.40, 0.38), 0.15, 0.6, 3),
             level=2, noise=0.3, seed=k, flat=True)
    save(a)


# ------------------------------------------------------------------ the castle
def castle():
    a = geo.Asset("castle")
    sides = 7
    corners = []
    for i in range(sides):
        ang = math.pi / 2 + (i + 0.5) / sides * 2 * math.pi       # one side faces +Z (the gate)
        corners.append(np.array([math.cos(ang) * WALL_R, 0, math.sin(ang) * WALL_R]))
    banner = C(0.62, 0.10, 0.12)
    gate_side = None
    for i in range(sides):
        p0, p1 = corners[i], corners[(i + 1) % sides]
        mid = (p0 + p1) / 2
        is_gate = mid[2] > WALL_R * 0.85
        # pull the wall ends in so they meet the towers
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        s0, s1 = p0 + d * (TOWER_R - 0.6), p1 - d * (TOWER_R - 0.6)
        if is_gate:
            gate_side = (s0, s1)
            gw = 4.6

            def hole(u, y, gw=gw, length=np.linalg.norm(s1 - s0)):
                return abs(u - length / 2) < gw / 2 and y < TOP + 5.0
            masonry_flat(a, s0, s1, TOP, TOP + WALL_H, WALL_T, 1, moss=0.6, skip=hole)
        else:
            masonry_flat(a, s0, s1, TOP, TOP + WALL_H, WALL_T, 1, moss=0.6)
        crenellations(a, s0, s1, TOP + WALL_H, WALL_T, outward_sign=-1)
        yaw = math.atan2(d[0], d[2]) - math.pi / 2
        length = np.linalg.norm(s1 - s0)
        if is_gate:
            gl = (length - 4.6) / 2
            for k, sgn in ((0, 1), (1, -1)):
                c = (s0 + d * gl / 2) if k == 0 else (s1 - d * gl / 2)
                collide(c[0], c[2], -yaw, gl / 2, WALL_T / 2, TOP - 1, TOP + WALL_H + 1)
            top_c = (s0 + s1) / 2
            collide(top_c[0], top_c[2], -yaw, 2.4, WALL_T / 2, TOP + 5.0, TOP + WALL_H + 1)
        else:
            c = (s0 + s1) / 2
            collide(c[0], c[2], -yaw, length / 2, WALL_T / 2, TOP - 1, TOP + WALL_H + 1)
    for i, p in enumerate(corners):
        round_tower(a, (p[0], p[2]), TOWER_R, TOP, TOWER_H, banner_col=banner if i % 2 == 0 else C(0.12, 0.2, 0.45))
        collide(p[0], p[2], 0.0, TOWER_R * 0.8, TOWER_R * 0.8, TOP - 1, TOP + TOWER_H + 2)
    # gatehouse: an arch with a raised portcullis and doors swung open
    s0, s1 = gate_side
    gmid = (s0 + s1) / 2
    d = (s1 - s0) / np.linalg.norm(s1 - s0)
    yaw = math.atan2(d[0], d[2]) - math.pi / 2
    with a.at((gmid[0], 0, gmid[2]), yaw=-yaw):
        for k in range(11):
            ang = k / 10 * math.pi
            box(a, "stone", (math.cos(ang) * 2.5, TOP + 3.6 + math.sin(ang) * 1.6, 0), (0.6, 0.55, WALL_T + 0.3),
                block_colour(int(rng.integers(0, 1 << 30)), C(0.7, 0.67, 0.6), C(0.62, 0.6, 0.55)), rot=rot_z(ang - math.pi / 2))
        for side in (-1, 1):
            for y in np.arange(TOP, TOP + 3.7, 0.6):
                box(a, "stone", (side * 2.55, y + 0.3, 0), (0.62, 0.56, WALL_T + 0.3), block_colour(int(rng.integers(0, 1 << 30)), C(0.7, 0.67, 0.6), C(0.62, 0.6, 0.55)))
        for i in range(9):       # portcullis, raised into the arch
            box(a, "iron", (-2.0 + i * 0.5, TOP + 4.6, 0.2), (0.07, 1.8, 0.07), C(0.18, 0.18, 0.2))
        for j in range(4):
            box(a, "iron", (0, TOP + 3.9 + j * 0.45, 0.2), (4.4, 0.07, 0.07), C(0.18, 0.18, 0.2))
        for side in (-1, 1):
            hinge = np.array([side * 2.2, TOP, -WALL_T / 2])
            with a.at(hinge, yaw=side * 1.4):
                for k in range(5):
                    box(a, "wood", (-side * (0.22 + k * 0.42), 1.9, 0), (0.4, 3.8, 0.12), grain(np.zeros((1, 3)), C(0.40, 0.27, 0.16), 0.1)[0] * rng.uniform(0.85, 1.1))
                for yy in (0.8, 3.0):
                    box(a, "iron", (-side * 1.05, yy, 0.08), (2.1, 0.12, 0.04), C(0.2, 0.2, 0.22))
        # banners hanging either side of the gate
        for side in (-1, 1):
            quad(a, "cloth", [(side * 4.2 - 0.6, TOP + 6.6, 1.2), (side * 4.2 + 0.6, TOP + 6.6, 1.2), (side * 4.2 + 0.6, TOP + 2.6, 1.2), (side * 4.2 - 0.6, TOP + 2.6, 1.2)],
                 banner)
            quad(a, "gold", [(side * 4.2 - 0.25, TOP + 5.2, 1.22), (side * 4.2 + 0.25, TOP + 5.2, 1.22), (side * 4.2 + 0.25, TOP + 4.6, 1.22), (side * 4.2 - 0.25, TOP + 4.6, 1.22)],
                 C(0.9, 0.72, 0.3))
            box(a, "glow_lamp", (side * 3.4, TOP + 3.2, 1.4), (0.25, 0.35, 0.25), C(1.0, 0.8, 0.45), shade=False)
    # courtyard paving
    n = 34
    xs = np.linspace(-WALL_R, WALL_R, n)
    for j in range(n - 1):
        for i in range(n - 1):
            x0, x1, z0, z1 = xs[i], xs[i + 1], xs[j], xs[j + 1]
            cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
            if math.hypot(cx, cz) > WALL_R * 0.86:
                continue
            seed = int(rng.integers(0, 1 << 30))
            box(a, "stone", (cx, TOP + 0.04, cz), (x1 - x0 - 0.06, 0.1, z1 - z0 - 0.06), block_colour(seed, C(0.56, 0.54, 0.5), C(0.44, 0.42, 0.40)), jitter=0.02, seed=seed % 97, shade=False)
    # the keep, at the back of the courtyard
    kx, kz, kw, kd, kh = 0.0, -6.0, 13.0, 9.0, 13.0
    hw, hd = kw / 2, kd / 2
    faces = [((kx - hw, kz + hd), (kx + hw, kz + hd)), ((kx + hw, kz + hd), (kx + hw, kz - hd)), ((kx + hw, kz - hd), (kx - hw, kz - hd)), ((kx - hw, kz - hd), (kx - hw, kz + hd))]
    door_w = 2.4
    for k, (p0, p1) in enumerate(faces):
        p0 = np.array([p0[0], 0, p0[1]])
        p1 = np.array([p1[0], 0, p1[1]])
        if k == 0:
            length = np.linalg.norm(p1 - p0)

            def door_hole(u, y, length=length):
                return abs(u - length / 2) < door_w / 2 and y < TOP + 3.4
            masonry_flat(a, p0, p1, TOP, TOP + kh, 1.4, 1, moss=0.3, skip=door_hole)
        else:
            masonry_flat(a, p0, p1, TOP, TOP + kh, 1.4, 1, moss=0.3)
        crenellations(a, p0, p1, TOP + kh, 1.4, outward_sign=-1, merlon=0.8, gap=0.6)
    collide(kx, kz, 0, hw + 0.7, hd + 0.7, TOP - 1, TOP + kh + 1)
    # door: a pointed arch, heavy doors (closed - it is entered through the hall)
    front = kz + hd + 0.72
    with a.at((kx, 0, front)):
        for k in range(9):
            ang = k / 8 * math.pi
            box(a, "stone", (math.cos(ang) * 1.45, TOP + 3.3 + math.sin(ang) * 1.0, 0), (0.45, 0.4, 0.5),
                block_colour(int(rng.integers(0, 1 << 30)), C(0.72, 0.69, 0.62), C(0.64, 0.62, 0.57)), rot=rot_z(ang - math.pi / 2))
        for side in (-1, 1):
            for k in range(3):
                box(a, "wood", (side * (0.2 + k * 0.4), TOP + 1.9, 0.05), (0.38, 3.8, 0.12), C(0.36, 0.23, 0.13) * rng.uniform(0.85, 1.1))
            box(a, "iron", (side * 0.35, TOP + 1.6, 0.14), (0.08, 0.3, 0.06), C(0.75, 0.6, 0.3))
        for yy in (TOP + 0.9, TOP + 2.8):
            box(a, "iron", (0, yy, 0.13), (2.4, 0.1, 0.04), C(0.2, 0.2, 0.22))
        for side in (-1, 1):
            box(a, "glow_lamp", (side * 1.9, TOP + 2.8, 0.3), (0.25, 0.4, 0.25), C(1.0, 0.8, 0.45), shade=False)
            box(a, "iron", (side * 1.9, TOP + 3.05, 0.3), (0.32, 0.06, 0.32), C(0.2, 0.2, 0.22))
        # steps up to the door
        for s in range(3):
            box(a, "stone", (0, TOP + 0.09 + s * 0.0, 1.0 - s * 0.35), (4.0 - s * 0.4, 0.18 * (s + 1), 0.4), C(0.6, 0.58, 0.54))
    # windows on the keep
    for side_x in (-1, 1):
        for row in (TOP + 6.0, TOP + 9.8):
            arch_window(a, (kx + side_x * 3.5, row, kz + hd + 0.72), 0.0)
        for zz in (-2.0, 2.0):
            arch_window(a, (kx + side_x * (hw + 0.72), TOP + 8.0, kz + zz), side_x * math.pi / 2)
    # a tall tower rising from the keep's back corner
    round_tower(a, (kx + hw - 1.5, kz - hd + 1.5), 3.0, TOP, 21.0, banner_col=banner)
    # courtyard furniture: braziers, training dummies, a cart, banners on poles
    for bx, bz in ((-5.5, 6.5), (5.5, 6.5), (-3.5, 0.5), (3.5, 0.5)):
        cylinder(a, "iron", (bx, TOP, bz), (bx, TOP + 1.0, bz), 0.06, colour=C(0.2, 0.2, 0.22), sides=6)
        lathe(a, "iron", [(0.1, TOP + 0.95), (0.42, TOP + 1.2), (0.45, TOP + 1.3), (0.0, TOP + 1.3)], 12, C(0.22, 0.22, 0.24), centre=(bx, 0, bz))
        blob(a, "glow_lamp", (bx, TOP + 1.42, bz), (0.3, 0.3, 0.3), C(1.0, 0.6, 0.2), level=1, noise=0.35, seed=bx)
    for dx in (-9.0, -7.5):
        cylinder(a, "wood", (dx, TOP, 8.0), (dx, TOP + 1.8, 8.0), 0.06, colour=C(0.4, 0.28, 0.16), sides=6)
        cylinder(a, "wood", (dx - 0.5, TOP + 1.4, 8.0), (dx + 0.5, TOP + 1.4, 8.0), 0.05, colour=C(0.4, 0.28, 0.16), sides=6)
        lathe(a, "cloth", [(0.22, TOP + 0.9), (0.28, TOP + 1.2), (0.25, TOP + 1.6), (0.15, TOP + 1.75), (0.0, TOP + 1.78)], 10, C(0.72, 0.62, 0.42), centre=(dx, 0, 8.0))
    for px in (-6.0, 6.0):
        cylinder(a, "wood", (px, TOP, -0.5), (px, TOP + 6.5, -0.5), 0.08, colour=C(0.36, 0.24, 0.14), sides=6)
        quad(a, "cloth", [(px, TOP + 6.3, -0.4), (px + 1.4, TOP + 6.3, -0.4), (px + 1.4, TOP + 3.6, -0.4), (px, TOP + 3.6, -0.4)], banner)
    save(a)
    door = {"x": kx, "z": front + 1.2, "y": TOP, "radius": 2.2}
    gate = {"x": float(gmid[0]), "z": float(gmid[2]), "width": 4.4}
    return door, gate


# ------------------------------------------------------------------ the bridge
def bridge(length, y_land, y_island):
    """A stone causeway on arches, rising from the shore (z = 0) to the island (z = -length)."""
    a = geo.Asset("bridge")
    width = 4.2
    spans = max(3, int(round(length / 9)))
    span_len = length / spans
    deck = lambda z: y_land + (y_island - y_land) * sstep(0, 1, -z / length)
    # piers and arches
    for s in range(spans + 1):
        z = -s * span_len
        if 0 < s < spans:
            for y in np.arange(-6, deck(z) - 0.9, 0.55):
                box(a, "stone", (0, y + 0.27, z), (width + 0.6, 0.5, 1.8), block_colour(int(rng.integers(0, 1 << 30)), moss=0.8, height=(-2, 1)), jitter=0.03)
            for side in (-1, 1):   # cutwaters
                for y in np.arange(-6, -0.5, 0.55):
                    box(a, "stone", (0, y + 0.27, z + side * 1.2), (width * 0.7, 0.5, 0.7), block_colour(int(rng.integers(0, 1 << 30)), moss=0.9, height=(-2, 0.5)),
                        rot=rot_y(math.pi / 4), jitter=0.03)
        if s < spans:
            z0, z1 = z, z - span_len
            n = 12
            for k in range(n + 1):
                t = k / n
                ang = t * math.pi
                zz = z0 + (z1 - z0) * t
                yy = (deck(z0) + deck(z1)) / 2 - 1.0 - (span_len / 2 - 0.9) * math.sin(ang) * 0.0
                arch_y = deck(zz) - 1.0 - (1 - math.sin(ang)) * (span_len * 0.42)
                box(a, "stone", (0, arch_y, zz), (width + 0.4, 0.55, span_len / n + 0.05), block_colour(int(rng.integers(0, 1 << 30)), moss=0.5, height=(-1, 1)),
                    rot=rot_x(math.cos(ang) * 0.6))
                del yy
    # deck, kerbs, parapets with lanterns
    steps = int(length / 0.6)
    for k in range(steps):
        z0, z1 = -k * length / steps, -(k + 1) * length / steps
        zm = (z0 + z1) / 2
        slope = math.atan2(deck(z1) - deck(z0), z0 - z1)
        for lane in range(4):
            box(a, "stone", (-width / 2 + (lane + 0.5) * width / 4, deck(zm) - 0.12, zm), (width / 4 - 0.04, 0.24, length / steps - 0.04),
                block_colour(int(rng.integers(0, 1 << 30)), C(0.6, 0.58, 0.54), C(0.5, 0.48, 0.45)), rot=rot_x(slope), shade=False)
        box(a, "stone", (0, deck(zm) - 0.6, zm), (width, 0.8, length / steps), C(0.3, 0.29, 0.28), rot=rot_x(slope), shade=False)
        for side in (-1, 1):
            box(a, "stone", (side * (width / 2 + 0.2), deck(zm) + 0.45, zm), (0.4, 0.9, length / steps + 0.02),
                block_colour(int(rng.integers(0, 1 << 30)), moss=0.4, height=(deck(zm), deck(zm) + 0.4)), rot=rot_x(slope), shade=False)
    for s in range(spans + 1):
        z = -s * span_len
        for side in (-1, 1):
            x = side * (width / 2 + 0.2)
            box(a, "stone", (x, deck(z) + 1.0, z), (0.6, 0.4, 0.6), C(0.58, 0.56, 0.52))
            cylinder(a, "iron", (x, deck(z) + 1.2, z), (x, deck(z) + 2.6, z), 0.04, colour=C(0.2, 0.2, 0.22), sides=6)
            box(a, "glow_lamp", (x, deck(z) + 2.75, z), (0.22, 0.32, 0.22), C(1.0, 0.8, 0.45), shade=False)
            box(a, "iron", (x, deck(z) + 2.97, z), (0.3, 0.05, 0.3), C(0.2, 0.2, 0.22))
    save(a)
    return {"length": length, "width": width, "y_land": y_land, "y_island": y_island}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    island()
    door, gate = castle()
    span = json.loads((OUT / "bridge_span.json").read_text()) if (OUT / "bridge_span.json").exists() else {"length": 36.0, "y_land": 2.0}
    info = bridge(span["length"], span["y_land"], TOP)
    (OUT / "castle.json").write_text(json.dumps({"top": TOP, "island_r": ISLAND_R, "wall_r": WALL_R, "door": door, "gate": gate,
                                                  "bridge": info, "colliders": colliders}, indent=1), encoding="utf-8")
    print("CASTLE colliders", len(colliders), "door", door, "gate", gate)


main()
