"""Model the two monsters that roam outside the village, as jointed rigs whose parts can be
animated, struck off and thrown about by the game:

  horror   a crouching, many-eyed thing: a ridged body with torn fins, two clawed arms and six
           segmented tentacles (each segment is its own joint)
  golem    an ice golem: stone blocks bound by ice crystals, chains and glowing runes, a crystal
           core in its chest, a crystal claw on one arm and a stone club on the other

Writes assets_src/monsters/<name>.glb (glTF axes, metres, standing on y = 0, facing +Z).
Run: blender --background --factory-startup --python tools/blender_monsters.py
"""
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geo  # noqa: E402
from geo import C, box, cylinder, lathe, blob, tube, quad, rounded_box, grain, fbm, vnoise, sstep, unit, rot_y, rot_x, rot_z, look_frame  # noqa: E402,F401

OUT = HERE.parent / "assets_src" / "monsters"
MATS = {"flesh": {"rough": 0.45}, "bone": {"rough": 0.5}, "membrane": {"rough": 0.6, "double": True}, "eye": {"rough": 0.1},
        "gore": {"rough": 0.3}, "stone": {"rough": 0.9}, "ice": {"rough": 0.12, "metal": 0.1, "emissive": (0.25, 0.65, 0.9), "strength": 0.35},
        "rune": {"rough": 0.4, "emissive": (0.35, 0.85, 1.0), "strength": 2.4}, "iron": {"rough": 0.45, "metal": 0.6},
        "glow_eye": {"rough": 0.3, "emissive": (0.45, 0.9, 1.0), "strength": 3.0}}
rng = np.random.default_rng(66)
SKIN, SKIN_D = C(0.46, 0.52, 0.47), C(0.28, 0.33, 0.30)


def skin(seed, wounds=0.4):
    def fn(unitv, p=None):
        p = unitv if p is None else p
        n = vnoise(p * 7.0, seed)
        c = SKIN_D[None] + (SKIN - SKIN_D)[None] * n[:, None]
        w = sstep(0.68, 0.78, vnoise(p * 4.5, seed + 5)) * wounds
        c = c * (1 - w[:, None]) + C(0.55, 0.12, 0.12) * w[:, None]
        return c
    return fn


def organic(asset, material, cage_pts, cage_quads, colour, levels=2, lumpy=0.0, seed=0.0):
    co, tri, nor = geo.subdivided(cage_pts, cage_quads, levels)
    if lumpy:
        co = co + nor * (fbm(co * 6 + seed, seed, 3) * lumpy)[:, None]
    asset.add(material, co, colour(co) if callable(colour) else colour, tri, nor)


def blob_part(asset, material, centre, radii, colour, level=3, noise=0.12, freq=2.5, seed=0.0, rot=None):
    blob(asset, material, centre, radii, (lambda u, p: colour(p)) if callable(colour) else colour, level=level, noise=noise, freq=freq, seed=seed, rot=rot)


def claw(asset, base, direction, length, radius, colour=C(0.82, 0.78, 0.66)):
    d = unit(direction)
    side = unit(np.cross(d, [0, 1, 0])) if abs(d[1]) < 0.9 else np.array([1.0, 0, 0])
    pts = [base + d * length * t + np.cross(side, d) * -0.25 * length * t * t for t in np.linspace(0, 1, 6)]
    tube(asset, "bone", pts, [radius * (1 - t) ** 0.9 + 0.002 for t in np.linspace(0, 1, 6)], colour, sides=6)


# ------------------------------------------------------------------ horror
def horror():
    parts = []
    S = 0.75      # overall size: the small, weak kind

    def add(name, asset, parent, pivot):
        parts.append({"name": name, "asset": asset, "parent": parent, "pivot": np.asarray(pivot, float) * 1.0})
        return len(parts) - 1

    # body
    body = geo.Asset("body")
    blob_part(body, "flesh", (0, 0.95 * S, -0.05 * S), (0.55 * S, 0.42 * S, 0.5 * S), skin(1), level=3, noise=0.1, seed=1)
    blob_part(body, "flesh", (0, 1.25 * S, -0.15 * S), (0.42 * S, 0.3 * S, 0.38 * S), skin(2), level=3, noise=0.1, seed=2)
    for k in range(7):      # ridges along the back
        t = k / 6
        blob_part(body, "flesh", (0, (1.45 - 0.2 * t) * S, (-0.35 + 0.55 * t - 0.25) * S), (0.12 * S, 0.06 * S, 0.1 * S), SKIN_D * 1.1, level=2, noise=0.2, seed=10 + k)
        blob_part(body, "bone", (0, (1.52 - 0.2 * t) * S, (-0.35 + 0.55 * t - 0.25) * S), (0.025 * S, 0.06 * S, 0.025 * S), C(0.8, 0.75, 0.62), level=1, noise=0.0)
    for k in range(18):     # pustules
        a, b = rng.uniform(0, 2 * math.pi), rng.uniform(0.1, 1.0)
        p = np.array([math.cos(a) * 0.5 * S * b, (0.95 + 0.35 * b) * S, math.sin(a) * 0.45 * S - 0.05 * S])
        blob_part(body, "gore", p, (0.04 * S, 0.04 * S, 0.04 * S), C(0.86, 0.80, 0.42) if k % 3 else C(0.62, 0.12, 0.12), level=1, noise=0.1, seed=k)
    b_i = add("body", body, None, (0, 0.95 * S, 0))
    # head
    head = geo.Asset("head")
    hc = np.array([0, 1.05 * S, 0.42 * S])
    blob_part(head, "flesh", hc, (0.26 * S, 0.22 * S, 0.24 * S), skin(3, 0.2), level=3, noise=0.1, seed=3)
    blob_part(head, "flesh", hc + [0, 0.18 * S, -0.06 * S], (0.16 * S, 0.1 * S, 0.14 * S), skin(4, 0.1), level=2, noise=0.1, seed=4)
    for sx in (-1, 1):       # horns
        tube(head, "bone", [hc + [sx * 0.15 * S, 0.12 * S, 0.05 * S], hc + [sx * 0.28 * S, 0.24 * S, 0.0], hc + [sx * 0.32 * S, 0.38 * S, -0.08 * S]],
             [0.035 * S, 0.025 * S, 0.006], C(0.78, 0.72, 0.6), sides=6)
    for row, (y, n) in enumerate(((0.08, 3), (0.0, 3))):     # six eyes in two rows
        for i in range(n):
            ex = (i - 1) * 0.1 * S
            ep = hc + [ex, y * S + 0.03 * S, 0.21 * S]
            blob_part(head, "flesh", ep, (0.045 * S, 0.035 * S, 0.03 * S), SKIN_D, level=1, noise=0.0)
            blob_part(head, "eye", ep + [0, 0, 0.012 * S], (0.03 * S, 0.03 * S, 0.025 * S), C(0.92, 0.82, 0.5), level=2, noise=0.0)
            blob_part(head, "eye", ep + [0, 0, 0.03 * S], (0.012 * S, 0.016 * S, 0.01 * S), C(0.05, 0.03, 0.03), level=1, noise=0.0)
    mouth = hc + [0, -0.1 * S, 0.2 * S]
    blob_part(head, "gore", mouth, (0.16 * S, 0.06 * S, 0.06 * S), C(0.28, 0.05, 0.06), level=2, noise=0.05)
    for i in range(10):      # fangs round the maw
        t = (i / 9 - 0.5) * 2.4
        for up in (1, -1):
            base = mouth + [math.sin(t) * 0.14 * S, up * 0.045 * S, math.cos(t) * 0.04 * S]
            tube(head, "bone", [base, base + [0, -up * (0.07 + 0.04 * (i % 2)) * S, 0.02 * S]], [0.015 * S, 0.002], C(0.85, 0.82, 0.72), sides=5)
    for i in range(6):      # face tendrils hanging from the jaw
        t = (i / 5 - 0.5) * 1.6
        base = mouth + [math.sin(t) * 0.12 * S, -0.06 * S, 0.0]
        pts = [base + [0, -k * 0.06 * S, 0.03 * k * S * math.cos(t)] for k in range(5)]
        tube(head, "flesh", pts, [0.02 * S * (1 - k / 5) + 0.004 for k in range(5)], C(0.5, 0.22, 0.22), sides=5)
    add("head", head, b_i, hc - [0, 0, 0.18 * S])
    # torn fins on the shoulders
    for sx in (-1, 1):
        fin = geo.Asset("fin")
        root = np.array([sx * 0.4 * S, 1.3 * S, -0.1 * S])
        spines = [root + np.array([sx * (0.45 + 0.1 * k) * S, (0.35 - 0.18 * k) * S, (-0.05 - 0.1 * k) * S]) for k in range(4)]
        for tip in spines:
            tube(fin, "bone", [root, (root + tip) / 2 + [0, 0.05 * S, 0], tip], [0.03 * S, 0.018 * S, 0.004], C(0.62, 0.6, 0.52), sides=5)
        for k in range(3):
            tri = np.array([root, spines[k], spines[k + 1]])
            mid = (spines[k] + spines[k + 1]) / 2 + (root - (spines[k] + spines[k + 1]) / 2) * 0.25
            fin.add("membrane", np.array([root, spines[k], mid, spines[k + 1]]), [C(0.6, 0.25, 0.25), C(0.72, 0.36, 0.34), C(0.5, 0.18, 0.2), C(0.72, 0.36, 0.34)],
                    [(0, 1, 2), (0, 2, 3)])
            del tri
        add(f"fin_{'l' if sx > 0 else 'r'}", fin, b_i, root)
    # arms: upper arm, forearm, clawed hand
    for sx in (-1, 1):
        tag = "l" if sx > 0 else "r"
        sh = np.array([sx * 0.42 * S, 1.12 * S, 0.12 * S])
        el = sh + np.array([sx * 0.22 * S, -0.3 * S, 0.22 * S])
        wr = el + np.array([-sx * 0.12 * S, 0.05 * S, 0.32 * S])
        up = geo.Asset("upper")
        tube(up, "flesh", [sh, (sh + el) / 2 + [sx * 0.05 * S, 0, 0], el], [0.14 * S, 0.12 * S, 0.09 * S], skin(20 + sx), sides=10)
        blob_part(up, "flesh", sh, (0.17 * S, 0.15 * S, 0.15 * S), skin(22 + sx), level=2)
        u_i = add(f"arm_{tag}", up, b_i, sh)
        fore = geo.Asset("fore")
        tube(fore, "flesh", [el, (el + wr) / 2, wr], [0.09 * S, 0.08 * S, 0.06 * S], skin(24 + sx), sides=10)
        for k in range(3):     # bony spurs on the forearm
            p = el + (wr - el) * (0.25 + 0.25 * k)
            tube(fore, "bone", [p + [sx * 0.05 * S, 0.04 * S, 0], p + [sx * 0.14 * S, 0.1 * S, -0.04 * S]], [0.02 * S, 0.003], C(0.75, 0.7, 0.58), sides=5)
        f_i = add(f"fore_{tag}", fore, u_i, el)
        hand = geo.Asset("hand")
        blob_part(hand, "flesh", wr + [0, 0, 0.04 * S], (0.08 * S, 0.06 * S, 0.09 * S), skin(26 + sx), level=2)
        for k in range(3):
            ang = (k - 1) * 0.45
            d = np.array([math.sin(ang) * 0.6 - sx * 0.1, -0.35, math.cos(ang)])
            claw(hand, wr + [math.sin(ang) * 0.05 * S, 0, 0.1 * S], d, 0.28 * S, 0.028 * S)
        add(f"hand_{tag}", hand, f_i, wr)
    # six tentacles, five segments each, every segment a joint
    for t in range(6):
        ang = (t + 0.5) / 6 * 2 * math.pi + 0.3
        out = np.array([math.cos(ang), 0, math.sin(ang)])
        start = np.array([0, 0.7 * S, -0.05 * S]) + out * 0.3 * S
        parent = b_i
        seg_len = 0.24 * S
        p = start
        r0 = 0.11 * S
        for k in range(5):
            # each segment curls outward and down, then lies along the ground
            d = unit(out * (0.6 + 0.12 * k) + np.array([0, -0.7 + 0.18 * k, 0]))
            q = p + d * seg_len
            if q[1] < 0.06 * S:
                q[1] = 0.06 * S
            seg = geo.Asset("seg")
            r1 = r0 * 0.78
            tube(seg, "flesh", [p, (p + q) / 2, q], [r0, (r0 + r1) / 2, r1], skin(40 + t * 5 + k, 0.25), sides=9, cap=(k == 4))
            for s in range(3):     # suckers underneath
                sp = p + (q - p) * (0.25 + 0.25 * s) + np.array([0, -r0 * 0.75, 0])
                blob_part(seg, "gore", sp, (r0 * 0.28, r0 * 0.12, r0 * 0.28), C(0.62, 0.22, 0.24), level=1, noise=0.0)
            parent = add(f"tent_{t}_{k}", seg, parent, p)
            p, r0 = q, r1
    geo.write_rig(OUT / "horror.glb", "horror", parts, MATS)


# ------------------------------------------------------------------ golem
STONE_C = [C(0.56, 0.50, 0.42), C(0.48, 0.44, 0.38), C(0.62, 0.56, 0.47)]


def stone_block(asset, centre, size, rot=None, seed=0, runes=1, frost=0.4):
    tone = STONE_C[seed % 3] * rng.uniform(0.9, 1.08)

    def col(v):
        c = grain(v, tone, 0.12, 3.0, seed)
        f = np.clip(sstep(0.55, 0.85, vnoise(v * 2.0, seed + 7)) * frost + sstep(0.6, 1.0, (v[:, 1] - np.min(v[:, 1])) / max(np.ptp(v[:, 1]), 1e-3)) * frost * 0.6, 0, 1)
        return c * (1 - f[:, None]) + C(0.82, 0.92, 0.98) * f[:, None]
    geo.rounded_box(asset, "stone", centre, size, col, rot=rot, levels=1, lumpy=0.02 * max(size), seed=seed)
    R = np.eye(3) if rot is None else np.asarray(rot)
    for k in range(runes):     # a glowing rune on the front face
        face = R @ np.array([0, 0, 1.0])
        right = R @ np.array([1.0, 0, 0])
        up = R @ np.array([0, 1.0, 0])
        base = np.asarray(centre) + face * (size[2] / 2 + 0.004) + up * rng.uniform(-0.15, 0.15) * size[1]
        s = min(size[0], size[1]) * 0.22
        strokes = [((0, -1), (0, 1)), ((0, 1), (0.6, 0.4)), ((0, 0), (-0.6, -0.5))] if k % 2 == 0 else [((-0.5, -1), (0, 1)), ((0, 1), (0.5, -1)), ((-0.3, 0), (0.3, 0))]
        for (x0, y0), (x1, y1) in strokes:
            p0 = base + right * x0 * s + up * y0 * s
            p1 = base + right * x1 * s + up * y1 * s
            dvec = unit(p1 - p0)
            side = np.cross(face, dvec) * 0.012
            asset.add("rune", np.array([p0 - side, p0 + side, p1 + side, p1 - side]), C(0.5, 0.9, 1.0), [(0, 1, 2), (0, 2, 3)], np.tile(face, (4, 1)))


def crystal(asset, base, direction, length, radius, seed=0):
    d = unit(direction)
    fr = look_frame(d)
    sides = 6
    ring = [np.array([math.cos(i / sides * 2 * math.pi), 0, math.sin(i / sides * 2 * math.pi)]) for i in range(sides)]
    pts = []
    for h, r in ((0.0, radius), (0.72, radius * 0.95), (1.0, 0.0)):
        for i in range(sides):
            pts.append(np.asarray(base) + fr @ (ring[i] * r) + d * length * h)
    pts = np.array(pts)
    faces = []
    for j in range(2):
        for i in range(sides):
            a, b = j * sides + i, j * sides + (i + 1) % sides
            faces += [(a, b, b + sides), (a, b + sides, a + sides)]
    faces = geo._orient(pts, faces, None, np.asarray(base) + d * length * 0.4)
    tri = pts[faces]
    fn = unit(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
    t = (np.einsum("ij,j->i", tri.reshape(-1, 3) - base, d) / length)
    col = C(0.70, 0.88, 0.98)[None] * (0.85 + 0.35 * t[:, None]) + C(0.1, 0.1, 0.1) * vnoise(tri.reshape(-1, 3) * 9, seed)[:, None]
    asset.add("ice", tri.reshape(-1, 3), col, np.arange(len(faces) * 3).reshape(-1, 3), np.repeat(fn, 3, axis=0))


def crystal_cluster(asset, base, direction, n, length, radius, spread=0.5, seed=0):
    r = np.random.default_rng(seed)
    d = unit(direction)
    for k in range(n):
        dk = unit(d + r.normal(0, spread, 3))
        crystal(asset, np.asarray(base) + r.normal(0, radius * 0.6, 3), dk, length * r.uniform(0.5, 1.0), radius * r.uniform(0.5, 1.0), seed + k)


def chain(asset, pts, link=0.09):
    pts = [np.asarray(p, float) for p in pts]
    for (a, b) in zip(pts, pts[1:]):
        n = max(1, int(np.linalg.norm(b - a) / link))
        for k in range(n):
            c = a + (b - a) * (k + 0.5) / n
            d = unit(b - a)
            fr = look_frame(d)
            ring = [c + fr @ np.array([math.cos(t) * 0.035, math.sin(t) * 0.06, 0]) for t in np.linspace(0, 2 * math.pi, 9)]
            if k % 2:
                ring = [c + fr @ np.array([0, math.sin(t) * 0.06, math.cos(t) * 0.035]) for t in np.linspace(0, 2 * math.pi, 9)]
            tube(asset, "iron", ring, 0.011, C(0.3, 0.3, 0.33), sides=4, cap=False)


def golem():
    parts = []
    S = 0.85

    def add(name, asset, parent, pivot):
        parts.append({"name": name, "asset": asset, "parent": parent, "pivot": np.asarray(pivot, float)})
        return len(parts) - 1

    hip_y = 1.0 * S
    pelvis = geo.Asset("pelvis")
    stone_block(pelvis, (0, hip_y, 0), (0.7 * S, 0.32 * S, 0.45 * S), seed=1, runes=0)
    p_i = add("pelvis", pelvis, None, (0, hip_y, 0))
    torso = geo.Asset("torso")
    tc = np.array([0, hip_y + 0.62 * S, 0])
    blocks = [((0, 0.0, 0.02), (0.9, 0.55, 0.5)), ((-0.32, 0.36, 0.0), (0.42, 0.4, 0.48)), ((0.32, 0.36, 0.0), (0.42, 0.4, 0.48)),
              ((0, 0.38, -0.08), (0.36, 0.38, 0.36)), ((0, -0.38, 0.0), (0.62, 0.3, 0.42))]
    for k, (off, size) in enumerate(blocks):
        stone_block(torso, tc + np.array(off) * S, np.array(size) * S, rot=rot_z(rng.normal(0, 0.05)) @ rot_y(rng.normal(0, 0.05)), seed=10 + k, runes=1 if k else 0)
    # the crystal heart, set into the chest, and chains across it
    crystal_cluster(torso, tc + np.array([0, 0.12, 0.26]) * S, (0, 0.2, 1), 5, 0.22 * S, 0.06 * S, spread=0.35, seed=5)
    blob(torso, "rune", tc + np.array([0, 0.12, 0.25]) * S, (0.1 * S, 0.1 * S, 0.05 * S), C(0.6, 0.95, 1.0), level=2, noise=0.0)
    chain(torso, [tc + np.array([-0.5, 0.5, 0.26]) * S, tc + np.array([0.0, 0.0, 0.32]) * S, tc + np.array([0.5, -0.35, 0.26]) * S])
    chain(torso, [tc + np.array([0.5, 0.5, 0.26]) * S, tc + np.array([0.2, 0.25, 0.3]) * S])
    for sx in (-1, 1):     # crystal pauldrons
        crystal_cluster(torso, tc + np.array([sx * 0.5, 0.55, -0.05]) * S, (sx * 0.4, 1, -0.2), 6, 0.55 * S, 0.08 * S, spread=0.3, seed=20 + sx)
    t_i = add("torso", torso, p_i, tc - np.array([0, 0.3, 0]) * S)
    head = geo.Asset("head")
    hcen = tc + np.array([0, 0.78, 0.06]) * S
    stone_block(head, hcen, (0.34 * S, 0.36 * S, 0.32 * S), seed=40, runes=0)
    for sx in (-1, 1):
        blob(head, "glow_eye", hcen + np.array([sx * 0.08, 0.04, 0.165]) * S, (0.035 * S, 0.025 * S, 0.012 * S), C(0.6, 0.95, 1.0), level=1, noise=0.0)
    crystal(head, hcen + np.array([0, 0.0, 0.168]) * S - np.array([0, 0.06, 0]) * S, (0, 1, 0.15), 0.12 * S, 0.035 * S, 3)
    crystal_cluster(head, hcen + np.array([0, 0.17, -0.05]) * S, (0, 1, -0.3), 3, 0.22 * S, 0.05 * S, spread=0.4, seed=41)
    add("head", head, t_i, hcen - np.array([0, 0.18, 0]) * S)
    for sx in (-1, 1):
        tag = "l" if sx > 0 else "r"
        sh = tc + np.array([sx * 0.62, 0.42, 0]) * S
        up = geo.Asset("upper")
        stone_block(up, sh + np.array([sx * 0.06, -0.12, 0]) * S, np.array([0.36, 0.42, 0.36]) * S, seed=50 + sx, runes=1)
        stone_block(up, sh + np.array([sx * 0.04, -0.46, 0]) * S, np.array([0.3, 0.32, 0.3]) * S, rot=rot_z(sx * 0.08), seed=52 + sx, runes=0)
        chain(up, [sh + np.array([sx * -0.05, -0.32, 0.18]) * S, sh + np.array([sx * 0.22, -0.32, 0.16]) * S])
        u_i = add(f"arm_{tag}", up, t_i, sh)
        el = sh + np.array([sx * 0.04, -0.66, 0]) * S
        fore = geo.Asset("fore")
        stone_block(fore, el + np.array([0, -0.2, 0.04]) * S, np.array([0.32, 0.42, 0.32]) * S, seed=60 + sx, runes=1)
        f_i = add(f"fore_{tag}", fore, u_i, el)
        wr = el + np.array([0, -0.45, 0.06]) * S
        hand = geo.Asset("hand")
        if sx > 0:   # crystal claw
            stone_block(hand, wr + np.array([0, -0.08, 0]) * S, np.array([0.3, 0.2, 0.3]) * S, seed=70, runes=0)
            crystal_cluster(hand, wr + np.array([0, -0.15, 0.05]) * S, (0, -1, 0.5), 6, 0.42 * S, 0.06 * S, spread=0.35, seed=71)
        else:        # stone club studded with crystal
            stone_block(hand, wr + np.array([0, -0.3, 0.02]) * S, np.array([0.34, 0.62, 0.34]) * S, rot=rot_x(0.1), seed=72, runes=1)
            for k in range(5):
                crystal(hand, wr + np.array([rng.uniform(-0.16, 0.16), -0.15 - 0.1 * k, 0.17]) * S, (rng.normal(0, 0.3), -0.2, 1), 0.12 * S, 0.03 * S, k)
        add(f"hand_{tag}", hand, f_i, wr)
        hp = np.array([sx * 0.22, hip_y - 0.12 * S, 0])
        thigh = geo.Asset("thigh")
        stone_block(thigh, hp + np.array([0, -0.22, 0]) * S, np.array([0.34, 0.46, 0.36]) * S, seed=80 + sx, runes=1)
        th_i = add(f"thigh_{tag}", thigh, p_i, hp)
        kn = hp + np.array([0, -0.48, 0]) * S
        shin = geo.Asset("shin")
        stone_block(shin, kn + np.array([0, -0.2, 0.02]) * S, np.array([0.3, 0.4, 0.32]) * S, seed=90 + sx, runes=0)
        crystal_cluster(shin, kn + np.array([0, -0.15, 0.12]) * S, (0, -0.3, 1), 4, 0.3 * S, 0.06 * S, spread=0.5, seed=91 + sx)
        stone_block(shin, kn + np.array([0, -0.42, 0.08]) * S, np.array([0.36, 0.14, 0.48]) * S, seed=93 + sx, runes=0)    # foot
        add(f"shin_{tag}", shin, th_i, kn)
    geo.write_rig(OUT / "golem.glb", "golem", parts, MATS)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    horror()
    golem()


main()
