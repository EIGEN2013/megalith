"""Model the villagers' props in Blender, fitted to Robin's head and hands in his rest pose:
a guard's helm and spear, a fisher's straw hat, the elder's beard and staff, braids with a
flower crown, a kerchief and a child's wooden sword.

Each prop is written to assets_src/npc/props/<name>.glb in Robin's model space (metres, glTF
axes, Robin facing +Z), so the game can hang it on a bone exactly where it was modelled.

Run: blender --background --factory-startup --python tools/blender_npc_props.py
"""
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blender_flora as F  # noqa: E402  (Asset, Part, subdivided, noise, colours)
from glb import Glb  # noqa: E402

ROOT = HERE.parent
F.OUT = ROOT / "assets_src" / "npc" / "props"
srgb, fbm, vnoise, unit = F.srgb, F.fbm, F.vnoise, F.unit


def head_frame():
    """Centre, half-size and top of Robin's head (hair included), in Blender axes."""
    g = Glb(str(ROOT / "live_src" / "hero.glb"))
    J = g.json
    prim = J["meshes"][0]["primitives"][0]
    pos = g.read(prim["attributes"]["POSITION"]).astype(np.float64)
    jnt = g.read(prim["attributes"]["JOINTS_0"]).astype(np.int64)
    wgt = g.read(prim["attributes"]["WEIGHTS_0"]).astype(np.float64)
    names = [J["nodes"][j]["name"] for j in J["skins"][0]["joints"]]
    sel = np.array([n in ("Head", "head_end", "headfront") for n in names])
    w = (wgt * sel[jnt]).sum(axis=1)
    head = pos[(w > 0.6) & (pos[:, 1] > 1.52)]
    lo, hi = head.min(axis=0), head.max(axis=0)
    centre = (lo + hi) / 2
    to_b = lambda p: np.array([p[0], -p[2], p[1]])          # glTF -> Blender
    return to_b(centre), (hi - lo) / 2, float(hi[1]), to_b


def lathe_solid(profile, seg, colour_fn, at, levels=1):
    """Revolve a profile; a profile that starts on the axis gets a proper closed crown."""
    if profile[0][0] < 0.01:
        v, quads, _ = F.lathe(profile[1:], seg, None)
        apex = len(v)
        v = np.vstack([v, [0, 0, profile[0][1]]])
        faces = list(quads) + [(i + 1 if i + 1 < seg else 0, i, apex) for i in range(seg)]
    else:
        v, faces, _ = F.lathe(profile, seg, None)
    co, tri, nor = F.subdivided(v, faces, levels)
    return co + at, tri, nor, colour_fn(co)


def helm(asset, c, half, top):
    """A rounded steel cap with a brim band and a nose guard."""
    r = max(half[0], half[2]) * 1.12
    prof = [(0.001, r * 1.05), (r * 0.5, r * 0.93), (r * 0.85, r * 0.55), (r * 1.02, r * 0.1), (r * 1.06, -r * 0.12), (r * 1.0, -r * 0.16), (r * 0.9, -r * 0.05)]
    co, tri, nor, col = lathe_solid(prof, 20, lambda p: srgb(0.58, 0.60, 0.64) * (0.8 + 0.25 * vnoise(p * 40, 3))[:, None],
                                    c + np.array([0, 0, top - c[2] - r * 0.8]))
    asset.part("metal").add(co, col, tri, nor)
    band = [(r * 1.07, -r * 0.02), (r * 1.1, -r * 0.1), (r * 1.07, -r * 0.18), (r * 1.0, -r * 0.18), (r * 1.0, -r * 0.02)]
    co, tri, nor, col = lathe_solid(band, 24, lambda p: srgb(0.45, 0.42, 0.40), c + np.array([0, 0, top - c[2] - r * 0.8]), 0)
    asset.part("metal").add(co, col, tri, nor)
    # nose guard down the front (Robin faces -Y in Blender)
    front = c + np.array([0, -r * 1.02, top - c[2] - r * 0.95])
    guard = np.array([[-0.012, 0, 0.02], [0.012, 0, 0.02], [0.009, -0.006, -0.085], [-0.009, -0.006, -0.085]]) + front
    asset.part("metal").add(guard, srgb(0.55, 0.57, 0.6), [(0, 2, 1), (0, 3, 2)], normals=np.tile([0, -1.0, 0], (4, 1)))


def spear(asset, hand):
    shaft_lo, shaft_hi = hand + np.array([0, 0, -1.05]), hand + np.array([0, 0, 1.2])
    pts = [shaft_lo + (shaft_hi - shaft_lo) * t for t in np.linspace(0, 1, 8)]
    F.tube_line(asset.part("wood"), pts, 0.016, srgb(0.42, 0.28, 0.16), 8)
    # leaf-shaped iron head
    tip = shaft_hi + np.array([0, 0, 0.3])
    base = shaft_hi
    v = [base + [0, 0, 0], base + [0.035, 0, 0.1], tip, base + [-0.035, 0, 0.1], base + [0, 0.012, 0.11], base + [0, -0.012, 0.11]]
    f = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4), (0, 5, 1), (1, 5, 2), (2, 5, 3), (3, 5, 0)]
    asset.part("metal").add(np.array(v), srgb(0.70, 0.72, 0.75), f)
    collar = [shaft_hi + [0, 0, -0.05 + 0.05 * k] for k in range(3)]
    F.tube_line(asset.part("metal"), collar, 0.022, srgb(0.4, 0.4, 0.42), 8)


def straw_hat(asset, c, half, top):
    r = max(half[0], half[2])
    prof = [(0.001, r * 0.85), (r * 0.6, r * 0.82), (r * 0.95, r * 0.6), (r * 1.05, r * 0.08), (r * 1.25, 0.0), (r * 2.4, -r * 0.18),
            (r * 2.45, -r * 0.24), (r * 2.3, -r * 0.2), (r * 1.2, -r * 0.05), (r * 0.9, r * 0.0)]

    def straw(p):
        weave = 0.5 + 0.5 * np.sin(np.arctan2(p[:, 1], p[:, 0]) * 60) * np.sin(np.hypot(p[:, 0], p[:, 1]) * 220)
        return srgb(0.82, 0.68, 0.40) * (0.8 + 0.22 * weave)[:, None] * (0.9 + 0.15 * vnoise(p * 30, 7))[:, None]
    co, tri, nor, col = lathe_solid(prof, 28, straw, c + np.array([0, 0, top - c[2] - r * 0.55]))
    band = (co[:, 2] - (top - r * 0.55)) < r * 0.25
    band &= np.hypot(co[:, 0] - c[0], co[:, 1] - c[1]) < r * 1.12
    col[band] = srgb(0.45, 0.18, 0.12)
    asset.part("straw").add(co, col, tri, nor)


def kerchief(asset, c, half, top):
    """A cloth cap over the crown and back of the head, knotted at the nape."""
    r = max(half[0], half[2]) * 1.08
    prof = [(0.001, r * 1.0), (r * 0.55, r * 0.9), (r * 0.9, r * 0.5), (r * 1.04, r * 0.05), (r * 1.02, -r * 0.25), (r * 0.94, -r * 0.3)]
    co, tri, nor, col = lathe_solid(prof, 22, lambda p: srgb(0.72, 0.56, 0.24) * (0.85 + 0.2 * vnoise(p * 25, 2))[:, None],
                                    c + np.array([0, 0, top - c[2] - r * 0.92]))
    # open the face: drop the triangles whose centre lies in front of the head and low down
    keep = []
    for k, (a, b, d) in enumerate(tri):
        m = (co[a] + co[b] + co[d]) / 3
        rel = m - c
        if rel[1] < -r * 0.35 and rel[2] < r * 0.55:
            continue
        keep.append(k)
    tri = tri[keep]
    asset.part("cloth").add(co, col, tri, nor)
    knot = c + np.array([0, r * 1.0, top - c[2] - r * 1.05])
    v = [knot + [0, 0, 0.03], knot + [0.05, 0.03, -0.12], knot + [0.02, 0.05, -0.14], knot + [-0.05, 0.03, -0.12], knot + [-0.02, 0.05, -0.14]]
    asset.part("cloth").add(np.array(v), srgb(0.66, 0.50, 0.20), [(0, 1, 2), (0, 3, 4)])


def beard(asset, c, half):
    """A full white beard: a soft wedge from the jaw down onto the chest, combed into strands."""
    rng = np.random.default_rng(5)
    jaw_z = c[2] - half[1] * 0.35
    front = -half[2] * 0.95
    for k in range(70):
        a = rng.uniform(-1.15, 1.15)
        start = c + np.array([math.sin(a) * half[0] * 0.92, math.cos(a) * front * 0.9, jaw_z + rng.uniform(-0.02, 0.05)])
        length = rng.uniform(0.13, 0.24) * (1 - 0.35 * abs(a))
        end = start + np.array([math.sin(a) * 0.02, -0.04 - 0.04 * (1 - abs(a)), -length])
        mid = (start + end) / 2 + np.array([0, -0.03, 0])
        tone = rng.uniform(0.82, 1.0)
        F.tube_line(asset.part("hair"), [start, mid, end], rng.uniform(0.012, 0.02), srgb(0.88, 0.86, 0.82) * tone, 5)
    # moustache
    for sgn in (-1, 1):
        pts = [c + np.array([sgn * 0.005, front * 1.05, c[2] * 0 + jaw_z - c[2] + c[2] + half[1] * 0.12]),
               c + np.array([sgn * 0.04, front * 1.02, jaw_z + half[1] * 0.08]), c + np.array([sgn * 0.07, front * 0.85, jaw_z - 0.02])]
        F.tube_line(asset.part("hair"), pts, 0.012, srgb(0.86, 0.84, 0.8), 5)


def staff(asset, hand):
    rng = np.random.default_rng(9)
    pts = [hand + np.array([0.02 * math.sin(t * 7), 0.02 * math.cos(t * 5), -0.95 + 1.6 * t]) for t in np.linspace(0, 1, 12)]
    F.tube_line(asset.part("wood"), pts, 0.024, srgb(0.36, 0.24, 0.14), 7)
    knob, tri, nor = F.subdivided(*_ico(), 1)
    knob = knob * np.array([0.05, 0.05, 0.065]) + pts[-1] + np.array([0, 0, 0.04])
    knob = knob + nor * (fbm(knob * 40, 2, 2) * 0.008)[:, None]
    asset.part("wood").add(knob, srgb(0.32, 0.21, 0.12) * (0.9 + 0.2 * vnoise(knob * 50, 1))[:, None], tri, nor)
    del rng


def _ico():
    t = (1 + 5 ** 0.5) / 2
    v = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
                  (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], dtype=np.float64)
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    return unit(v), f


def braids(asset, c, half, colour):
    """Two plaits from behind the ears down over the shoulders, tied with ribbon."""
    for sgn in (-1, 1):
        start = c + np.array([sgn * half[0] * 0.95, half[2] * 0.2, -half[1] * 0.2])
        end = start + np.array([sgn * 0.03, -0.09, -0.36])
        steps = 14
        for strand in range(3):
            pts = []
            for k in range(steps + 1):
                t = k / steps
                p = start + (end - start) * t + np.array([0, -0.05 * math.sin(t * math.pi), 0])
                ang = strand * 2.1 + t * 18
                p = p + np.array([math.cos(ang) * 0.016, math.sin(ang) * 0.016, 0]) * (1 - 0.4 * t)
                pts.append(p)
            F.tube_line(asset.part("hair"), pts, 0.017, colour * (0.9 + 0.1 * strand), 6)
        tie = [end + np.array([0, 0, 0.03]), end + np.array([0, 0, 0.0])]
        F.tube_line(asset.part("cloth"), tie, 0.024, srgb(0.85, 0.85, 0.75), 8)


def flower_crown(asset, c, half, top):
    rng = np.random.default_rng(3)
    r = max(half[0], half[2]) * 1.02
    z = top - half[1] * 0.42
    ring = [c + np.array([math.cos(a) * r, math.sin(a) * r * 1.05, z - c[2] + 0.01 * math.sin(a * 3)]) for a in np.linspace(0, 2 * math.pi, 25)]
    F.tube_line(asset.part("plant"), ring, 0.008, srgb(0.25, 0.45, 0.2), 5)
    for k in range(9):
        p = ring[int(k * 24 / 9)]
        petal = (srgb(0.96, 0.62, 0.78), srgb(0.98, 0.95, 0.9), srgb(0.78, 0.62, 0.98))[k % 3]
        outward = unit(p - np.array([c[0], c[1], p[2]]))
        side = np.cross(outward, [0, 0, 1.0])
        v = [p + outward * 0.012]
        for i in range(6):
            a = i / 6 * 2 * math.pi
            v.append(p + (side * math.cos(a) + np.array([0, 0, 1.0]) * math.sin(a)) * 0.026 + outward * 0.008)
        asset.part("plant").add(np.array(v), [srgb(0.98, 0.85, 0.4)] + [petal] * 6, [(0, 1 + i, 1 + (i + 1) % 6) for i in range(6)],
                                normals=np.tile(outward, (7, 1)))
    del rng


def toy_sword(asset, hand):
    blade = [hand + np.array([0, -0.02, 0.06]), hand + np.array([0, -0.02, 0.55])]
    v = [blade[0] + [-0.025, 0, 0], blade[0] + [0.025, 0, 0], blade[1] + [0.02, 0, 0], blade[1] + [0, 0, 0.05], blade[1] + [-0.02, 0, 0]]
    asset.part("wood").add(np.array(v), srgb(0.72, 0.56, 0.36), [(0, 1, 2), (0, 2, 4), (4, 2, 3)], normals=np.tile([0, -1.0, 0], (5, 1)))
    asset.part("wood").add(np.array(v) + [0, 0.012, 0], srgb(0.64, 0.5, 0.32), [(0, 2, 1), (0, 4, 2), (4, 3, 2)], normals=np.tile([0, 1.0, 0], (5, 1)))
    guard = [hand + np.array([-0.07, -0.02, 0.05]), hand + np.array([0.07, -0.02, 0.05])]
    F.tube_line(asset.part("wood"), guard, 0.014, srgb(0.42, 0.28, 0.16), 6)
    grip = [hand + np.array([0, -0.02, -0.05]), hand + np.array([0, -0.02, 0.05])]
    F.tube_line(asset.part("wood"), grip, 0.016, srgb(0.36, 0.22, 0.12), 6)


def main():
    F.OUT.mkdir(parents=True, exist_ok=True)
    F.clear_scene()
    c, half, top, to_b = head_frame()
    right_hand = to_b(np.array([-0.345, 0.88, 0.07]))       # palm, a little below the wrist joint
    print(f"PROPS head centre {np.round(c, 3)} half {np.round(half, 3)} top {top:.3f}")
    for name, build in (("helm", lambda a: helm(a, c, half, top)), ("spear", lambda a: spear(a, right_hand)),
                        ("straw_hat", lambda a: straw_hat(a, c, half, top)), ("kerchief", lambda a: kerchief(a, c, half, top)),
                        ("beard", lambda a: beard(a, c, half)), ("staff", lambda a: staff(a, right_hand)),
                        ("braids", lambda a: braids(a, c, half, srgb(0.42, 0.22, 0.10))), ("flower_crown", lambda a: flower_crown(a, c, half, top)),
                        ("toy_sword", lambda a: toy_sword(a, right_hand))):
        asset = F.Asset(name)
        build(asset)
        asset.save()


main()
