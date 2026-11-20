"""Model the things Robin can carry: a sword, an axe and a leather satchel, placed in Robin's
model space (rest pose) so the game can hang them on his hand and hip bones as modelled.
Also writes small 'icon' copies centred on the origin, for pickups on the ground.

Writes assets_src/items/<name>.glb
Run: blender --background --factory-startup --python tools/blender_items.py
"""
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geo  # noqa: E402
from geo import C, box, cylinder, lathe, blob, tube, quad, rounded_box, grain, vnoise, unit, look_frame, rot_y, rot_x  # noqa: E402,F401

OUT = HERE.parent / "assets_src" / "items"
MATS = {"steel": {"rough": 0.25, "metal": 0.9}, "iron": {"rough": 0.45, "metal": 0.7}, "wood": {"rough": 0.8},
        "leather": {"rough": 0.7}, "gold": {"rough": 0.3, "metal": 0.85}, "cloth": {"rough": 0.9}}
PALM = np.array([-0.345, 0.885, 0.075])       # Robin's right palm, rest pose (glTF)
HIP_LEFT = np.array([0.215, 0.86, 0.02])


def place(asset_fn, origin, direction):
    """Model along +Y at the origin, then aim +Y along `direction` and move the grip to `origin`."""
    a = geo.Asset("tmp")
    asset_fn(a)
    frame = look_frame(direction)
    out = geo.Asset(a.name)
    for mname, part in a.parts.items():
        v, n, c, f = part.arrays()
        out.add(mname, v @ frame.T + origin, c, f, n @ frame.T)
    return out


def sword(a):
    a.name = "sword"
    # grip from y=-0.09 to 0.09, guard at 0.1, blade to 0.95, pommel below
    cylinder(a, "leather", (0, -0.09, 0), (0, 0.09, 0), 0.018, colour=C(0.30, 0.18, 0.10), sides=8)
    for k in range(8):      # leather wrap ridges
        y = -0.08 + k * 0.022
        cylinder(a, "leather", (0, y, 0), (0, y + 0.008, 0), 0.0205, colour=C(0.24, 0.14, 0.08), sides=8)
    blob(a, "gold", (0, -0.115, 0), (0.028, 0.028, 0.028), C(0.82, 0.64, 0.28), level=2, noise=0.0)
    box(a, "gold", (0, 0.1, 0), (0.2, 0.026, 0.034), C(0.82, 0.64, 0.28))
    for s in (-1, 1):
        blob(a, "gold", (s * 0.105, 0.1, 0), (0.018, 0.018, 0.018), C(0.85, 0.68, 0.3), level=1, noise=0.0)
    # blade: diamond cross-section, tapering to a point, with a fuller
    L, w0, t0 = 0.86, 0.034, 0.008
    stations = np.linspace(0, 1, 8)
    verts, faces = [], []
    for s in stations:
        w = w0 * (1 - 0.25 * s) * (1 if s < 0.88 else (1 - s) / 0.12)
        y = 0.113 + s * L
        verts += [(-w, y, 0), (0, y, t0), (w, y, 0), (0, y, -t0)]
    for i in range(len(stations) - 1):
        for k in range(4):
            p0, p1 = i * 4 + k, i * 4 + (k + 1) % 4
            faces += [(p0, p1, p1 + 4), (p0, p1 + 4, p0 + 4)]
    verts = np.array(verts)
    faces = geo._orient(verts, faces, None, None) if False else np.array(faces)
    tri = verts[faces]
    centre = np.stack([np.zeros(len(faces)), tri[:, :, 1].mean(axis=1), np.zeros(len(faces))], axis=1)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.einsum("ij,ij->i", fn, tri.mean(axis=1) - centre) < 0
    faces[flip] = faces[flip][:, ::-1]
    tri = verts[faces]
    fn = unit(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
    col = np.where((np.abs(tri.reshape(-1, 3)[:, 0]) < 0.004)[:, None], C(0.62, 0.64, 0.68), C(0.86, 0.88, 0.92))
    a.add("steel", tri.reshape(-1, 3), col, np.arange(len(faces) * 3).reshape(-1, 3), np.repeat(fn, 3, axis=0))


def axe(a):
    a.name = "axe"
    cylinder(a, "wood", (0, -0.18, 0), (0, 0.62, 0), 0.02, 0.018, colour=lambda v: grain(v, C(0.48, 0.32, 0.18), 0.15, 4, 3, (1, 6, 1)), sides=8)
    for y in (-0.16, -0.06):
        cylinder(a, "leather", (0, y, 0), (0, y + 0.07, 0), 0.023, colour=C(0.28, 0.17, 0.1), sides=8)
    # head: a wedge cheek flaring into a curved bit, with a butt on the back
    top = 0.56
    pts = []
    for k in range(9):
        t = k / 8
        ang = (t - 0.5) * 1.2
        pts.append((0.06 + 0.12 * math.cos(ang) * 1.0 + 0.0, top + math.sin(ang) * 0.12))
    outline = [(-0.04, top + 0.035), (0.06, top + 0.04)] + pts[::-1] + [(0.06, top - 0.04), (-0.04, top - 0.035)]
    outline = np.array(outline)
    n = len(outline)
    verts = []
    for x, y in outline:
        th = 0.02 * (1 - min(1, max(0, (x - 0.02) / 0.16))) + 0.002
        verts += [(x, y, th), (x, y, -th)]
    verts = np.array(verts)
    faces = []
    cx = outline.mean(axis=0)
    centre_f = np.array([cx[0], cx[1], 0.02])
    centre_b = np.array([cx[0], cx[1], -0.02])
    verts = np.vstack([verts, centre_f, centre_b])
    cf, cb = len(verts) - 2, len(verts) - 1
    for i in range(n):
        j = (i + 1) % n
        faces += [(cf, 2 * i, 2 * j), (cb, 2 * j + 1, 2 * i + 1), (2 * i, 2 * i + 1, 2 * j + 1), (2 * i, 2 * j + 1, 2 * j)]
    faces = geo._orient(verts, faces, None, np.array([cx[0], cx[1], 0]))
    tri = verts[faces]
    fn = unit(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
    flat = tri.reshape(-1, 3)
    col = np.where((flat[:, 0] > 0.14)[:, None], C(0.85, 0.86, 0.88), C(0.42, 0.42, 0.45))
    a.add("steel", flat, col, np.arange(len(faces) * 3).reshape(-1, 3), np.repeat(fn, 3, axis=0))
    box(a, "iron", (-0.045, top, 0), (0.05, 0.07, 0.05), C(0.35, 0.35, 0.38))


def bag(a):
    a.name = "bag"
    rounded_box(a, "leather", (0, 0, 0), (0.08, 0.2, 0.26), lambda v: grain(v, C(0.45, 0.28, 0.15), 0.12, 6, 2), levels=2, lumpy=0.004)
    rounded_box(a, "leather", (0.025, 0.05, 0), (0.05, 0.12, 0.27), C(0.38, 0.23, 0.12), levels=2)        # flap
    box(a, "gold", (0.052, -0.005, 0), (0.008, 0.035, 0.03), C(0.82, 0.64, 0.28))
    tube(a, "leather", [(0.0, 0.1, -0.12), (-0.06, 0.32, -0.12), (-0.2, 0.55, -0.1)], 0.012, C(0.32, 0.2, 0.1), sides=5, cap=False)
    tube(a, "leather", [(0.0, 0.1, 0.12), (-0.06, 0.32, 0.12), (-0.2, 0.55, 0.12)], 0.012, C(0.32, 0.2, 0.1), sides=5, cap=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fn, direction in ((sword, (0.0, -0.18, 1.0)), (axe, (0.0, -0.18, 1.0))):
        held = place(fn, PALM, np.array(direction))
        held.name = fn.__name__
        held.write(OUT / f"{fn.__name__}.glb", MATS)
        icon = place(fn, np.zeros(3), np.array([0, 0, 1.0]))
        icon.name = fn.__name__ + "_icon"
        icon.write(OUT / f"{fn.__name__}_icon.glb", MATS)
    a = geo.Asset("bag")
    bag(a)
    out = geo.Asset("bag")
    for mname, part in a.parts.items():
        v, n, c, f = part.arrays()
        out.add(mname, v + HIP_LEFT, c, f, n)
    out.write(OUT / "bag.glb", MATS)


main()
