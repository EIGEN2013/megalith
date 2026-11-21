"""Robin's sword, axe and satchel, modelled in detail and baked to PBR textures.

  sword   diamond-section steel blade with a fuller, ground edges and fine scratches; curved brass
          crossguard with knobbed ends; leather-wrapped grip (spiral ridges); faceted brass pommel
  axe     bearded forged-iron head with a bright ground edge, hammered faces; tapered ash haft with grain,
          leather grip wrap and an iron butt cap
  bag     leather satchel: rounded body, flap, stitched seams, brass buckle, strap

Each is placed in Robin's rest-pose model space (sword and axe at the right palm, bag at the left hip) so the
game can hang it on his bones. Writes assets_src/items/items_hq.glb (nodes sword, axe, bag).
Run: blender --background --factory-startup --python tools/hq_items.py
"""
import math
import pathlib
import sys

import bmesh
import numpy as np
from mathutils import Matrix, Vector

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bl_pbr as B  # noqa: E402

PALM = np.array([-0.345, 0.885, 0.075])            # glTF, Robin's right palm (rest pose)
HIP_LEFT = np.array([0.215, 0.86, 0.02])
AIM = np.array([0.0, -0.18, 1.0])


def bl(p):
    p = np.asarray(p, float)
    return Vector((p[0], -p[2], p[1]))


def lin(*c):
    return tuple(x ** 2.2 for x in c)


def steel(nd):
    oc = nd.coords("Object")
    m = nd.sep(nd.attr("mask"))                       # R: edge bevel, G: fuller
    scratch = nd.noise(nd.vmath("MULTIPLY", oc, (60.0, 60.0, 2.0)), 1.0, 2.0, 0.5)
    sc = nd.ramp(scratch, 0.62, 0.7)
    col = nd.mix(m[1], lin(0.70, 0.71, 0.72), lin(0.52, 0.53, 0.55))
    col = nd.mix(m[0], col, lin(0.88, 0.89, 0.9))
    col = nd.mix(nd.ramp(nd.noise(oc, 25.0, 6.0), 0.55, 0.75, 0, 0.35), col, lin(0.36, 0.33, 0.30))   # faint tarnish
    rough = nd.mixf(sc, nd.mixf(m[0], 0.22, 0.12), 0.42)
    nd.done(col, rough, 1.0, (0, 0, 0))


def brass(nd):
    oc = nd.coords("Object")
    pt = nd.geometry("Pointiness")
    col = nd.mix(nd.ramp(pt, 0.5, 0.42, 0, 0.8), lin(0.80, 0.60, 0.30), lin(0.25, 0.17, 0.08))
    col = nd.mix(nd.ramp(nd.noise(oc, 40.0, 4.0), 0.6, 0.75, 0, 0.4), col, lin(0.45, 0.38, 0.22))
    nd.done(col, nd.ramp(pt, 0.5, 0.42, 0.3, 0.7), 1.0, (0, 0, 0))


def leather(nd):
    oc = nd.coords("Object")
    m = nd.sep(nd.attr("mask"))                       # R: stitches
    grain = nd.noise(oc, 180.0, 3.0, 0.6)
    col = nd.mix(nd.ramp(nd.noise(oc, 9.0, 5.0), 0.3, 0.7), lin(0.20, 0.11, 0.06), lin(0.36, 0.21, 0.11))
    col = nd.mix(nd.ramp(grain, 0.45, 0.6, 0, 0.35), col, lin(0.12, 0.07, 0.04))
    pt = nd.geometry("Pointiness")
    col = nd.mix(nd.ramp(pt, 0.5, 0.58, 0, 0.5), col, lin(0.48, 0.32, 0.18))                    # worn edges
    col = nd.mix(m[0], col, lin(0.62, 0.55, 0.42))                                              # thread
    nd.done(col, nd.mixf(nd.ramp(grain, 0.3, 0.7), 0.55, 0.78), 0.0, (0, 0, 0))


def iron(nd):
    oc = nd.coords("Object")
    m = nd.sep(nd.attr("mask"))                       # R: ground cutting edge
    hammer = nd.voronoi(oc, 45.0, "F1", "Distance")
    col = nd.mix(nd.ramp(hammer, 0.1, 0.5), lin(0.16, 0.16, 0.17), lin(0.30, 0.30, 0.31))
    col = nd.mix(nd.ramp(nd.noise(oc, 12.0, 6.0), 0.58, 0.72, 0, 0.5), col, lin(0.30, 0.15, 0.07))   # rust bloom
    col = nd.mix(m[0], col, lin(0.80, 0.81, 0.82))
    nd.done(col, nd.mixf(m[0], nd.ramp(hammer, 0.1, 0.5, 0.35, 0.6), 0.18), 1.0, (0, 0, 0))


def wood(nd):
    oc = nd.coords("Object")
    warp = nd.noise(oc, 3.0, 4.0)
    rings = nd.math("SINE", nd.math("MULTIPLY", nd.math("ADD", nd.sep(oc)[0], nd.math("MULTIPLY", warp, 0.03)), 900.0))
    fibre = nd.noise(nd.vmath("MULTIPLY", oc, (150.0, 150.0, 4.0)), 1.0, 3.0)
    col = nd.mix(nd.ramp(rings, -1, 1), lin(0.42, 0.30, 0.18), lin(0.58, 0.44, 0.28))
    col = nd.mix(nd.ramp(fibre, 0.4, 0.7, 0, 0.4), col, lin(0.30, 0.20, 0.12))
    nd.done(col, 0.62, 0.0, (0, 0, 0))


MAT = {}


def mats():
    B.MATS.clear()
    for k, f in (("steel", steel), ("brass", brass), ("leather", leather), ("iron", iron), ("wood", wood)):
        MAT[k] = B.new_material(k, f)


def piece(model, part, bm, mat, M, mask=None, smooth=True, subdiv=0):
    B.transform_bm(bm, M)
    hi = B.from_bmesh(mat, bm)
    if subdiv:
        s = hi.modifiers.new("s", "SUBSURF")
        s.levels = subdiv
        B.apply_mods(hi)
    if mask is not None:
        v = B.verts_np(hi)
        B.set_point_colour(hi, "mask", mask(v))
    if smooth:
        B.smooth(hi, 40)
    B.material(hi, MAT[mat])
    lo = B.copy_obj(hi, mat + "_lo")
    model.add(part, hi, lo)
    return hi


def frame_at(origin, direction, roll=0.0):
    return Matrix.Translation(bl(origin)) @ B.look(bl(direction) if False else Vector(bl(np.array(direction)) - bl(np.zeros(3)))) @ Matrix.Rotation(roll, 4, "Z")


def sword(model, M):
    # blade along local +Z from the guard (z = 0.11) to the tip (z = 0.97)
    L0, L1, n = 0.115, 0.97, 24
    bm = bmesh.new()
    rings = []
    for i in range(n + 1):
        t = i / n
        z = L0 + (L1 - L0) * t
        w = 0.026 * (1 - 0.22 * t) * (1 if t < 0.86 else max(0.02, (1 - t) / 0.14) ** 0.8)
        th = 0.0055 * (1 - 0.5 * t) + 0.0008
        fd = th * (0.35 if t < 0.72 else 1.0)              # the fuller: a shallow groove down the middle
        prof = [(-w, 0), (-w * 0.55, th), (-w * 0.16, th), (0, fd), (w * 0.16, th), (w * 0.55, th), (w, 0),
                (w * 0.55, -th), (w * 0.16, -th), (0, -fd), (-w * 0.16, -th), (-w * 0.55, -th)]
        rings.append([bm.verts.new((x, y, z)) for x, y in prof])
    for a, b in zip(rings, rings[1:]):
        for k in range(12):
            bm.faces.new((a[k], a[(k + 1) % 12], b[(k + 1) % 12], b[k]))
    bm.faces.new(rings[0][::-1])
    tip = bm.verts.new((0, 0, L1 + 0.012))
    for k in range(12):
        bm.faces.new((rings[-1][k], rings[-1][(k + 1) % 12], tip))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    piece(model, "sword", bm, "steel", M,
          mask=lambda v: np.stack([np.clip((np.abs(v[:, 0]) - 0.0) * 0, 0, 0) + 0, 0 * v[:, 0], 0 * v[:, 0], 0 * v[:, 0]], 1), smooth=False)
    # guard: a curved bar with knobbed ends
    pts = [Vector((x, 0, 0.1 + 0.012 * (x / 0.11) ** 2)) for x in np.linspace(-0.11, 0.11, 9)]
    piece(model, "sword", B.tube_bm(pts, [0.009 + 0.004 * (1 - abs(p.x) / 0.11) for p in pts], 12, True), "brass", M, subdiv=1)
    for sx in (-1, 1):
        piece(model, "sword", B.sphere_bm((0.014,) * 3, 14, 10), "brass", M @ Matrix.Translation((sx * 0.115, 0, 0.115)))
    piece(model, "sword", B.box_bm((0.05, 0.022, 0.024), bevel=0.004, segments=2), "brass", M @ Matrix.Translation((0, 0, 0.1)))
    # grip: leather in a spiral wrap, brass ferrule, faceted pommel
    gp = [Vector((0, 0, z)) for z in np.linspace(-0.1, 0.088, 20)]
    g = B.tube_bm(gp, [0.016] * 20, 16, True)
    hi = piece(model, "sword", g, "leather", M, subdiv=1,
               mask=lambda v: np.stack([np.clip(1 - np.abs(np.sin((np.arctan2(v[:, 1], v[:, 0]) * 1 + v[:, 2] * 120))) * 6, 0, 1) * 0, 0 * v[:, 0], 0 * v[:, 0], 0 * v[:, 0]], 1))
    vv = B.verts_np(hi)
    loc = np.array(M.inverted()) @ np.c_[vv, np.ones(len(vv))].T
    ang = np.arctan2(loc[1], loc[0])
    ridge = np.abs(np.sin(ang * 0.5 + loc[2] * 110.0))
    n = B.vnormals(hi)
    B.set_verts(hi, vv + n * (0.0018 * ridge)[:, None])
    piece(model, "sword", B.tube_bm([Vector((0, 0, -0.112)), Vector((0, 0, -0.098))], 0.019, 16, True), "brass", M)
    piece(model, "sword", B.sphere_bm((0.024, 0.024, 0.02), 8, 6), "brass", M @ Matrix.Translation((0, 0, -0.13)))


def axe(model, M):
    # haft: tapered oval, slightly swelling at the grip, along local +Z
    hz = np.linspace(-0.2, 0.6, 18)
    pts = [Vector((0.006 * math.sin(z * 4), 0, z)) for z in hz]
    piece(model, "axe", B.tube_bm(pts, [0.019 - 0.004 * (z + 0.2) / 0.8 for z in hz], 14, True, profile=[1.25 if k % 7 in (0,) else 1.0 for k in range(14)]),
          "wood", M, subdiv=1)
    for z0 in (-0.17, -0.1, -0.03):
        piece(model, "axe", B.tube_bm([Vector((0, 0, z0)), Vector((0, 0, z0 + 0.06))], 0.0215, 14, True), "leather", M)
    piece(model, "axe", B.tube_bm([Vector((0, 0, -0.215)), Vector((0, 0, -0.195))], 0.021, 14, True), "iron", M)
    # head: bearded profile in the XZ plane, thick at the eye, thin at the edge
    prof = [(-0.035, 0.53), (0.02, 0.535), (0.07, 0.548), (0.13, 0.585), (0.165, 0.6), (0.172, 0.56), (0.17, 0.5),
            (0.16, 0.44), (0.13, 0.41), (0.075, 0.45), (0.03, 0.47), (-0.035, 0.47)]
    bm = bmesh.new()
    k = len(prof)
    front, back, mid = [], [], []
    for x, z in prof:
        th = 0.0045 + 0.017 * max(0.0, 1 - (x + 0.035) / 0.2) ** 1.6
        front.append(bm.verts.new((x, th, z)))
        back.append(bm.verts.new((x, -th, z)))
    bm.faces.new(front)
    bm.faces.new(back[::-1])
    for i in range(k):
        j = (i + 1) % k
        bm.faces.new((front[i], back[i], back[j], front[j]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    piece(model, "axe", bm, "iron", M, subdiv=2, smooth=True,
          mask=lambda v: np.stack([np.clip((np.array(M.inverted()) @ np.c_[v, np.ones(len(v))].T)[0] * 40 - 5.6, 0, 1), 0 * v[:, 0], 0 * v[:, 0], 0 * v[:, 0]], 1))
    piece(model, "axe", B.box_bm((0.07, 0.05, 0.075), bevel=0.008, segments=3), "iron", M @ Matrix.Translation((-0.005, 0, 0.5)))
    piece(model, "axe", B.box_bm((0.03, 0.04, 0.05), bevel=0.006, segments=2), "iron", M @ Matrix.Translation((-0.05, 0, 0.5)))


def bag(model, M):
    body = B.box_bm((0.085, 0.27, 0.2), bevel=0.03, segments=4)
    piece(model, "bag", body, "leather", M, subdiv=1,
          mask=lambda v: np.stack([np.clip(1 - np.abs(np.abs((np.array(M.inverted()) @ np.c_[v, np.ones(len(v))].T)[1]) - 0.118) * 300, 0, 1)
                                   * (np.sin((np.array(M.inverted()) @ np.c_[v, np.ones(len(v))].T)[2] * 500) > 0), 0 * v[:, 0], 0 * v[:, 0], 0 * v[:, 0]], 1))
    flap = B.box_bm((0.02, 0.28, 0.13), bevel=0.008, segments=3)
    piece(model, "bag", flap, "leather", M @ Matrix.Translation((0.05, 0, 0.04)) @ Matrix.Rotation(0.12, 4, "Y"), subdiv=1)
    piece(model, "bag", B.box_bm((0.012, 0.045, 0.035), bevel=0.004, segments=2), "brass", M @ Matrix.Translation((0.063, 0, -0.015)))
    piece(model, "bag", B.box_bm((0.006, 0.03, 0.05), bevel=0.002, segments=1), "leather", M @ Matrix.Translation((0.06, 0, 0.0)))
    for sy in (-1, 1):
        pts = [Vector((0, sy * 0.13, 0.09)), Vector((-0.06, sy * 0.12, 0.25)), Vector((-0.2, sy * 0.1, 0.42))]
        piece(model, "bag", B.tube_bm(pts, 0.007, 6, True, profile=[2.2, 1, 0.5, 1, 2.2, 1]), "leather", M)


def main():
    B.reset()
    mats()
    G = B.Model("items")
    G.part("sword", bl(PALM))
    G.part("axe", bl(PALM))
    G.part("bag", bl(HIP_LEFT))
    aim = (bl(AIM) - bl(np.zeros(3))).normalized()
    Mw = Matrix.Translation(bl(PALM)) @ B.look(aim)
    sword(G, Mw)
    axe(G, Mw)
    bag(G, Matrix.Translation(bl(HIP_LEFT)))
    G.bake(size=1024, extrusion=0.006, ray=0.02, ao=True)
    G.export(B.ROOT / "assets_src" / "items" / "items_hq.glb")


main()
