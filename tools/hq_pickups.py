"""Things Robin can pick up and carry in the satchel.

  meat    a severed tentacle segment from the horror: wet violet-grey skin with two rows of suckers, the cut
          end raw red muscle with a pale core ring
  core    a golem's heart-stone: a fist-sized split rock whose cracks glow a cold blue
  log     a short split length of pine (scanned pine bark, end grain with rings and radial checks)
  apple, pear   Poly Haven scans (CC0), decimated, textures reduced to 512

Writes assets_src/items/pickups_hq.glb (meat, core, log) and assets_src/items/pickups_food.glb (apple, pear).
Run: blender --background --factory-startup --python tools/hq_pickups.py
"""
import math
import pathlib
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bl_pbr as B  # noqa: E402

PH = B.ROOT / "ph"
OUT = B.ROOT / "assets_src" / "items"
rng = np.random.default_rng(7)


def lin(*c):
    return tuple(x ** 2.2 for x in c)


def skin(nd):
    oc = nd.coords("Object")
    m = nd.sep(nd.attr("mask"))                     # R: cut face, G: sucker rim, B: underside
    blot = nd.noise(oc, 14.0, 6.0, 0.6)
    col = nd.mix(nd.ramp(blot, 0.35, 0.7), lin(0.17, 0.11, 0.17), lin(0.30, 0.19, 0.25))
    col = nd.mix(nd.ramp(nd.noise(oc, 60.0, 3.0), 0.55, 0.7, 0, 0.6), col, lin(0.09, 0.05, 0.10))   # freckles
    vein = nd.ramp(nd.voronoi(nd.vmath("MULTIPLY", oc, (1.0, 1.0, 0.35)), 22.0, "DISTANCE_TO_EDGE", "Distance"), 0.01, 0.0)
    col = nd.mix(nd.math("MULTIPLY", vein, 0.45), col, lin(0.36, 0.10, 0.16))                        # dark red veins
    col = nd.mix(m[2], col, lin(0.50, 0.36, 0.38))                                                   # pale underside
    col = nd.mix(m[1], col, lin(0.66, 0.50, 0.50))                                                   # sucker rims
    # the cut: dark red muscle, fibrous, with a pale nerve ring and a glossy wet film
    fibre = nd.noise(nd.vmath("MULTIPLY", oc, (90.0, 90.0, 8.0)), 1.0, 4.0, 0.6)
    flesh = nd.mix(nd.ramp(fibre, 0.35, 0.65), lin(0.36, 0.05, 0.06), lin(0.58, 0.12, 0.12))
    r = nd.math("SQRT", nd.math("ADD", nd.math("POWER", nd.sep(oc)[0], 2.0), nd.math("POWER", nd.sep(oc)[1], 2.0)))
    ring = nd.ramp(nd.math("ABSOLUTE", nd.math("SUBTRACT", r, 0.022)), 0.006, 0.0)
    flesh = nd.mix(ring, flesh, lin(0.80, 0.66, 0.60))
    col = nd.mix(m[0], col, flesh)
    rough = nd.mixf(m[0], nd.mixf(nd.ramp(blot, 0.3, 0.7), 0.30, 0.42), 0.18)
    nd.bump(nd.mixf(m[0], nd.noise(oc, 120.0, 4.0), fibre), 0.35, 0.004)
    nd.done(col, rough, 0.0, (0, 0, 0))


def heartstone(nd):
    oc = nd.coords("Object")
    scan = nd.image(next((PH / "rock_face_03").glob("*_diff_2k.jpg")), oc, 3.0)
    height = nd.image(next((PH / "rock_face_03").glob("*_disp_2k.png")), oc, 3.0, "Non-Color")
    cool = nd.mix(0.45, scan, lin(0.42, 0.45, 0.50), "MULTIPLY")
    # a few thin, wandering fissures: cell edges of a noise-warped voronoi, thinned where a second noise is low
    warped = nd.vmath("ADD", oc, nd.vmath("SCALE", nd.noise(oc, 7.0, 4.0, 0.6, out="Color"), (0, 0, 0), 0.06))
    crack = nd.voronoi(warped, 5.0, "DISTANCE_TO_EDGE", "Distance")
    width = nd.ramp(nd.noise(oc, 4.0, 2.0), 0.35, 0.65, 0.0, 0.016)
    glow = nd.ramp(nd.math("SUBTRACT", crack, width), 0.006, 0.0)
    col = nd.mix(glow, cool, lin(0.55, 0.85, 1.0))
    nd.bump(nd.math("SUBTRACT", nd.sep(height)[0], nd.math("MULTIPLY", glow, 0.6)), 0.8, 0.01)
    emit = nd.mix(glow, (0, 0, 0), lin(0.35, 0.75, 1.0))
    nd.done(col, nd.mixf(glow, 0.86, 0.3), 0.0, emit)


def logwood(nd):
    oc = nd.coords("Object")
    m = nd.sep(nd.attr("mask"))                     # R: end grain
    bark = nd.image(next((PH / "pine_bark").glob("*_diff_2k.jpg")), oc, 2.2)
    bh = nd.image(next((PH / "pine_bark").glob("*_disp_2k.png")), oc, 2.2, "Non-Color")
    s = nd.sep(oc)
    r = nd.math("SQRT", nd.math("ADD", nd.math("POWER", s[0], 2.0), nd.math("POWER", s[1], 2.0)))
    warp = nd.noise(oc, 6.0, 3.0)
    rings = nd.math("SINE", nd.math("MULTIPLY", nd.math("ADD", r, nd.math("MULTIPLY", warp, 0.006)), 520.0))
    endc = nd.mix(nd.ramp(rings, -1, 1), lin(0.62, 0.46, 0.28), lin(0.80, 0.64, 0.42))
    endc = nd.mix(nd.ramp(r, 0.085, 0.11), endc, lin(0.70, 0.52, 0.30))                  # sapwood
    check = nd.ramp(nd.voronoi(nd.vmath("MULTIPLY", oc, (1.0, 1.0, 0.0)), 14.0, "DISTANCE_TO_EDGE", "Distance"), 0.012, 0.0)
    endc = nd.mix(nd.math("MULTIPLY", check, 0.85), endc, lin(0.20, 0.13, 0.08))         # radial checks
    col = nd.mix(m[0], bark, endc)
    nd.bump(nd.mixf(m[0], nd.sep(bh)[0], nd.math("MULTIPLY", nd.math("SUBTRACT", 1.0, check), 0.3)), 0.9, 0.01)
    nd.done(col, nd.mixf(m[0], 0.85, 0.7), 0.0, (0, 0, 0))


MAT = {}


def add(model, part, bm, mat, M, mask=None, subdiv=0, smooth=True):
    B.transform_bm(bm, M)
    hi = B.from_bmesh(mat, bm)
    if subdiv:
        s = hi.modifiers.new("s", "SUBSURF")
        s.levels = subdiv
        B.apply_mods(hi)
    if mask is not None:
        B.set_point_colour(hi, "mask", mask(B.verts_np(hi), M))
    if smooth:
        B.smooth(hi, 50)
    B.material(hi, MAT[mat])
    return hi


def local(v, M):
    return (np.array(M.inverted()) @ np.c_[v, np.ones(len(v))].T)[:3].T


def cup(s, sides=20):
    """A sucker: a revolved profile, a raised fleshy rim around a deep hollow."""
    prof = [(0.0, -0.05), (0.22, -0.02), (0.42, 0.14), (0.55, 0.42), (0.68, 0.52), (0.84, 0.46), (0.98, 0.18), (1.04, -0.12)]
    bm = bmesh.new()
    rings = [[bm.verts.new((r * s * math.cos(a), r * s * math.sin(a), z * s)) for a in np.linspace(0, 2 * math.pi, sides, endpoint=False)]
             for r, z in prof[1:]]
    centre = bm.verts.new((0, 0, prof[0][1] * s))
    for k in range(sides):
        bm.faces.new((centre, rings[0][(k + 1) % sides], rings[0][k]))
    for a, b in zip(rings, rings[1:]):
        for k in range(sides):
            bm.faces.new((a[k], a[(k + 1) % sides], b[(k + 1) % sides], b[k]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def tentacle(model, M):
    # tapering, gently curled, slightly flattened underneath; cut square at the thick end (z = 0)
    n, L = 26, 0.42
    pts = [Vector((0.04 * math.sin(t * 2.2) - 0.04 * t * t, 0.0, L * t)) for t in np.linspace(0, 1, n)]
    rad = [0.048 * (1 - 0.72 * t) + 0.004 for t in np.linspace(0, 1, n)]
    bm = B.tube_bm(pts, rad, 28, True, profile=[0.86 if 10 <= k <= 18 else 1.0 for k in range(28)])
    hi = add(model, "meat", bm, "skin", M, subdiv=1, smooth=True)
    v = B.verts_np(hi)
    lv = local(v, M)
    # wrinkles around the body, a little noise
    nrm = B.vnormals(hi)
    wr = 0.0016 * np.sin(lv[:, 2] * 260 + np.sin(lv[:, 0] * 40) * 2)
    B.set_verts(hi, v + nrm * wr[:, None])
    rr = np.hypot(lv[:, 0] - np.interp(lv[:, 2], [p.z for p in pts], [p.x for p in pts]), lv[:, 1]) + 1e-6
    under = np.clip((-lv[:, 1] / rr - 0.55) * 4, 0, 1)          # a pale strip along the sucker side only
    cut = (lv[:, 2] < 0.004).astype(float)
    B.set_point_colour(hi, "mask", np.stack([cut, 0 * cut, under * (1 - cut), np.ones_like(cut)], 1))
    parts = [hi]
    # two rows of suckers on the underside, shrinking toward the tip
    for row in (-1, 1):
        for k in range(9):
            t = 0.06 + k * 0.1
            if t > 0.92:
                break
            i = min(int(t * (n - 1)), n - 2)
            p = pts[i]
            r = rad[i]
            a = math.pi * 1.5 + row * 0.42
            c = Vector((p.x + math.cos(a) * r * 0.9, math.sin(a) * r * 0.86 * 0.95, p.z))
            s = r * 0.32
            sb = cup(s)
            out = (c - Vector((p.x, 0, p.z))).normalized()
            Ms = M @ Matrix.Translation(c) @ B.look(out)
            h = add(model, "meat", sb, "skin", Ms, subdiv=1)
            B.set_point_colour(h, "mask", np.tile([0, 1, 0, 1], (len(h.data.vertices), 1)).astype(float))
            parts.append(h)
    lo = B.join([B.copy_obj(o, "meat_piece") for o in parts], "meat_lo")
    B.decimate(lo, 2400)
    model.add("meat", parts, lo)


def core(model, M):
    bm = B.sphere_bm((0.09, 0.075, 0.08), 32, 22)
    hi = add(model, "core", bm, "heartstone", M, subdiv=0, smooth=False)
    v = B.verts_np(hi)
    lv = local(v, M)
    # break it into planes: clamp against a few random cleavage planes, then roughen
    for _ in range(9):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        off = rng.uniform(0.055, 0.075)
        proj = lv @ d
        over = np.maximum(proj - off, 0)
        lv = lv - over[:, None] * d[None, :] * 0.92
    lv *= 1 + 0.06 * np.sin(lv[:, [1]] * 90 + lv[:, [2]] * 70)
    B.set_verts(hi, (np.array(M) @ np.c_[lv, np.ones(len(lv))].T)[:3].T)
    B.smooth(hi, 35)
    lo = B.copy_obj(hi, "core_lo")
    B.decimate(lo, 900)
    model.add("core", hi, lo)


def log(model, M):
    L, R = 0.62, 0.105
    pts = [Vector((0, 0, z)) for z in np.linspace(-L / 2, L / 2, 12)]
    bm = B.tube_bm(pts, [R * (1 + 0.04 * math.sin(i * 1.7)) for i in range(12)], 36, True)
    hi = add(model, "log", bm, "logwood", M, subdiv=1, smooth=True)
    v = B.verts_np(hi)
    lv = local(v, M)
    endg = (np.abs(lv[:, 2]) > L / 2 - 0.004).astype(float)
    nrm = B.vnormals(hi)
    side = 1 - endg
    bumpy = 0.004 * np.sin(np.arctan2(lv[:, 1], lv[:, 0]) * 11 + lv[:, 2] * 9) * side
    B.set_verts(hi, v + nrm * bumpy[:, None])
    B.set_point_colour(hi, "mask", np.stack([endg, 0 * endg, 0 * endg, np.ones_like(endg)], 1))
    lo = B.copy_obj(hi, "log_lo")
    B.decimate(lo, 1100)
    model.add("log", hi, lo)


def food():
    """The two scans: largest mesh of each file, centred on its base, decimated, textures at 512."""
    B.reset()
    out = []
    for name, rel in (("apple", "food_apple_01/food_apple_01_1k.gltf"), ("pear", "food_pears_asian_01/food_pears_asian_01_1k.gltf")):
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(PH / rel))
        new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
        new.sort(key=lambda o: -len(o.data.polygons))
        keep = new[0]
        for o in new[1:]:
            bpy.data.objects.remove(o, do_unlink=True)
        mw = keep.matrix_world.copy()
        keep.parent = None
        keep.matrix_world = mw
        bpy.context.view_layer.objects.active = keep
        bpy.ops.object.select_all(action="DESELECT")
        keep.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        v = B.verts_np(keep)
        c = np.array([(v[:, 0].min() + v[:, 0].max()) / 2, (v[:, 1].min() + v[:, 1].max()) / 2, v[:, 2].min()])
        B.set_verts(keep, v - c)
        keep.location = (0, 0, 0)
        B.decimate(keep, 2400)
        B.smooth(keep, 60)
        keep.name = keep.data.name = name
        for slot in keep.material_slots:
            for n in slot.material.node_tree.nodes:
                if n.type == "TEX_IMAGE" and n.image and max(n.image.size) > 512:
                    n.image.scale(512, 512)
        size = (v.max(0) - v.min(0))
        print(f"[food] {name}: {len(keep.data.polygons)} faces, size {size.round(3)}")
        out.append(keep)
    for o in list(bpy.data.objects):
        if o not in out:
            bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    for o in out:
        o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(OUT / "pickups_food.glb"), export_format="GLB", use_selection=True, export_yup=True,
                              export_image_format="JPEG", export_animations=False)
    print("[export]", OUT / "pickups_food.glb")


def main():
    B.reset()
    B.MATS.clear()
    for k, f in (("skin", skin), ("heartstone", heartstone), ("logwood", logwood)):
        MAT[k] = B.new_material(k, f)
    G = B.Model("pickups")
    spots = {"meat": Vector((0, 0, 0)), "core": Vector((1.0, 0, 0)), "log": Vector((2.0, 0, 0))}
    for k, p in spots.items():
        G.part(k, p)
    # meat lies on its side: the tentacle runs along +X, suckers face down-ish
    tentacle(G, Matrix.Translation(spots["meat"] + Vector((-0.2, 0, 0.05))) @ Matrix.Rotation(math.pi / 2, 4, "Y") @ Matrix.Rotation(-0.5, 4, "Z"))
    core(G, Matrix.Translation(spots["core"] + Vector((0, 0, 0.07))))
    log(G, Matrix.Translation(spots["log"] + Vector((0, 0, 0.105))) @ Matrix.Rotation(math.pi / 2, 4, "Y"))
    G.bake(size=1024, extrusion=0.008, ray=0.03, ao=True)
    G.export(OUT / "pickups_hq.glb", emissive_strength=3.0)
    food()


main()
