"""The tentacle horror, sculpted and baked: fused clay-like flesh (voxel remesh) with wrinkles,
veins, warts and a pale moist belly; six glossy slit-pupil eyes in lidded sockets; a toothed maw;
bone horns, spines and claws; veined, torn fins; tentacles ringed with wrinkles and suckers.

Parts (unchanged names): body > head, fin_l/r, arm_l/r > fore_l/r > hand_l/r, tent_{0..5}_{0..4} chains.
Run: blender --background --factory-startup --python tools/hq_horror.py
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
import noise3 as N  # noqa: E402

rng = np.random.default_rng(66)
S = 0.85
VOX = 0.0075


def lin(*c):
    return tuple(x ** 2.2 for x in c)


def bl(p):
    """glTF (x, y up, z forward) -> Blender (x, -z, y)."""
    p = np.asarray(p, float)
    return np.array([p[0], -p[2], p[1]])


# ----------------------------------------------------------------------------- materials
def flesh_mat(nd):
    oc = nd.coords("Object")
    m = nd.sep(nd.attr("mask"))
    wart = nd.attr("mask", "Alpha")
    n1 = nd.noise(oc, 3.5, 8.0, 0.6)
    col = nd.mix(nd.ramp(n1, 0.35, 0.65), lin(0.17, 0.20, 0.17), lin(0.33, 0.37, 0.30))
    blot = nd.noise(oc, 13.0, 5.0, 0.6)
    col = nd.mix(nd.ramp(blot, 0.55, 0.72, 0, 0.75), col, lin(0.12, 0.13, 0.11))
    col = nd.mix(nd.ramp(nd.noise(oc, 7.0, 4.0), 0.6, 0.75, 0, 0.5), col, lin(0.40, 0.30, 0.26))          # bruised patches
    col = nd.mix(m[1], col, nd.mix(nd.ramp(n1, 0.3, 0.7), lin(0.56, 0.52, 0.42), lin(0.68, 0.62, 0.50)))  # pale belly
    col = nd.mix(m[0], col, lin(0.30, 0.09, 0.17))                                                          # veins
    col = nd.mix(wart, col, lin(0.52, 0.47, 0.30))                                                          # warts
    pt = nd.geometry("Pointiness")
    col = nd.mix(nd.ramp(pt, 0.5, 0.42, 0, 0.75), col, nd.mix(1.0, col, (0.3, 0.28, 0.3), "MULTIPLY"))      # creases
    col = nd.mix(nd.ramp(pt, 0.5, 0.57, 0, 0.4), col, nd.mix(1.0, col, (1.4, 1.4, 1.35), "MULTIPLY"))
    col = nd.mix(m[2], col, nd.mix(nd.ramp(nd.noise(oc, 30.0, 3.0), 0.4, 0.6), lin(0.55, 0.10, 0.10), lin(0.72, 0.32, 0.34)))  # raw / suckers
    rough = nd.mixf(m[1], nd.ramp(blot, 0.3, 0.8, 0.42, 0.58), 0.3)
    rough = nd.mixf(m[2], rough, 0.18)
    nd.done(col, rough, 0.0, (0, 0, 0))


def bone_mat(nd):
    oc = nd.coords("Object")
    t = nd.sep(nd.attr("mask"))[0]
    col = nd.mix(nd.ramp(t, 0.0, 0.7), lin(0.30, 0.24, 0.17), lin(0.82, 0.76, 0.60))
    col = nd.mix(nd.ramp(nd.noise(oc, 40.0, 4.0), 0.55, 0.7, 0, 0.5), col, lin(0.45, 0.38, 0.28))
    col = nd.mix(nd.ramp(nd.voronoi(oc, 25.0, "DISTANCE_TO_EDGE"), 0.03, 0.0, 0, 0.6), col, lin(0.25, 0.2, 0.15))
    nd.done(col, nd.ramp(t, 0, 1, 0.55, 0.32), 0.0, (0, 0, 0))


def eye_mat(nd):
    m = nd.sep(nd.attr("mask"))
    u, v, front = m[0], m[1], m[2]          # u, v in [-1, 1] across the eye, front = facing forward
    r = nd.math("SQRT", nd.math("ADD", nd.math("MULTIPLY", u, u), nd.math("MULTIPLY", v, v)))
    iris = nd.math("MULTIPLY", nd.ramp(r, 0.72, 0.62), nd.ramp(front, 0.2, 0.5))
    slit = nd.math("MULTIPLY", nd.ramp(nd.math("ABSOLUTE", u), 0.13, 0.08), nd.ramp(nd.math("ABSOLUTE", v), 0.62, 0.5))
    slit = nd.math("MULTIPLY", slit, iris)
    streak = nd.noise(nd.attr("mask"), 9.0, 3.0)
    sclera = nd.mix(nd.ramp(front, 0.6, -0.4), lin(0.86, 0.80, 0.55), lin(0.55, 0.30, 0.22))
    irisc = nd.mix(nd.ramp(r, 0.0, 0.62), lin(0.98, 0.78, 0.20), lin(0.62, 0.30, 0.04))
    irisc = nd.mix(nd.ramp(streak, 0.45, 0.6, 0, 0.5), irisc, lin(0.38, 0.16, 0.02))
    col = nd.mix(iris, sclera, irisc)
    col = nd.mix(slit, col, lin(0.02, 0.015, 0.01))
    emit = nd.mix(nd.math("MULTIPLY", iris, nd.math("SUBTRACT", 1.0, slit)), (0, 0, 0), lin(0.55, 0.35, 0.05))
    nd.done(col, 0.04, 0.0, emit)


def membrane_mat(nd):
    oc = nd.coords("Object")
    m = nd.sep(nd.attr("mask"))
    col = nd.mix(nd.ramp(nd.noise(oc, 6.0, 6.0), 0.3, 0.7), lin(0.42, 0.07, 0.08), lin(0.70, 0.22, 0.18))
    col = nd.mix(m[0], col, lin(0.20, 0.03, 0.06))                                   # veins
    col = nd.mix(m[1], col, lin(0.22, 0.12, 0.10))                                   # dried, torn edges
    nd.done(col, 0.35, 0.0, nd.mix(m[0], lin(0.06, 0.0, 0.0), (0, 0, 0)))


def gum_mat(nd):
    oc = nd.coords("Object")
    col = nd.mix(nd.ramp(nd.noise(oc, 20.0, 4.0), 0.3, 0.7), lin(0.12, 0.01, 0.02), lin(0.40, 0.06, 0.08))
    nd.done(col, 0.15, 0.0, (0, 0, 0))


FLESH = BONE = EYE = MEMB = GUM = None


def make_materials():
    global FLESH, BONE, EYE, MEMB, GUM
    B.MATS.clear()
    FLESH = B.new_material("flesh", flesh_mat)
    BONE = B.new_material("bone", bone_mat)
    EYE = B.new_material("eye", eye_mat)
    MEMB = B.new_material("membrane", membrane_mat)
    GUM = B.new_material("gum", gum_mat)


# ----------------------------------------------------------------------------- sculpting
def ellipsoid(c, r, rot=None):
    bm = B.sphere_bm(r, 28, 18)
    return B.transform_bm(bm, B.place(bl(c), rot))


def capsule(p0, p1, r0, r1, sides=16):
    a, b = bl(p0), bl(p1)
    pts = [a + (b - a) * t for t in np.linspace(0, 1, 6)]
    bm = B.tube_bm(pts, list(np.linspace(r0, r1, 6)), sides, cap=True)
    for p, r in ((a, r0), (b, r1)):
        s = B.sphere_bm((r, r, r), 16, 10)
        B.transform_bm(s, B.place(p))
        tmp = bmesh.new()
        me = bpy.data.meshes.new("t")
        s.to_mesh(me)
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)
        s.free()
        tmp.free()
    return bm


def flesh(name, bms, seed, smooth_iter=4, warts=1.0, veins=1.0, wrinkle=1.0, belly=True, rings_axis=None, suckers=None, carve=None, lumps=1.0):
    """Fuse primitives into one sculpted flesh mesh. Returns (hi, lo_faces_hint)."""
    hi = B.join_bms(name, bms)
    B.remesh(hi, VOX)
    sm = hi.modifiers.new("sm", "SMOOTH")
    sm.factor = 0.6
    sm.iterations = smooth_iter
    B.apply_mods(hi)
    v = B.verts_np(hi)
    n = B.vnormals(hi)
    p = v * (1 / S) + seed * 3.7
    d = lumps * 0.045 * N.fbm(p * 1.6, 4, seed) + 0.012 * N.fbm(p * 4.5, 4, seed + 9)
    # deep folds (ridged creases), then fine wrinkles, stretched into rings along limbs when given
    d -= wrinkle * 0.016 * N.smoothstep(0.72, 0.95, N.ridged(p * 3.2, 3, seed + 10))
    if rings_axis is not None:
        a0, a1 = rings_axis
        ax = (a1 - a0) / np.linalg.norm(a1 - a0)
        s = (v - a0) @ ax
        ring = np.sin(s * 120 / S + 2.5 * N.fbm(p * 4, 2, seed + 1))
        d += wrinkle * 0.006 * (np.abs(ring) * 2 - 1) * (0.6 + 0.4 * N.fbm(p * 6, 2, seed + 2))
    d += wrinkle * 0.008 * (N.ridged(p * 10.0, 3, seed + 3) - 0.62)
    c1, c2, _ = N.worley(p * 4.0, seed + 4)
    vein = N.smoothstep(0.06, 0.0, c2 - c1) * N.smoothstep(-0.1, 0.3, N.fbm(p * 2.5, 3, seed + 5)) * veins
    d += 0.004 * vein
    w1, _, _ = N.worley(p * 12.0, seed + 6)
    wart = N.smoothstep(0.32, 0.0, w1) * N.smoothstep(0.05, 0.4, N.fbm(p * 2.0, 3, seed + 7)) * warts
    d += 0.012 * wart
    raw = np.zeros(len(v))
    if suckers is not None:
        down = suckers                           # direction the suckers face (Blender coords)
        face = np.clip(n @ down, 0, 1)
        s1, _, _ = N.worley(p * 16.0, seed + 8)
        cup = N.smoothstep(0.3, 0.12, s1) * N.smoothstep(0.55, 0.85, face)
        rim = N.smoothstep(0.42, 0.3, s1) * (1 - N.smoothstep(0.3, 0.12, s1)) * N.smoothstep(0.55, 0.85, face)
        d += 0.004 * rim - 0.006 * cup
        raw = np.maximum(raw, np.clip(cup + 0.6 * rim, 0, 1))
        wart *= 1 - face
    v = v + n * (d * S)[:, None]
    B.set_verts(hi, v)
    bel = N.smoothstep(-0.15, -0.65, n[:, 2]) if belly else np.zeros(len(v))
    masks = np.stack([vein, bel, raw, wart], axis=1)
    if carve:                                   # cut sockets after sculpting, carry the masks over by nearest vertex
        from mathutils.kdtree import KDTree
        kd = KDTree(len(v))
        for i, co in enumerate(v):
            kd.insert(co, i)
        kd.balance()
        carve(hi)
        nv = B.verts_np(hi)
        masks = masks[[kd.find(co)[1] for co in nv]]
    B.set_point_colour(hi, "mask", masks)
    B.material(hi, FLESH)
    return hi


def lowpoly(hi, faces):
    lo = B.copy_obj(hi, hi.name + "_lo")
    B.decimate(lo, faces)
    B.smooth(lo, 70)
    return lo


def bone_spike(points, r0, sides=10, curve=1.0):
    bm = B.cone_path([bl(p) for p in points], r0, 0.0, sides, curve)
    ob = B.from_bmesh("bone", bm)
    sub = ob.modifiers.new("ss", "SUBSURF")
    sub.levels = 1
    B.apply_mods(ob)
    v = B.verts_np(ob)
    base = bl(points[0])
    tip = bl(points[-1])
    ax = tip - base
    t = np.clip((v - base) @ ax / (ax @ ax), 0, 1)
    nrm = B.vnormals(ob)
    v = v + nrm * (0.0012 * np.sin(t * 60) * (1 - t))[:, None]           # growth rings
    B.set_verts(ob, v)
    B.set_point_colour(ob, "mask", np.stack([t, 0 * t, 0 * t, 0 * t], axis=1))
    B.material(ob, BONE)
    lo = B.from_bmesh("bone_lo", B.cone_path([bl(p) for p in points], r0, 0.0, 5, curve))
    B.smooth(lo, 60)
    return ob, lo


def eyeball(centre, gaze, radius):
    c = bl(centre)
    g = bl(gaze)
    g /= np.linalg.norm(g)
    bm = B.sphere_bm((radius, radius, radius), 24, 16)
    M = B.place(c, B.look(g))
    B.transform_bm(bm, M)
    ob = B.from_bmesh("eye", bm)
    v = B.verts_np(ob)
    Minv = np.array(M.inverted())
    loc = (v @ Minv[:3, :3].T + Minv[:3, 3]) / radius
    B.set_point_colour(ob, "mask", np.stack([loc[:, 0] * 1.25, loc[:, 1] * 1.25, loc[:, 2], 0 * loc[:, 0]], axis=1))
    B.material(ob, EYE)
    lo = B.from_bmesh("eye_lo", B.transform_bm(B.sphere_bm((radius, radius, radius), 10, 7), M))
    B.smooth(lo, 80)
    return ob, lo


def torus(centre, axis, R, r):
    bm = bmesh.new()
    seg, wseg = 20, 8
    rings = []
    for i in range(seg):
        t = i / seg * 2 * math.pi
        ring = []
        for j in range(wseg):
            s = j / wseg * 2 * math.pi
            ring.append(bm.verts.new(((R + r * math.cos(s)) * math.cos(t), (R + r * math.cos(s)) * math.sin(t), r * math.sin(s))))
        rings.append(ring)
    for i in range(seg):
        for j in range(wseg):
            bm.faces.new((rings[i][j], rings[(i + 1) % seg][j], rings[(i + 1) % seg][(j + 1) % wseg], rings[i][(j + 1) % wseg]))
    return B.transform_bm(bm, B.place(bl(centre), B.look(bl(axis))))


# ----------------------------------------------------------------------------- the horror
def build():
    B.reset()
    make_materials()
    H = B.Model("horror")
    body_piv = (0, 0.95 * S, 0)
    hc = np.array([0, 1.05 * S, 0.42 * S])
    H.part("body", bl(body_piv))
    H.part("head", bl(hc - [0, 0, 0.18 * S]), "body")

    # ---- body: hunched sack with a ridge of spines and pustules
    bms = [ellipsoid((0, 0.95 * S, -0.05 * S), (0.56 * S, 0.5 * S, 0.44 * S)),
           ellipsoid((0, 1.24 * S, -0.16 * S), (0.43 * S, 0.38 * S, 0.3 * S), B.euler(20, 0, 0)),
           ellipsoid((0, 0.72 * S, 0.0), (0.42 * S, 0.36 * S, 0.24 * S)),
           ellipsoid((0, 1.05 * S, 0.22 * S), (0.3 * S, 0.22 * S, 0.24 * S))]               # neck into the head
    for sx in (-1, 1):
        bms.append(ellipsoid((sx * 0.42 * S, 1.12 * S, 0.1 * S), (0.2 * S, 0.2 * S, 0.17 * S)))  # shoulder masses
    body = flesh("body", bms, 1)
    H.add("body", body, lowpoly(body, 2600))
    for k in range(8):
        t = k / 7
        base = (0, (1.5 - 0.28 * t) * S, (-0.38 + 0.6 * t - 0.2) * S)
        L = (0.16 - 0.08 * abs(t - 0.4)) * S
        tip = (rng.normal(0, 0.02), base[1] + L, base[2] - 0.06 * S)
        hi, lo = bone_spike([base, (0, base[1] + L * 0.6, base[2] - 0.02 * S), tip], 0.035 * S)
        H.add("body", hi, lo)

    # ---- head: bulbous, six lidded eyes, a toothed maw, horns, jaw tendrils
    # an irregular cluster of eyes of different sizes over the front and top of the head
    eyes = []
    HR = np.array([0.27, 0.24, 0.24]) * S
    cands = [(0.0, 0.32, 0.05)] + [(rng.uniform(-1.0, 1.0), rng.uniform(0.1, 0.85), rng.uniform(0.022, 0.04)) for _ in range(60)]
    for az, el, rad in cands:
        dirn = np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
        surf = hc + dirn * HR
        nrm = dirn / HR
        nrm /= np.linalg.norm(nrm)
        r = rad * S
        if any(np.linalg.norm(surf - e[0]) < (r + e[2]) * 1.5 for e in eyes):
            continue
        eyes.append((surf + nrm * r * 0.05, nrm, r))
        if len(eyes) == 7:
            break
    mouth = hc + np.array([0, -0.115 * S, 0.19 * S])
    bms = [ellipsoid(hc, (0.27 * S, 0.24 * S, 0.24 * S)),
           ellipsoid(hc + [0, 0.17 * S, -0.06 * S], (0.18 * S, 0.12 * S, 0.16 * S)),
           ellipsoid(hc + [0, -0.12 * S, 0.1 * S], (0.22 * S, 0.11 * S, 0.14 * S))]           # jaw
    for ep, gaze, rad in eyes:
        bms.append(torus(ep, gaze, rad * 1.02, rad * 0.38))

    def carve_head(ob):
        cut = []
        for ep, gaze, rad in eyes:
            c = B.sphere_bm((rad * 1.04,) * 3, 20, 14)
            B.transform_bm(c, B.place(bl(ep)))
            cut.append(B.from_bmesh("ec", c))
        mc = B.sphere_bm((0.165 * S, 0.1 * S, 0.075 * S), 28, 16)
        B.transform_bm(mc, B.place(bl(mouth + [0, 0, 0.05 * S])))
        cut.append(B.from_bmesh("mc", mc))
        B.boolean(ob, cut)
    head = flesh("head", bms, 3, warts=0.7, carve=carve_head, lumps=0.35)
    H.add("head", head, lowpoly(head, 2200))
    for ep, gaze, rad in eyes:
        H.add("head", *eyeball(ep, gaze, rad))
    gm = B.from_bmesh("gum", B.transform_bm(B.sphere_bm((0.15 * S, 0.085 * S, 0.06 * S), 24, 14), B.place(bl(mouth - [0, 0, 0.0 * S]))))
    B.material(gm, GUM)
    gml = B.from_bmesh("gum_lo", B.transform_bm(B.sphere_bm((0.15 * S, 0.085 * S, 0.06 * S), 10, 6), B.place(bl(mouth))))
    H.add("head", gm, gml)
    for row, (depth, scale) in enumerate(((0.035, 1.0), (0.012, 0.6))):      # two irregular rows of fangs
        for up in (1, -1):
            for i in range(11):
                if rng.random() < 0.18:
                    continue                                 # a gap: lost tooth
                t = (i / 10 - 0.5) * 2.2 + rng.normal(0, 0.06)
                base = mouth + [math.sin(t) * 0.15 * S, up * (0.07 - 0.012 * abs(math.sin(t))) * S, math.cos(t) * 0.055 * S + depth * S]
                L = rng.uniform(0.035, 0.09) * S * scale * (1.25 if abs(t) > 0.6 else 1.0)
                lean = np.array([rng.normal(0, 0.25), -up, rng.uniform(-0.1, 0.35)])
                lean /= np.linalg.norm(lean)
                tip = base + lean * L
                mid = base + lean * L * 0.5 + [0, 0, -0.008 * S]
                H.add("head", *bone_spike([base - lean * 0.01 * S, mid, tip], rng.uniform(0.009, 0.014) * S * scale, 8))
    for sx in (-1, 1):                       # horns, one broken
        b0 = hc + [sx * 0.15 * S, 0.14 * S, 0.0]
        pts = [b0, b0 + [sx * 0.1 * S, 0.1 * S, -0.03 * S], b0 + [sx * 0.15 * S, 0.22 * S, -0.1 * S], b0 + [sx * 0.13 * S, 0.3 * S, -0.2 * S]]
        if sx < 0:
            pts = pts[:3]
        H.add("head", *bone_spike(pts, 0.042 * S, 12, 0.8))
    for i in range(9):                       # thin tendrils hanging from the jaw, curling
        t = (i / 8 - 0.5) * 1.8
        base = mouth + [math.sin(t) * 0.13 * S, -0.08 * S, -0.02 * S]
        L = rng.uniform(0.2, 0.36) * S
        ph = rng.uniform(0, 6.28)
        pts = [bl(base + [math.sin(ph + k * 0.7) * 0.02 * S * k / 4, -L * k / 9, 0.03 * S * math.sin(k * 0.5) + 0.015 * k * S * math.cos(t)]) for k in range(10)]
        radii = [0.011 * S * (1 - k / 10) + 0.0025 for k in range(10)]
        hi = B.from_bmesh("tendril", B.tube_bm(pts, radii, 10, True))
        sub = hi.modifiers.new("ss", "SUBSURF")
        sub.levels = 1
        B.apply_mods(hi)
        z = np.zeros(len(hi.data.vertices))
        B.set_point_colour(hi, "mask", np.stack([z + 0.3, z, z + 0.12, z], axis=1))
        B.material(hi, FLESH)
        lo = B.from_bmesh("tendril_lo", B.tube_bm(pts, radii, 5, True))
        B.smooth(lo, 80)
        H.add("head", hi, lo)

    # ---- torn, veined fins on the shoulders
    for sx in (-1, 1):
        tag = "l" if sx > 0 else "r"
        root = np.array([sx * 0.4 * S, 1.3 * S, -0.1 * S])
        H.part(f"fin_{tag}", bl(root), "body")
        spines = [root + np.array([sx * (0.48 + 0.1 * k) * S, (0.4 - 0.2 * k) * S, (-0.05 - 0.12 * k) * S]) for k in range(4)]
        for tip in spines:
            mid = (root + tip) / 2 + [0, 0.06 * S, 0]
            H.add(f"fin_{tag}", *bone_spike([root, mid, tip], 0.026 * S, 8))
        # membrane: a fan of strips between neighbouring spines, rippled, thin, torn at the rim
        verts, faces, vein, edge = [], [], [], []
        RES = 18
        for k in range(3):
            a, b = spines[k], spines[k + 1]
            base_i = len(verts)
            for i in range(RES + 1):
                for j in range(RES + 1):
                    s, t = i / RES, j / RES
                    rim = (1 - t) * a + t * b
                    sag = (root - rim) * 0.22 * math.sin(math.pi * t)        # the rim sags between the spines
                    p = root + (rim + sag - root) * s
                    p = p + np.array([0, 0, 1.0]) * 0.02 * S * math.sin(s * 9 + t * 3) * s
                    verts.append(bl(p))
                    vein.append(max(math.exp(-(t * RES / 2.2) ** 2), math.exp(-((1 - t) * RES / 2.2) ** 2)))
                    edge.append(s)
            for i in range(RES):
                for j in range(RES):
                    if i > RES - 3 and rng.random() < 0.35:
                        continue                                       # tears at the rim
                    q = base_i + i * (RES + 1) + j
                    faces.append((q, q + 1, q + RES + 2, q + RES + 1))
        ob = B.mesh_obj("membrane", verts, faces)
        vv = np.array(verts)
        pp = vv / S * 6 + sx * 5
        c1, c2, _ = N.worley(pp, 31 + sx)
        vein_m = np.clip(np.array(vein) + N.smoothstep(0.06, 0.0, c2 - c1) * 0.8, 0, 1)
        edge_m = N.smoothstep(0.78, 1.0, np.array(edge) + 0.15 * N.fbm(pp, 2, 3))
        B.set_point_colour(ob, "mask", np.stack([vein_m, edge_m, 0 * vein_m, 0 * vein_m], axis=1))
        lo = B.copy_obj(ob, "membrane_lo")
        so = ob.modifiers.new("so", "SOLIDIFY")
        so.thickness = 0.006 * S
        so.offset = 0
        B.apply_mods(ob)
        B.material(ob, MEMB)
        so = lo.modifiers.new("so", "SOLIDIFY")
        so.thickness = 0.006 * S
        so.offset = 0
        B.apply_mods(lo)
        B.decimate(lo, 900)
        H.add(f"fin_{tag}", ob, lo)

    # ---- arms: upper arm, forearm with spurs, clawed hand
    for sx in (-1, 1):
        tag = "l" if sx > 0 else "r"
        sh = np.array([sx * 0.42 * S, 1.12 * S, 0.12 * S])
        el = sh + np.array([sx * 0.22 * S, -0.3 * S, 0.22 * S])
        wr = el + np.array([-sx * 0.12 * S, 0.05 * S, 0.32 * S])
        H.part(f"arm_{tag}", bl(sh), "body")
        H.part(f"fore_{tag}", bl(el), f"arm_{tag}")
        H.part(f"hand_{tag}", bl(wr), f"fore_{tag}")
        up = flesh(f"up{tag}", [capsule(sh, el, 0.13 * S, 0.095 * S), ellipsoid((sh + el) / 2 + [sx * 0.03 * S, 0.02 * S, 0], (0.12 * S, 0.1 * S, 0.11 * S))],
                   10 + sx, rings_axis=(bl(sh), bl(el)), belly=False)
        H.add(f"arm_{tag}", up, lowpoly(up, 900))
        fo = flesh(f"fo{tag}", [capsule(el, wr, 0.092 * S, 0.065 * S)], 12 + sx, rings_axis=(bl(el), bl(wr)), belly=False)
        H.add(f"fore_{tag}", fo, lowpoly(fo, 700))
        for k in range(3):
            p = el + (wr - el) * (0.22 + 0.26 * k)
            H.add(f"fore_{tag}", *bone_spike([p + [sx * 0.04 * S, 0.04 * S, 0], p + [sx * 0.1 * S, 0.09 * S, -0.03 * S], p + [sx * 0.15 * S, 0.12 * S, -0.06 * S]], 0.022 * S, 8))
        ha = flesh(f"ha{tag}", [ellipsoid(wr + [0, 0, 0.05 * S], (0.085 * S, 0.06 * S, 0.1 * S))] +
                   [capsule(wr + [math.sin((k - 1) * 0.45) * 0.05 * S, 0, 0.08 * S], wr + [math.sin((k - 1) * 0.45) * 0.09 * S, -0.02 * S, 0.15 * S], 0.03 * S, 0.025 * S, 10) for k in range(3)],
                   14 + sx, belly=False)
        H.add(f"hand_{tag}", ha, lowpoly(ha, 600))
        for k in range(3):
            ang = (k - 1) * 0.45
            d = np.array([math.sin(ang) * 0.6 - sx * 0.1, -0.35, math.cos(ang)])
            d /= np.linalg.norm(d)
            base = wr + [math.sin(ang) * 0.09 * S, -0.02 * S, 0.15 * S]
            side = np.cross(d, [0, 1, 0])
            side /= np.linalg.norm(side) + 1e-9
            pts = [base + d * 0.25 * S * t + np.cross(side, d) * -0.22 * 0.25 * S * t * t for t in np.linspace(0, 1, 6)]
            H.add(f"hand_{tag}", *bone_spike(pts, 0.026 * S, 10, 0.9))

    # ---- six tentacles, five jointed segments each, wrinkled rings and suckers underneath
    for t in range(6):
        ang = (t + 0.5) / 6 * 2 * math.pi + 0.3
        out = np.array([math.cos(ang), 0, math.sin(ang)])
        p = np.array([0, 0.7 * S, -0.05 * S]) + out * 0.3 * S
        parent = "body"
        seg_len = 0.25 * S
        r0 = 0.115 * S
        for k in range(5):
            d = out * (0.6 + 0.12 * k) + np.array([0, -0.7 + 0.18 * k, 0])
            d /= np.linalg.norm(d)
            q = p + d * seg_len
            q[1] = max(q[1], 0.07 * S)
            r1 = r0 * 0.78
            name = f"tent_{t}_{k}"
            H.part(name, bl(p), parent)
            mid = (p + q) / 2 + np.array([0, 0.01 * S, 0])
            bms = [capsule(p, mid, r0, (r0 + r1) / 2, 18), capsule(mid, q, (r0 + r1) / 2, r1 if k < 4 else r1 * 0.35, 18)]
            hi = flesh(name, bms, 40 + t * 5 + k, warts=0.5, rings_axis=(bl(p), bl(q)), suckers=np.array([0, 0, -1.0]), belly=True)
            H.add(name, hi, lowpoly(hi, 360))
            parent, p, r0 = name, q, r1
    return H


def main():
    H = build()
    H.bake(size=2048, extrusion=0.02, ray=0.06)
    H.export(B.ROOT / "assets_src" / "monsters" / "horror.glb", emissive_strength=1.0)


main()
