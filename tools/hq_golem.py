"""Ice golem, sculpted and baked (high-poly stone carved with chips, cracks, strata, glowing runes,
ice crystal clusters, frost and iron chains -> baked PBR textures on a game mesh).

Parts (same names as before): pelvis > torso > head, arm_l/r > fore_l/r > hand_l/r,
pelvis > thigh_l/r > shin_l/r.
Run: blender --background --factory-startup --python tools/hq_golem.py
Writes assets_src/monsters/golem.glb (+ textures in assets_src/hq/golem/)
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

rng = np.random.default_rng(7)


def lin(*c):
    return tuple(x ** 2.2 for x in c)


# ----------------------------------------------------------------------------- materials
ROCK = pathlib.Path(__file__).resolve().parent.parent / "ph" / "rock_face_03"


def stone_mat(nd):
    oc = nd.coords("Object")
    m = nd.sep(nd.attr("mask"))
    tint = nd.attr("mask", "Alpha")
    # photographed rock (Poly Haven rock_face_03), projected triplanar onto every block, cooled toward grey-blue
    scan = nd.image(next(ROCK.glob("*_diff_2k.jpg")), oc, 0.8)
    height = nd.image(next(ROCK.glob("*_disp_2k.png")), oc, 0.8, "Non-Color")
    nd.bump(height, 0.55, 0.04)
    cool = nd.mix(1.0, scan, (0.78, 0.86, 1.0), "MULTIPLY")
    grey = nd.mix(0.5, scan, nd.mix(1.0, scan, (0.9, 0.9, 0.9), "MULTIPLY"))
    col = nd.mix(nd.ramp(tint, 0.2, 0.8), grey, cool)
    n1 = nd.noise(oc, 2.6, 8.0, 0.62)
    col = nd.mix(nd.ramp(n1, 0.3, 0.7, 0.0, 0.5), col, nd.mix(1.0, col, (0.7, 0.7, 0.72), "MULTIPLY"))
    n2 = nd.noise(oc, 9.0, 6.0, 0.7)
    col = nd.mix(nd.ramp(n2, 0.62, 0.8, 0.0, 0.35), col, lin(0.33, 0.30, 0.28))          # mottled stains
    sp = nd.noise(oc, 70.0, 2.0, 0.5)
    col = nd.mix(nd.ramp(sp, 0.66, 0.74, 0.0, 0.55), col, lin(0.18, 0.17, 0.17))         # dark grains
    fl = nd.noise(oc, 95.0, 1.0, 0.5, w=None)
    col = nd.mix(nd.ramp(fl, 0.70, 0.76, 0.0, 0.6), col, lin(0.78, 0.76, 0.72))          # quartz flecks
    pt = nd.geometry("Pointiness")
    col = nd.mix(nd.ramp(pt, 0.5, 0.58, 0.0, 0.55), col, nd.mix(1.0, col, (1.55, 1.52, 1.5), "MULTIPLY"))   # worn edges
    col = nd.mix(nd.ramp(pt, 0.5, 0.42, 0.0, 0.7), col, nd.mix(1.0, col, (0.35, 0.35, 0.36), "MULTIPLY"))   # grime in hollows
    col = nd.mix(nd.math("MULTIPLY", m[0], 0.9), col, lin(0.07, 0.07, 0.08))                               # cracks
    lich = nd.mix(nd.noise(oc, 25.0, 3.0), lin(0.58, 0.60, 0.48), lin(0.40, 0.46, 0.36))
    col = nd.mix(nd.math("MULTIPLY", m[2], nd.ramp(nd.noise(oc, 40.0, 4.0), 0.42, 0.58)), col, lich)        # lichen
    frost_col = nd.mix(nd.ramp(nd.noise(oc, 60.0, 2.0), 0.6, 0.72), lin(0.80, 0.87, 0.94), lin(0.97, 0.99, 1.0))
    frost = nd.math("MULTIPLY", m[1], nd.ramp(nd.noise(oc, 14.0, 5.0), 0.3, 0.55, 0.55, 1.0))
    col = nd.mix(frost, col, frost_col)
    rough = nd.mixf(frost, nd.ramp(n2, 0.3, 0.8, 0.9, 0.78), 0.45)
    nd.done(col, rough, 0.0, (0, 0, 0))


def glow_mat(strength):
    def build(nd):
        nd.done(lin(0.45, 0.85, 0.98), 0.35, 0.0, tuple(c * strength for c in lin(0.55, 0.93, 1.0)))
    return build


def ice_mat(nd):
    oc = nd.coords("Object")
    t = nd.sep(nd.attr("mask"))[0]
    edge = nd.voronoi(oc, 7.0, "DISTANCE_TO_EDGE")
    streak = nd.ramp(edge, 0.06, 0.0)
    cloud = nd.noise(oc, 5.0, 6.0, 0.6)
    col = nd.mix(nd.ramp(t, 0.0, 0.55), lin(0.05, 0.22, 0.42), lin(0.30, 0.62, 0.86))
    col = nd.mix(nd.ramp(t, 0.45, 1.0), col, lin(0.80, 0.94, 1.0))
    col = nd.mix(nd.ramp(cloud, 0.45, 0.7, 0.0, 0.5), col, lin(0.75, 0.9, 1.0))
    col = nd.mix(nd.math("MULTIPLY", streak, 0.75), col, lin(0.92, 0.98, 1.0))
    emit = nd.mix(nd.ramp(t, 0.0, 0.8), lin(0.10, 0.42, 0.62), lin(0.01, 0.05, 0.08))
    emit = nd.mix(nd.math("MULTIPLY", streak, 0.5), emit, lin(0.35, 0.75, 0.9))
    nd.done(col, nd.mixf(streak, 0.06, 0.3), 0.0, emit)


def iron_mat(nd):
    oc = nd.coords("Object")
    rust = nd.ramp(nd.noise(oc, 7.0, 8.0, 0.65), 0.5, 0.68)
    pit = nd.noise(oc, 60.0, 3.0)
    base = nd.mix(nd.ramp(pit, 0.4, 0.7, 0, 0.5), lin(0.20, 0.20, 0.21), lin(0.10, 0.10, 0.11))
    col = nd.mix(rust, base, nd.mix(nd.ramp(pit, 0.3, 0.7), lin(0.40, 0.20, 0.09), lin(0.26, 0.12, 0.06)))
    nd.done(col, nd.mixf(rust, 0.4, 0.88), nd.mixf(rust, 0.92, 0.15), (0, 0, 0))


STONE = RUNE = EYE = ICE = IRON = None


def make_materials():
    global STONE, RUNE, EYE, ICE, IRON
    B.MATS.clear()
    STONE = B.new_material("stone", stone_mat)
    RUNE = B.new_material("rune", glow_mat(1.0))
    EYE = B.new_material("eye", glow_mat(1.0))
    ICE = B.new_material("ice", ice_mat)
    IRON = B.new_material("iron", iron_mat)


# ----------------------------------------------------------------------------- pieces
GLYPHS = [
    [((0, -.5), (0, .5)), ((0, .5), (.38, .18)), ((0, .18), (.38, -.12))],               # F
    [((0, -.5), (0, .5)), ((.38, -.5), (.38, .5)), ((0, .25), (.38, -.1))],              # H
    [((0, -.5), (0, .5)), ((0, .5), (.35, .25)), ((.35, .25), (0, 0)), ((0, 0), (.35, -.5))],   # R
    [((0, -.5), (0, .5)), ((-.3, .45), (0, .2)), ((.3, .45), (0, .2))],                   # Y
    [((-.3, .5), (.3, -.5)), ((.3, .5), (-.3, -.5))],                                      # X
    [((0, -.5), (0, .5)), ((0, .5), (.35, .15)), ((0, -.5), (.35, -.15))],               # P-ish
    [((-.3, -.5), (-.3, .5)), ((-.3, .5), (.3, -.1)), ((.3, -.1), (.3, .5)), ((.3, .5), (.3, -.5))],   # M
    [((0, -.5), (0, .5)), ((-.32, .1), (.32, .1)), ((-.32, .1), (0, .5)), ((.32, .1), (0, .5))],   # arrow
]


def stone_block(model, part, centre, size, rot=(0, 0, 0), seed=0, runes=0, chips=7, tint=None, face=(0, -1, 0), frost=1.0, lichen=1.0):
    r = np.random.default_rng(seed)
    size = np.array(size, float)
    rotm = B.euler(*rot)
    M = B.place(centre, rotm)
    bm = B.box_bm(size, bevel=float(min(size)) * 0.035, segments=2, taper=(r.uniform(0.9, 1.0), r.uniform(0.9, 1.0)), jitter=0.025, rng=r)
    B.transform_bm(bm, M)
    hi = B.from_bmesh(f"blk{seed}", bm)
    # chipped corners and edges
    cutters = []
    for k in range(chips):
        corner = size / 2 * r.choice([-1, 1], 3)
        if r.random() < 0.5:                       # an edge rather than a corner
            ax = r.integers(3)
            corner[ax] = r.uniform(-0.35, 0.35) * size[ax]
        cs = float(min(size)) * r.uniform(0.18, 0.34)
        cb = B.box_bm((cs, cs * r.uniform(0.6, 1.4), cs * r.uniform(0.6, 1.2)))
        B.transform_bm(cb, M @ Matrix.Translation(Vector(corner * 0.98)) @ B.euler(*r.uniform(0, 360, 3)))
        cutters.append(B.from_bmesh("cut", cb))
    B.boolean(hi, cutters)
    B.remesh(hi, 0.011)
    # sculpt: broad unevenness, strata, pitting, cracks
    v = B.verts_np(hi)
    n = B.vnormals(hi)
    Minv = np.array(M.inverted())
    pl = (v @ Minv[:3, :3].T) + Minv[:3, 3]
    p = pl + seed * 7.31
    amp = float(np.clip(min(size) / 0.45, 0.6, 1.5))
    d = 0.018 * N.fbm(p * 2.0, 4, seed) + 0.010 * N.fbm(p * 7.0, 4, seed + 5) + 0.0035 * N.fbm(p * 24.0, 2, seed + 8)
    d += 0.0025 * np.sin(pl[:, 2] * 52.0 + 3.0 * N.fbm(p * 2.5, 3, seed + 2))
    # chiselled facets: each Worley cell is pushed in by its own amount, leaving flat-ish chips with sharp rims
    f1, f2, cid = N.worley(p * 4.5, seed + 1)
    d -= 0.009 * cid * N.smoothstep(0.0, 0.03, f2 - f1)
    f1b, _, _ = N.worley(p * 9.0, seed + 11)
    d -= 0.005 * N.smoothstep(0.3, 0.0, f1b)
    # crumbling edges and corners
    q = np.sort(np.abs(pl) / (size / 2), axis=1)
    edge = N.smoothstep(0.78, 0.97, q[:, 1])
    d -= edge * 0.035 * np.clip(N.fbm(p * 5.0, 4, seed + 12) + 0.35, 0, 1)
    c1, c2, _ = N.worley(p * 2.2, seed + 3)
    crack = N.smoothstep(0.085, 0.0, c2 - c1) * N.smoothstep(-0.15, 0.2, N.fbm(p * 3.0, 3, seed + 4))
    d -= 0.024 * crack
    v = v + n * (d * amp)[:, None]
    B.set_verts(hi, v)
    up = n[:, 2]
    fr = N.smoothstep(0.25, 0.8, up + 0.3 * N.fbm(p * 4.0, 3, seed + 6)) * frost
    li = N.smoothstep(0.5, 0.78, N.fbm(p * 3.2, 4, seed + 9) * 0.5 + 0.5) * (1 - fr) * lichen
    t = np.full(len(v), r.random() if tint is None else tint)
    B.set_point_colour(hi, "mask", np.stack([crack, fr, li, t], axis=1))
    # game mesh
    area = 2 * (size[0] * size[1] + size[1] * size[2] + size[0] * size[2])
    lo = B.copy_obj(hi, f"lo{seed}")
    B.decimate(lo, int(np.clip(area * 300, 220, 1800)))
    B.smooth(lo)
    # runes carved into one face, filled with glowing ice
    glows = []
    if runes:
        fl = np.array(face, float)
        ax = int(np.argmax(np.abs(fl)))
        others = [a for a in range(3) if a != ax]
        u_ax, v_ax = (others[0], 2) if ax != 2 else (0, 1)
        if v_ax == ax:
            v_ax = others[1]
        half_u, half_v = size[u_ax] / 2, size[v_ax] / 2
        h = float(np.clip(min(half_u, half_v) * 0.55, 0.09, 0.2))
        from mathutils.bvhtree import BVHTree
        bvh = BVHTree.FromObject(hi, bpy.context.evaluated_depsgraph_get())
        sgn = float(np.sign(fl[ax]))
        nrm_w = (M.to_3x3() @ Vector(fl)).normalized()

        def surface_at(uv):
            """Signed distance of the real (sculpted) surface from the centre along the face axis."""
            loc = np.zeros(3)
            loc[u_ax], loc[v_ax] = uv
            loc[ax] = sgn * (size[ax] / 2 + 0.3)
            hit = bvh.ray_cast(M @ Vector(loc), -nrm_w, 1.0)[0]
            if hit is None:
                return size[ax] / 2 - 0.05
            local = M.inverted() @ hit
            return sgn * local[ax]
        cutters = []
        placed = 0
        tries = 0
        while placed < runes and tries < 30:
            tries += 1
            cu = r.uniform(-half_u + h * 0.6, half_u - h * 0.6)
            cv = r.uniform(-half_v + h * 0.7, half_v - h * 0.7)
            g = GLYPHS[r.integers(len(GLYPHS))]
            for (a, b) in g:
                a2 = np.array([cu + a[0] * h, cv + a[1] * h])
                b2 = np.array([cu + b[0] * h, cv + b[1] * h])
                mid, L = (a2 + b2) / 2, np.linalg.norm(b2 - a2)
                ang = math.atan2(b2[1] - a2[1], b2[0] - a2[0])
                inner = min(surface_at(a2), surface_at(mid), surface_at(b2))      # deepest point of the real surface
                cyclic = (u_ax, v_ax, ax) in ((0, 1, 2), (1, 2, 0), (2, 0, 1))
                # groove: from 0.03 below the deepest surface point out past the face; glow strip: sunk in the groove
                for width, d_lo, d_hi, store in ((0.026, inner - 0.03, inner + 0.12, cutters), (0.017, inner - 0.026, inner - 0.011, glows)):
                    loc = np.zeros(3)
                    loc[u_ax], loc[v_ax] = mid
                    loc[ax] = sgn * (d_lo + d_hi) / 2
                    dims = np.zeros(3)
                    dims[u_ax], dims[v_ax], dims[ax] = L + width, width, d_hi - d_lo
                    sb = B.box_bm(dims)
                    rot_axis = np.zeros(3)
                    rot_axis[ax] = 1
                    R = Matrix.Rotation(ang if cyclic else -ang, 4, Vector(rot_axis))
                    B.transform_bm(sb, M @ Matrix.Translation(Vector(loc)) @ R)
                    store.append(B.from_bmesh("rune", sb))
            placed += 1
        B.boolean(hi, cutters)
        for g in glows:
            B.material(g, RUNE)
    B.material(hi, STONE)
    model.add(part, [hi] + glows, lo)
    return M


def crystal(model, part, base, direction, length, radius, seed, sink=0.06):
    r = np.random.default_rng(seed)
    sides = 6
    bm = bmesh.new()
    rings = []
    tilt = r.normal(0, 0.12, 2)
    for z, rad in ((-sink, radius * 0.8), (length * 0.72, radius * r.uniform(0.9, 1.08))):
        ring = []
        for k in range(sides):
            a = k / sides * 2 * math.pi + r.normal(0, 0.06)
            rr = rad * r.uniform(0.86, 1.08)
            ring.append(bm.verts.new((math.cos(a) * rr + tilt[0] * z * 0.2, math.sin(a) * rr + tilt[1] * z * 0.2, z)))
        rings.append(ring)
    tip = bm.verts.new((tilt[0] * length * 0.2 + r.normal(0, radius * 0.15), tilt[1] * length * 0.2, length))
    bm.faces.new(rings[0][::-1])
    for k in range(sides):
        a, b = k, (k + 1) % sides
        bm.faces.new((rings[0][a], rings[0][b], rings[1][b], rings[1][a]))
        bm.faces.new((rings[1][a], rings[1][b], tip))
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    M = B.place(base, B.look(direction) @ Matrix.Rotation(r.uniform(0, 6.28), 4, "Z"))
    B.transform_bm(bm, M)
    lo = B.from_bmesh(f"cr_lo{seed}", bm)
    hi = B.copy_obj(lo, f"cr_hi{seed}")
    bv = hi.modifiers.new("bv", "BEVEL")
    bv.width = radius * 0.07
    bv.segments = 3
    bv.limit_method = "ANGLE"
    sub = hi.modifiers.new("ss", "SUBSURF")
    sub.levels = 1
    sub.subdivision_type = "SIMPLE"
    B.apply_mods(hi)
    v = B.verts_np(hi)
    Minv = np.array(M.inverted())
    zl = (v @ Minv[:3, :3].T + Minv[:3, 3])[:, 2]
    n = B.vnormals(hi)
    v = v + n * (0.004 * N.fbm(v * 9.0 + seed, 3, seed))[:, None] * (radius / 0.06)
    B.set_verts(hi, v)
    t = np.clip(zl / length, 0, 1)
    B.set_point_colour(hi, "mask", np.stack([t, 0 * t, 0 * t, 0 * t], axis=1))
    B.material(hi, ICE)
    B.smooth(lo, 30)
    model.add(part, hi, lo)


def cluster(model, part, base, direction, count, length, radius, spread, seed):
    r = np.random.default_rng(seed)
    d0 = np.array(direction, float)
    d0 /= np.linalg.norm(d0)
    for k in range(count):
        d = d0 + r.normal(0, spread, 3)
        d /= np.linalg.norm(d)
        off = r.normal(0, radius * 0.9, 3)
        off -= d0 * np.dot(off, d0)
        L = length * (1.0 if k == 0 else r.uniform(0.4, 0.85))
        crystal(model, part, np.array(base) + off, d, L, radius * (1.0 if k == 0 else r.uniform(0.55, 0.85)), seed * 31 + k)


def icicles(model, part, points, seed):
    r = np.random.default_rng(seed)
    for k, p in enumerate(points):
        L = r.uniform(0.07, 0.2)
        crystal(model, part, np.array(p) + [0, 0, 0.03], (r.normal(0, 0.08), r.normal(0, 0.08), -1), L, r.uniform(0.012, 0.022), seed * 13 + k, sink=0.03)


def torus_bm(a, b, wire, seg, wseg):
    bm = bmesh.new()
    rings = []
    for i in range(seg):
        t = i / seg * 2 * math.pi
        c = Vector((math.cos(t) * a, math.sin(t) * b, 0))
        tang = Vector((-math.sin(t) * a, math.cos(t) * b, 0)).normalized()
        nrm = Vector((tang.y, -tang.x, 0))
        ring = []
        for j in range(wseg):
            s = j / wseg * 2 * math.pi
            ring.append(bm.verts.new(c + nrm * (math.cos(s) * wire) + Vector((0, 0, math.sin(s) * wire))))
        rings.append(ring)
    for i in range(seg):
        for j in range(wseg):
            a0, a1 = rings[i][j], rings[i][(j + 1) % wseg]
            b0, b1 = rings[(i + 1) % seg][j], rings[(i + 1) % seg][(j + 1) % wseg]
            bm.faces.new((a0, b0, b1, a1))
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    return bm


def chain(model, part, pts, link=0.05, wire=0.013, seed=0):
    pts = [np.array(p, float) for p in pts]
    step = link * 1.45
    k = 0
    r = np.random.default_rng(seed)
    for a, b in zip(pts, pts[1:]):
        seg = b - a
        L = np.linalg.norm(seg)
        d = seg / L
        n = int(L / step)
        for i in range(n):
            c = a + d * (i + 0.5) * step
            twist = (math.pi / 2 if k % 2 else 0) + r.normal(0, 0.15)
            M = B.place(c, B.look(d) @ Matrix.Rotation(twist, 4, "Z") @ Matrix.Rotation(math.pi / 2, 4, "Y"))
            hb = torus_bm(link * 0.95, link * 0.6, wire, 20, 8)
            B.transform_bm(hb, M)
            lb = torus_bm(link * 0.95, link * 0.6, wire, 8, 4)
            B.transform_bm(lb, M)
            hi = B.from_bmesh("ch_hi", hb)
            lo = B.from_bmesh("ch_lo", lb)
            B.material(hi, IRON)
            B.smooth(hi, 80)
            B.smooth(lo, 80)
            model.add(part, hi, lo)
            k += 1


def eye_sockets(model, part, head_obj_index, centres, size):
    """Carve slits into the head (last stone hi of the part) and set glowing eyes inside."""
    info = model.parts[part]
    head = [h for h in info["hi"] if h.data.materials and h.data.materials[0].name == "stone"][head_obj_index]
    cutters = []
    for c in centres:
        cb = B.box_bm((size[0], 0.12, size[1]), bevel=0.012, segments=2)
        B.transform_bm(cb, B.place(c))
        cutters.append(B.from_bmesh("eyecut", cb))
        gb = bmesh.new()
        bmesh.ops.create_uvsphere(gb, u_segments=16, v_segments=10, radius=1.0)
        B.transform_bm(gb, B.place(np.array(c) + [0, 0.05, 0]) @ Matrix.Diagonal((size[0] * 0.45, 0.03, size[1] * 0.42, 1)))
        g = B.from_bmesh("eye", gb)
        B.material(g, EYE)
        info["hi"].append(g)
    B.boolean(head, cutters)


# ----------------------------------------------------------------------------- the golem
def build():
    B.reset()
    make_materials()
    G = B.Model("golem")
    G.part("pelvis", (0, 0, 1.12))
    G.part("torso", (0, 0, 1.3), "pelvis")
    G.part("head", (0, -0.22, 2.05), "torso")
    for sx, tag in ((1, "l"), (-1, "r")):
        G.part(f"arm_{tag}", (sx * 0.84, 0.02, 2.06), "torso")
        G.part(f"fore_{tag}", (sx * 0.96, 0.0, 1.42), f"arm_{tag}")
        G.part(f"hand_{tag}", (sx * 0.99, -0.05, 0.8), f"fore_{tag}")
        G.part(f"thigh_{tag}", (sx * 0.3, 0.0, 1.0), "pelvis")
        G.part(f"shin_{tag}", (sx * 0.33, 0.0, 0.6), f"thigh_{tag}")

    # pelvis and belly
    stone_block(G, "pelvis", (0, 0.02, 1.1), (0.92, 0.6, 0.38), (0, 0, 2), seed=1, runes=1, tint=0.2)
    # torso: a great chest block, shoulder masses and a hunched back
    stone_block(G, "torso", (0, -0.02, 1.4), (0.78, 0.58, 0.32), (3, 0, -2), seed=2, runes=1, tint=0.45)
    stone_block(G, "torso", (0, 0.02, 1.76), (1.18, 0.78, 0.5), (-4, 0, 1), seed=3, runes=2, tint=0.3)
    for sx in (1, -1):
        stone_block(G, "torso", (sx * 0.56, 0.04, 2.13), (0.6, 0.72, 0.44), (0, sx * 7, sx * 5), seed=4 + (sx > 0), runes=1 if sx > 0 else 0,
                    tint=0.7 if sx > 0 else 0.15, face=(sx, 0, 0))
    stone_block(G, "torso", (0, 0.24, 2.2), (0.62, 0.5, 0.42), (8, 0, 0), seed=6, runes=0, tint=0.85)
    # crystal heart in the chest, chains, crystals bursting from shoulders and back
    cluster(G, "torso", (0, -0.4, 1.8), (0, -1, 0.15), 4, 0.2, 0.05, 0.4, seed=11)
    chain(G, "torso", [(-0.62, -0.44, 2.02), (-0.2, -0.47, 1.82), (0.25, -0.46, 1.6), (0.62, -0.42, 1.44)], seed=1)
    chain(G, "torso", [(0.62, -0.4, 2.0), (0.32, -0.46, 1.9)], seed=2)
    chain(G, "torso", [(-0.42, -0.33, 1.3), (0.0, -0.36, 1.25), (0.42, -0.33, 1.3)], seed=3)
    for sx in (1, -1):
        cluster(G, "torso", (sx * 0.62, 0.08, 2.32), (sx * 0.35, 0.15, 1), 7, 0.62, 0.075, 0.32, seed=20 + (sx > 0))
    cluster(G, "torso", (0, 0.42, 2.25), (0, 0.8, 0.7), 5, 0.45, 0.06, 0.35, seed=24)
    icicles(G, "torso", [(x, -0.38, 1.52) for x in np.linspace(-0.5, 0.5, 6)] + [(x, -0.32, 1.25) for x in (-0.28, 0.05, 0.3)], seed=5)
    # head, sunk between the shoulders, with glowing eye slits and a crystal crest
    stone_block(G, "head", (0, -0.3, 2.2), (0.4, 0.38, 0.32), (4, 0, 0), seed=40, runes=0, chips=5, tint=0.55)
    eye_sockets(G, "head", 0, [(-0.1, -0.5, 2.19), (0.1, -0.5, 2.19)], (0.085, 0.04))
    stone_block(G, "head", (0, -0.45, 2.285), (0.48, 0.17, 0.12), (-12, 0, 0), seed=42, runes=0, chips=4, tint=0.5, frost=1.3)      # brow ridge
    stone_block(G, "head", (0, -0.4, 2.03), (0.36, 0.3, 0.13), (6, 0, 0), seed=43, runes=0, chips=4, tint=0.6)                       # jaw
    cluster(G, "head", (0, -0.28, 2.38), (0, 0.2, 1), 3, 0.26, 0.035, 0.35, seed=41)
    for sx, tag in ((1, "l"), (-1, "r")):
        # upper arm
        stone_block(G, f"arm_{tag}", (sx * 0.93, 0.02, 1.76), (0.44, 0.48, 0.6), (0, 0, sx * 4), seed=50 + (sx > 0), runes=1, tint=0.35 + 0.3 * (sx > 0), face=(sx, 0, 0))
        chain(G, f"arm_{tag}", [(sx * 0.68, -0.27, 1.62), (sx * 0.93, -0.28, 1.58), (sx * 1.18, -0.24, 1.62)], seed=4 + (sx > 0))
        # forearm, heavy, crystals on the outside
        stone_block(G, f"fore_{tag}", (sx * 0.98, -0.02, 1.1), (0.5, 0.54, 0.62), (0, 0, sx * -3), seed=60 + (sx > 0), runes=1, tint=0.6 - 0.4 * (sx > 0), face=(0, -1, 0))
        cluster(G, f"fore_{tag}", (sx * 1.22, 0.04, 1.14), (sx * 1, 0.1, 0.35), 5, 0.36, 0.05, 0.35, seed=62 + (sx > 0))
        icicles(G, f"fore_{tag}", [(sx * 0.98 + dx, -0.02 + dy, 0.8) for dx, dy in ((-0.15, -0.15), (0.1, -0.18), (0.16, 0.1))], seed=63 + (sx > 0))
        # hands: crystal claw (left), stone maul (right)
        if sx > 0:
            stone_block(G, "hand_l", (1.0, -0.07, 0.62), (0.42, 0.46, 0.34), (6, 0, 0), seed=70, runes=0, tint=0.5)
            for k, dx in enumerate((-0.13, 0.0, 0.13)):
                crystal(G, "hand_l", (1.0 + dx, -0.2, 0.52), (dx * 1.5, -0.55, -1), 0.34, 0.04, 700 + k)
        else:
            stone_block(G, "hand_r", (-1.0, -0.06, 0.52), (0.48, 0.5, 0.56), (0, 0, 4), seed=72, runes=1, tint=0.25, face=(-1, 0, 0))
            for k in range(5):
                crystal(G, "hand_r", (-1.0 + rng.uniform(-0.16, 0.16), -0.3, 0.4 + 0.07 * k), (rng.normal(0, 0.3), -1, rng.normal(0, 0.3)), 0.16, 0.03, 720 + k)
        # legs: thick and short
        stone_block(G, f"thigh_{tag}", (sx * 0.32, 0.0, 0.82), (0.44, 0.48, 0.4), (0, 0, sx * 3), seed=80 + (sx > 0), runes=1 if sx < 0 else 0, tint=0.4)
        stone_block(G, f"shin_{tag}", (sx * 0.34, -0.02, 0.42), (0.42, 0.46, 0.42), (0, 0, 0), seed=90 + (sx > 0), runes=1 if sx > 0 else 0, tint=0.65)
        stone_block(G, f"shin_{tag}", (sx * 0.35, -0.1, 0.11), (0.5, 0.64, 0.22), (0, 0, sx * 4), seed=93 + (sx > 0), runes=0, chips=5, tint=0.2)
        cluster(G, f"shin_{tag}", (sx * 0.34, -0.24, 0.55), (sx * 0.2, -1, 0.5), 3, 0.24, 0.04, 0.4, seed=95 + (sx > 0))
    return G


def main():
    G = build()
    G.bake(size=2048)
    out = B.ROOT / "assets_src" / "monsters" / "golem.glb"
    G.export(out)


main()
