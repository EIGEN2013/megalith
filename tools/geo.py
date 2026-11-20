"""Shared modelling kit for the Blender scripts: primitives built as numpy geometry, painted per
vertex, collected per material and written as GLB.

Axes: glTF (+Y up, models face +Z). Units: metres. Colours: sRGB in, linear out.
Blender (bpy) is used when available to Catmull-Clark smooth organic parts; everything else is
plain numpy so the same code runs anywhere.
"""
import math
import pathlib

import numpy as np

try:
    import bpy  # noqa: F401
    HAVE_BPY = True
except ImportError:
    HAVE_BPY = False

from glb import Glb


# ------------------------------------------------------------------ maths
def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def look_frame(direction):
    """Rotation whose +Y axis points along `direction`."""
    d = unit(direction)
    ref = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 0, 1.0])
    x = unit(np.cross(d, ref))
    z = np.cross(x, d)
    return np.stack([x, d, z], axis=1)


def _hash(ix, iy, iz, seed):
    v = np.sin(ix * 127.1 + iy * 311.7 + iz * 74.7 + seed * 19.19) * 43758.5453
    return v - np.floor(v)


def vnoise(p, seed=0.0):
    p = np.asarray(p, dtype=np.float64)
    i = np.floor(p)
    f = p - i
    u = f * f * (3 - 2 * f)
    x0, y0, z0 = i[..., 0], i[..., 1], i[..., 2]

    def h(dx, dy, dz):
        return _hash(x0 + dx, y0 + dy, z0 + dz, seed)
    a = h(0, 0, 0) * (1 - u[..., 0]) + h(1, 0, 0) * u[..., 0]
    b = h(0, 1, 0) * (1 - u[..., 0]) + h(1, 1, 0) * u[..., 0]
    c = h(0, 0, 1) * (1 - u[..., 0]) + h(1, 0, 1) * u[..., 0]
    d = h(0, 1, 1) * (1 - u[..., 0]) + h(1, 1, 1) * u[..., 0]
    return (a * (1 - u[..., 1]) + b * u[..., 1]) * (1 - u[..., 2]) + (c * (1 - u[..., 1]) + d * u[..., 1]) * u[..., 2]


def fbm(p, seed=0.0, octaves=4):
    total, amp, freq = 0.0, 0.5, 1.0
    for o in range(octaves):
        total = total + amp * (vnoise(np.asarray(p) * freq, seed + o * 7.31) * 2 - 1)
        amp *= 0.5
        freq *= 2.0
    return total


def sstep(lo, hi, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def C(*rgb):
    return np.array(rgb, dtype=np.float64)


def grain(verts, base, amount=0.12, scale=6.0, seed=1.0, streak=None):
    """Per-vertex colour variation; streak=(sx, sy, sz) stretches the noise (wood grain)."""
    p = np.asarray(verts) * (np.array(streak) if streak is not None else 1.0) * scale
    n = vnoise(p, seed)
    return np.asarray(base)[None] * (1 - amount + 2 * amount * n)[:, None]


# ------------------------------------------------------------------ collector
class Part:
    def __init__(self):
        self.v, self.n, self.c, self.f = [], [], [], []
        self.count = 0

    def add(self, verts, colors, faces, normals=None):
        verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        colors = np.broadcast_to(np.asarray(colors, dtype=np.float64), verts.shape).copy()
        if normals is None:
            normals = np.zeros_like(verts)
            tri = verts[faces]
            fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            for k in range(3):
                np.add.at(normals, faces[:, k], fn)
        self.v.append(verts)
        self.n.append(unit(np.broadcast_to(np.asarray(normals, dtype=np.float64), verts.shape)))
        self.c.append(colors)
        self.f.append(faces + self.count)
        self.count += len(verts)

    def tris(self):
        return sum(len(f) for f in self.f)

    def arrays(self):
        v = np.concatenate(self.v)
        n = np.concatenate(self.n)
        c = np.clip(np.concatenate(self.c), 0, 1)
        f = np.concatenate(self.f)
        return v, n, c, f


class Asset:
    """Geometry grouped by material; transforms can be stacked while building."""

    def __init__(self, name):
        self.name = name
        self.parts = {}
        self.extras = {}
        self._stack = [(np.eye(3), np.zeros(3))]

    def part(self, material):
        return self.parts.setdefault(material, Part())

    # transform stack: everything added inside `with asset.at(...)` is moved there
    class _At:
        def __init__(self, asset, m, t):
            self.asset, self.m, self.t = asset, m, t

        def __enter__(self):
            pm, pt = self.asset._stack[-1]
            self.asset._stack.append((pm @ self.m, pm @ self.t + pt))
            return self.asset

        def __exit__(self, *exc):
            self.asset._stack.pop()

    def at(self, offset=(0, 0, 0), yaw=0.0, matrix=None, scale=1.0):
        m = (rot_y(yaw) if matrix is None else np.asarray(matrix)) * scale
        return Asset._At(self, m, np.asarray(offset, dtype=np.float64))

    def add(self, material, verts, colors, faces, normals=None):
        m, t = self._stack[-1]
        verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3) @ m.T + t
        if normals is not None:
            normals = unit(np.broadcast_to(np.asarray(normals, dtype=np.float64), verts.shape) @ np.linalg.inv(m))
        self.part(material).add(verts, colors, faces, normals)

    def bounds(self):
        allv = np.concatenate([np.concatenate(p.v) for p in self.parts.values() if p.count])
        return allv.min(axis=0), allv.max(axis=0)

    def tris(self):
        return sum(p.tris() for p in self.parts.values())

    def write(self, path, materials=None):
        """materials: name -> dict(rough=, metal=, double=, emissive=(r,g,b), strength=)"""
        materials = materials or {}
        out = Glb()
        J = out.json
        J["asset"]["generator"] = "geo.py"
        J["materials"], J["meshes"] = [], []
        prims = []
        for name, part in self.parts.items():
            if not part.count:
                continue
            v, n, c, f = part.arrays()
            spec = materials.get(name, {})
            linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
            colour = np.concatenate([linear, np.ones((len(linear), 1))], axis=1)
            m = {"name": name, "doubleSided": bool(spec.get("double", False)),
                 "pbrMetallicRoughness": {"metallicFactor": spec.get("metal", 0.0), "roughnessFactor": spec.get("rough", 0.85)}}
            if spec.get("emissive"):
                m["emissiveFactor"] = list(spec["emissive"])
                m["extensions"] = {"KHR_materials_emissive_strength": {"emissiveStrength": spec.get("strength", 1.0)}}
                J["extensionsUsed"] = ["KHR_materials_emissive_strength"]
            J["materials"].append(m)
            prims.append({"attributes": {
                "POSITION": out.add_accessor(v.astype(np.float32), "VEC3", minmax=True, target=34962),
                "NORMAL": out.add_accessor(unit(n).astype(np.float32), "VEC3", target=34962),
                "COLOR_0": out.add_accessor(np.round(colour * 255).astype(np.uint8), "VEC4", target=34962, normalized=True)},
                "indices": out.add_accessor(f.reshape(-1).astype(np.uint16 if len(v) < 65536 else np.uint32), "SCALAR", target=34963),
                "material": len(J["materials"]) - 1})
        J["meshes"].append({"name": self.name, "primitives": prims})
        J["nodes"] = [{"name": self.name, "mesh": 0, "extras": self.extras} if self.extras else {"name": self.name, "mesh": 0}]
        J["scenes"] = [{"nodes": [0]}]
        J["scene"] = 0
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        out.save(str(path))
        lo, hi = self.bounds()
        print(f"MODEL {self.name:22s} {self.tris():7d} tris  size {np.round(hi - lo, 2).tolist()}")


def write_rig(path, name, parts, materials=None):
    """Several Assets as a node hierarchy in one GLB.
    parts: list of dicts {name, asset, parent (index or None), pivot (model-space xyz)};
    each part's geometry is modelled in model space and stored relative to its pivot, and each
    node sits at its pivot relative to its parent's pivot - so rotating a node bends at its joint."""
    materials = materials or {}
    out = Glb()
    J = out.json
    J["asset"]["generator"] = "geo.py rig"
    J["materials"], J["meshes"], J["nodes"] = [], [], []
    mat_index = {}
    for i, p in enumerate(parts):
        pivot = np.asarray(p["pivot"], float)
        prims = []
        for mname, part in p["asset"].parts.items():
            if not part.count:
                continue
            if mname not in mat_index:
                spec = materials.get(mname, {})
                m = {"name": mname, "doubleSided": bool(spec.get("double", False)),
                     "pbrMetallicRoughness": {"metallicFactor": spec.get("metal", 0.0), "roughnessFactor": spec.get("rough", 0.85)}}
                if spec.get("emissive"):
                    m["emissiveFactor"] = list(spec["emissive"])
                    m["extensions"] = {"KHR_materials_emissive_strength": {"emissiveStrength": spec.get("strength", 1.0)}}
                    J["extensionsUsed"] = ["KHR_materials_emissive_strength"]
                J["materials"].append(m)
                mat_index[mname] = len(J["materials"]) - 1
            v, n, c, f = part.arrays()
            v = v - pivot
            linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
            colour = np.concatenate([linear, np.ones((len(linear), 1))], axis=1)
            prims.append({"attributes": {
                "POSITION": out.add_accessor(v.astype(np.float32), "VEC3", minmax=True, target=34962),
                "NORMAL": out.add_accessor(unit(n).astype(np.float32), "VEC3", target=34962),
                "COLOR_0": out.add_accessor(np.round(colour * 255).astype(np.uint8), "VEC4", target=34962, normalized=True)},
                "indices": out.add_accessor(f.reshape(-1).astype(np.uint16 if len(v) < 65536 else np.uint32), "SCALAR", target=34963),
                "material": mat_index[mname]})
        parent = p.get("parent")
        rel = pivot - (np.asarray(parts[parent]["pivot"], float) if parent is not None else 0)
        node = {"name": p["name"], "translation": [float(x) for x in rel]}
        if prims:
            J["meshes"].append({"name": p["name"], "primitives": prims})
            node["mesh"] = len(J["meshes"]) - 1
        J["nodes"].append(node)
    for i, p in enumerate(parts):
        if p.get("parent") is not None:
            J["nodes"][p["parent"]].setdefault("children", []).append(i)
    J["scenes"] = [{"name": name, "nodes": [i for i, p in enumerate(parts) if p.get("parent") is None]}]
    J["scene"] = 0
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.save(str(path))
    total = sum(p["asset"].tris() for p in parts)
    print(f"RIG {name:20s} {len(parts)} parts, {total} tris")


# ------------------------------------------------------------------ primitives
FACE_DIRS = [((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
             ((0, 1, 0), (0, 0, 1), (1, 0, 0)), ((0, -1, 0), (0, 0, -1), (1, 0, 0)),
             ((0, 0, 1), (0, 1, 0), (-1, 0, 0)), ((0, 0, -1), (0, 1, 0), (1, 0, 0))]


def box(asset, material, centre, size, colour, rot=None, jitter=0.0, seed=0.0, shade=True):
    """A flat-shaded box. size = full extents. colour: rgb or callable(verts)->colours."""
    hs = np.asarray(size, dtype=np.float64) / 2
    rot = np.eye(3) if rot is None else np.asarray(rot)
    verts, norms, faces = [], [], []
    for n, a, b in FACE_DIRS:
        n, a, b = np.array(n, float), np.array(a, float), np.array(b, float)
        base = len(verts)
        for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            verts.append(n * hs + a * sa * hs + b * sb * hs)
            norms.append(n)
        faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    verts = np.array(verts)
    if jitter:
        verts = verts + (vnoise(verts * 3 + seed, seed)[:, None] - 0.5) * jitter
    verts = verts @ rot.T + np.asarray(centre)
    norms = np.array(norms) @ rot.T
    # fix winding to match normals
    faces = np.array(faces)
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.einsum("ij,ij->i", fn, norms[faces[:, 0]]) < 0
    faces[flip] = faces[flip][:, ::-1]
    col = colour(verts) if callable(colour) else np.broadcast_to(colour, verts.shape)
    if shade:   # a touch of baked ambient occlusion: darker toward the bottom
        rel = (verts[:, 1] - verts[:, 1].min()) / max(np.ptp(verts[:, 1]), 1e-6)
        col = col * (0.82 + 0.18 * rel)[:, None]
    asset.add(material, verts, col, faces, norms)


def cylinder(asset, material, p0, p1, r0, r1=None, colour=(0.5, 0.5, 0.5), sides=12, caps=True, smooth=True, twist=0.0):
    r1 = r0 if r1 is None else r1
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    frame = look_frame(p1 - p0)
    ang = np.arange(sides) / sides * 2 * math.pi
    ring = np.stack([np.cos(ang), np.zeros(sides), np.sin(ang)], axis=1)
    bottom = ring * r0 @ frame.T + p0
    top = (ring @ rot_y(twist).T) * r1 @ frame.T + p1
    verts = np.vstack([bottom, top])
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces += [(i, j, sides + j), (i, sides + j, sides + i)]
    normals = None
    if smooth:
        normals = np.vstack([ring @ frame.T, (ring @ rot_y(twist).T) @ frame.T])
    col = colour(verts) if callable(colour) else colour
    asset.add(material, verts, col, _orient(verts, faces, normals, (p0 + p1) / 2), normals)
    if caps:
        for centre, rr, sign, rr_ring in ((p0, r0, -1, bottom), (p1, r1, 1, top)):
            v = np.vstack([centre, rr_ring])
            f = [(0, 1 + i, 1 + (i + 1) % sides) for i in range(sides)]
            nrm = np.tile(frame[:, 1] * sign, (len(v), 1))
            c2 = colour(v) if callable(colour) else colour
            asset.add(material, v, c2, _orient(v, f, nrm, None), nrm)


def _orient(verts, faces, normals, inside):
    faces = np.array(faces)
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if normals is not None:
        ref = np.asarray(normals)[faces[:, 0]]
    else:
        ref = tri.mean(axis=1) - inside
    flip = np.einsum("ij,ij->i", fn, ref) < 0
    faces[flip] = faces[flip][:, ::-1]
    return faces


def lathe(asset, material, profile, seg, colour, centre=(0, 0, 0), closed_top=False, rot=None):
    """Revolve (radius, height) points about +Y."""
    profile = list(profile)
    ang = np.arange(seg) / seg * 2 * math.pi
    verts = []
    for r, y in profile:
        for a in ang:
            verts.append((r * math.cos(a), y, r * math.sin(a)))
    verts = np.array(verts, dtype=np.float64)
    faces = []
    for j in range(len(profile) - 1):
        for i in range(seg):
            a, b = j * seg + i, j * seg + (i + 1) % seg
            faces += [(a, b, b + seg), (a, b + seg, a + seg)]
    if closed_top:
        apex = len(verts)
        verts = np.vstack([verts, [0, profile[-1][1], 0]])
        base = (len(profile) - 1) * seg
        faces += [(base + i, base + (i + 1) % seg, apex) for i in range(seg)]
    # normals from the profile's slope
    normals = np.zeros_like(verts)
    faces_np = np.array(faces)
    tri = verts[faces_np]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    axis_pts = np.stack([np.zeros(len(faces_np)), tri[:, :, 1].mean(axis=1), np.zeros(len(faces_np))], axis=1)
    flip = np.einsum("ij,ij->i", fn, tri.mean(axis=1) - axis_pts) < 0
    faces_np[flip] = faces_np[flip][:, ::-1]
    tri = verts[faces_np]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(normals, faces_np[:, k], fn)
    if rot is not None:
        verts, normals = verts @ np.asarray(rot).T, normals @ np.asarray(rot).T
    verts = verts + np.asarray(centre, float)
    col = colour(verts) if callable(colour) else colour
    asset.add(material, verts, col, faces_np, normals)


ICO = {}


def icosphere(level):
    if level in ICO:
        return ICO[level]
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, dtype=np.float64) / np.linalg.norm(p) for p in v]
    for _ in range(level):
        cache, nf = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[key] = len(v) - 1
            return cache[key]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    ICO[level] = (np.array(v), np.array(f))
    return ICO[level]


def blob(asset, material, centre, radii, colour, level=2, noise=0.15, freq=1.6, seed=0.0, rot=None, flat=False):
    sv, sf = icosphere(level)
    wob = 1 + noise * fbm(sv * freq + seed, seed, 3)
    p = sv * wob[:, None] * np.asarray(radii, float)
    if rot is not None:
        p = p @ np.asarray(rot).T
    p = p + np.asarray(centre, float)
    col = colour(sv, p) if callable(colour) else colour
    if flat:
        tri = p[sf]
        fn = unit(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
        cc = np.asarray(col)[sf] if np.ndim(col) == 2 else np.broadcast_to(col, tri.shape)
        asset.add(material, tri.reshape(-1, 3), np.asarray(cc).reshape(-1, 3), np.arange(len(sf) * 3).reshape(-1, 3), np.repeat(fn, 3, axis=0))
    else:
        asset.add(material, p, col, _orient(p, sf, None, np.asarray(centre, float)))


def tube(asset, material, pts, radii, colour, sides=7, cap=True):
    pts = [np.asarray(p, float) for p in pts]
    radii = np.broadcast_to(np.asarray(radii, float), (len(pts),))
    verts, faces, normals = [], [], []
    for i, p in enumerate(pts):
        d = unit(pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)])
        fr = look_frame(d)
        for s in range(sides):
            a = s / sides * 2 * math.pi
            off = fr[:, 0] * math.cos(a) + fr[:, 2] * math.sin(a)
            verts.append(p + off * radii[i])
            normals.append(off)
    for i in range(len(pts) - 1):
        for s in range(sides):
            a0, a1 = i * sides + s, i * sides + (s + 1) % sides
            faces += [(a0, a1, a1 + sides), (a0, a1 + sides, a0 + sides)]
    verts, normals = np.array(verts), np.array(normals)
    if cap:
        tip = len(verts)
        verts = np.vstack([verts, pts[-1] + unit(pts[-1] - pts[-2]) * radii[-1] * 0.5])
        normals = np.vstack([normals, unit(pts[-1] - pts[-2])])
        base = (len(pts) - 1) * sides
        faces += [(base + s, base + (s + 1) % sides, tip) for s in range(sides)]
    col = colour(verts) if callable(colour) else colour
    asset.add(material, verts, col, _orient(verts, faces, normals, None), normals)


def quad(asset, material, corners, colour, double=True):
    corners = np.asarray(corners, float)
    n = unit(np.cross(corners[1] - corners[0], corners[2] - corners[0]))
    asset.add(material, corners, colour, [(0, 1, 2), (0, 2, 3)], np.tile(n, (4, 1)))


def subdivided(verts, faces, levels=1):
    """Catmull-Clark smoothing through Blender (falls back to the raw mesh without bpy)."""
    if not HAVE_BPY:
        faces = [f if len(f) == 3 else (f[0], f[1], f[2]) for f in faces]
        return np.asarray(verts, float), np.asarray([t for f in faces for t in ([f] if len(f) == 3 else [])]), None
    import bpy
    me = bpy.data.meshes.new("part")
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    obj = bpy.data.objects.new("part", me)
    bpy.context.collection.objects.link(obj)
    sub = obj.modifiers.new("smooth", "SUBSURF")
    sub.levels = levels
    obj.modifiers.new("tri", "TRIANGULATE")
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    m2 = ev.to_mesh()
    m2.calc_loop_triangles()
    co = np.empty(len(m2.vertices) * 3)
    m2.vertices.foreach_get("co", co)
    nor = np.empty(len(m2.vertices) * 3)
    m2.vertex_normals.foreach_get("vector", nor)
    tri = np.empty(len(m2.loop_triangles) * 3, dtype=np.int64)
    m2.loop_triangles.foreach_get("vertices", tri)
    ev.to_mesh_clear()
    bpy.data.objects.remove(obj)
    bpy.data.meshes.remove(me)
    co, tri, nor = co.reshape(-1, 3), tri.reshape(-1, 3), nor.reshape(-1, 3)
    if np.mean(np.einsum("ij,ij->i", nor, co - co.mean(axis=0))) < 0:
        tri, nor = tri[:, ::-1].copy(), -nor
    return co, tri, nor


def rounded_box(asset, material, centre, size, colour, rot=None, levels=2, lumpy=0.0, seed=0.0):
    """A softened block (stone, cushion, loaf): a cube cage smoothed in Blender."""
    hs = np.asarray(size, float) / 2
    # an inset cage so the smoothed result keeps roughly the requested size
    cage = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    # add edge loops near the faces to keep the shape boxy
    k = 0.72
    pts, quads = _beveled_cage(hs, k)
    co, tri, nor = subdivided(pts, quads, levels)
    if lumpy:
        co = co + nor * (fbm(co * (2.5 / max(hs.min(), 0.05)) + seed, seed, 3) * lumpy)[:, None]
    if rot is not None:
        co, nor = co @ np.asarray(rot).T, nor @ np.asarray(rot).T
    co = co + np.asarray(centre, float)
    col = colour(co) if callable(colour) else colour
    asset.add(material, co, col, tri, nor)
    del cage, faces


def _beveled_cage(hs, k):
    """A cube cage with extra loops 1-k from each face (gives crisp-but-round corners)."""
    xs = [-1, -k, k, 1]
    pts, index = [], {}
    for i, x in enumerate(xs):
        for j, y in enumerate(xs):
            for l, z in enumerate(xs):
                if i in (0, 3) or j in (0, 3) or l in (0, 3):
                    index[(i, j, l)] = len(pts)
                    pts.append((x * hs[0], y * hs[1], z * hs[2]))
    quads = []
    n = 4
    for axis in range(3):
        for side in (0, 3):
            for a in range(3):
                for b in range(3):
                    def key(u, v):
                        idx = [0, 0, 0]
                        idx[axis] = side
                        idx[(axis + 1) % 3] = u
                        idx[(axis + 2) % 3] = v
                        return index[tuple(idx)]
                    q = (key(a, b), key(a + 1, b), key(a + 1, b + 1), key(a, b + 1))
                    quads.append(q if side == 3 else q[::-1])
    del n
    return np.array(pts), quads
