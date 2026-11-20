"""Model the forest's living things in Blender: trees, mushrooms, ferns and flowers, grass tufts
and pebbles, in the village's look (detailed geometry, painted per vertex, no textures).

Trunks, roots and branches are a skeleton of points that Blender's Skin modifier turns into one
continuous solid, smoothed with Subdivision Surface and thinned with Decimate; bark ridges and
colours are then worked into the vertices. Leaves, mushroom caps and gills, fronds and grass
blades are built leaf by leaf, blade by blade.

Every asset is written to assets_src/flora/<name>.glb with one primitive per material:
    bark  leaf  plant  grass  mush  rock     painted per vertex (COLOR_0, linear)
    glow_teal  glow_violet  glow_gold        the same, and lit from within by the game
Units are metres, the base sits on y=0 (glTF, +Y up), the front faces +Z.

Run: blender --background --factory-startup --python tools/blender_flora.py
"""
import math
import pathlib
import sys

import bpy
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from glb import Glb  # noqa: E402

OUT = HERE.parent / "assets_src" / "flora"


# ------------------------------------------------------------------ noise
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


def srgb(*rgb):
    return np.array(rgb, dtype=np.float64)


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def perpendicular(d):
    ref = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
    a = unit(np.cross(d, ref))
    return a, np.cross(d, a)


# ------------------------------------------------------------------ geometry collector
class Part:
    """Triangles for one material. Positions in Blender axes (Z up) until written out."""

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
        self.n.append(unit(np.asarray(normals, dtype=np.float64).reshape(-1, 3)))
        self.c.append(colors)
        self.f.append(faces + self.count)
        self.count += len(verts)

    def tris(self):
        return sum(len(f) for f in self.f)


class Asset:
    def __init__(self, name):
        self.name = name
        self.parts = {}

    def part(self, material):
        return self.parts.setdefault(material, Part())

    def save(self):
        out = Glb()
        J = out.json
        J["asset"]["generator"] = "blender_flora.py (Blender %s)" % bpy.app.version_string
        J["materials"], J["meshes"] = [], []
        prims = []
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for material, part in self.parts.items():
            if not part.count:
                continue
            v = np.concatenate(part.v)
            n = np.concatenate(part.n)
            c = np.clip(np.concatenate(part.c), 0, 1)
            f = np.concatenate(part.f)
            v = v[:, [0, 2, 1]] * np.array([1, 1, -1])          # Blender Z-up -> glTF Y-up
            n = n[:, [0, 2, 1]] * np.array([1, 1, -1])
            lo, hi = np.minimum(lo, v.min(axis=0)), np.maximum(hi, v.max(axis=0))
            linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
            colour = np.concatenate([linear, np.ones((len(linear), 1))], axis=1)
            J["materials"].append({"name": material, "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.85},
                                   "doubleSided": material in ("leaf", "plant", "grass")})
            prims.append({"attributes": {
                "POSITION": out.add_accessor(v.astype(np.float32), "VEC3", minmax=True, target=34962),
                "NORMAL": out.add_accessor(unit(n).astype(np.float32), "VEC3", target=34962),
                "COLOR_0": out.add_accessor(np.round(colour * 65535).astype(np.uint16), "VEC4", target=34962, normalized=True)},
                "indices": out.add_accessor(f.reshape(-1).astype(np.uint16 if len(v) < 65536 else np.uint32), "SCALAR", target=34963),
                "material": len(J["materials"]) - 1})
        J["meshes"].append({"name": self.name, "primitives": prims})
        J["nodes"] = [{"name": self.name, "mesh": 0}]
        J["scenes"] = [{"nodes": [0]}]
        J["scene"] = 0
        out.save(str(OUT / f"{self.name}.glb"))
        detail = ", ".join(f"{m} {p.tris()}" for m, p in self.parts.items() if p.count)
        print(f"FLORA {self.name:18s} {sum(p.tris() for p in self.parts.values()):6d} tris  ({detail})  "
              f"size {np.round(hi - lo, 2).tolist()}")


# ------------------------------------------------------------------ Blender helpers
def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


def evaluated(obj):
    """Vertices, triangles and normals of an object with its modifiers applied."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    nor = np.empty(len(me.vertices) * 3)
    me.vertex_normals.foreach_get("vector", nor)
    tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", tri)
    ev.to_mesh_clear()
    return co.reshape(-1, 3), tri.reshape(-1, 3), nor.reshape(-1, 3)


def skin_solid(verts, edges, radii, subdiv=2, target_tris=None):
    """Skin modifier over a branching skeleton -> one smooth closed solid."""
    me = bpy.data.meshes.new("skeleton")
    me.from_pydata([tuple(v) for v in verts], [tuple(e) for e in edges], [])
    obj = bpy.data.objects.new("skeleton", me)
    bpy.context.collection.objects.link(obj)
    skin = obj.modifiers.new("skin", "SKIN")
    skin.branch_smoothing = 0.6
    skin.use_x_symmetry = skin.use_y_symmetry = skin.use_z_symmetry = False
    layer = me.skin_vertices[0].data
    for i, r in enumerate(radii):
        layer[i].radius = (float(r), float(r))
    layer[0].use_root = True
    sub = obj.modifiers.new("smooth", "SUBSURF")
    sub.levels = subdiv
    if target_tris:
        co, tri, _ = evaluated(obj)
        if len(tri) > target_tris:
            dec = obj.modifiers.new("decimate", "DECIMATE")
            dec.ratio = target_tris / len(tri)
    obj.modifiers.new("tri", "TRIANGULATE")
    co, tri, nor = evaluated(obj)
    bpy.data.objects.remove(obj)
    return co, tri, nor


def subdivided(verts, faces, levels=1):
    """Catmull-Clark a small closed mesh (mushroom caps, pebbles) in Blender."""
    me = bpy.data.meshes.new("part")
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    obj = bpy.data.objects.new("part", me)
    bpy.context.collection.objects.link(obj)
    sub = obj.modifiers.new("smooth", "SUBSURF")
    sub.levels = levels
    obj.modifiers.new("tri", "TRIANGULATE")
    co, tri, nor = evaluated(obj)
    bpy.data.objects.remove(obj)
    # the faces were given in no particular winding: make the surface face outward
    if np.mean(np.einsum("ij,ij->i", nor, co - co.mean(axis=0))) < 0:
        tri, nor = tri[:, ::-1].copy(), -nor
    return co, tri, nor


# ------------------------------------------------------------------ skeleton for trees
class Skeleton:
    def __init__(self):
        self.verts, self.radii, self.edges = [], [], []
        self.tips = []           # (position, direction, size) where leaves go

    def add(self, pos, radius, parent=None):
        self.verts.append(np.asarray(pos, dtype=np.float64))
        self.radii.append(radius)
        i = len(self.verts) - 1
        if parent is not None:
            self.edges.append((parent, i))
        return i

    def limb(self, parent, points, radii):
        """Chain of points hanging off vertex `parent`; returns their indices."""
        ids, prev = [], parent
        for p, r in zip(points, radii):
            prev = self.add(p, r, prev)
            ids.append(prev)
        return ids


def curve(start, direction, length, steps, bend_to=None, bend=0.0, wobble=0.0, rng=None, gravity=0.0):
    """Points along a gently curving limb."""
    d = unit(direction)
    side, up = perpendicular(d)
    phase = rng.uniform(0, 6.28) if rng is not None else 0.0
    pts, p = [], np.asarray(start, dtype=np.float64)
    step = length / steps
    for k in range(1, steps + 1):
        t = k / steps
        if bend_to is not None:
            d = unit(d + unit(bend_to) * bend / steps)
        d = unit(d + np.array([0, 0, -gravity / steps]))
        w = wobble * math.sin(t * 5.2 + phase)
        p = p + d * step + (side * w + up * w * 0.5) * step
        pts.append(p.copy())
    return pts


# ------------------------------------------------------------------ leaves
def leaf_shape(length, width, fold=0.25, droop=0.25):
    """A lance leaf along +x: centre vein and two edges at three stations, then the tip."""
    stations = np.array([0.0, 0.3, 0.62])
    w = width * np.sin(np.pi * np.clip(stations, 0.05, 1)) ** 0.75 * (1 - 0.25 * stations)
    w[0] = width * 0.12
    pts = []
    for s, ww in zip(stations, w):
        z = -droop * length * s * s
        pts += [(s * length, -ww, z - fold * ww), (s * length, 0, z), (s * length, ww, z - fold * ww)]
    pts.append((length, 0, -droop * length))
    faces = [(0, 3, 4), (0, 4, 1), (1, 4, 5), (1, 5, 2), (3, 6, 7), (3, 7, 4), (4, 7, 8), (4, 8, 5), (6, 9, 7), (7, 9, 8)]
    along = np.array([0, 0, 0, 0.3, 0.3, 0.3, 0.62, 0.62, 0.62, 1.0])
    return np.array(pts), np.array(faces), along


LEAF_PALETTES = {
    # (base of the leaf, tip of the leaf) pairs the crown mixes between
    "teal": [(srgb(0.05, 0.34, 0.38), srgb(0.36, 0.86, 0.78)), (srgb(0.10, 0.30, 0.48), srgb(0.40, 0.70, 0.95))],
    "violet": [(srgb(0.25, 0.14, 0.45), srgb(0.66, 0.50, 0.95)), (srgb(0.16, 0.20, 0.46), srgb(0.48, 0.62, 0.98))],
    "magenta": [(srgb(0.40, 0.10, 0.34), srgb(0.94, 0.48, 0.78)), (srgb(0.32, 0.12, 0.46), srgb(0.78, 0.50, 0.96))],
    "green": [(srgb(0.10, 0.25, 0.10), srgb(0.42, 0.66, 0.26)), (srgb(0.12, 0.28, 0.16), srgb(0.36, 0.62, 0.40))],
}
GOLD = srgb(0.98, 0.82, 0.40)


CROWN_LOG = None     # set to a list to record (centre, radius) of every crown built


def crown(part, rng, centre, radius, palettes, leaf_len, count, flat=0.28, gold=0.08, droop_edge=0.6):
    """A shelf-like crown: overlapping leaves on a flattened dome, drooping at the rim."""
    if CROWN_LOG is not None:
        CROWN_LOG.append((np.array(centre, dtype=np.float64), float(radius)))
    base_shape, faces, along = leaf_shape(1.0, 1.0)
    for _ in range(count):
        r = radius * math.sqrt(rng.uniform(0.0, 1.0))
        a = rng.uniform(0, 2 * math.pi)
        rim = r / radius
        pos = centre + np.array([math.cos(a) * r, math.sin(a) * r, flat * radius * (1 - rim * rim) + rng.normal(0, 0.06) * radius])
        heading = a + rng.normal(0, 0.55)
        tilt = math.radians(rng.uniform(5, 30) * (1 - rim) - droop_edge * 40 * rim * rng.uniform(0.6, 1.2))
        length = leaf_len * rng.uniform(0.7, 1.25)
        width = length * rng.uniform(0.16, 0.24)
        pts = base_shape * np.array([length, width, length])
        pts = pts + np.outer(along, [0, 0, 0]) * 0
        ct, st = math.cos(tilt), math.sin(tilt)
        pts = np.stack([pts[:, 0] * ct - pts[:, 2] * st, pts[:, 1], pts[:, 0] * st + pts[:, 2] * ct], axis=1)
        roll = rng.normal(0, 0.35)
        cr, sr = math.cos(roll), math.sin(roll)
        pts = np.stack([pts[:, 0], pts[:, 1] * cr - pts[:, 2] * sr, pts[:, 1] * sr + pts[:, 2] * cr], axis=1)
        ch, sh = math.cos(heading), math.sin(heading)
        pts = np.stack([pts[:, 0] * ch - pts[:, 1] * sh, pts[:, 0] * sh + pts[:, 1] * ch, pts[:, 2]], axis=1) + pos
        low, high = palettes[int(rng.integers(0, len(palettes)))]
        other = palettes[int(rng.integers(0, len(palettes)))][1]
        high = high * (1 - 0.35 * rng.random()) + other * 0.35 * rng.random()
        light = 0.75 + 0.35 * (1 - rim) + rng.normal(0, 0.06)        # outer leaves sit in the crown's shade
        cols = low[None] + (high - low)[None] * along[:, None] ** 0.8
        if rng.random() < gold:
            cols[-4:] = cols[-4:] * 0.4 + GOLD * 0.6
        part.add(pts, cols * light, faces)


# ------------------------------------------------------------------ bark
def bark_finish(co, tri, nor, rng, scale, base_colour, sheen, moss=0.6, ridge=0.025):
    """Ridged bark worked into the trunk's surface, then painted."""
    stretch = np.array([7.0, 7.0, 1.1]) / scale
    seed = rng.uniform(0, 50)
    ridges = np.abs(fbm(co * stretch, seed, 3))
    lumps = fbm(co * (1.6 / scale), seed + 9, 3)
    co = co + nor * ((0.5 - ridges) * ridge * scale + lumps * 0.03 * scale)[:, None]
    shade = np.clip(0.55 + 0.9 * (ridges - 0.25), 0.25, 1.15)
    col = base_colour[None] * shade[:, None]
    shimmer = sstep(0.15, 0.55, fbm(co * (0.45 / scale), seed + 3, 2))
    col = col * (1 - 0.5 * shimmer[:, None]) + sheen[None] * 0.5 * shimmer[:, None] * shade[:, None]
    green = (1 - sstep(0.15 * scale, 1.1 * scale, co[:, 2])) * 0.8 + sstep(0.55, 0.9, nor[:, 2]) * 0.6
    green = np.clip(green * (0.4 + 0.8 * vnoise(co * (2.5 / scale), seed + 5)) * moss, 0, 1)
    mossy = srgb(0.20, 0.36, 0.14) * (0.8 + 0.4 * vnoise(co * 6 / scale, seed + 7))[:, None]
    col = col * (1 - green[:, None]) + mossy * green[:, None]
    return co, col


def tree(name, seed, height, kind, palettes, trunk_r, pads=None, leaves_per_pad=170, gold=0.08, brackets=False):
    """kind: 'shelf' twisted painted tree, 'umbrella' broad green tree, 'giant' framing tree."""
    rng = np.random.default_rng(seed)
    sk = Skeleton()
    asset = Asset(name)
    # trunk: an S-curve that leans and sways, flaring into roots at the foot
    lean_dir = rng.uniform(0, 2 * math.pi)
    lean = np.array([math.cos(lean_dir), math.sin(lean_dir), 0]) * rng.uniform(0.08, 0.22)
    side = np.array([-lean[1], lean[0], 0]) / max(np.linalg.norm(lean), 1e-6)
    sway = rng.uniform(0.55, 1.0) * (1.4 if kind == "giant" else 1.0) * (0.6 if kind == "umbrella" else 1.0)
    phase = rng.uniform(0, 6.28)
    steps = 14
    trunk_pts = []
    for k in range(steps + 1):
        t = k / steps
        p = np.array([0, 0, -0.35]) + np.array([0, 0, height * 0.72 * t]) + lean * height * 0.72 * t
        p = p + side * math.sin(t * math.pi * 1.5 + phase) * sway * t + lean * math.sin(t * 3.0) * 0.3
        trunk_pts.append(p)
    trunk_radii = [trunk_r * ((1 - t) ** 0.9 * 0.82 + 0.18) * (1 + 0.8 * math.exp(-t * 12)) for t in np.linspace(0, 1, steps + 1)]
    root = sk.add(trunk_pts[0], trunk_radii[0])
    trunk = [root] + sk.limb(root, trunk_pts[1:], trunk_radii[1:])
    # roots: buttresses that leave the trunk a little above the ground and dive into it
    n_roots = int(rng.integers(5, 8))
    for k in range(n_roots):
        a = k / n_roots * 2 * math.pi + rng.uniform(-0.3, 0.3)
        start = trunk[2]
        length = trunk_r * rng.uniform(2.6, 4.2)
        out = np.array([math.cos(a), math.sin(a), -0.55])
        pts = curve(sk.verts[start], out, length, 5, bend_to=np.array([math.cos(a), math.sin(a), -1.6]), bend=0.7, wobble=0.15, rng=rng)
        sk.limb(start, pts, [trunk_r * 0.62 * (1 - i / 5.5) ** 1.1 + 0.03 for i in range(1, 6)])
    # painted trunks twist: a couple of thinner strands wind round the main one and fuse into it
    if kind != "umbrella":
        for s in range(2 if kind == "shelf" else 3):
            off = s / 2 * 2 * math.pi + rng.uniform(0, 1)
            strand = []
            for k in range(1, 10):
                t = k / 14
                centre = trunk_pts[k]
                ang = off + t * 2 * math.pi * 1.1
                strand.append(centre + np.array([math.cos(ang), math.sin(ang), 0]) * trunk_radii[k] * 0.78)
            sk.limb(trunk[1], strand, [trunk_radii[k] * 0.42 for k in range(1, 10)])
    # branches
    n_branch = {"shelf": rng.integers(4, 6), "umbrella": rng.integers(5, 7), "giant": rng.integers(6, 8)}[kind]
    tips = [(sk.verts[trunk[-1]], np.array([0, 0, 1.0]), 1.15)]
    for b in range(int(n_branch)):
        t = rng.uniform(0.42, 0.9) if kind != "umbrella" else rng.uniform(0.55, 0.92)
        at = trunk[int(round(t * steps))]
        a = b / n_branch * 2 * math.pi + rng.uniform(-0.45, 0.45)
        rise = rng.uniform(0.35, 0.8) if kind != "umbrella" else rng.uniform(0.15, 0.45)
        out = np.array([math.cos(a), math.sin(a), rise])
        length = height * rng.uniform(0.28, 0.42) * (1.15 if kind == "giant" else 1.0)
        r0 = sk.radii[at] * 0.55
        pts = curve(sk.verts[at], out, length, 6, bend_to=np.array([math.cos(a), math.sin(a), 0.1]), bend=0.8, wobble=0.25, rng=rng)
        ids = sk.limb(at, pts, [r0 * (1 - i / 6.5) + 0.02 for i in range(1, 7)])
        tips.append((sk.verts[ids[-1]], unit(out), rng.uniform(0.8, 1.05)))
        for s in range(int(rng.integers(1, 3))):          # side shoots
            j = ids[int(rng.integers(2, 5))]
            a2 = a + rng.choice([-1, 1]) * rng.uniform(0.6, 1.2)
            out2 = np.array([math.cos(a2), math.sin(a2), rng.uniform(0.2, 0.7)])
            pts2 = curve(sk.verts[j], out2, length * rng.uniform(0.35, 0.55), 4, wobble=0.3, rng=rng)
            ids2 = sk.limb(j, pts2, [sk.radii[j] * 0.6 * (1 - i / 4.5) + 0.015 for i in range(1, 5)])
            tips.append((sk.verts[ids2[-1]], unit(out2), rng.uniform(0.6, 0.85)))
    co, tri, nor = skin_solid(sk.verts, sk.edges, sk.radii, subdiv=2, target_tris=5200 if kind != "giant" else 7500)
    bark_base = srgb(0.30, 0.21, 0.16) if kind != "umbrella" else srgb(0.33, 0.27, 0.20)
    sheen = srgb(0.30, 0.34, 0.62) if kind != "umbrella" else srgb(0.34, 0.33, 0.30)
    co, col = bark_finish(co, tri, nor, rng, height / 7.0, bark_base, sheen)
    asset.part("bark").add(co, col, tri)
    # leaves
    leaf = asset.part("leaf")
    for pos, direction, size in tips:
        if kind == "umbrella":
            for layer in range(2):
                radius = height * 0.24 * size * (1 - 0.3 * layer)
                crown(leaf, rng, pos + np.array([0, 0, 0.25 + layer * 0.55]), radius, palettes, 0.32, int(leaves_per_pad * size * (1 - 0.35 * layer)),
                      flat=0.2, gold=0.0, droop_edge=0.4)
        else:
            radius = height * (0.21 if kind == "shelf" else 0.19) * size
            crown(leaf, rng, pos + np.array([0, 0, 0.2]), radius, palettes, 0.55 if kind != "giant" else 0.62,
                  int(leaves_per_pad * size), gold=gold)
    # bracket fungus climbing the trunk of the big ones
    if brackets:
        mush = asset.part("mush")
        for k in range(int(rng.integers(5, 9))):
            t = rng.uniform(0.08, 0.35)
            centre = trunk_pts[int(t * steps)] + np.array([0, 0, rng.uniform(-0.2, 0.2)])
            a = rng.uniform(0, 2 * math.pi)
            r = trunk_radii[int(t * steps)]
            shelf(mush, rng, centre + np.array([math.cos(a), math.sin(a), 0]) * r * 0.85, a, rng.uniform(0.22, 0.4) * height / 7)
    asset.save()


def shelf(part, rng, at, heading, size):
    """A bracket fungus: a half-disc shelf with a rolled lip, gold to ochre in rings."""
    seg, rings = 12, 5
    pts, cols = [], []
    for j in range(rings + 1):
        rr = j / rings
        for i in range(seg + 1):
            a = (i / seg - 0.5) * math.pi * 1.1
            x = math.cos(a) * size * rr
            y = math.sin(a) * size * rr * 1.25
            z = -0.25 * size * rr * rr + 0.05 * size * math.sin(rr * math.pi)
            pts.append((x, y, z))
            band = 0.5 + 0.5 * math.sin(rr * 13)
            cols.append(srgb(0.70, 0.46, 0.14) * (1 - rr) + srgb(0.98, 0.84, 0.42) * rr * (0.85 + 0.15 * band))
    faces = []
    for j in range(rings):
        for i in range(seg):
            a = j * (seg + 1) + i
            faces += [(a, a + 1, a + seg + 2), (a, a + seg + 2, a + seg + 1)]
    pts = np.array(pts)
    under = pts.copy()
    under[:, 2] -= 0.06 * size * (1 - (np.linalg.norm(pts[:, :2], axis=1) / size) ** 2)
    ch, sh = math.cos(heading), math.sin(heading)
    allp = np.vstack([pts, under])
    allp = np.stack([allp[:, 0] * ch - allp[:, 1] * sh, allp[:, 0] * sh + allp[:, 1] * ch, allp[:, 2]], axis=1) + at
    n = len(pts)
    faces = faces + [(c, b, a) for a, b, c in [(f[0] + n, f[1] + n, f[2] + n) for f in faces]]
    part.add(allp, np.vstack([np.array(cols), np.array(cols) * 0.55]), faces)


# ------------------------------------------------------------------ mushrooms
def lathe(profile, seg, colour_fn, twist=0.0):
    """Revolve a (radius, height) profile around Z. Returns verts, quad faces, (t, angle)."""
    verts, uvs = [], []
    for j, (r, z) in enumerate(profile):
        for i in range(seg):
            a = i / seg * 2 * math.pi + twist * j
            verts.append((r * math.cos(a), r * math.sin(a), z))
            uvs.append((j / (len(profile) - 1), a))
    faces = []
    for j in range(len(profile) - 1):
        for i in range(seg):
            a, b = j * seg + i, j * seg + (i + 1) % seg
            faces.append((a, b, b + seg, a + seg))
    return np.array(verts), faces, np.array(uvs)


def mushroom(asset, rng, at, tall, cap_r, cap_col, stem_col, glow=None, spots=0.0, shape="dome", lean=0.0, skirt=False):
    """One mushroom: bent stem with a bulb, a smooth cap with a rolled rim, gills underneath."""
    body = asset.part(glow or "mush")
    stem_r = cap_r * (0.17 if shape != "bell" else 0.12)
    heading = rng.uniform(0, 2 * math.pi)
    bend = np.array([math.cos(heading), math.sin(heading), 0]) * lean * tall
    # stem
    prof = [(stem_r * 1.45, -0.03), (stem_r * 1.35, 0.04 * tall), (stem_r * 1.05, 0.15 * tall), (stem_r * 0.95, 0.5 * tall), (stem_r * 0.85, 0.92 * tall), (stem_r * 0.7, tall)]
    v, quads, uv = lathe(prof, 8 if cap_r > 0.08 else 6, None)
    t = np.clip(v[:, 2] / tall, 0, 1)
    v = v + np.outer(np.sin(t * math.pi * 0.5), bend)
    tri = [(a, b, c) for a, b, c, d in quads] + [(a, c, d) for a, b, c, d in quads]
    col = stem_col[None] * (0.7 + 0.35 * t[:, None]) * (0.92 + 0.12 * vnoise(v * 30, 4)[:, None])
    asset.part("mush" if glow is None else glow).add(v + at, col, tri)
    top = at + np.array([0, 0, tall]) + bend
    if skirt:
        sv, sq, _ = lathe([(stem_r * 0.8, -0.02 * tall), (stem_r * 2.4, -0.11 * tall), (stem_r * 2.3, -0.14 * tall)], 12, None)
        sv[:, 2] += -0.18 * tall
        asset.part("mush").add(sv + top, stem_col * 1.05, [(a, b, c) for a, b, c, d in sq] + [(a, c, d) for a, b, c, d in sq])
    # cap: a profile closed at the crown, Catmull-Clark smoothed
    h = cap_r * {"dome": 0.62, "flat": 0.32, "bell": 1.15, "wide": 0.45}[shape]
    # crown -> shoulder -> rim, which curls under, then the underside climbs back up into the
    # cap so the gills sit tucked inside the lip instead of hanging below it
    if shape == "bell":
        prof = [(cap_r * 0.0, h), (cap_r * 0.35, h * 0.95), (cap_r * 0.62, h * 0.72), (cap_r * 0.86, h * 0.35), (cap_r, h * 0.03),
                (cap_r * 0.9, -h * 0.03), (cap_r * 0.62, h * 0.22), (stem_r * 1.2, h * 0.42)]
    else:
        prof = [(cap_r * 0.0, h), (cap_r * 0.45, h * 0.94), (cap_r * 0.78, h * 0.68), (cap_r * 0.97, h * 0.3), (cap_r * 1.02, h * 0.04),
                (cap_r * 0.95, -h * 0.07), (cap_r * 0.72, h * 0.14), (cap_r * 0.4, h * 0.24), (stem_r * 1.1, h * 0.27)]
    gill_top = prof[-1][1]
    seg = 12 if cap_r > 0.08 else 8
    v, quads, uv = lathe(prof[1:], seg, None)
    crown_i = len(v)
    v = np.vstack([v, [prof[0][0], 0, prof[0][1]]])
    faces = list(quads) + [(crown_i, i, (i + 1) % seg) for i in range(seg)][::1]
    faces = [tuple(reversed(f)) if len(f) == 3 else f for f in faces]
    v = v + np.c_[np.zeros((len(v), 2)), fbm(v * (3.0 / cap_r), rng.uniform(0, 9), 2) * 0.05 * cap_r]
    co, tri, nor = subdivided(v, faces, 1 if cap_r > 0.05 else 0)
    upness = np.clip(co[:, 2] / max(h, 1e-6), -0.3, 1)
    col = cap_col[None] * (0.62 + 0.5 * np.clip(upness, 0, 1)[:, None] ** 0.7)
    col = col + (col - col.mean(axis=1, keepdims=True)) * 0.25                 # a touch more saturated
    gill_colour = stem_col * 0.5 + cap_col * 0.2
    underside = nor[:, 2] < -0.25
    col[underside] = gill_colour
    if spots:
        dots = sstep(0.68, 0.74, vnoise(co * (6.5 / cap_r), rng.uniform(0, 9))) * (upness > 0.15)
        col = col * (1 - dots[:, None] * spots) + srgb(0.98, 0.96, 0.88) * dots[:, None] * spots
    body.add(co + top, col, tri, nor)
    # gills: thin radial fins under the cap
    gill_part = asset.part("mush" if glow is None else glow)
    fins = 18 if cap_r > 0.08 else 8
    gv, gf, gc = [], [], []
    for i in range(fins):
        a = i / fins * 2 * math.pi
        d = np.array([math.cos(a), math.sin(a), 0])
        base = len(gv)
        r0, r1 = stem_r * 1.3, cap_r * 0.82
        z0, z1 = gill_top * 0.9, h * 0.02
        gv += [d * r0 + [0, 0, z0], d * r1 + [0, 0, z1 + 0.02 * cap_r], d * r1 + [0, 0, z1], d * r0 + [0, 0, z0 - 0.12 * cap_r]]
        gf += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        gc += [gill_colour * 0.8] * 4
    gill_part.add(np.array(gv) + top, np.array(gc), gf, normals=np.tile([0, 0, -1.0], (len(gv), 1)))


def mushroom_cluster(name, seed, items, ring=0.18):
    rng = np.random.default_rng(seed)
    asset = Asset(name)
    for i, item in enumerate(items):
        a = i / len(items) * 2 * math.pi + rng.uniform(-0.5, 0.5)
        r = 0 if i == 0 else ring * rng.uniform(0.6, 1.2)
        at = np.array([math.cos(a) * r, math.sin(a) * r, 0])
        mushroom(asset, rng, at, **item)
    # a few short sprigs of moss the cluster grows from
    moss = asset.part("plant")
    for k in range(10):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.0, ring * 1.6)
        blade(moss, rng, np.array([math.cos(a) * r, math.sin(a) * r, 0]), rng.uniform(0.03, 0.06), 0.01,
              srgb(0.10, 0.28, 0.12), srgb(0.36, 0.58, 0.24), lean=0.8)
    asset.save()


# ------------------------------------------------------------------ plants and grass
def blade(part, rng, at, length, width, low, high, lean=0.5, segments=3, heading=None, curl=0.0):
    heading = rng.uniform(0, 2 * math.pi) if heading is None else heading
    d = np.array([math.cos(heading), math.sin(heading), 0])
    side = np.array([-d[1], d[0], 0])
    pts, cols = [], []
    for k in range(segments + 1):
        t = k / segments
        out = lean * length * t * t
        p = at + d * out + np.array([0, 0, length * t * (1 - 0.25 * lean * t)])
        if curl:
            p = p + side * curl * length * t * t
        w = width * (1 - t) ** 0.9
        pts += [p - side * w, p + side * w]
        c = low * (1 - t) + high * t
        cols += [c, c]
    pts[-2] = pts[-1] = (pts[-2] + pts[-1]) / 2
    faces = []
    for k in range(segments):
        a = k * 2
        faces += [(a, a + 1, a + 3), (a, a + 3, a + 2)]
    normal = np.cross(side, [0, 0, 1.0]) * -1 + np.array([0, 0, 0.8])
    part.add(np.array(pts), np.array(cols), faces, normals=np.tile(unit(normal), (len(pts), 1)))


def grass_tuft(name, seed, blades, height, low, high, spread=0.12):
    rng = np.random.default_rng(seed)
    asset = Asset(name)
    part = asset.part("grass")
    for _ in range(blades):
        a, r = rng.uniform(0, 2 * math.pi), spread * math.sqrt(rng.uniform(0, 1))
        tone = rng.uniform(0.82, 1.12)
        blade(part, rng, np.array([math.cos(a) * r, math.sin(a) * r, 0]), height * rng.uniform(0.55, 1.15), rng.uniform(0.010, 0.017),
              low * tone, high * tone, lean=rng.uniform(0.25, 0.9), heading=a + rng.normal(0, 0.6), curl=rng.normal(0, 0.15))
    asset.save()


def fern(name, seed, size, low, high, fronds=9):
    rng = np.random.default_rng(seed)
    asset = Asset(name)
    part = asset.part("plant")
    for f in range(fronds):
        a = f / fronds * 2 * math.pi + rng.normal(0, 0.25)
        d = np.array([math.cos(a), math.sin(a), 0])
        side = np.array([-d[1], d[0], 0])
        length = size * rng.uniform(0.75, 1.1)
        rise = rng.uniform(0.7, 1.1)
        stations = 12
        spine = [np.array([0, 0, 0.0])]
        for k in range(1, stations + 1):
            t = k / stations
            spine.append(d * length * t + np.array([0, 0, length * rise * (t - 0.85 * t * t)]))
        for k in range(1, stations):
            t = k / stations
            p = spine[k]
            tangent = unit(spine[k + 1] - spine[k - 1])
            ll = length * 0.2 * math.sin(math.pi * min(1, t * 1.1)) + 0.01
            for sgn in (-1, 1):
                tip = p + side * sgn * ll + tangent * ll * 0.45 + np.array([0, 0, -ll * 0.25])
                mid = (p + tip) / 2 + np.array([0, 0, ll * 0.08])
                w = ll * 0.28
                quad = [p, mid + tangent * w, tip, mid - tangent * w]
                col = low * (1 - t) + high * t
                part.add(np.array(quad), [col * 0.8, col, col * 1.12, col], [(0, 1, 2), (0, 2, 3)])
        tube_line(part, spine, 0.008 * size / 0.6, low * 0.8, 4)
    asset.save()


def tube_line(part, pts, radius, colour, sides=5):
    pts = [np.asarray(p, dtype=np.float64) for p in pts]
    verts, faces = [], []
    for i, p in enumerate(pts):
        d = unit(pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)])
        a, b = perpendicular(d)
        r = radius * (1 - 0.7 * i / max(1, len(pts) - 1))
        for s in range(sides):
            ang = s / sides * 2 * math.pi
            verts.append(p + (a * math.cos(ang) + b * math.sin(ang)) * r)
    for i in range(len(pts) - 1):
        for s in range(sides):
            a0, a1 = i * sides + s, i * sides + (s + 1) % sides
            faces += [(a0, a1, a1 + sides), (a0, a1 + sides, a0 + sides)]
    part.add(np.array(verts), colour, faces)


def curl_plant(name, seed, size, low, high, stems=6):
    """Fiddleheads: thin stems rolling into spirals, like the purple curls in the painting."""
    rng = np.random.default_rng(seed)
    asset = Asset(name)
    part = asset.part("plant")
    for s in range(stems):
        a = s / stems * 2 * math.pi + rng.normal(0, 0.3)
        d = np.array([math.cos(a), math.sin(a), 0])
        tall = size * rng.uniform(0.6, 1.05)
        pts = [d * 0.02 * k + np.array([0, 0, tall * k / 10]) for k in range(11)]
        top = pts[-1]
        turns = rng.uniform(1.2, 1.9)
        r0 = tall * 0.22
        for k in range(1, 15):
            t = k / 14
            ang = t * turns * 2 * math.pi
            r = r0 * (1 - 0.85 * t)
            pts.append(top + d * r * math.sin(ang) + np.array([0, 0, r * (1 - math.cos(ang))]) * 0.9 + d * r0 * 0.0)
        verts = []
        tube_line(part, pts, 0.012 * size / 0.5, [low * (1 - k / len(pts)) + high * (k / len(pts)) for k in range(len(pts)) for _ in range(4)], 4)
        # a few small leaflets on the stem
        for k in range(2, 9, 2):
            p = pts[k]
            for sgn in (-1, 1):
                side = np.array([-d[1], d[0], 0]) * sgn
                tip = p + side * 0.07 * size + np.array([0, 0, 0.03 * size])
                part.add(np.array([p, tip + np.array([0, 0, 0.02 * size]), tip, tip - np.array([0, 0, 0.015 * size])]), low * 1.1, [(0, 1, 2), (0, 2, 3)])
        del verts
    asset.save()


def flower_clump(name, seed, size, petal, centre, stems=9):
    rng = np.random.default_rng(seed)
    asset = Asset(name)
    plant = asset.part("plant")
    leaf_low, leaf_high = srgb(0.10, 0.26, 0.14), srgb(0.30, 0.52, 0.28)
    for k in range(12):
        blade(plant, rng, np.zeros(3), size * rng.uniform(0.35, 0.6), 0.02, leaf_low, leaf_high, lean=0.9)
    for s in range(stems):
        a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0, 0.1)
        base = np.array([math.cos(a) * r, math.sin(a) * r, 0])
        tall = size * rng.uniform(0.6, 1.0)
        bend = np.array([math.cos(a), math.sin(a), 0]) * rng.uniform(0.03, 0.12)
        pts = [base + np.array([0, 0, tall * t]) + bend * t * t for t in np.linspace(0, 1, 6)]
        tube_line(plant, pts, 0.006, leaf_low * 1.2, 4)
        head = pts[-1]
        n = 6
        pr = size * rng.uniform(0.06, 0.09)
        tone = rng.uniform(0.85, 1.1)
        verts, cols, faces = [head + np.array([0, 0, 0.004])], [centre], []
        for i in range(n):
            ang = i / n * 2 * math.pi
            for j, (rr, zz) in enumerate(((0.45, 0.01), (1.0, 0.0))):
                for o in (-0.28, 0.28):
                    verts.append(head + np.array([math.cos(ang + o / (1 + j)) * pr * rr, math.sin(ang + o / (1 + j)) * pr * rr, zz * pr]))
                    cols.append(petal * tone * (0.8 + 0.3 * j))
            b = 1 + i * 4
            faces += [(0, b, b + 1), (b, b + 2, b + 3), (b, b + 3, b + 1)]
        plant.add(np.array(verts), np.array(cols), faces, normals=np.tile([0, 0, 1.0], (len(verts), 1)))
        # a small golden centre
        plant.add(np.array([head + [0.012, 0, 0.012], head + [-0.006, 0.011, 0.012], head + [-0.006, -0.011, 0.012]]), centre, [(0, 1, 2)],
                  normals=np.tile([0, 0, 1.0], (3, 1)))
    asset.save()


def broad_blades(name, seed, size, low, high, count=10):
    """Agave-like clump of long arching blades (the teal and violet tufts of the painting)."""
    rng = np.random.default_rng(seed)
    asset = Asset(name)
    part = asset.part("plant")
    for k in range(count):
        a = k / count * 2 * math.pi + rng.normal(0, 0.3)
        blade(part, rng, np.zeros(3), size * rng.uniform(0.7, 1.15), size * rng.uniform(0.05, 0.08), low, high,
              lean=rng.uniform(0.6, 1.1), heading=a, segments=5, curl=rng.normal(0, 0.12))
    asset.save()


def pebble(name, seed, size, mossy=0.5):
    rng = np.random.default_rng(seed)
    asset = Asset(name)
    t = (1 + 5 ** 0.5) / 2
    v = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
                  (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], dtype=np.float64)
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = unit(v) * np.array([1.0, 0.8, 0.55])
    v = v * (1 + 0.22 * fbm(v * 1.3, rng.uniform(0, 9), 2))[:, None]
    co, tri, nor = subdivided(v, f, 2)
    co = co + nor * (fbm(co * 3.2, rng.uniform(0, 9), 3) * 0.06)[:, None]
    co = co * size
    co[:, 2] += size * 0.35
    tone = vnoise(co * (2.5 / size), rng.uniform(0, 9))
    col = srgb(0.16, 0.15, 0.14)[None] + (srgb(0.38, 0.35, 0.31) - srgb(0.16, 0.15, 0.14))[None] * tone[:, None]
    green = np.clip(sstep(0.3, 0.9, nor[:, 2]) * mossy * (0.4 + vnoise(co * 5 / size, 3)), 0, 1)
    col = col * (1 - green[:, None]) + srgb(0.22, 0.38, 0.16) * green[:, None]
    asset.part("rock").add(co, col, tri, nor)
    asset.save()


# ------------------------------------------------------------------ the catalogue
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clear_scene()
    P = LEAF_PALETTES
    # trees (heights in metres); the forest assembler picks one per tree spot
    tree("tree_teal", 11, 7.5, "shelf", P["teal"] + P["violet"][:1], 0.42, gold=0.06)
    tree("tree_violet", 23, 7.0, "shelf", P["violet"] + P["teal"][1:], 0.40, gold=0.08)
    tree("tree_magenta", 37, 6.6, "shelf", P["magenta"] + P["violet"][:1], 0.38, gold=0.12)
    tree("tree_green", 41, 7.8, "umbrella", P["green"], 0.40, leaves_per_pad=72)
    tree("tree_green_b", 53, 6.4, "umbrella", P["green"] + P["teal"][:1], 0.36, leaves_per_pad=68)
    tree("tree_giant", 67, 10.5, "giant", P["violet"] + P["teal"] + P["magenta"][:1], 0.62, leaves_per_pad=200, gold=0.1, brackets=True)
    tree("tree_giant_b", 79, 9.6, "giant", P["teal"] + P["magenta"], 0.58, leaves_per_pad=190, gold=0.1, brackets=True)
    clear_scene()
    # mushrooms
    tan, orange = srgb(0.86, 0.66, 0.40), srgb(0.92, 0.46, 0.18)
    cream = srgb(0.86, 0.82, 0.70)
    mushroom_cluster("mush_tan", 101, [dict(tall=0.30, cap_r=0.14, cap_col=tan, stem_col=cream, spots=0.6),
                                       dict(tall=0.22, cap_r=0.10, cap_col=tan, stem_col=cream, spots=0.6, lean=0.2),
                                       dict(tall=0.14, cap_r=0.07, cap_col=tan * 1.05, stem_col=cream, spots=0.5, lean=0.3),
                                       dict(tall=0.10, cap_r=0.05, cap_col=tan, stem_col=cream, lean=0.4, shape="flat")], ring=0.13)
    mushroom_cluster("mush_orange", 113, [dict(tall=0.34, cap_r=0.16, cap_col=orange, stem_col=cream * 0.95, shape="wide", spots=0.25),
                                          dict(tall=0.25, cap_r=0.12, cap_col=orange * 1.05, stem_col=cream, shape="wide", lean=0.25),
                                          dict(tall=0.18, cap_r=0.09, cap_col=orange, stem_col=cream, shape="dome", lean=0.3),
                                          dict(tall=0.13, cap_r=0.06, cap_col=orange, stem_col=cream, shape="dome", lean=0.35),
                                          dict(tall=0.09, cap_r=0.045, cap_col=orange * 0.95, stem_col=cream, shape="dome", lean=0.4)], ring=0.14)
    teal_cap, teal_stem = srgb(0.35, 0.92, 0.86), srgb(0.62, 0.86, 0.92)
    mushroom_cluster("mush_glow_teal", 127, [dict(tall=0.42, cap_r=0.09, cap_col=teal_cap, stem_col=teal_stem, glow="glow_teal", shape="bell", lean=0.15),
                                             dict(tall=0.32, cap_r=0.07, cap_col=teal_cap, stem_col=teal_stem, glow="glow_teal", shape="bell", lean=0.25),
                                             dict(tall=0.24, cap_r=0.06, cap_col=teal_cap, stem_col=teal_stem, glow="glow_teal", shape="bell", lean=0.3),
                                             dict(tall=0.18, cap_r=0.045, cap_col=teal_cap, stem_col=teal_stem, glow="glow_teal", shape="bell", lean=0.35),
                                             dict(tall=0.12, cap_r=0.035, cap_col=teal_cap, stem_col=teal_stem, glow="glow_teal", shape="bell", lean=0.4)], ring=0.12)
    violet_cap, blue_stem = srgb(0.62, 0.36, 0.86), srgb(0.40, 0.70, 0.98)
    mushroom_cluster("mush_glow_violet", 139, [dict(tall=1.25, cap_r=0.55, cap_col=violet_cap, stem_col=blue_stem, glow="glow_violet", shape="wide", spots=0.45, lean=0.12),
                                               dict(tall=0.7, cap_r=0.32, cap_col=violet_cap * 1.05, stem_col=blue_stem, glow="glow_violet", shape="dome", spots=0.4, lean=0.25)], ring=0.55)
    mushroom_cluster("mush_glow_small", 151, [dict(tall=rt, cap_r=rt * 0.38, cap_col=srgb(0.45, 0.95, 0.98), stem_col=srgb(0.70, 0.92, 0.96), glow="glow_teal", lean=0.3)
                                              for rt in (0.09, 0.075, 0.07, 0.06, 0.055, 0.05, 0.045, 0.04)], ring=0.1)
    mushroom_cluster("mush_gold", 163, [dict(tall=0.55, cap_r=0.2, cap_col=srgb(0.96, 0.86, 0.52), stem_col=srgb(0.95, 0.92, 0.80), glow="glow_gold", shape="dome", lean=0.15),
                                        dict(tall=0.36, cap_r=0.14, cap_col=srgb(0.92, 0.84, 0.56), stem_col=srgb(0.95, 0.92, 0.80), glow="glow_gold", shape="dome", lean=0.3),
                                        dict(tall=0.24, cap_r=0.09, cap_col=srgb(0.92, 0.84, 0.56), stem_col=srgb(0.95, 0.92, 0.80), glow="glow_gold", shape="dome", lean=0.35)], ring=0.16)
    clear_scene()
    # plants
    fern("fern_teal", 201, 0.75, srgb(0.06, 0.30, 0.30), srgb(0.30, 0.78, 0.66))
    fern("fern_green", 213, 0.65, srgb(0.10, 0.28, 0.12), srgb(0.40, 0.66, 0.30), fronds=8)
    curl_plant("curl_violet", 227, 0.55, srgb(0.30, 0.16, 0.44), srgb(0.78, 0.60, 0.98))
    curl_plant("curl_teal", 239, 0.45, srgb(0.12, 0.34, 0.40), srgb(0.55, 0.92, 0.90), stems=5)
    flower_clump("flower_pink", 251, 0.42, srgb(0.95, 0.52, 0.78), srgb(0.98, 0.86, 0.40))
    flower_clump("flower_violet", 263, 0.38, srgb(0.66, 0.50, 0.98), srgb(0.98, 0.90, 0.55))
    flower_clump("flower_gold", 277, 0.34, srgb(0.98, 0.82, 0.38), srgb(0.90, 0.46, 0.20))
    broad_blades("blades_teal", 281, 0.62, srgb(0.05, 0.26, 0.28), srgb(0.32, 0.86, 0.76))
    broad_blades("blades_violet", 293, 0.55, srgb(0.20, 0.10, 0.32), srgb(0.70, 0.48, 0.92))
    # grass: deep forest green into sunlit yellow-green, like the village's meadow
    grass_tuft("grass_a", 301, 24, 0.5, srgb(0.13, 0.27, 0.07), srgb(0.60, 0.72, 0.28), spread=0.2)
    grass_tuft("grass_b", 313, 22, 0.4, srgb(0.12, 0.25, 0.08), srgb(0.50, 0.68, 0.26), spread=0.18)
    grass_tuft("grass_c", 327, 26, 0.62, srgb(0.14, 0.28, 0.08), srgb(0.66, 0.74, 0.32), spread=0.22)
    grass_tuft("grass_teal", 331, 22, 0.45, srgb(0.08, 0.24, 0.14), srgb(0.40, 0.70, 0.46), spread=0.18)
    # pebbles
    for i, (size, moss) in enumerate(((0.22, 0.5), (0.3, 0.7), (0.16, 0.3), (0.42, 0.8))):
        pebble(f"pebble_{i}", 401 + i * 7, size, moss)


if __name__ == "__main__":
    main()
