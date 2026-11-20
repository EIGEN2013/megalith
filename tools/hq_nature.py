"""Realistic trees and bushes for the land around the village.

1. Sprig textures: single leaves are modelled (lobed oak, serrated apple, glossy pear, pine needles,
   bush leaves, blossoms, berries) and arranged on twigs; each sprig is rendered top-down in Cycles
   (ambient light, so overlapping leaves shade each other) into a colour+alpha atlas, plus a
   normal-map render of the same sprigs.
2. Bark textures: tileable height fields (fissures, plates, moss) -> colour + normal maps.
3. Trees: recursive branching skeletons with root flare -> bark tubes (UV-tiled bark), and
   hundreds of sprig cards at the twig ends, with normals bent outward from the crown so the
   foliage shades softly like a real canopy.

Writes assets_src/hq/nature.glb (one root node per species) and assets_src/hq/fruit_spots.json.
Run: blender --background --factory-startup --python tools/hq_nature.py
"""
import json
import math
import pathlib
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import noise3 as N  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "assets_src" / "hq"
TEX = OUT / "nature_tex"
OUT.mkdir(parents=True, exist_ok=True)
TEX.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(11)


def lin(*c):
    return tuple(x ** 2.2 for x in c)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def link(ob):
    bpy.context.scene.collection.objects.link(ob)
    return ob


def mesh_from(name, verts, faces, uvs=None, attrs=None, normals=None, mat=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(int(i) for i in f) for f in faces])
    me.update()
    if uvs is not None:
        uvl = me.uv_layers.new(name="UVMap")
        loop_v = np.empty(len(me.loops), dtype=np.int64)
        me.loops.foreach_get("vertex_index", loop_v)
        uvl.data.foreach_set("uv", np.asarray(uvs, dtype=np.float32)[loop_v].ravel())
    for aname, vals in (attrs or {}).items():
        at = me.attributes.new(aname, "FLOAT_COLOR", "POINT")
        v4 = np.ones((len(verts), 4), dtype=np.float32)
        v4[:, :np.asarray(vals).shape[1]] = vals
        at.data.foreach_set("color", v4.ravel())
    for p in me.polygons:
        p.use_smooth = True
    if normals is not None:
        nn = np.asarray(normals, dtype=np.float64)
        nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-9
        me.normals_split_custom_set_from_vertices([tuple(n) for n in nn])
    if mat is not None:
        me.materials.append(mat)
    return link(bpy.data.objects.new(name, me))


# ============================================================================ 1. sprig renders
def leaf_outline(kind, y):
    """Half-width of a leaf (length 1) at y in [0, 1]."""
    s = math.sin(math.pi * min(max(y, 0), 1))
    if kind == "oak":
        lobes = 0.62 + 0.38 * abs(math.cos(y * math.pi * 4.5))
        return 0.30 * s ** 0.8 * lobes * (0.6 + 0.4 * y)
    if kind == "apple":
        return 0.30 * math.sin(math.pi * y ** 0.75) ** 0.9 * (1 + 0.04 * math.sin(y * 70))
    if kind == "pear":
        return 0.36 * math.sin(math.pi * y ** 0.85) ** 0.8
    if kind == "bush":
        return 0.33 * math.sin(math.pi * y ** 0.8) ** 0.9
    if kind == "reed":
        return 0.035 * (1 - y) ** 0.6
    return 0.3 * s


class MB:
    """Accumulates geometry with per-vertex attributes for one material."""

    def __init__(self):
        self.v, self.f, self.uv, self.n, self.a, self.b = [], [], [], [], [], []

    def grid(self, pts, rows, cols, attr_a, attr_b, uv=None, nrm=None):
        base = len(self.v)
        self.v += list(pts)
        self.a += list(attr_a)
        self.b += list(attr_b)
        self.uv += list(uv) if uv is not None else [(0, 0)] * len(pts)
        self.n += list(nrm) if nrm is not None else [(0, 0, 1)] * len(pts)
        for i in range(rows - 1):
            for j in range(cols - 1):
                q = base + i * cols + j
                self.f.append((q, q + 1, q + cols + 1, q + cols))

    def obj(self, name, mat, normals=False):
        return mesh_from(name, self.v, self.f, uvs=self.uv, attrs={"lc": np.array(self.a), "tint": np.array(self.b)},
                         normals=self.n if normals else None, mat=mat)


def leaf(mb, kind, base, direction, length, tint, roll=0.0, droop=0.0, curl=0.15, fold=0.18, rngl=rng):
    """A leaf from `base` along `direction` (in the XY render plane), bent in Z."""
    d = np.array([direction[0], direction[1], 0.0])
    d /= np.linalg.norm(d)
    side = np.array([-d[1], d[0], 0.0])
    up = np.array([0, 0, 1.0])
    rows, cols = 14, 7
    pts, la, lb = [], [], []
    for i in range(rows):
        y = i / (rows - 1)
        w = leaf_outline(kind, y)
        for j in range(cols):
            x = (j / (cols - 1)) * 2 - 1
            p = base + d * (y * length) + side * (x * w * length)
            z = (abs(x) * fold - curl * y * y + droop * y) * length + x * roll * w * length
            p = p + up * z
            pts.append(p)
            la.append((x, y, abs(x)))
            lb.append(tint)
    mb.grid(pts, rows, cols, la, lb)


def twig(mb, pts, r0, r1, colour):
    pts = [np.asarray(p, float) for p in pts]
    sides = 6
    rows = []
    for i, p in enumerate(pts):
        a, b = pts[max(i - 1, 0)], pts[min(i + 1, len(pts) - 1)]
        t = (b - a) / (np.linalg.norm(b - a) + 1e-9)
        n = np.cross(t, [0, 0, 1.0])
        n /= np.linalg.norm(n) + 1e-9
        bn = np.cross(t, n)
        r = r0 + (r1 - r0) * i / (len(pts) - 1)
        rows += [p + (n * math.cos(k / (sides - 1) * 2 * math.pi) + bn * math.sin(k / (sides - 1) * 2 * math.pi)) * r for k in range(sides)]
    mb.grid(rows, len(pts), sides, [(0, 0, 2.0)] * len(rows), [colour] * len(rows))


def blossom(mb, centre, r, colour, rngl):
    """Five petals around a yellow centre."""
    for k in range(5):
        a = k / 5 * 2 * math.pi + rngl.uniform(0, 0.4)
        d = np.array([math.cos(a), math.sin(a)])
        leaf(mb, "pear", np.array(centre) + np.array([*(d * r * 0.15), 0.0]), d, r, colour, curl=-0.1, fold=0.05)
    leaf(mb, "bush", np.array(centre) + [0, -r * 0.12, 0.01], (0, 1), r * 0.25, lin(0.95, 0.75, 0.2), curl=0, fold=0)


def berry(mb, centre, r, colour):
    rows, cols = 7, 8
    pts = []
    for i in range(rows):
        th = i / (rows - 1) * math.pi
        for j in range(cols):
            ph = j / (cols - 1) * 2 * math.pi
            pts.append(np.array(centre) + r * np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)]))
    mb.grid(pts, rows, cols, [(0, 0, 3.0)] * len(pts), [colour] * len(pts))


def build_sprig(kind, seed, extent=1.0):
    """Leaves on a twig, laid out in the XY plane inside [-0.5, 0.5] x [0, 1] (scaled by extent)."""
    r = np.random.default_rng(seed)
    mb = MB()
    E = extent
    bark = lin(0.30, 0.22, 0.15)
    if kind == "pine":
        # a fan of shoots, each densely clothed in needles
        stem = [np.array([r.normal(0, 0.015) * E, y * E * 0.95, 0]) for y in np.linspace(0.0, 1.0, 8)]
        twig(mb, stem, 0.014 * E, 0.005 * E, bark)
        shoots = [stem]
        for k in range(6):
            t = 0.2 + 0.13 * k
            o = stem[0] + (stem[-1] - stem[0]) * t
            side = 1 if k % 2 else -1
            d = np.array([side * r.uniform(0.6, 0.9), r.uniform(0.4, 0.7), 0])
            d /= np.linalg.norm(d)
            L = E * (0.42 - 0.18 * t)
            pts = [o + d * L * s + np.array([0, 0.05 * E * s * s, 0]) for s in np.linspace(0, 1, 5)]
            twig(mb, pts, 0.008 * E, 0.003 * E, bark)
            shoots.append(pts)
        for pts in shoots:
            a0, a1 = pts[0], pts[-1]
            axis = (a1 - a0) / np.linalg.norm(a1 - a0)
            for _ in range(int(150 * np.linalg.norm(a1 - a0) / E) + 30):
                t = r.uniform(0.0, 1.0)
                p = a0 + (a1 - a0) * t
                a = r.uniform(0, 2 * math.pi)
                side = np.array([-axis[1], axis[0], 0])
                d = axis * 0.55 + side * math.cos(a)
                L = (0.085 + 0.03 * r.random()) * E * (1.1 - 0.35 * t)
                tint = lin(0.11 + 0.05 * r.random(), 0.24 + 0.08 * r.random(), 0.13 + 0.04 * r.random())
                leaf(mb, "reed", p + np.array([0, 0, math.sin(a) * 0.012 * E]), d[:2], L, tint, curl=0.04, fold=0.0, droop=0.04 * math.sin(a))
        return mb
    if kind == "reed":
        for k in range(26):
            x0 = r.normal(0, 0.06) * E
            L = r.uniform(0.6, 1.0) * E
            d = np.array([r.normal(0, 0.15) + x0 * 0.6, 1.0])
            tint = lin(0.30 + 0.12 * r.random(), 0.42 + 0.1 * r.random(), 0.18)
            leaf(mb, "reed", np.array([x0, 0.0, 0.0]), d, L, tint, curl=0.0, fold=0.1, droop=r.normal(0, 0.15))
        for k in range(3):                     # cattail heads
            x = r.normal(0, 0.05) * E
            y = r.uniform(0.75, 0.92) * E
            twig(mb, [np.array([x * 0.5, 0, 0]), np.array([x, y, 0])], 0.006 * E, 0.004 * E, lin(0.40, 0.42, 0.22))
            twig(mb, [np.array([x, y, 0.01]), np.array([x, y + 0.11 * E, 0.01])], 0.022 * E, 0.022 * E, lin(0.30, 0.18, 0.09))
        return mb
    # broad-leaved clusters: a stem fanning into side twigs, every twig carrying leaves to its tip
    tip = np.array([r.normal(0, 0.04) * E, 0.9 * E, 0])
    stem = [tip * t + np.array([math.sin(t * 3 + seed) * 0.03 * E, 0, 0]) for t in np.linspace(0, 1, 7)]
    twig(mb, stem, 0.013 * E, 0.004 * E, bark)
    base_tint = {"oak": lin(0.24, 0.35, 0.11), "apple": lin(0.26, 0.42, 0.14), "pear": lin(0.18, 0.33, 0.10),
                 "bush": lin(0.22, 0.36, 0.13), "berry": lin(0.17, 0.30, 0.12), "flower": lin(0.22, 0.38, 0.14)}[kind]
    lkind = {"berry": "bush", "flower": "bush"}.get(kind, kind)
    size = {"oak": 0.20, "apple": 0.17, "pear": 0.18, "bush": 0.13, "berry": 0.13, "flower": 0.13}[kind]
    twigs = [stem]
    for k in range(5):
        t = 0.18 + 0.15 * k
        o = tip * t
        side = 1 if k % 2 else -1
        d = np.array([side * r.uniform(0.55, 0.85), r.uniform(0.5, 0.8), 0])
        d /= np.linalg.norm(d)
        L = E * (0.4 - 0.2 * t) * r.uniform(0.9, 1.1)
        pts = [o + d * L * s + np.array([0, 0.06 * E * s * s, 0]) for s in np.linspace(0, 1, 5)]
        twig(mb, pts, 0.008 * E, 0.003 * E, bark)
        twigs.append(pts)
    for ti, pts in enumerate(twigs):
        a0, a1 = pts[0], pts[-1]
        axis = (a1 - a0) / (np.linalg.norm(a1 - a0) + 1e-9)
        n_leaves = max(4, int(np.linalg.norm(a1 - a0) / (size * E * 0.42)))
        for k in range(n_leaves):
            t = 0.15 + 0.85 * k / (n_leaves - 1)
            p = a0 + (a1 - a0) * t
            sideways = 1 if (k + ti) % 2 else -1
            ang = math.radians(r.uniform(30, 70)) * sideways
            ca, sa = math.cos(ang), math.sin(ang)
            d = np.array([axis[0] * ca - axis[1] * sa, axis[0] * sa + axis[1] * ca])
            L = E * size * (1.1 - 0.3 * t) * r.uniform(0.85, 1.12)
            var = r.uniform(0.8, 1.2)
            tint = tuple(np.clip(np.array(base_tint) * np.array([var * r.uniform(0.9, 1.25), var, var * r.uniform(0.8, 1.1)]), 0, 1))
            if r.random() < 0.05:
                tint = lin(0.55, 0.50, 0.18)                       # the odd yellowing leaf
            leaf(mb, lkind, p + np.array([d[0], d[1], 0]) * 0.01 * E, d, L, tint, roll=r.normal(0, 0.25), droop=r.normal(0, 0.12), curl=r.uniform(0.05, 0.25))
        leaf(mb, lkind, a1, axis[:2], E * size * 0.85, base_tint)
    if kind == "flower" or (kind == "apple" and seed % 2 == 0):
        col = lin(0.98, 0.88, 0.92) if kind == "apple" else [lin(0.95, 0.55, 0.70), lin(0.98, 0.95, 0.92), lin(0.75, 0.55, 0.95)][seed % 3]
        for _ in range(7 if kind == "flower" else 3):
            c = stem[r.integers(2, 7)] + np.array([r.normal(0, 0.15), r.normal(0, 0.08), 0.05]) * E
            blossom(mb, c, 0.06 * E, col, r)
    if kind == "berry":
        for _ in range(5):
            c = stem[r.integers(2, 7)] + np.array([r.normal(0, 0.12), r.normal(0, 0.06), 0.04]) * E
            for b in range(r.integers(3, 7)):
                berry(mb, c + np.array([r.normal(0, 0.025), r.normal(0, 0.025), 0.01]) * E, 0.02 * E, lin(0.62, 0.05, 0.08))
    return mb


def sprig_material():
    mat = bpy.data.materials.new("sprig")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    lc = nt.nodes.new("ShaderNodeAttribute")
    lc.attribute_name = "lc"
    tint = nt.nodes.new("ShaderNodeAttribute")
    tint.attribute_name = "tint"
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(lc.outputs["Vector"], sep.inputs[0])

    def m(op, a, b=0.0):
        n = nt.nodes.new("ShaderNodeMath")
        n.operation = op
        for i, v in enumerate((a, b)):
            if isinstance(v, bpy.types.NodeSocket):
                nt.links.new(v, n.inputs[i])
            else:
                n.inputs[i].default_value = v
        return n.outputs[0]
    ax = sep.outputs[2]                 # |x| across the leaf, 0 at the midrib (2 = twig, 3 = berry)
    is_leaf = m("LESS_THAN", ax, 1.5)
    midrib = m("MULTIPLY", m("SUBTRACT", 1.0, m("MINIMUM", m("DIVIDE", ax, 0.06), 1.0)), is_leaf)
    side = m("FRACT", m("MULTIPLY", m("SUBTRACT", sep.outputs[1], m("MULTIPLY", ax, 0.55)), 9.0))
    sidev = m("MULTIPLY", m("MULTIPLY", m("GREATER_THAN", side, 0.9), m("LESS_THAN", ax, 0.85)), is_leaf)
    edge = m("MULTIPLY", m("GREATER_THAN", ax, 0.88), is_leaf)
    light = m("ADD", 1.0, m("ADD", m("MULTIPLY", midrib, 0.5), m("MULTIPLY", sidev, 0.22)))
    light = m("SUBTRACT", light, m("MULTIPLY", edge, 0.25))
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 60.0
    blot = m("ADD", 0.85, m("MULTIPLY", noise.outputs["Fac"], 0.3))
    col = nt.nodes.new("ShaderNodeVectorMath")
    col.operation = "SCALE"
    nt.links.new(tint.outputs["Color"], col.inputs[0])
    nt.links.new(m("MULTIPLY", light, blot), col.inputs["Scale"])
    diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
    nt.links.new(col.outputs[0], diff.inputs["Color"])
    # normal pass: emission of the facing world normal mapped to 0..1
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    flip = nt.nodes.new("ShaderNodeVectorMath")
    flip.operation = "SCALE"
    nt.links.new(geo.outputs["Normal"], flip.inputs[0])
    nt.links.new(m("SUBTRACT", 1.0, m("MULTIPLY", geo.outputs["Backfacing"], 2.0)), flip.inputs["Scale"])
    half = nt.nodes.new("ShaderNodeVectorMath")
    half.operation = "MULTIPLY_ADD"
    nt.links.new(flip.outputs[0], half.inputs[0])
    half.inputs[1].default_value = (0.5, 0.5, 0.5)
    half.inputs[2].default_value = (0.5, 0.5, 0.5)
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(half.outputs[0], em.inputs["Color"])
    return mat, nt, out, diff, em


def render_sprigs(name, kinds, extent):
    """Render 4 sprigs into a 2x2 atlas (colour+alpha, normal). Returns file paths."""
    reset()
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.use_denoising = False
    sc.render.film_transparent = True
    sc.render.resolution_x = sc.render.resolution_y = 512
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.view_settings.view_transform = "Standard"
    sc.render.filter_size = 1.0
    w = bpy.data.worlds.new("w")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = extent
    cam.location = (0, extent / 2, 5)
    link(cam)
    sc.camera = cam
    mat, nt, out, diff, em = sprig_material()
    tiles_c, tiles_n = [], []
    for i, (kind, seed) in enumerate(kinds):
        for ob in list(sc.objects):
            if ob.type == "MESH":
                bpy.data.objects.remove(ob, do_unlink=True)
        mb = build_sprig(kind, seed, extent * 0.98)
        ob = mb.obj("sprig", mat)
        for passname, node, samples in (("c", diff, 48), ("n", em, 8)):
            for l in list(out.inputs["Surface"].links):
                nt.links.remove(l)
            nt.links.new(node.outputs[0], out.inputs["Surface"])
            sc.cycles.samples = samples
            raw_ok = True
            try:
                sc.view_settings.view_transform = "Standard" if passname == "c" else "Raw"
            except TypeError:
                raw_ok = passname == "c"
            path = TEX / f"_{name}_{i}_{passname}.png"
            sc.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            im = bpy.data.images.load(str(path))
            px = np.empty(512 * 512 * 4, dtype=np.float32)
            im.pixels.foreach_get(px)
            if not raw_ok:
                px = px.reshape(-1, 4)
                px[:, :3] = px[:, :3] ** 2.2
                px = px.ravel()
            (tiles_c if passname == "c" else tiles_n).append(px.reshape(512, 512, 4))
            bpy.data.images.remove(im)
    atlas_c = np.zeros((1024, 1024, 4), dtype=np.float32)
    atlas_n = np.zeros((1024, 1024, 4), dtype=np.float32)
    for i in range(4):
        y0, x0 = (i // 2) * 512, (i % 2) * 512
        atlas_c[y0:y0 + 512, x0:x0 + 512] = tiles_c[i]
        atlas_n[y0:y0 + 512, x0:x0 + 512] = tiles_n[i]
    # bleed colours into the transparent margin so mip-maps don't fringe dark
    a = atlas_c[..., 3].copy()
    rgb = atlas_c[..., :3] * (a[..., None] > 0.02)
    nrm = atlas_n[..., :3].copy()
    filled = a > 0.02
    for _ in range(24):
        acc = np.zeros_like(rgb)
        accn = np.zeros_like(nrm)
        cnt = np.zeros(a.shape, dtype=np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.roll(filled, (dy, dx), (0, 1))
            acc += np.roll(rgb, (dy, dx), (0, 1)) * sh[..., None]
            accn += np.roll(nrm, (dy, dx), (0, 1)) * sh[..., None]
            cnt += sh
        grow = (~filled) & (cnt > 0)
        rgb[grow] = acc[grow] / cnt[grow][:, None]
        nrm[grow] = accn[grow] / cnt[grow][:, None]
        filled = filled | grow
    rgb[~filled] = np.array([0.1, 0.15, 0.05])
    nrm[~filled] = np.array([0.5, 0.5, 1.0])
    atlas_c[..., :3] = rgb
    atlas_n[..., :3] = nrm
    atlas_n[..., 3] = 1
    paths = {}
    for key, arr, fmt in (("c", atlas_c, "PNG"), ("n", atlas_n, "JPEG")):
        im = bpy.data.images.new(f"{name}_{key}", 1024, 1024, alpha=(key == "c"))
        im.colorspace_settings.name = "sRGB" if key == "c" else "Non-Color"
        im.pixels.foreach_set(arr.ravel())
        ext = "png" if fmt == "PNG" else "jpg"
        path = TEX / f"leaf_{name}_{key}.{ext}"
        im.filepath_raw = str(path)
        im.file_format = fmt
        im.save()
        paths[key] = path
    print(f"[sprigs] {name}")
    return paths


# ============================================================================ 2. bark
def bark_texture(name, kind, base, dark, moss=0.3, W=256, H=512):
    """Tileable bark: u wraps around the trunk (0.5 m), v runs up it (1 m)."""
    u = (np.arange(W) + 0.5) / W
    v = (np.arange(H) + 0.5) / H
    U, V = np.meshgrid(u, v)
    R = 0.5 / (2 * math.pi) * 6.0

    def field(Vv):
        p = np.stack([np.cos(2 * math.pi * U) * R, np.sin(2 * math.pi * U) * R, Vv * 6.0], axis=-1).reshape(-1, 3)
        if kind == "oak":            # deep vertical fissures between corky ridges
            q = p * np.array([1.0, 1.0, 0.25])
            h = 1 - np.abs(N.fbm(q * 1.6 + N.fbm(p * 0.8, 2, 3)[:, None] * 0.6, 5, 1))
            h = h ** 2.2 + 0.15 * N.fbm(p * 5, 3, 2)
        elif kind == "pine":         # flaky plates
            f1, f2, cid = N.worley(p * np.array([1.4, 1.4, 0.8]), 5)
            h = N.smoothstep(0.0, 0.12, f2 - f1) * (0.6 + 0.4 * cid) + 0.1 * N.fbm(p * 6, 3, 4)
        else:                        # apple / pear: scaly, shallow cracks
            f1, f2, cid = N.worley(p * np.array([2.2, 2.2, 1.6]), 7)
            h = N.smoothstep(0.0, 0.08, f2 - f1) * (0.7 + 0.3 * cid) + 0.15 * N.fbm(p * 4, 4, 8)
        return h.reshape(U.shape)
    h = field(V) * (1 - V) + field(V - 1.0) * V
    h = (h - h.min()) / (h.max() - h.min())
    p = np.stack([np.cos(2 * math.pi * U) * R, np.sin(2 * math.pi * U) * R, V * 6.0], axis=-1).reshape(-1, 3)
    var = N.fbm(p * 1.2, 3, 21).reshape(U.shape) * 0.5 + 0.5
    col = np.array(dark)[None, None] + (np.array(base) - np.array(dark))[None, None] * (h[..., None] ** 0.8)
    col = col * (0.85 + 0.3 * var[..., None])
    mossm = N.smoothstep(0.55, 0.75, N.fbm(p * 0.9, 4, 23).reshape(U.shape) * 0.5 + 0.5) * moss * (1 - h * 0.5)
    col = col * (1 - mossm[..., None]) + np.array(lin(0.20, 0.30, 0.10))[None, None] * mossm[..., None]
    lich = N.smoothstep(0.72, 0.8, N.fbm(p * 2.5, 3, 29).reshape(U.shape) * 0.5 + 0.5) * 0.7
    col = col * (1 - lich[..., None]) + np.array(lin(0.55, 0.58, 0.48))[None, None] * lich[..., None]
    gy, gx = np.gradient(h)
    k = 6.0
    nrm = np.stack([-gx * k * W / 64, -gy * k * H / 128, np.ones_like(h)], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    paths = {}
    for key, arr, cs in (("c", col ** (1 / 2.2), "sRGB"), ("n", nrm * 0.5 + 0.5, "Non-Color")):
        rgba = np.concatenate([np.clip(arr, 0, 1), np.ones((H, W, 1))], axis=-1).astype(np.float32)
        im = bpy.data.images.new(f"bark_{name}_{key}", W, H)
        im.colorspace_settings.name = cs
        im.pixels.foreach_set(rgba.ravel())
        path = TEX / f"bark_{name}_{key}.jpg"
        im.filepath_raw = str(path)
        im.file_format = "JPEG"
        im.save()
        paths[key] = path
    print(f"[bark] {name}")
    return paths


# ============================================================================ 3. trees
class Tree:
    def __init__(self, spec, seed):
        self.s = spec
        self.r = np.random.default_rng(seed)
        self.bark = {"v": [], "f": [], "uv": []}
        self.cards = []          # (base, outward direction, size)
        self.tips = []

    def tube(self, pts, radii, flare=None):
        r = self.r
        sides = int(np.clip(round(radii[0] * 90), 5, 16))
        pts = [np.asarray(p, float) for p in pts]
        L = 0.0
        base = len(self.bark["v"])
        nref = None
        rep = max(1, round(2 * math.pi * radii[0] / 0.5))
        for i, (p, rad) in enumerate(zip(pts, radii)):
            if i:
                L += np.linalg.norm(p - pts[i - 1])
            a, b = pts[max(i - 1, 0)], pts[min(i + 1, len(pts) - 1)]
            t = (b - a) / (np.linalg.norm(b - a) + 1e-9)
            if nref is None:
                nref = np.cross(t, [0.3, 0.1, 1.0])
            n = nref - t * np.dot(nref, t)
            n /= np.linalg.norm(n) + 1e-9
            nref = n
            bn = np.cross(t, n)
            for k in range(sides + 1):
                a_ = k / sides * 2 * math.pi
                m = 1.0
                if flare is not None and i < len(flare):
                    m = flare[i][k % sides]
                self.bark["v"].append(p + (n * math.cos(a_) + bn * math.sin(a_)) * rad * m)
                self.bark["uv"].append((k / sides * rep, L / 1.0))
        rows = len(pts)
        for i in range(rows - 1):
            for k in range(sides):
                q = base + i * (sides + 1) + k
                self.bark["f"].append((q, q + 1, q + sides + 2, q + sides + 1))

    def branch(self, p, d, L, rad, depth):
        s, r = self.s, self.r
        n = max(3, int(L / 0.35) + 2)
        pts, radii = [np.array(p, float)], [rad]
        dd = np.array(d, float)
        for i in range(1, n + 1):
            t = i / n
            dd = dd + r.normal(0, s["wiggle"], 3) + np.array([0, 0, s["photo"]]) - np.array([0, 0, s["droop"] * depth * t])
            dd /= np.linalg.norm(dd)
            pts.append(pts[-1] + dd * L / n)
            radii.append(max(rad * (1 - s["taper"] * t), 0.006))
        flare = None
        if depth == 0:
            flare = []
            for i in range(len(pts)):
                k = max(0.0, 1 - i / 2.5)
                prof = [1 + k * (1.1 + 0.6 * math.sin(j * 2 * math.pi / 5 + 0.7) ** 2) for j in range(17)]
                flare.append(prof)
            pts[0] = pts[0] - np.array([0, 0, 0.3])
        self.tube(pts, radii, flare)
        if depth >= s["depth"]:
            for q in pts[1:]:
                self.tips.append(q)
            return
        kids = s["kids"][depth]
        start = s["start"][depth]
        golden = r.uniform(0, 2 * math.pi)
        for k in range(kids):
            t = start + (1 - start) * (k + r.uniform(0.0, 0.8)) / kids
            i = min(int(t * n), n - 1)
            q = pts[i] + (pts[i + 1] - pts[i]) * (t * n - i)
            parent_d = (pts[i + 1] - pts[i]) / (np.linalg.norm(pts[i + 1] - pts[i]) + 1e-9)
            golden += 2.4
            ang = math.radians(r.uniform(*s["angle"][depth]))
            ax = np.cross(parent_d, [math.cos(golden), math.sin(golden), 0.3])
            ax /= np.linalg.norm(ax) + 1e-9
            nd = np.array(Matrix.Rotation(ang, 3, Vector(ax)) @ Vector(parent_d))
            nd = nd + np.array([0, 0, s["up"][depth]])
            nd /= np.linalg.norm(nd)
            cl = L * s["ratio"][depth] * (1.15 - 0.45 * t if depth else 1.0) * r.uniform(0.85, 1.12)
            self.branch(q, nd, cl, radii[i] * s["rratio"], depth + 1)
        if s.get("leader") and depth < s["depth"]:
            self.branch(pts[-1], (pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2]), L * 0.55, radii[-1], depth + 1)

    def leaves(self):
        s, r = self.s, self.r
        tips = np.array(self.tips)
        centre = tips.mean(axis=0)
        cards = []
        order = np.concatenate([r.permutation(len(tips)) for _ in range(1 + s["cards"] // max(len(tips), 1))])[: s["cards"]]
        for n_, i in enumerate(order):
            p = tips[i] + (r.normal(0, 0.18, 3) if n_ >= len(tips) else 0)
            out = p - centre
            out = out / (np.linalg.norm(out) + 1e-9)
            d = out + r.normal(0, 0.45, 3) + np.array([0, 0, 0.3])
            d /= np.linalg.norm(d)
            cards.append((p - d * 0.15 * s["card"], d, s["card"] * r.uniform(0.8, 1.2)))
        self.cards = cards
        self.centre = centre


def build_species(spec, seed):
    T = Tree(spec, seed)
    T.branch(np.zeros(3), np.array([0, 0, 1.0]) + rng.normal(0, 0.04, 3), spec["trunk"], spec["r"], 0)
    T.leaves()
    return T


def pine_tree(spec, seed):
    r = np.random.default_rng(seed)
    T = Tree(spec, seed)
    H = spec["height"]
    pts = [np.array([r.normal(0, 0.03), r.normal(0, 0.03), z]) for z in np.linspace(0, H, 14)]
    radii = [spec["r"] * (1 - 0.92 * (z / H)) + 0.01 for z in np.linspace(0, H, 14)]
    flare = [[1 + max(0, 1 - i / 2) * (0.6 + 0.4 * math.sin(j * 1.9) ** 2) for j in range(17)] for i in range(14)]
    pts[0] = pts[0] - np.array([0, 0, 0.3])
    T.tube(pts, radii, flare)
    z = H * 0.22
    whorl = 0
    cards = []
    while z < H - 0.3:
        frac = (z - H * 0.22) / (H * 0.78)
        L = spec["spread"] * (1 - frac) ** 0.9 + 0.35
        nb = r.integers(5, 8)
        a0 = r.uniform(0, 6.28)
        for k in range(nb):
            a = a0 + k / nb * 2 * math.pi + r.normal(0, 0.15)
            d = np.array([math.cos(a), math.sin(a), 0.25 - 0.55 * (1 - frac)])
            d /= np.linalg.norm(d)
            seg = 5
            bp = [np.array([0, 0, z])]
            for i in range(1, seg + 1):
                t = i / seg
                bp.append(bp[-1] + (d + np.array([0, 0, -0.35 * t * (1 - frac) + 0.15 * t * t])) * L / seg)
            T.tube(bp, [0.05 * (1 - frac) + 0.02, 0.03, 0.02, 0.012, 0.008, 0.005], None)
            for i in range(1, seg + 1):
                for _ in range(2 if i > 1 else 1):
                    side = np.cross(d, [0, 0, 1.0])
                    side /= np.linalg.norm(side) + 1e-9
                    dd = d * 0.6 + side * r.choice([-1, 1]) * 0.7 + np.array([0, 0, r.normal(0.05, 0.15)])
                    dd /= np.linalg.norm(dd)
                    cards.append((bp[i] - dd * 0.05, dd, spec["card"] * r.uniform(0.8, 1.15) * (1.1 - 0.3 * frac)))
            cards.append((bp[-1] - d * 0.1, d, spec["card"] * 1.1))
        z += r.uniform(0.38, 0.55)
        whorl += 1
    for k in range(4):                       # leader tip
        dd = np.array([r.normal(0, 0.3), r.normal(0, 0.3), 1.0])
        dd /= np.linalg.norm(dd)
        cards.append((np.array([0, 0, H - 0.5]), dd, spec["card"] * 0.9))
    T.cards = cards
    T.centre = np.array([0, 0, H * 0.5])
    return T


def card_geometry(T, cols=2, rows=2):
    """Two crossed quads per card, UVs in one of the 4 atlas cells, normals bent away from the crown centre."""
    v, f, uv, nn = [], [], [], []
    r = T.r
    for base, d, size in T.cards:
        cell = r.integers(4)
        cu, cv = (cell % 2) * 0.5, (cell // 2) * 0.5
        side = np.cross(d, r.normal(0, 1, 3))
        side /= np.linalg.norm(side) + 1e-9
        for twist in (0.0, math.pi / 2 * r.uniform(0.6, 1.0)):
            sd = np.array(Matrix.Rotation(twist, 3, Vector(d)) @ Vector(side))
            fn = np.cross(sd, d)
            if fn[2] < 0:
                fn = -fn
            k0 = len(v)
            for (x, y) in ((-0.5, 0), (0.5, 0), (0.5, 1), (-0.5, 1)):
                p = base + sd * x * size + d * y * size
                v.append(p)
                uv.append((cu + (x + 0.5) * 0.5, cv + y * 0.5))
                out = p - T.centre
                out /= np.linalg.norm(out) + 1e-9
                bent = out * 0.75 + fn * 0.25 + np.array([0, 0, 0.25])
                nn.append(bent / np.linalg.norm(bent))
            f.append((k0, k0 + 1, k0 + 2, k0 + 3))
    return v, f, uv, nn


def textured_material(name, tex, alpha=False, rough=0.8):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    c = nt.nodes.new("ShaderNodeTexImage")
    c.image = bpy.data.images.load(str(tex["c"]))
    nt.links.new(c.outputs["Color"], bsdf.inputs["Base Color"])
    if alpha:
        nt.links.new(c.outputs["Alpha"], bsdf.inputs["Alpha"])
    n = nt.nodes.new("ShaderNodeTexImage")
    n.image = bpy.data.images.load(str(tex["n"]))
    n.image.colorspace_settings.name = "Non-Color"
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(n.outputs["Color"], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = rough
    return mat


SPECIES = {
    "tree_oak": dict(kind="oak", trunk=3.6, r=0.5, depth=4, kids=[5, 4, 3, 3], start=[0.8, 0.3, 0.25, 0.3], angle=[(35, 55), (35, 60), (35, 60), (30, 55)],
                     ratio=[1.05, 0.62, 0.55, 0.55], rratio=0.6, up=[0.2, 0.12, 0.1, 0.1], wiggle=0.12, photo=0.03, droop=0.05, taper=0.45,
                     cards=1500, card=1.45, bark="oak", leaf="oak", seed=1),
    "tree_oak_b": dict(kind="oak", trunk=3.0, r=0.42, depth=4, kids=[4, 4, 3, 3], start=[0.75, 0.3, 0.25, 0.3], angle=[(28, 48), (35, 60), (35, 60), (30, 55)],
                       ratio=[1.05, 0.62, 0.55, 0.55], rratio=0.6, up=[0.35, 0.15, 0.1, 0.1], wiggle=0.14, photo=0.05, droop=0.04, taper=0.45,
                       cards=1200, card=1.35, bark="oak", leaf="oak", seed=2),
    "tree_apple": dict(kind="apple", trunk=1.2, r=0.18, depth=3, kids=[4, 4, 3], start=[0.85, 0.3, 0.25], angle=[(40, 60), (35, 60), (30, 55)],
                       ratio=[1.9, 0.6, 0.55], rratio=0.6, up=[0.12, 0.1, 0.05], wiggle=0.2, photo=0.03, droop=0.06, taper=0.5,
                       cards=650, card=0.95, bark="fruit", leaf="apple", seed=3),
    "tree_pear": dict(kind="pear", trunk=2.0, r=0.22, depth=3, kids=[5, 4, 3], start=[0.7, 0.3, 0.25], angle=[(22, 38), (30, 50), (30, 55)],
                      ratio=[1.5, 0.6, 0.55], rratio=0.6, up=[0.45, 0.2, 0.1], wiggle=0.14, photo=0.06, droop=0.03, taper=0.5,
                      cards=750, card=0.95, bark="fruit", leaf="pear", seed=4, leader=True),
    "bush_round": dict(kind="bush", trunk=0.25, r=0.05, depth=2, kids=[8, 4], start=[0.2, 0.3], angle=[(30, 70), (30, 60)],
                       ratio=[3.4, 0.5], rratio=0.6, up=[0.0, 0.0], wiggle=0.25, photo=0.05, droop=0.1, taper=0.5,
                       cards=240, card=0.7, bark="fruit", leaf="bush", seed=5),
    "bush_berry": dict(kind="berry", trunk=0.2, r=0.045, depth=2, kids=[7, 4], start=[0.2, 0.3], angle=[(35, 70), (30, 60)],
                       ratio=[3.4, 0.5], rratio=0.6, up=[0.0, 0.0], wiggle=0.25, photo=0.04, droop=0.12, taper=0.5,
                       cards=220, card=0.7, bark="fruit", leaf="berry", seed=6),
    "bush_flower": dict(kind="flower", trunk=0.2, r=0.04, depth=2, kids=[7, 4], start=[0.2, 0.3], angle=[(30, 65), (30, 60)],
                        ratio=[3.0, 0.5], rratio=0.6, up=[0.05, 0.0], wiggle=0.25, photo=0.05, droop=0.1, taper=0.5,
                        cards=200, card=0.6, bark="fruit", leaf="flower", seed=7),
}
PINES = {"tree_pine": dict(height=15.0, r=0.38, spread=3.4, card=1.35, seed=8), "tree_pine_b": dict(height=11.0, r=0.28, spread=2.8, card=1.25, seed=9)}


def main():
    sprig_sets = {
        "oak": [("oak", 1), ("oak", 2), ("oak", 3), ("oak", 4)],
        "apple": [("apple", 1), ("apple", 2), ("apple", 3), ("apple", 4)],
        "pear": [("pear", 1), ("pear", 2), ("pear", 3), ("pear", 4)],
        "bush": [("bush", 1), ("bush", 2), ("bush", 3), ("bush", 4)],
        "berry": [("berry", 1), ("berry", 2), ("berry", 3), ("berry", 4)],
        "flower": [("flower", 1), ("flower", 2), ("flower", 3), ("flower", 4)],
        "pine": [("pine", 1), ("pine", 2), ("pine", 3), ("pine", 4)],
        "reed": [("reed", 1), ("reed", 2), ("reed", 3), ("reed", 4)],
    }
    leaf_tex = {k: render_sprigs(k, v, 1.0) for k, v in sprig_sets.items()}
    reset()
    bark_tex = {"oak": bark_texture("oak", "oak", lin(0.42, 0.38, 0.33), lin(0.12, 0.10, 0.08), moss=0.6),
                "pine": bark_texture("pine", "pine", lin(0.50, 0.32, 0.22), lin(0.16, 0.10, 0.07), moss=0.2),
                "fruit": bark_texture("fruit", "fruit", lin(0.40, 0.36, 0.32), lin(0.17, 0.14, 0.12), moss=0.35)}
    bark_mats = {k: textured_material(f"bark_{k}", t, rough=0.92) for k, t in bark_tex.items()}
    leaf_mats = {k: textured_material(f"leaf_{k}", t, alpha=True, rough=0.65) for k, t in leaf_tex.items()}
    roots = []
    fruit_spots = {}
    for name, spec in list(SPECIES.items()) + list(PINES.items()):
        if name in PINES:
            T = pine_tree(spec, spec["seed"])
            bark_mat, leaf_mat = bark_mats["pine"], leaf_mats["pine"]
        else:
            T = build_species(spec, spec["seed"])
            bark_mat, leaf_mat = bark_mats[spec["bark"]], leaf_mats[spec["leaf"]]
        root = link(bpy.data.objects.new(name, None))
        b = T.bark
        bo = mesh_from(name + "_bark", b["v"], b["f"], uvs=b["uv"], mat=bark_mat)
        cv, cf, cuv, cn = card_geometry(T)
        lo = mesh_from(name + "_leaves", cv, cf, uvs=cuv, normals=cn, mat=leaf_mat)
        bo.parent = root
        lo.parent = root
        roots.append(root)
        tris = len(b["f"]) * 2 + len(cf) * 2
        print(f"[tree] {name}: {tris} tris, {len(T.cards)} cards")
        if name in ("tree_apple", "tree_pear"):
            tips = np.array([c[0] for c in T.cards])
            out = tips - T.centre
            order = np.argsort(-np.linalg.norm(out * [1, 1, 0.6], axis=1))
            spots = []
            for i in order:
                p = tips[i] + np.array([0, 0, -0.12])
                if p[2] < 1.3 or any(np.linalg.norm(p - q) < 0.45 for q in spots):
                    continue
                spots.append(p)
                if len(spots) == 16:
                    break
            fruit_spots[name] = [[round(float(p[0]), 3), round(float(p[2]), 3), round(float(-p[1]), 3)] for p in spots]
    # reeds: a few crossed cards
    root = link(bpy.data.objects.new("reeds", None))
    v, f, uv, nn = [], [], [], []
    for k in range(5):
        a = k / 5 * math.pi + rng.normal(0, 0.2)
        off = rng.normal(0, 0.25, 2)
        d = np.array([math.cos(a), math.sin(a), 0])
        h = rng.uniform(1.4, 1.9)
        cell = k % 4
        cu, cv_ = (cell % 2) * 0.5, (cell // 2) * 0.5
        k0 = len(v)
        for (x, y) in ((-0.5, 0), (0.5, 0), (0.5, 1), (-0.5, 1)):
            v.append(np.array([off[0], off[1], 0]) + d * x * h + np.array([0, 0, y * h]))
            uv.append((cu + (x + 0.5) * 0.5, cv_ + y * 0.5))
            nn.append((0, 0, 1))
        f.append((k0, k0 + 1, k0 + 2, k0 + 3))
    mesh_from("reeds_leaves", v, f, uvs=uv, normals=nn, mat=leaf_mats["reed"]).parent = root
    roots.append(root)
    bpy.ops.object.select_all(action="DESELECT")
    for r_ in roots:
        r_.select_set(True)
        for c in r_.children:
            c.select_set(True)
    out = OUT / "nature.glb"
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB", use_selection=True, export_yup=True, export_apply=False,
                              export_animations=False, export_image_format="AUTO", export_normals=True, export_texcoords=True,
                              export_materials="EXPORT", export_extras=False)
    # alpha-tested, double-sided foliage
    from glb import Glb
    g = Glb(str(out))
    for m in g.json.get("materials", []):
        if m["name"].startswith("leaf_"):
            m["alphaMode"] = "MASK"
            m["alphaCutoff"] = 0.45
            m["doubleSided"] = True
    g.save(str(out))
    (OUT / "fruit_spots.json").write_text(json.dumps(fruit_spots))
    print("[export]", out)


if __name__ == "__main__":
    main()
