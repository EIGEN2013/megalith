"""High-poly -> low-poly PBR pipeline for Blender (run inside blender --background).

Each model is made of pieces. A piece is a dense, sculpted high-poly mesh (`hi`) with a procedural
material, and a light game mesh (`lo`) that follows it. All pieces of a model are baked together
into one texture atlas:

  colour (with ambient occlusion folded in), tangent-space normal map, ORM (R occlusion, G roughness,
  B metalness) and emission,

then the low meshes are split back into the model's parts (one node per part, origin at its pivot,
parented like a skeleton) and written as a GLB with JPEG textures.

Coordinates are Blender's: +Z up, the model faces -Y (glTF +Z after export), +X is the model's left.
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
import noise3  # noqa: E402

ROOT = HERE.parent


# ----------------------------------------------------------------------------- scene helpers
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 1
    w = bpy.data.worlds.new("w")
    sc.world = w
    w.light_settings.distance = 0.35


def link(ob):
    bpy.context.scene.collection.objects.link(ob)
    return ob


def mesh_obj(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(int(i) for i in f) for f in faces])
    me.update()
    return link(bpy.data.objects.new(name, me))


def from_bmesh(name, bm):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return link(bpy.data.objects.new(name, me))


def apply_mods(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    ob.modifiers.clear()
    old = ob.data
    ob.data = me
    if old.users == 0:
        bpy.data.meshes.remove(old)
    return ob


def copy_obj(ob, name):
    c = ob.copy()
    c.data = ob.data.copy()
    c.name = name
    return link(c)


def delete(obs):
    for ob in obs:
        me = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)


def verts_np(ob):
    me = ob.data
    v = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", v)
    return v.reshape(-1, 3)


def set_verts(ob, v):
    ob.data.vertices.foreach_set("co", np.asarray(v, dtype=np.float64).ravel())
    ob.data.update()


def vnormals(ob):
    me = ob.data
    n = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("normal", n)
    return n.reshape(-1, 3)


def set_point_colour(ob, name, rgba):
    me = ob.data
    if name in me.attributes:
        me.attributes.remove(me.attributes[name])
    at = me.attributes.new(name, "FLOAT_COLOR", "POINT")
    at.data.foreach_set("color", np.asarray(rgba, dtype=np.float32).ravel())


def remesh(ob, voxel):
    m = ob.modifiers.new("rm", "REMESH")
    m.mode = "VOXEL"
    m.voxel_size = voxel
    m.adaptivity = 0.0
    m.use_smooth_shade = True
    return apply_mods(ob)


def boolean(ob, cutters, op="DIFFERENCE", solver="EXACT"):
    if not cutters:
        return ob
    col = bpy.data.collections.new("cut")
    bpy.context.scene.collection.children.link(col)
    for c in cutters:
        for uc in list(c.users_collection):
            uc.objects.unlink(c)
        col.objects.link(c)
    m = ob.modifiers.new("b", "BOOLEAN")
    m.operation = op
    m.operand_type = "COLLECTION"
    m.collection = col
    m.solver = solver
    apply_mods(ob)
    delete(list(col.objects))
    bpy.data.collections.remove(col)
    return ob


def decimate(ob, faces_target):
    n = sum(len(p.vertices) - 2 for p in ob.data.polygons)      # triangles
    if n > faces_target:
        m = ob.modifiers.new("d", "DECIMATE")
        m.decimate_type = "COLLAPSE"
        m.ratio = max(0.002, faces_target / n)
        m.use_collapse_triangulate = True
        apply_mods(ob)
    return ob


def smooth(ob, angle=50):
    for p in ob.data.polygons:
        p.use_smooth = True
    try:
        ob.data.set_sharp_from_angle(angle=math.radians(angle))
    except Exception:
        pass


def material(ob, mat):
    ob.data.materials.clear()
    ob.data.materials.append(mat)


# ----------------------------------------------------------------------------- shapes
def box_bm(size, bevel=0.0, segments=2, taper=(1.0, 1.0), jitter=0.0, rng=None):
    """A box centred on the origin; the top face can be tapered and the corners jittered."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    sx, sy, sz = size
    for v in bm.verts:
        x, y, z = v.co
        k = taper[0] if z > 0 else 1.0
        kk = taper[1] if z > 0 else 1.0
        v.co = Vector((x * sx * k, y * sy * kk, z * sz))
        if jitter and rng is not None:
            v.co += Vector(rng.normal(0, jitter, 3)) * min(size)
    if bevel:
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bevel, segments=segments, profile=0.5, affect="EDGES", clamp_overlap=True)
    return bm


def sphere_bm(radii=(1, 1, 1), u=24, v=16):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=1.0)
    bmesh.ops.scale(bm, vec=Vector(radii), verts=list(bm.verts))
    return bm


def _frames(points):
    pts = [np.asarray(p, float) for p in points]
    tans = []
    for i in range(len(pts)):
        a, b = pts[max(i - 1, 0)], pts[min(i + 1, len(pts) - 1)]
        t = b - a
        tans.append(t / (np.linalg.norm(t) + 1e-9))
    ref = np.array([0, 0, 1.0]) if abs(tans[0][2]) < 0.9 else np.array([1.0, 0, 0])
    n = np.cross(tans[0], ref)
    n /= np.linalg.norm(n)
    frames = []
    for t in tans:
        n = n - t * np.dot(n, t)
        n /= np.linalg.norm(n) + 1e-9
        frames.append((t, n, np.cross(t, n)))
    return pts, frames


def tube_bm(points, radii, sides=12, cap=True, profile=None):
    """A tube along points with per-point radius; optional (sides,) radial profile multipliers."""
    pts, frames = _frames(points)
    if np.isscalar(radii):
        radii = [radii] * len(pts)
    bm = bmesh.new()
    rings = []
    for p, (t, n, b), r in zip(pts, frames, radii):
        ring = []
        for k in range(sides):
            a = k / sides * 2 * math.pi
            m = profile[k] if profile is not None else 1.0
            ring.append(bm.verts.new(tuple(p + (n * math.cos(a) + b * math.sin(a)) * r * m)))
        rings.append(ring)
    for i in range(len(rings) - 1):
        for k in range(sides):
            bm.faces.new((rings[i][k], rings[i][(k + 1) % sides], rings[i + 1][(k + 1) % sides], rings[i + 1][k]))
    if cap:
        bm.faces.new(rings[0][::-1])
        tipr = radii[-1]
        if tipr < 1e-4:
            pass
        bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    return bm


def cone_path(points, r0, r1=0.0, sides=10, curve=1.0):
    """A tapering spike along points ending in a point."""
    n = len(points)
    radii = [max(r0 * (1 - i / (n - 1)) ** curve + r1 * (i / (n - 1)), 1e-4) for i in range(n)]
    bm = tube_bm(points[:-1], radii[:-1], sides, cap=False)
    tip = bm.verts.new(tuple(np.asarray(points[-1], float)))
    bm.verts.ensure_lookup_table()
    last = list(bm.verts)[-1 - sides:-1]
    first = list(bm.verts)[:sides]
    for k in range(sides):
        bm.faces.new((last[k], last[(k + 1) % sides], tip))
    bm.faces.new(first[::-1])
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    return bm


def join_bms(name, bms):
    obs = [from_bmesh(name, b) for b in bms]
    return join(obs, name)


def transform_bm(bm, mat):
    bmesh.ops.transform(bm, matrix=mat, verts=list(bm.verts))
    return bm


def euler(rx=0, ry=0, rz=0):
    from mathutils import Euler
    return Euler((math.radians(rx), math.radians(ry), math.radians(rz)), "XYZ").to_matrix().to_4x4()


def place(centre, rot=None):
    m = Matrix.Translation(Vector(centre))
    return m @ rot if rot is not None else m


def look(direction):
    """Rotation taking +Z to `direction`."""
    d = Vector(direction).normalized()
    return d.to_track_quat("Z", "Y").to_matrix().to_4x4()


# ----------------------------------------------------------------------------- shader node building
class Nodes:
    """Tiny helper for writing procedural materials whose channels are baked one at a time."""

    def __init__(self, mat):
        mat.use_nodes = True
        self.nt = mat.node_tree
        self.nt.nodes.clear()
        self.out = self.nt.nodes.new("ShaderNodeOutputMaterial")
        self.emit = self.nt.nodes.new("ShaderNodeEmission")
        self.diff = self.nt.nodes.new("ShaderNodeBsdfDiffuse")      # carries bump for the normal bake
        self.nt.links.new(self.emit.outputs[0], self.out.inputs["Surface"])
        self.channels = {}

    def image(self, path, vec, scale, colourspace="sRGB", blend=0.3):
        """A box-projected (triplanar) image, `scale` repeats per object-space metre."""
        n = self.node("ShaderNodeTexImage")
        n.image = bpy.data.images.load(str(path), check_existing=True)
        n.image.colorspace_settings.name = colourspace
        n.projection = "BOX"
        n.projection_blend = blend
        sc = self.vmath("SCALE", vec, (0, 0, 0), scale)
        self.nt.links.new(sc, n.inputs["Vector"])
        return n.outputs["Color"]

    def bump(self, height, strength=0.5, distance=0.02):
        b = self.node("ShaderNodeBump")
        self.put(b, "Height", height)
        self.put(b, "Strength", strength)
        self.put(b, "Distance", distance)
        self.nt.links.new(b.outputs["Normal"], self.diff.inputs["Normal"])

    def _sock(self, coll, key):
        if isinstance(key, int):
            return coll[key]
        for s in coll:
            if s.identifier == key and s.enabled:
                return s
        for s in coll:
            if s.name == key and s.enabled:
                return s
        return coll[key]

    def put(self, node, key, val):
        s = self._sock(node.inputs, key)
        if isinstance(val, bpy.types.NodeSocket):
            self.nt.links.new(val, s)
        elif isinstance(val, (tuple, list)):
            s.default_value = tuple(val) + ((1.0,) if len(val) == 3 and len(s.default_value) == 4 else ())
        else:
            s.default_value = val

    def node(self, kind, **props):
        n = self.nt.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def coords(self, kind="Object"):
        return self.node("ShaderNodeTexCoord").outputs[kind]

    def geometry(self, out):
        return self.node("ShaderNodeNewGeometry").outputs[out]

    def attr(self, name, out="Color"):
        n = self.node("ShaderNodeAttribute")
        n.attribute_name = name
        return n.outputs[out]

    def sep(self, col):
        n = self.node("ShaderNodeSeparateColor")
        self.put(n, 0, col)
        return n.outputs

    def noise(self, vec, scale, detail=6.0, rough=0.55, distortion=0.0, out="Fac", w=None):
        n = self.node("ShaderNodeTexNoise")
        if w is not None:
            n.noise_dimensions = "4D"
            self.put(n, "W", w)
        self.put(n, "Vector", vec)
        self.put(n, "Scale", scale)
        self.put(n, "Detail", detail)
        self.put(n, "Roughness", rough)
        self.put(n, "Distortion", distortion)
        return n.outputs[out]

    def voronoi(self, vec, scale, feature="F1", out="Distance", rand=1.0):
        n = self.node("ShaderNodeTexVoronoi")
        n.feature = feature
        self.put(n, "Vector", vec)
        self.put(n, "Scale", scale)
        self.put(n, "Randomness", rand)
        return n.outputs[out]

    def math(self, op, a, b=0.0, clamp=False):
        n = self.node("ShaderNodeMath", operation=op)
        n.use_clamp = clamp
        self.put(n, 0, a)
        self.put(n, 1, b)
        return n.outputs[0]

    def vmath(self, op, a, b=(0, 0, 0), scale=None):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self.put(n, 0, a)
        self.put(n, 1, b)
        if scale is not None:
            self.put(n, "Scale", scale)
        return n.outputs[0] if op not in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else n.outputs[1]

    def ramp(self, v, a, b, c=0.0, d=1.0, clamp=True, smooth=True):
        n = self.node("ShaderNodeMapRange")
        n.clamp = clamp
        if smooth:
            n.interpolation_type = "SMOOTHSTEP"
        self.put(n, "Value", v)
        self.put(n, "From Min", a)
        self.put(n, "From Max", b)
        self.put(n, "To Min", c)
        self.put(n, "To Max", d)
        return n.outputs["Result"]

    def mix(self, fac, a, b, blend="MIX"):
        n = self.node("ShaderNodeMix")
        n.data_type = "RGBA"
        n.blend_type = blend
        self.put(n, "Factor_Float", fac)
        self.put(n, "A_Color", a)
        self.put(n, "B_Color", b)
        return self._sock(n.outputs, "Result_Color")

    def mixf(self, fac, a, b):
        n = self.node("ShaderNodeMix")
        n.data_type = "FLOAT"
        self.put(n, "Factor_Float", fac)
        self.put(n, "A_Float", a)
        self.put(n, "B_Float", b)
        return self._sock(n.outputs, "Result_Float")

    def rgb(self, c):
        n = self.node("ShaderNodeRGB")
        n.outputs[0].default_value = tuple(c) + (1.0,)
        return n.outputs[0]

    def value(self, x):
        n = self.node("ShaderNodeValue")
        n.outputs[0].default_value = x
        return n.outputs[0]

    def done(self, colour, rough, metal=0.0, emit=(0, 0, 0)):
        self.channels = {"colour": colour, "rough": rough, "metal": metal, "emit": emit}

    def show(self, channel):
        for l in list(self.out.inputs["Surface"].links):
            self.nt.links.remove(l)
        if channel == "normal":
            self.nt.links.new(self.diff.outputs[0], self.out.inputs["Surface"])
            return
        self.nt.links.new(self.emit.outputs[0], self.out.inputs["Surface"])
        v = self.channels[channel]
        for l in list(self.emit.inputs[0].links):
            self.nt.links.remove(l)
        if isinstance(v, bpy.types.NodeSocket):
            self.nt.links.new(v, self.emit.inputs[0])
        else:
            vv = (v, v, v) if isinstance(v, (int, float)) else v
            self.emit.inputs[0].default_value = tuple(vv) + (1.0,)
        self.emit.inputs[1].default_value = 1.0


MATS = {}


def new_material(name, build):
    """build(Nodes) must end with nodes.done(...)."""
    mat = bpy.data.materials.new(name)
    nd = Nodes(mat)
    build(nd)
    MATS[name] = nd
    return mat


# ----------------------------------------------------------------------------- the model
class Model:
    def __init__(self, name):
        self.name = name
        self.parts = {}          # part -> {"pivot", "parent", "hi": [], "lo": []}
        self.order = []

    def part(self, name, pivot, parent=None):
        self.parts[name] = {"pivot": Vector(pivot), "parent": parent, "hi": [], "lo": []}
        self.order.append(name)

    def add(self, part, hi, lo):
        """hi may be a single object or a list; lo is the matching game mesh (or None)."""
        his = hi if isinstance(hi, list) else [hi]
        self.parts[part]["hi"] += his
        if lo is not None:
            self.parts[part]["lo"].append(lo)

    # ------------------------------------------------------------------ bake + export
    def bake(self, size=2048, extrusion=0.03, ray=0.09, ao=True, out_dir=None):
        out_dir = pathlib.Path(out_dir or ROOT / "assets_src" / "hq" / self.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        # join all hi pieces, and all lo pieces (one material slot per part, so they can be split again)
        his = [h for p in self.order for h in self.parts[p]["hi"]]
        lows = []
        part_mats = []
        for i, p in enumerate(self.order):
            pm = bpy.data.materials.new(f"part__{p}")
            part_mats.append(pm)
            for lo in self.parts[p]["lo"]:
                lo.data.materials.clear()
                lo.data.materials.append(pm)
                lows.append(lo)
        hi = join(his, f"{self.name}_HI")
        lo = join(lows, f"{self.name}_LO")
        tris = sum(len(p.vertices) - 2 for p in lo.data.polygons)
        print(f"[bake] {self.name}: hi {len(hi.data.polygons)} faces, lo {tris} tris")
        # UVs: one atlas
        bpy.ops.object.select_all(action="DESELECT")
        lo.select_set(True)
        bpy.context.view_layer.objects.active = lo
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=math.radians(58), island_margin=0.0, area_weight=0.0, scale_to_bounds=False)
        bpy.ops.uv.pack_islands(margin=0.006, rotate=True)
        bpy.ops.object.mode_set(mode="OBJECT")
        # target images
        imgs = {}
        for key, cs, fill in (("colour", "sRGB", (0.5, 0.5, 0.5, 1)), ("rough", "Non-Color", (0.8, 0.8, 0.8, 1)), ("metal", "Non-Color", (0, 0, 0, 1)),
                              ("emit", "sRGB", (0, 0, 0, 1)), ("normal", "Non-Color", (0.5, 0.5, 1, 1)), ("ao", "Non-Color", (1, 1, 1, 1))):
            im = bpy.data.images.new(f"{self.name}_{key}", size, size, alpha=False, float_buffer=False)
            im.colorspace_settings.name = cs
            im.generated_color = fill
            imgs[key] = im
        tnodes = []
        for pm in part_mats:
            pm.use_nodes = True
            tn = pm.node_tree.nodes.new("ShaderNodeTexImage")
            pm.node_tree.nodes.active = tn
            tnodes.append(tn)
        for ob in (lo,):
            for attr in ("visible_camera", "visible_diffuse", "visible_glossy", "visible_transmission", "visible_volume_scatter", "visible_shadow"):
                setattr(ob, attr, False)
        sc = bpy.context.scene
        bk = sc.render.bake
        bk.use_selected_to_active = True
        bk.cage_extrusion = extrusion
        bk.max_ray_distance = ray
        bk.margin = 12
        bk.use_clear = True
        bk.target = "IMAGE_TEXTURES"

        def run(kind, key, **kw):
            for tn in tnodes:
                tn.image = imgs[key]
            bpy.ops.object.select_all(action="DESELECT")
            hi.select_set(True)
            lo.select_set(True)
            bpy.context.view_layer.objects.active = lo
            import time
            t = time.time()
            bpy.ops.object.bake(type=kind, use_selected_to_active=True, cage_extrusion=extrusion, max_ray_distance=ray, margin=12, use_clear=True, **kw)
            print(f"[bake] {key} {time.time() - t:.1f}s")

        for ch in ("colour", "rough", "metal", "emit"):
            for nd in MATS.values():
                nd.show(ch)
            sc.cycles.samples = 1
            run("EMIT", ch)
        for nd in MATS.values():
            nd.show("normal")
        run("NORMAL", "normal", normal_space="TANGENT")
        if ao:
            sc.cycles.samples = 48
            run("AO", "ao")
            sc.cycles.samples = 1
        # compose colour * AO, and ORM
        px = {k: pixels(im) for k, im in imgs.items()}
        occ = px["ao"][..., 0] if ao else np.ones(px["colour"].shape[:2])
        occ = 0.25 + 0.75 * occ
        col = px["colour"].copy()
        col[..., :3] = srgb_mul(col[..., :3], occ)
        orm = np.stack([occ, px["rough"][..., 0], px["metal"][..., 0], np.ones_like(occ)], axis=-1)
        files = {"colour": col, "normal": px["normal"], "orm": orm, "emit": px["emit"]}
        paths = {}
        for k, arr in files.items():
            im = bpy.data.images.new(f"{self.name}_{k}_final", size, size, alpha=False)
            im.colorspace_settings.name = "sRGB" if k in ("colour", "emit") else "Non-Color"
            im.pixels.foreach_set(arr.astype(np.float32).ravel())
            path = out_dir / f"{self.name}_{k}.png"
            im.filepath_raw = str(path)
            im.file_format = "PNG"
            im.save()
            paths[k] = path
        self.lo, self.hi, self.tex = lo, hi, paths
        return paths

    def export(self, path, emissive_strength=1.0):
        lo = self.lo
        self.hi.hide_render = True
        # final material
        mat = bpy.data.materials.new(self.name)
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes.get("Principled BSDF")
        if bsdf is None:
            nt.nodes.clear()
            bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
            outn = nt.nodes.new("ShaderNodeOutputMaterial")
            nt.links.new(bsdf.outputs[0], outn.inputs[0])

        def tex(k, cs):
            n = nt.nodes.new("ShaderNodeTexImage")
            n.image = bpy.data.images.load(str(self.tex[k]))
            n.image.colorspace_settings.name = cs
            return n
        nt.links.new(tex("colour", "sRGB").outputs["Color"], bsdf.inputs["Base Color"])
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(tex("normal", "Non-Color").outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
        sp = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(tex("orm", "Non-Color").outputs["Color"], sp.inputs[0])
        nt.links.new(sp.outputs[1], bsdf.inputs["Roughness"])
        nt.links.new(sp.outputs[2], bsdf.inputs["Metallic"])
        nt.links.new(tex("emit", "sRGB").outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = emissive_strength
        # split the low mesh back into parts
        bpy.ops.object.select_all(action="DESELECT")
        lo.select_set(True)
        bpy.context.view_layer.objects.active = lo
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.separate(type="MATERIAL")
        bpy.ops.object.mode_set(mode="OBJECT")
        pieces = {}
        for ob in list(bpy.context.scene.objects):
            if ob.type != "MESH" or ob is self.hi or len(ob.data.polygons) == 0:
                continue
            used = ob.material_slots[ob.data.polygons[0].material_index].material
            if used is not None and used.name.startswith("part__"):
                pieces[used.name[len("part__"):]] = ob
        objs = {}
        for p in self.order:
            info = self.parts[p]
            ob = pieces.get(p)
            if ob is None:       # a part with no mesh of its own: an empty node
                ob = link(bpy.data.objects.new(p, None))
                ob.location = info["pivot"]
            else:
                ob.data.transform(Matrix.Translation(-info["pivot"]))
                ob.location = info["pivot"]
                ob.data.materials.clear()
                ob.data.materials.append(mat)
                ob.data.polygons.foreach_set("material_index", np.zeros(len(ob.data.polygons), dtype=np.int32))
                ob.name = p
                ob.data.name = p
                for attr in ("visible_camera", "visible_diffuse", "visible_glossy", "visible_shadow"):
                    setattr(ob, attr, True)
            objs[p] = ob
        bpy.context.view_layer.update()
        for p in self.order:
            par = self.parts[p]["parent"]
            if par:
                ch = objs[p]
                mw = ch.matrix_world.copy()
                ch.parent = objs[par]
                ch.matrix_world = mw
        bpy.ops.object.select_all(action="DESELECT")
        for ob in objs.values():
            ob.select_set(True)
        kw = dict(filepath=str(path), export_format="GLB", use_selection=True, export_yup=True, export_apply=False,
                  export_animations=False, export_image_format="JPEG", export_texcoords=True, export_normals=True,
                  export_tangents=False, export_materials="EXPORT", export_extras=False)
        try:
            bpy.ops.export_scene.gltf(export_jpeg_quality=90, **kw)
        except TypeError:
            bpy.ops.export_scene.gltf(**kw)
        print(f"[export] {path}")
        return objs


def join(obs, name):
    obs = [o for o in obs if o is not None]
    bpy.ops.object.select_all(action="DESELECT")
    for o in obs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]
    if len(obs) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    ob.name = name
    return ob


def pixels(im):
    w, h = im.size
    a = np.empty(w * h * 4, dtype=np.float32)
    im.pixels.foreach_get(a)
    return a.reshape(h, w, 4)


def srgb_mul(c, k):
    """Multiply sRGB-encoded colours by a linear factor."""
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    lin = lin * k[..., None]
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(np.maximum(lin, 0), 1 / 2.4) - 0.055)
