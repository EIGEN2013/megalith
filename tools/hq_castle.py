"""The castle on its island, and the stone bridge to the shore, in photoscanned materials.

Curtain wall (ten sides, crenellated wall-walk) with round towers and slate cones, a gatehouse facing the
shore with an arched gate, a courtyard, and a keep you can walk into: a great hall with flagstones,
plastered stone walls, timber trusses, pillars, a dais and throne, a long table, chandeliers, banners and a
fireplace. Box-mapped UVs at real-world scale; ambient occlusion and grime baked into vertex colours.

Coordinates are Blender's (z up, -Y faces the shore = world +Z). Origin = island centre at courtyard level.
Writes assets_src/hq/castle.glb and castle.json (colliders and props in world-relative glTF axes).
Run: blender --background --factory-startup --python tools/hq_castle.py
"""
import base64
import json
import math
import pathlib
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PH = ROOT / "ph"
OUT = ROOT / "assets_src" / "hq"
LAND = json.loads((OUT / "land.json").read_text())
CX, CZ, TOP = LAND["castle"]["x"], LAND["castle"]["z"], LAND["castle"]["top"]
BASE_Y = TOP + 0.2                       # world height of the courtyard
rng = np.random.default_rng(5)

# ---------------------------------------------------------------- materials (scans, metres per tile)
SCANS = {"stone": ("castle_brick_07", 3.0), "stone": ("castle_wall_varriation", 4.0), "blocks": ("medieval_blocks_05", 3.0),
         "moss": ("mossy_stone_wall", 3.0), "slate": ("roof_slates_03", 2.2), "wood": ("medieval_wood", 2.0),
         "beam": ("rough_wood", 1.6), "floor": ("monastery_stone_floor", 3.0), "plaster": ("plastered_stone_wall", 3.0)}
FLAT = {"cloth": ((0.30, 0.03, 0.04), 0.85, 0.0), "gold": ((0.85, 0.62, 0.25), 0.35, 1.0), "iron": ((0.06, 0.06, 0.065), 0.55, 0.9),
        "glow": ((1.0, 0.62, 0.25), 0.6, 0.0), "dark": ((0.01, 0.01, 0.012), 0.9, 0.0), "carpet": ((0.22, 0.03, 0.04), 0.95, 0.0)}
MATS = {}
TILE = {}


def scan_tex(asset, key):
    hits = sorted((PH / asset).glob(f"*_{key}_2k.jpg"))
    return hits[0]


TEXDIR = OUT / "castle_tex"
TEXDIR.mkdir(parents=True, exist_ok=True)


def shrink(image, size, name):
    """Downscale and save under a new file, so the glTF exporter ships the small version."""
    if image.size[0] > size:
        image.scale(size, size)
    path = TEXDIR / f"{name}.jpg"
    image.filepath_raw = str(path)
    image.file_format = "JPEG"
    bpy.context.scene.render.image_settings.quality = 88
    image.save()
    image.filepath = str(path)


def make_materials():
    for name, (asset, tile) in SCANS.items():
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        b = nt.nodes.get("Principled BSDF")

        def img(key, cs, size=1024):
            n = nt.nodes.new("ShaderNodeTexImage")
            n.image = bpy.data.images.load(str(scan_tex(asset, key)))
            n.image.colorspace_settings.name = cs
            shrink(n.image, size, f"{name}_{key}")
            return n
        big = 2048 if name in ("stone", "stone") else 1024
        nt.links.new(img("diff", "sRGB", big).outputs["Color"], b.inputs["Base Color"])
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(img("nor_gl", "Non-Color", big).outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
        sp = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(img("arm", "Non-Color").outputs["Color"], sp.inputs[0])
        nt.links.new(sp.outputs[1], b.inputs["Roughness"])
        nt.links.new(sp.outputs[2], b.inputs["Metallic"])
        # vertex colour (baked occlusion + grime) multiplies the base colour
        MATS[name] = m
        TILE[name] = tile
    for name, (col, rough, metal) in FLAT.items():
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        b = m.node_tree.nodes.get("Principled BSDF")
        b.inputs["Base Color"].default_value = (*col, 1)
        b.inputs["Roughness"].default_value = rough
        b.inputs["Metallic"].default_value = metal
        if name == "glow":
            b.inputs["Emission Color"].default_value = (*col, 1)
            b.inputs["Emission Strength"].default_value = 4.0
        MATS[name] = m
        TILE[name] = 1.0


# ---------------------------------------------------------------- geometry kit
PARTS = []          # (bmesh, material)
COLLIDERS = []      # glTF/world-relative boxes


def collider(cx, cy, hx, hy, z0, z1, yaw=0.0):
    """A box in Blender coords -> game collider (x, z, yaw, half extents, y range), relative to the castle origin."""
    COLLIDERS.append({"x": round(cx, 3), "z": round(-cy, 3), "yaw": round(-yaw, 4), "hx": round(hx, 3), "hz": round(hy, 3),
                      "y0": round(z0, 3), "y1": round(z1, 3)})


def grid_face(bm, o, u, v, nu, nv):
    """A quad grid on the plane o + s*u + t*v, s,t in [0,1]."""
    verts = [[bm.verts.new(o + u * (i / nu) + v * (j / nv)) for j in range(nv + 1)] for i in range(nu + 1)]
    for i in range(nu):
        for j in range(nv):
            bm.faces.new((verts[i][j], verts[i + 1][j], verts[i + 1][j + 1], verts[i][j + 1]))


def box(mat, centre, size, yaw=0.0, cell=1.4, coll=None):
    sx, sy, sz = size
    bm = bmesh.new()
    c = Vector(centre)
    R = Matrix.Rotation(yaw, 3, "Z")
    X, Y, Z = R @ Vector((sx, 0, 0)), R @ Vector((0, sy, 0)), Vector((0, 0, sz))
    o = c - X / 2 - Y / 2 - Z / 2
    n = lambda L: max(1, int(math.ceil(L / cell)))
    grid_face(bm, o, Y, X, n(sy), n(sx))                    # bottom
    grid_face(bm, o + Z, X, Y, n(sx), n(sy))                # top
    grid_face(bm, o, X, Z, n(sx), n(sz))                    # -Y side
    grid_face(bm, o + Y, Z, X, n(sz), n(sx))                # +Y side
    grid_face(bm, o, Z, Y, n(sz), n(sy))                    # -X side
    grid_face(bm, o + X, Y, Z, n(sy), n(sz))                # +X side
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    PARTS.append((bm, mat))
    if coll:
        collider(c.x, c.y, sx / 2, sy / 2, c.z - sz / 2, c.z + sz / 2, yaw)
    return bm


def cylinder(mat, centre, r, h, sides=28, cell=1.4, cap=True, r_top=None, coll=False):
    bm = bmesh.new()
    c = Vector(centre)
    r_top = r if r_top is None else r_top
    rows = max(1, int(math.ceil(h / cell)))
    rings = []
    for k in range(rows + 1):
        z = c.z + h * k / rows
        rr = r + (r_top - r) * k / rows
        rings.append([bm.verts.new((c.x + math.cos(a) * rr, c.y + math.sin(a) * rr, z)) for a in np.linspace(0, 2 * math.pi, sides, endpoint=False)])
    for k in range(rows):
        for i in range(sides):
            j = (i + 1) % sides
            bm.faces.new((rings[k][i], rings[k][j], rings[k + 1][j], rings[k + 1][i]))
    if cap:
        bm.faces.new(rings[-1])
        bm.faces.new(rings[0][::-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    PARTS.append((bm, mat))
    if coll:
        collider(c.x, c.y, r * 0.8, r * 0.8, c.z, c.z + h)
    return bm


def cone(mat, centre, r, h, sides=28, rings=6):
    bm = bmesh.new()
    c = Vector(centre)
    tip = bm.verts.new((c.x, c.y, c.z + h))
    prev = None
    allr = []
    for k in range(rings):
        t = k / rings
        rr = r * (1 - t)
        bulge = 1 + 0.05 * math.sin(math.pi * t)               # a slight bell
        allr.append([bm.verts.new((c.x + math.cos(a) * rr * bulge, c.y + math.sin(a) * rr * bulge, c.z + h * t ** 1.15))
                     for a in np.linspace(0, 2 * math.pi, sides, endpoint=False)])
    for k in range(rings - 1):
        for i in range(sides):
            j = (i + 1) % sides
            bm.faces.new((allr[k][i], allr[k][j], allr[k + 1][j], allr[k + 1][i]))
    for i in range(sides):
        bm.faces.new((allr[-1][i], allr[-1][(i + 1) % sides], tip))
    bm.faces.new(allr[0][::-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    PARTS.append((bm, mat))
    return bm


def cut(bm_target, cutters):
    """Boolean difference of cutter bmeshes from a target bmesh (in place)."""
    me_t = bpy.data.meshes.new("t")
    bm_target.to_mesh(me_t)
    ob = bpy.data.objects.new("t", me_t)
    bpy.context.scene.collection.objects.link(ob)
    col = bpy.data.collections.new("cut")
    bpy.context.scene.collection.children.link(col)
    for c in cutters:
        me = bpy.data.meshes.new("c")
        c.to_mesh(me)
        c.free()
        col.objects.link(bpy.data.objects.new("c", me))
    m = ob.modifiers.new("b", "BOOLEAN")
    m.operation = "DIFFERENCE"
    m.operand_type = "COLLECTION"
    m.collection = col
    m.solver = "EXACT"
    dg = bpy.context.evaluated_depsgraph_get()
    res = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    bm_target.clear()
    bm_target.from_mesh(res)
    for o in list(col.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.data.collections.remove(col)
    bpy.data.objects.remove(ob, do_unlink=True)


def arch_cutter(centre, width, height, depth, yaw=0.0, sides=16):
    """A door/window cutter: a box with a semicircular head, long axis = depth along local Y."""
    bm = bmesh.new()
    r = width / 2
    spring = height - r
    pts = [(-r, 0), (r, 0), (r, spring)] + [(math.cos(a) * r, spring + math.sin(a) * r) for a in np.linspace(0, math.pi, sides)[1:-1]] + [(-r, spring)]
    R = Matrix.Rotation(yaw, 3, "Z")
    c = Vector(centre)
    front = [bm.verts.new(c + R @ Vector((x, -depth / 2, z))) for x, z in pts]
    back = [bm.verts.new(c + R @ Vector((x, depth / 2, z))) for x, z in pts]
    n = len(pts)
    bm.faces.new(front[::-1])
    bm.faces.new(back)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((front[i], front[j], back[j], back[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def wall(mat, p0, p1, height, thick, z0=0.0, coll=True, cell=1.4):
    p0, p1 = Vector((*p0, 0)), Vector((*p1, 0))
    d = p1 - p0
    yaw = math.atan2(d.y, d.x)
    mid = (p0 + p1) / 2
    return box(mat, (mid.x, mid.y, z0 + height / 2), (d.length, thick, height), yaw=yaw, cell=cell, coll=coll)


# ---------------------------------------------------------------- the castle
WALL_R, WALL_H, WALL_T, SIDES = 19.0, 9.0, 2.4, 10
TOWER_R, TOWER_H = 3.3, 13.5
KEEP = dict(x=0.0, y=4.5, w=18.0, d=13.0, h=15.0, t=1.6, floor=1.2)


def curtain():
    angs = [(-90 + 18) * math.pi / 180 + k * 2 * math.pi / SIDES for k in range(SIDES)]
    corners = [(math.cos(a) * WALL_R, math.sin(a) * WALL_R) for a in angs]
    gate_seg = SIDES - 1                                       # between corners 9 and 0: centred on -90 degrees
    for k in range(SIDES):
        p0, p1 = corners[k], corners[(k + 1) % SIDES]
        mat = "stone"
        w = wall(mat, p0, p1, WALL_H + 4.0, WALL_T, z0=-4.0, coll=(k != gate_seg))
        d = Vector((p1[0] - p0[0], p1[1] - p0[1], 0))
        yaw = math.atan2(d.y, d.x)
        mid = Vector(((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, 0))
        out = Vector((math.cos(yaw - math.pi / 2), math.sin(yaw - math.pi / 2), 0))
        if out.dot(mid) < 0:
            out = -out
        cutters = []
        if k == gate_seg:
            cutters.append(arch_cutter((mid.x, mid.y, -0.05), 4.6, 6.2, WALL_T * 3, yaw))
            # the gate's walls either side, as colliders
            half = d.length / 2
            for s in (-1, 1):
                cc = mid + d.normalized() * s * (half + 2.3) / 2
                collider(cc.x, cc.y, (half - 2.3) / 2, WALL_T / 2, -4, WALL_H, yaw)
        else:
            for s in np.linspace(-0.3, 0.3, 2):                 # arrow slits
                c = mid + d * s
                cutters.append(arch_cutter((c.x, c.y, 3.0), 0.28, 1.6, WALL_T * 3, yaw, sides=6))
        cut(w, cutters)
        # crenellations on the outer edge, low parapet inside
        n = int(d.length / 1.9)
        for i in range(n):
            t = (i + 0.5) / n - 0.5
            c = mid + d * t + out * (WALL_T / 2 - 0.3)
            box("stone", (c.x, c.y, WALL_H + 0.65), (1.0, 0.6, 1.3), yaw=yaw)
        c = mid - out * (WALL_T / 2 - 0.2)
        box("stone", (c.x, c.y, WALL_H + 0.3), (d.length, 0.4, 0.6), yaw=yaw)
        # a machicolation corbel line under the parapet
        c = mid + out * (WALL_T / 2 + 0.15)
        box("stone", (c.x, c.y, WALL_H - 0.35), (d.length, 0.3, 0.5), yaw=yaw)
        # walkable wall-walk
        collider(mid.x, mid.y, d.length / 2, WALL_T / 2 - 0.4, WALL_H - 0.2, WALL_H, yaw)
    # towers at every corner
    for k, (x, y) in enumerate(corners):
        gate_tower = k in (gate_seg, 0)
        r = TOWER_R * (1.12 if gate_tower else 1.0)
        h = TOWER_H + (1.5 if gate_tower else 0.0)
        t = cylinder("stone", (x, y, -4.0), r, h + 4.0, sides=32, coll=True)
        cutters = []
        for a in np.linspace(0, 2 * math.pi, 4, endpoint=False) + k:
            cutters.append(arch_cutter((x + math.cos(a) * r, y + math.sin(a) * r, 5.5), 0.3, 1.5, 1.6, a + math.pi / 2, sides=6))
        cut(t, cutters)
        cylinder("stone", (x, y, h - 0.9), r + 0.35, 0.9, sides=32)            # corbelled ring under the roof
        for a in np.linspace(0, 2 * math.pi, 12, endpoint=False):              # merlons
            box("stone", (x + math.cos(a) * (r + 0.1), y + math.sin(a) * (r + 0.1), h + 0.55), (0.9, 0.5, 1.1), yaw=a + math.pi / 2)
        cone("slate", (x, y, h + 1.1), r + 0.55, 7.5 if gate_tower else 6.5, sides=32)
        cylinder("iron", (x, y, h + 1.1 + (7.5 if gate_tower else 6.5) - 0.2), 0.05, 1.6, sides=6)   # finial
    return corners


def keep():
    K = KEEP
    x0, x1 = K["x"] - K["w"] / 2, K["x"] + K["w"] / 2
    y0, y1 = K["y"] - K["d"] / 2, K["y"] + K["d"] / 2
    t, h = K["t"], K["h"]
    # outer shell in four walls, plaster inside
    walls = {
        "front": wall("blocks", (x0, y0), (x1, y0), h + 2, t, z0=-2.0, coll=False),
        "back": wall("blocks", (x0, y1), (x1, y1), h + 2, t, z0=-2.0, coll=True),
        "left": wall("blocks", (x0, y0), (x0, y1), h + 2, t, z0=-2.0, coll=True),
        "right": wall("blocks", (x1, y0), (x1, y1), h + 2, t, z0=-2.0, coll=True),
    }
    door_w, door_h = 3.2, 4.9
    cut(walls["front"], [arch_cutter((K["x"], y0, K["floor"] - 0.05), door_w, door_h, t * 3)] +
        [arch_cutter((K["x"] + s * 5.5, y0, 5.0), 1.5, 3.8, t * 3) for s in (-1, 1)])
    cut(walls["back"], [arch_cutter((K["x"] + s * 5.0, y1, 5.0), 1.5, 3.8, t * 3) for s in (-1, 0, 1)])
    for side, xx in (("left", x0), ("right", x1)):
        cut(walls[side], [arch_cutter((xx, K["y"] + s * 3.0, 5.0), 1.5, 3.8, t * 3, math.pi / 2) for s in (-1, 1)])
    # front wall colliders either side of the door
    half = K["w"] / 2
    for s in (-1, 1):
        cx = K["x"] + s * (half + door_w / 2) / 2
        collider(cx, y0, (half - door_w / 2) / 2, t / 2, -2, h)
    # plaster lining inside (slightly inset), with the same openings
    li = 0.06
    lin = {
        "front": wall("plaster", (x0 + t / 2, y0 + t / 2 + li), (x1 - t / 2, y0 + t / 2 + li), h - 1.0, 0.1, z0=K["floor"], coll=False),
        "back": wall("plaster", (x0 + t / 2, y1 - t / 2 - li), (x1 - t / 2, y1 - t / 2 - li), h - 1.0, 0.1, z0=K["floor"], coll=False),
        "left": wall("plaster", (x0 + t / 2 + li, y0 + t / 2), (x0 + t / 2 + li, y1 - t / 2), h - 1.0, 0.1, z0=K["floor"], coll=False),
        "right": wall("plaster", (x1 - t / 2 - li, y0 + t / 2), (x1 - t / 2 - li, y1 - t / 2), h - 1.0, 0.1, z0=K["floor"], coll=False),
    }
    cut(lin["front"], [arch_cutter((K["x"], y0 + t / 2, K["floor"] - 0.05), door_w, door_h, 1.0)] +
        [arch_cutter((K["x"] + s * 5.5, y0 + t / 2, 5.0), 1.5, 3.8, 1.0) for s in (-1, 1)])
    cut(lin["back"], [arch_cutter((K["x"] + s * 5.0, y1 - t / 2, 5.0), 1.5, 3.8, 1.0) for s in (-1, 0, 1)])
    for side, xx in (("left", x0 + t / 2), ("right", x1 - t / 2)):
        cut(lin[side], [arch_cutter((xx, K["y"] + s * 3.0, 5.0), 1.5, 3.8, 1.0, math.pi / 2) for s in (-1, 1)])
    # stone plinth course and buttresses outside
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x0, y1, x1, y1), (x0, y0, x0, y1), (x1, y0, x1, y1)):
        wall("moss", (ax, ay), (bx, by), 1.4, t + 0.5, z0=-0.4, coll=False)
    for s in (-1, 1):
        for yy in (y0 + 3.5, y1 - 3.5):
            xx = K["x"] + s * (K["w"] / 2 + 0.6)
            box("stone", (xx, yy, 3.0), (1.2, 1.4, 6.0))
            box("stone", (xx + s * 0.2, yy, 6.5), (0.8, 1.2, 1.4))
    # corner turrets
    for sx in (-1, 1):
        for sy in (-1, 1):
            tx, ty = K["x"] + sx * (K["w"] / 2), K["y"] + sy * (K["d"] / 2)
            cylinder("stone", (tx, ty, h - 3.0), 1.9, 7.5, sides=24)
            for a in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                box("stone", (tx + math.cos(a) * 1.95, ty + math.sin(a) * 1.95, h + 4.9), (0.7, 0.4, 0.9), yaw=a + math.pi / 2)
            cone("slate", (tx, ty, h + 4.6), 2.3, 5.5, sides=24)
    # floor of the great hall
    box("floor", (K["x"], K["y"], K["floor"] - 0.3), (K["w"] - t, K["d"] - t, 0.6), cell=1.0, coll=True)
    # entrance steps
    for i in range(4):
        z = K["floor"] * (i + 1) / 5
        box("stone", (K["x"], y0 - t / 2 - 0.45 * (4 - i) - 0.2, z / 2), (door_w + 2.0, 0.45 * (5 - i) + 0.4, z), coll=True)
    # hip roof over the keep, timber trusses beneath
    rh = 7.0
    bm = bmesh.new()
    e = 0.9
    a = bm.verts.new((x0 - e, y0 - e, h))
    b = bm.verts.new((x1 + e, y0 - e, h))
    c = bm.verts.new((x1 + e, y1 + e, h))
    d = bm.verts.new((x0 - e, y1 + e, h))
    r0 = bm.verts.new((x0 + K["d"] / 2, K["y"], h + rh))
    r1 = bm.verts.new((x1 - K["d"] / 2, K["y"], h + rh))
    ridge_x0, ridge_x1 = r0.co.x, r1.co.x
    for f in ((a, b, r1, r0), (b, c, r1), (c, d, r0, r1), (d, a, r0)):
        bm.faces.new(f)
    bm.faces.new((a, d, c, b))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=6, use_grid_fill=True)
    PARTS.append((bm, "slate"))
    # ridge cap
    box("stone", ((ridge_x0 + ridge_x1) / 2, K["y"], h + rh), (ridge_x1 - ridge_x0 + 0.4, 0.5, 0.35))
    # ceiling boards and trusses inside
    box("wood", (K["x"], K["y"], h - 0.3), (K["w"] - t, K["d"] - t, 0.2), cell=1.0)
    for i in range(5):
        xx = x0 + t / 2 + (i + 0.5) * (K["w"] - t) / 5
        box("beam", (xx, K["y"], h - 0.75), (0.45, K["d"] - t, 0.6))
        for s in (-1, 1):
            box("beam", (xx, K["y"] + s * (K["d"] / 2 - t / 2 - 0.3), h - 2.2), (0.4, 0.5, 2.8))
            # curved brace approximated by a slanted beam
            bb = box("beam", (xx, K["y"] + s * (K["d"] / 2 - t / 2 - 1.1), h - 1.5), (0.3, 1.8, 0.3))
            bmesh.ops.rotate(bb, verts=bb.verts, cent=Vector((xx, K["y"] + s * (K["d"] / 2 - t / 2 - 1.1), h - 1.5)), matrix=Matrix.Rotation(-s * 0.6, 3, "X"))
    # pillars flanking the walk from the door to the dais
    for s in (-1, 1):
        for i in range(3):
            px = K["x"] + s * 2.4
            py = K["y"] + (i - 1) * 3.4
            cylinder("stone", (px, py, K["floor"]), 0.42, h - K["floor"] - 1.0, sides=12, coll=True)
            box("stone", (px, py, K["floor"] + 0.2), (1.2, 1.2, 0.4))
            box("stone", (px, py, h - 1.25), (1.1, 1.1, 0.5))
    hall_props(K)


def hall_props(K):
    f = K["floor"]
    y1 = K["y"] + K["d"] / 2 - K["t"] / 2
    # dais and throne at the far end
    box("floor", (K["x"], y1 - 1.6, f + 0.25), (6.0, 3.0, 0.5), coll=True)
    box("floor", (K["x"], y1 - 3.2, f + 0.12), (4.0, 0.6, 0.24), coll=True)
    tx, ty, tz = K["x"], y1 - 1.4, f + 0.5
    box("beam", (tx, ty, tz + 0.25), (1.3, 1.0, 0.5))
    box("beam", (tx, ty + 0.45, tz + 1.6), (1.3, 0.18, 2.6))
    for s in (-1, 1):
        box("beam", (tx + s * 0.6, ty, tz + 0.75), (0.14, 1.0, 0.5))
        box("gold", (tx + s * 0.62, ty + 0.45, tz + 2.95), (0.16, 0.16, 0.3))
    box("cloth", (tx, ty + 0.34, tz + 1.4), (1.0, 0.06, 1.8))
    box("cloth", (tx, ty, tz + 0.52), (1.1, 0.85, 0.06))
    collider(tx, ty, 0.7, 0.55, tz, tz + 0.5)
    # carpet runner from the door to the dais
    box("carpet", (K["x"], K["y"] - 0.6, f + 0.012), (2.2, K["d"] - K["t"] - 4.0, 0.02))
    # long table and benches
    for s in (-1, 1):
        lx = K["x"] + s * 4.6
        box("beam", (lx, K["y"] - 0.5, f + 0.78), (1.3, 7.0, 0.1))
        for yy in (-3.6, 2.6):
            box("beam", (lx, K["y"] - 0.5 + yy * 0.95, f + 0.38), (1.1, 0.12, 0.76))
        for b_ in (-1, 1):
            box("beam", (lx + b_ * 1.05, K["y"] - 0.5, f + 0.45), (0.38, 6.6, 0.08))
            box("beam", (lx + b_ * 1.05, K["y"] - 0.5, f + 0.22), (0.1, 6.0, 0.44))
        collider(lx, K["y"] - 0.5, 1.4, 3.5, f, f + 0.82)
        for k in range(6):                                                  # tankards and plates
            cylinder("iron", (lx + rng.uniform(-0.4, 0.4), K["y"] - 3.5 + k * 1.2, f + 0.83), 0.05, 0.14, sides=8)
            cylinder("gold", (lx + rng.uniform(-0.3, 0.3), K["y"] - 3.2 + k * 1.2, f + 0.835), 0.14, 0.01, sides=12)
    # chandeliers
    for yy in (K["y"] - 2.5, K["y"] + 2.0):
        cz = K["h"] - 4.5
        ring = cylinder("iron", (K["x"], yy, cz), 1.2, 0.08, sides=24)
        cylinder("iron", (K["x"], yy, cz), 1.1, 0.1, sides=24)
        cylinder("iron", (K["x"], yy, cz + 0.08), 0.03, K["h"] - cz - 1.1, sides=6)
        for a in np.linspace(0, 2 * math.pi, 10, endpoint=False):
            cx_, cy_ = K["x"] + math.cos(a) * 1.15, yy + math.sin(a) * 1.15
            cylinder("cloth", (cx_, cy_, cz + 0.08), 0.035, 0.18, sides=8)
            cone("glow", (cx_, cy_, cz + 0.26), 0.03, 0.09, sides=6, rings=2)
    # banners between the windows, fireplace on the right wall
    x0 = K["x"] - K["w"] / 2 + K["t"] / 2
    x1 = K["x"] + K["w"] / 2 - K["t"] / 2
    for xx, side in ((x0 + 0.12, 1), (x1 - 0.12, -1)):
        for yy in (K["y"] - 5.0, K["y"], K["y"] + 4.6):
            box("cloth", (xx, yy, 7.5), (0.04, 1.3, 4.0))
            box("gold", (xx + side * 0.03, yy, 7.9), (0.03, 0.55, 0.55))
            box("beam", (xx + side * 0.05, yy, 9.6), (0.08, 1.6, 0.1))
    fx = x1 - 0.5
    box("stone", (fx, K["y"] + 0.0, f + 1.6), (1.0, 3.2, 3.2), coll=True)
    box("stone", (fx - 0.2, K["y"], f + 3.4), (1.4, 3.6, 0.4))
    box("dark", (fx - 0.52, K["y"], f + 0.95), (0.06, 1.8, 1.5))
    for k in range(5):
        cone("glow", (fx - 0.62, K["y"] + (k - 2) * 0.3, f + 0.25), 0.18, 0.5 + 0.2 * (k % 2), sides=8, rings=2)
    for k in range(3):
        box("beam", (fx - 0.65, K["y"] + (k - 1) * 0.35, f + 0.2), (0.3, 0.9, 0.18), yaw=0.4 * (k - 1))
    # wall torches
    for xx, s in ((x0 + 0.2, 1), (x1 - 0.2, -1)):
        for yy in (K["y"] - 2.6, K["y"] + 2.4):
            box("iron", (xx + s * 0.1, yy, 4.0), (0.2, 0.08, 0.5))
            cone("glow", (xx + s * 0.25, yy, 4.35), 0.08, 0.3, sides=6, rings=2)


def courtyard(corners):
    # paving inside the walls, with a stone apron outside
    cylinder("floor", (0, 0, -0.6), WALL_R - 0.9, 0.6, sides=40, cell=1.2)
    collider(0, 0, 13.5, 13.5, -1.0, 0.0)
    for k in range(SIDES):
        p0, p1 = corners[k], corners[(k + 1) % SIDES]
        mid = ((p0[0] + p1[0]) / 2 * 0.86, (p0[1] + p1[1]) / 2 * 0.86)
        collider(mid[0], mid[1], 6.0, 2.2, -1.0, 0.0, math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    # well
    cylinder("stone", (-9.0, -3.0, 0.0), 1.1, 0.9, sides=20, coll=True)
    cylinder("dark", (-9.0, -3.0, 0.85), 0.85, 0.06, sides=20)
    for s in (-1, 1):
        box("beam", (-9.0 + s * 1.0, -3.0, 1.6), (0.15, 0.15, 1.6))
    box("beam", (-9.0, -3.0, 2.4), (2.3, 0.15, 0.15))
    box("slate", (-9.0, -3.0, 2.7), (2.6, 1.6, 0.12))
    # stairs up to the wall-walk, along the inside of the wall left of the gate
    ang = math.radians(-90 - 18 - 10)
    for i in range(18):
        z = (i + 1) * WALL_H / 18
        r = WALL_R - WALL_T / 2 - 1.0
        a = ang - i * 0.055
        box("stone", (math.cos(a) * r, math.sin(a) * r, z / 2), (1.6, 0.6, z), yaw=a + math.pi / 2, coll=True)


def bridge():
    """Stone arch bridge from the island to the shore at the end of the west road."""
    road_end = LAND["roads"][1][-1]                                       # world (x, z)
    shore_y = -(road_end[1] - CZ)                                          # Blender y of the shore end
    G = LAND["near"]
    hv = np.frombuffer(base64.b64decode(G["int16_le_base64"]), dtype="<i2").astype(np.float64).reshape(G["size"], G["size"]) * G["unit"]

    def ground(wx, wz):
        ix = int(round(wx + G["half"]))
        iz = int(round(wz + G["half"]))
        return float(hv[iz, ix])
    land_h = ground(CX, road_end[1]) - BASE_Y
    start_y, end_y = -WALL_R - 4.0, shore_y + 1.0
    L = start_y - end_y
    n_arch = max(3, int(L / 9.5))
    water = LAND["water"] - BASE_Y
    deck_t, width = 1.1, 5.0
    pts = []
    for i in range(n_arch + 1):
        t = i / n_arch
        y = start_y - L * t
        z = (1 - t) * 0.0 + t * land_h + 1.2 * math.sin(math.pi * t)       # a gentle hump
        pts.append((y, z))
    def deck_z(y):
        t = (start_y - y) / L
        return t * land_h + 1.2 * math.sin(math.pi * t)

    for i in range(n_arch):
        (ya, za), (yb, zb) = pts[i], pts[i + 1]
        ym, zm = (ya + yb) / 2, (za + zb) / 2
        lowest = min(za, zb, zm) - 0.7
        seg = box("stone", (0, ym, lowest - deck_t / 2 - 2.0), (width, ya - yb + 0.02, deck_t + 4.0))
        span = (ya - yb) - 2.4
        cut(seg, [arch_cutter((0, ym, lowest - deck_t - 6.5), span, 4.4, width * 3, yaw=math.pi / 2, sides=20)])
    # the deck, spandrel and parapets follow the bridge's curve as continuous swept shapes
    ys = list(np.linspace(start_y, end_y, int(L / 0.8) + 1))

    def sweep(mat, profile):
        """Extrude a closed (x, dz) profile along the deck curve."""
        bm = bmesh.new()
        rings = [[bm.verts.new((x, y, deck_z(y) + dz)) for x, dz in profile] for y in ys]
        n = len(profile)
        for a, b in zip(rings, rings[1:]):
            for i in range(n):
                j = (i + 1) % n
                bm.faces.new((a[i], a[j], b[j], b[i]))
        bm.faces.new(rings[0][::-1])
        bm.faces.new(rings[-1])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        PARTS.append((bm, mat))
    hw = width / 2
    sweep("stone", [(-hw, -2.6), (hw, -2.6), (hw, -0.02), (-hw, -0.02)])                       # deck body
    sweep("floor", [(-hw + 0.45, -0.05), (hw - 0.45, -0.05), (hw - 0.45, 0.04), (-hw + 0.45, 0.04)])  # paving
    for s in (-1, 1):
        x0_, x1_ = s * (hw - 0.5), s * hw
        lo_, hi_ = min(x0_, x1_), max(x0_, x1_)
        sweep("blocks", [(lo_, -0.05), (hi_, -0.05), (hi_, 1.0), (lo_, 1.0)])                   # parapet
        sweep("stone", [(lo_ - 0.06, 0.98), (hi_ + 0.06, 0.98), (hi_ + 0.06, 1.14), (lo_ - 0.06, 1.14)])   # coping
        sweep("stone", [(lo_ - (0.12 if s < 0 else 0), -0.35), (hi_ + (0.12 if s > 0 else 0), -0.35),
                        (hi_ + (0.12 if s > 0 else 0), -0.05), (lo_ - (0.12 if s < 0 else 0), -0.05)])   # string course
    step = 0.8
    n_steps = int(math.ceil(L / step))
    for j in range(n_steps):
        ya = start_y - L * j / n_steps
        yb = start_y - L * (j + 1) / n_steps
        ym = (ya + yb) / 2
        z = deck_z(ym)
        for s in (-1, 1):
            collider(s * (width / 2 - 0.25), ym, 0.25, (ya - yb) / 2, z, z + 1.1)
        collider(0, ym, width / 2 - 0.5, (ya - yb) / 2 + 0.05, z - 1.0, z + 0.05)
    for i in range(1, n_arch):                                              # piers into the sea, with cutwaters
        y, z = pts[i]
        box("moss", (0, y, (z - deck_t + water - 3.0) / 2), (width + 0.6, 2.4, (z - deck_t) - (water - 3.0)))
        bm = cylinder("moss", (0, y, water - 3.0), 1.5, (z - deck_t - 2.0) - (water - 3.0), sides=3)
        bmesh.ops.rotate(bm, verts=bm.verts, cent=Vector((0, y, 0)), matrix=Matrix.Rotation(math.pi / 2, 3, "Z"))
    return {"start": [0, round(-start_y, 2)], "end": [0, round(-end_y, 2)], "land_h": land_h}


# ---------------------------------------------------------------- finishing: UVs, vertex colours, export
def finish_and_export():
    by_mat = {}
    for bm, mat in PARTS:
        by_mat.setdefault(mat, []).append(bm)
    objs = []
    for mat, bms in by_mat.items():
        me = bpy.data.meshes.new(mat)
        big = bmesh.new()
        for bm in bms:
            tmp = bpy.data.meshes.new("tmp")
            bm.to_mesh(tmp)
            big.from_mesh(tmp)
            bpy.data.meshes.remove(tmp)
            bm.free()
        # box-mapped UVs at the material's real-world scale
        uvl = big.loops.layers.uv.new("UVMap")
        tile = TILE[mat]
        for f in big.faces:
            n = f.normal
            ax = max(range(3), key=lambda i: abs(n[i]))
            for l in f.loops:
                co = l.vert.co
                if ax == 0:
                    uv = (co.y * (1 if n.x > 0 else -1), co.z)
                elif ax == 1:
                    uv = (co.x * (-1 if n.y > 0 else 1), co.z)
                else:
                    uv = (co.x, co.y)
                l[uvl].uv = (uv[0] / tile, uv[1] / tile)
        big.to_mesh(me)
        big.free()
        for p in me.polygons:
            p.use_smooth = False
        me.materials.append(MATS[mat])
        ob = bpy.data.objects.new("castle_" + mat, me)
        bpy.context.scene.collection.objects.link(ob)
        objs.append(ob)
    # vertex colours: baked ambient occlusion x grime rising from the ground and moss low on the stone
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 24
    w = bpy.data.worlds.new("w")
    sc.world = w
    w.light_settings.distance = 6.0
    for ob in objs:
        ca = ob.data.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
        ob.data.color_attributes.active_color = ca
    import time
    for ob in objs:
        if ob.name in ("castle_glow",):
            continue
        bpy.ops.object.select_all(action="DESELECT")
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        t0 = time.time()
        try:
            bpy.ops.object.bake(type="AO", target="VERTEX_COLORS")
        except Exception as e:
            print("[ao] failed", ob.name, e)
        print(f"[ao] {ob.name} {len(ob.data.vertices)} verts {time.time() - t0:.1f}s")
    for ob in objs:
        me = ob.data
        ca = me.color_attributes["Col"]
        n = len(me.vertices)
        c = np.ones(n * 4, dtype=np.float32)
        if ob.name != "castle_glow":
            ca.data.foreach_get("color", c)
        c = c.reshape(-1, 4)
        v = np.empty(n * 3)
        me.vertices.foreach_get("co", v)
        v = v.reshape(-1, 3)
        ao = np.clip(c[:, 0], 0, 1) ** 0.8
        grime = 1 - 0.35 * np.clip(1 - (v[:, 2] + 0.5) / 3.0, 0, 1)
        noise = 0.9 + 0.2 * np.sin(v[:, 0] * 0.7 + np.sin(v[:, 1] * 0.5) * 2) * np.sin(v[:, 2] * 0.9 + v[:, 1] * 0.3)
        k = np.clip((0.35 + 0.65 * ao) * grime * noise, 0, 1)
        if ob.name in ("castle_glow", "castle_gold", "castle_iron", "castle_cloth", "castle_carpet", "castle_dark"):
            k = np.clip(0.55 + 0.45 * ao, 0, 1) if ob.name != "castle_glow" else np.ones(n)
        tint = np.stack([k, k * (1 + 0.04 * (v[:, 2] < 1.0)), k, np.ones(n)], axis=1)
        ca.data.foreach_set("color", tint.astype(np.float32).ravel())
    # the scanned castle door in the keep's doorway, standing open
    before = set(bpy.data.objects)
    door = next((PH / "large_castle_door").glob("*.gltf"))
    bpy.ops.import_scene.gltf(filepath=str(door))
    new = [o for o in bpy.data.objects if o not in before]
    done = set()
    for o in new:
        if o.type == "MESH":
            for mat in o.data.materials:
                for nd in (mat.node_tree.nodes if mat and mat.node_tree else []):
                    if nd.type == "TEX_IMAGE" and nd.image and nd.image.name not in done:
                        done.add(nd.image.name)
                        shrink(nd.image, 1024, f"door{len(done)}")
    root = bpy.data.objects.new("castle_door", None)
    bpy.context.scene.collection.objects.link(root)
    for o in new:
        if o.parent is None:
            o.parent = root
    K = KEEP
    root.location = (K["x"] - 1.55, K["y"] - K["d"] / 2 - 0.2, K["floor"])
    root.rotation_euler = (0, 0, math.radians(-100))
    root.scale = (1.55, 1.55, 1.55)
    # export everything
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs + [root] + new:
        o.select_set(True)
    out = OUT / "castle.glb"
    kw = dict(filepath=str(out), export_format="GLB", use_selection=True, export_yup=True, export_apply=True,
              export_animations=False, export_image_format="JPEG", export_normals=True, export_texcoords=True, export_materials="EXPORT")
    try:
        bpy.ops.export_scene.gltf(export_vertex_color="ACTIVE", **kw)
    except TypeError:
        bpy.ops.export_scene.gltf(export_colors=True, **kw)
    print("[export]", out, round(out.stat().st_size / 1e6, 1), "MB", sum(len(o.data.polygons) for o in objs), "faces")


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    make_materials()
    corners = curtain()
    keep()
    courtyard(corners)
    br = bridge()
    finish_and_export()
    info = {"origin": [CX, BASE_Y, CZ], "colliders": COLLIDERS, "bridge": br, "wall_r": WALL_R,
            "keep": {"x": KEEP["x"], "z": -KEEP["y"], "w": KEEP["w"], "d": KEEP["d"], "floor": KEEP["floor"]}}
    (OUT / "castle.json").write_text(json.dumps(info))
    print("[castle] colliders", len(COLLIDERS), "bridge", br)


main()
