"""Step 2 of the stone pipeline: model each megalith and rock in Blender.

For every prop described in assets_src/tex/<name>.json:
  1. extrude its traced outline into a slab and voxel-remesh it (this also fuses the trilithon's
     two uprights and lintel into one solid),
  2. round it: full thickness in the middle, thinner toward the outline, tapering upward,
  3. roughen the surface with fractal noise,
  4. decimate to a game-sized mesh,
  5. unwrap by projecting the painting onto the front, folding the image around the sides,
  6. export assets_src/models/<name>.glb (model front = -Y in Blender = +Z in glTF, base at z=0,
     one unit tall).

Run: blender --background --factory-startup --python tools/blender_stones.py
"""
import json
import pathlib

import bmesh
import bpy
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEX = ROOT / "assets_src" / "tex"
OUT = ROOT / "assets_src" / "models"
CELL = 0.006          # resolution of the outline's distance field, in prop heights


# ------------------------------------------------------------------ noise
def _hash(ix, iy, iz, seed):
    v = np.sin(ix * 127.1 + iy * 311.7 + iz * 74.7 + seed * 19.19) * 43758.5453
    return v - np.floor(v)


def vnoise(p, seed):
    i = np.floor(p)
    f = p - i
    u = f * f * (3 - 2 * f)
    x0, y0, z0 = i[:, 0], i[:, 1], i[:, 2]

    def h(dx, dy, dz):
        return _hash(x0 + dx, y0 + dy, z0 + dz, seed)
    a = h(0, 0, 0) * (1 - u[:, 0]) + h(1, 0, 0) * u[:, 0]
    b = h(0, 1, 0) * (1 - u[:, 0]) + h(1, 1, 0) * u[:, 0]
    c = h(0, 0, 1) * (1 - u[:, 0]) + h(1, 0, 1) * u[:, 0]
    d = h(0, 1, 1) * (1 - u[:, 0]) + h(1, 1, 1) * u[:, 0]
    return (a * (1 - u[:, 1]) + b * u[:, 1]) * (1 - u[:, 2]) + (c * (1 - u[:, 1]) + d * u[:, 1]) * u[:, 2]


def fbm(p, seed, octaves=4):
    total, amp, freq = 0.0, 0.5, 1.0
    for o in range(octaves):
        total = total + amp * (vnoise(p * freq, seed + o * 7.31) * 2 - 1)
        amp *= 0.5
        freq *= 2.0
    return total


# ------------------------------------------------------------------ outline distance field
def edge_distance(polygons, half_width):
    """Grid of 'how far inside the outline' (prop heights). The base counts as interior, so
    stones stay thick where they meet the ground."""
    xs = np.arange(-half_width - 0.03, half_width + 0.03, CELL)
    zs = np.arange(-0.03, 1.03, CELL)
    gx, gz = np.meshgrid(xs, zs)
    inside = np.zeros(gx.shape, dtype=bool)
    for poly in polygons:
        hit = np.zeros(gx.shape, dtype=bool)
        for (x1, z1), (x2, z2) in zip(poly, poly[1:] + poly[:1]):
            if z1 == z2:
                continue
            cross = ((z1 > gz) != (z2 > gz)) & (gx < (x2 - x1) * (gz - z1) / (z2 - z1) + x1)
            hit ^= cross
        inside |= hit
    base_row = int(round((0.05 - zs[0]) / CELL))
    inside[:base_row] = inside[base_row]
    pad = 40
    inside = np.pad(inside, ((pad, 0), (0, 0)), mode="edge")
    dist = np.zeros(inside.shape)
    cur = inside.copy()
    for _ in range(60):
        nxt = cur.copy()
        nxt[1:] &= cur[:-1]
        nxt[:-1] &= cur[1:]
        nxt[:, 1:] &= cur[:, :-1]
        nxt[:, :-1] &= cur[:, 1:]
        nxt[0] = cur[0]                    # the padded bottom never erodes from below
        cur = nxt
        dist += cur
    return xs, zs, dist[pad:] * CELL


def sample(xs, zs, grid, x, z):
    fx = np.clip((x - xs[0]) / CELL, 0, len(xs) - 1.001)
    fz = np.clip((z - zs[0]) / CELL, 0, len(zs) - 1.001)
    ix, iz = fx.astype(int), fz.astype(int)
    tx, tz = fx - ix, fz - iz
    return (grid[iz, ix] * (1 - tx) + grid[iz, ix + 1] * tx) * (1 - tz) + (grid[iz + 1, ix] * (1 - tx) + grid[iz + 1, ix + 1] * tx) * tz


# ------------------------------------------------------------------ mesh helpers
def evaluated_copy(obj, name):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph))
    mesh.name = name
    return mesh


def get_co(mesh):
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def get_normals(mesh):
    n = np.empty(len(mesh.vertices) * 3)
    mesh.vertex_normals.foreach_get("vector", n)
    return n.reshape(-1, 3)


def build(meta, seed):
    name, depth, rock = meta["name"], meta["depth"], meta.get("rock", False)
    half_width = meta["aspect"] / 2

    # 1. outline -> slab -> one fused, evenly tessellated solid
    bm = bmesh.new()
    for poly in meta["polygons"]:
        front = [bm.verts.new((x, -depth / 2, z)) for x, z in poly]
        back = [bm.verts.new((x, depth / 2, z)) for x, z in poly]
        count = len(poly)
        bm.faces.new(front)
        bm.faces.new(back[::-1])
        for i in range(count):
            bm.faces.new((front[i], front[(i + 1) % count], back[(i + 1) % count], back[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    mesh = bpy.data.meshes.new(name + "_slab")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    remesh = obj.modifiers.new("remesh", "REMESH")
    remesh.mode = "VOXEL"
    remesh.voxel_size = 0.013 if not rock else 0.016
    remesh.adaptivity = 0.0
    smooth = obj.modifiers.new("smooth", "SMOOTH")
    smooth.factor = 0.6
    smooth.iterations = 6
    mesh = evaluated_copy(obj, name + "_dense")
    obj.modifiers.clear()
    obj.data = mesh

    # 2. round it and taper it
    co = get_co(mesh)
    xs, zs, grid = edge_distance(meta["polygons"], half_width)
    inside = sample(xs, zs, grid, co[:, 0], co[:, 2])
    reach = min(depth * 0.5, 0.14) if not rock else 0.28
    t = np.clip(inside / reach, 0, 1)
    thickness = (0.42 if not rock else 0.16) + (0.58 if not rock else 0.84) * np.sqrt(1 - (1 - t) ** 2)
    height = np.clip(co[:, 2], 0, 1)
    thickness *= 1 - (0.62 if rock else 0.22) * height ** 1.2
    co[:, 1] *= thickness
    # 3. roughen: broad lumps plus finer chips, lighter on the faces that carry the painting
    mesh.vertices.foreach_set("co", co.ravel())
    mesh.update()
    normals = get_normals(mesh)
    facing = np.abs(normals[:, 1])
    lumps = fbm(co * (3.0 if rock else 2.2), seed, 3)
    chips = fbm(co * 11.0, seed + 40, 3)
    amount = (0.045 if rock else 0.02) * lumps + (0.014 if rock else 0.008) * chips
    amount *= 1 - 0.55 * facing if not rock else 1.0
    co += normals * amount[:, None]
    if rock:   # a rock is not symmetrical front to back
        co[:, 1] += 0.06 * fbm(co[:, [0, 2, 1]] * 2.0, seed + 90, 2) * height
    mesh.vertices.foreach_set("co", co.ravel())
    mesh.update()

    # 4. decimate
    tris = sum(len(p.vertices) - 2 for p in mesh.polygons)
    decimate = obj.modifiers.new("decimate", "DECIMATE")
    decimate.ratio = min(1.0, meta["tris"] / max(tris, 1))
    tri = obj.modifiers.new("triangulate", "TRIANGULATE")
    del tri
    mesh = evaluated_copy(obj, name)
    obj.modifiers.clear()
    obj.data = mesh
    mesh.shade_smooth()

    # 5. unwrap: project from the front; on the way round the edge, slide inward so the sides
    #    show the stone next to the outline instead of smeared edge pixels
    co = get_co(mesh)
    normals = get_normals(mesh)
    #    Going from the front (q=0) over the edge to the back (q=1), the lookup point moves
    #    inward from the outline by up to `wrap`, so the image continues unbroken round the side
    #    and the back shows it slightly squeezed. Far from the outline the shift fades out.
    half = np.abs(co[:, 1]).max()
    q = np.clip((half + co[:, 1]) / (2 * half), 0, 1)        # front is -Y
    wrap = min(depth * 0.9, 0.3)
    soft = grid.copy()
    for _ in range(3):
        soft[1:-1, 1:-1] = (soft[1:-1, 1:-1] + soft[:-2, 1:-1] + soft[2:, 1:-1] + soft[1:-1, :-2] + soft[1:-1, 2:]) / 5
    grad_z, grad_x = np.gradient(soft, CELL)
    here = np.clip(sample(xs, zs, grid, co[:, 0], co[:, 2]), 0, None)
    gx, gz = sample(xs, zs, grad_x, co[:, 0], co[:, 2]), sample(xs, zs, grad_z, co[:, 0], co[:, 2])
    length = np.maximum(np.hypot(gx, gz), 1e-6)
    shift = wrap * q * (1 - np.clip(here / (2 * wrap), 0, 1))
    px = co[:, 0] + gx / length * shift
    pz = co[:, 2] + gz / length * shift
    del normals
    u = np.clip(meta["u0"] + px * meta["ku"], 0.004, 0.996)
    v = 1.0 - np.clip(meta["v0"] - pz * meta["kv"], 0.004, 0.996)
    loops = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", loops)
    layer = mesh.uv_layers.new(name="UVMap")
    layer.data.foreach_set("uv", np.stack([u[loops], v[loops]], axis=1).ravel())

    # 6. preview material + export
    image = bpy.data.images.load(str(TEX / f"{name}_color.jpg"))
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    material.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.95
    mesh.materials.append(material)

    for other in bpy.context.selected_objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    target = OUT / f"{name}.glb"
    bpy.ops.export_scene.gltf(filepath=str(target), export_format="GLB", use_selection=True, export_yup=True,
                              export_apply=False, export_animations=False, export_image_format="JPEG",
                              export_jpeg_quality=80, export_tangents=False)
    final = sum(len(p.vertices) - 2 for p in mesh.polygons)
    size = co.max(axis=0) - co.min(axis=0)
    print(f"STONE {name}: {len(mesh.vertices)} verts, {final} tris, size {size.round(3).tolist()}")
    obj.select_set(False)
    obj.hide_set(True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    for i, path in enumerate(sorted(TEX.glob("*.json"))):
        build(json.loads(path.read_text(encoding="utf-8")), seed=float(i * 13 + 3))


main()
