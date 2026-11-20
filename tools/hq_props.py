"""Scanned ground props (Poly Haven) made game-ready: each object of each scan becomes its own node
(origin at the centre of its base, resting on y = 0), reduced to a sensible triangle budget with
its scanned UVs kept, textures downsized, and all of it written to one GLB with WebP textures.

Writes assets_src/hq/props.glb and props.json (node name -> size, triangle count, kind).
Run: blender --background --factory-startup --python tools/hq_props.py
"""
import json
import pathlib

import bpy
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
PH = ROOT / "ph"
OUT = ROOT / "assets_src" / "hq"

# asset: (kind, triangles per object, texture size, split into objects?)
ASSETS = {
    "boulder_01": ("rock", 6000, 1024, False),
    "namaqualand_boulder_02": ("rock", 6000, 1024, False),
    "rock_moss_set_01": ("rock", 3000, 1024, True),
    "rock_moss_set_02": ("rock", 3000, 1024, True),
    "rock_face_01": ("cliff", 8000, 1024, False),
    "rock_face_02": ("cliff", 8000, 1024, False),
    "rock_07": ("pebble", 400, 512, False),
    "rock_09": ("pebble", 300, 512, False),
    "stone_01": ("pebble", 400, 512, False),
    "tree_stump_01": ("stump", 4000, 1024, False),
    "tree_stump_02": ("stump", 4000, 1024, False),
    "dead_tree_trunk_02": ("log", 5000, 1024, False),
    "root_cluster_01": ("roots", 6000, 1024, False),
    "dry_branches_medium_01": ("twigs", 3000, 1024, True),
    "fern_02": ("fern", 4000, 1024, True),
    "shrub_01": ("shrub", 20000, 1024, False),
    "shrub_02": ("shrub", 8000, 1024, True),
    "shrub_03": ("shrub", 4000, 1024, True),
    "shrub_04": ("shrub", 10000, 1024, False),
    "grass_medium_01": ("grass", 3000, 1024, True),
    "nettle_plant": ("weed", 4000, 1024, True),
    "dandelion_01": ("flower", 4000, 1024, True),
    "moss_01": ("moss", 400, 512, True),
}


def tris(ob):
    return sum(len(p.vertices) - 2 for p in ob.data.polygons)


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    meta = {}
    keep = []
    resized = set()
    for asset, (kind, budget, texsize, split) in ASSETS.items():
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(next((PH / asset).glob("*.gltf"))))
        new = [o for o in bpy.data.objects if o not in before]
        meshes = [o for o in new if o.type == "MESH"]
        bpy.ops.object.select_all(action="DESELECT")
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        if not split and len(meshes) > 1:
            bpy.ops.object.join()
            meshes = [bpy.context.view_layer.objects.active]
        for i, ob in enumerate(meshes):
            ob.parent = None
            n = tris(ob)
            if n > budget:
                m = ob.modifiers.new("d", "DECIMATE")
                m.ratio = budget / n
                m.use_collapse_triangulate = True
                bpy.ops.object.select_all(action="DESELECT")
                ob.select_set(True)
                bpy.context.view_layer.objects.active = ob
                bpy.ops.object.modifier_apply(modifier="d")
            v = np.empty(len(ob.data.vertices) * 3)
            ob.data.vertices.foreach_get("co", v)
            v = v.reshape(-1, 3)
            lo, hi = v.min(axis=0), v.max(axis=0)
            base = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
            ob.data.vertices.foreach_set("co", (v - base).ravel())
            ob.data.update()
            ob.location = (0, 0, 0)
            name = asset if len(meshes) == 1 else f"{asset}_{i}"
            ob.name = name
            size = hi - lo
            meta[name] = {"kind": kind, "tris": tris(ob), "size": [round(float(size[0]), 3), round(float(size[2]), 3), round(float(size[1]), 3)]}
            keep.append(ob)
            for mat in ob.data.materials:
                if mat and mat.node_tree:
                    for nd in mat.node_tree.nodes:
                        if nd.type == "TEX_IMAGE" and nd.image and nd.image.name not in resized:
                            w, h = nd.image.size
                            if max(w, h) > texsize:
                                nd.image.scale(texsize, texsize)
                            resized.add(nd.image.name)
        for o in new:
            if o.type != "MESH" and o.name in bpy.data.objects:
                bpy.data.objects.remove(o, do_unlink=True)
        print(f"[prop] {asset}: {[ (k, meta[k]['tris']) for k in meta if k.startswith(asset)]}")
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from glb import Glb
    hard = ("rock", "cliff", "pebble", "stump", "log", "roots", "twigs")
    for part, pick in (("props_hard", lambda k: k in hard), ("props_soft", lambda k: k not in hard)):
        bpy.ops.object.select_all(action="DESELECT")
        for ob in keep:
            ob.select_set(pick(meta[ob.name]["kind"]))
        out = OUT / f"{part}.glb"
        bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB", use_selection=True, export_yup=True, export_apply=False,
                                  export_animations=False, export_image_format="WEBP", export_normals=True, export_texcoords=True,
                                  export_materials="EXPORT", export_extras=False)
        g = Glb(str(out))
        for m in g.json.get("materials", []):      # the scans model every leaf and blade: no blending needed
            m.pop("alphaMode", None)
            m.pop("alphaCutoff", None)
            if any(k in m["name"] for k in ("grass", "fern", "shrub", "nettle", "dandelion", "moss", "branches")):
                m["doubleSided"] = True
        g.save(str(out))
        print("[export]", out, round(out.stat().st_size / 1e6, 1), "MB")
    (OUT / "props.json").write_text(json.dumps(meta, indent=1))


main()

