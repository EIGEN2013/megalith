"""Report size, triangle count and materials of every downloaded Poly Haven model.
Run: blender --background --factory-startup --python tools/ph_inspect.py
"""
import pathlib

import bpy
from mathutils import Vector

ROOT = pathlib.Path(__file__).resolve().parent.parent
for d in sorted((ROOT / "ph").iterdir()):
    g = list(d.glob("*.gltf"))
    if not g:
        continue
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(g[0]))
    lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
    tris = 0
    mats = {}
    objs = 0
    for ob in bpy.context.scene.objects:
        if ob.type != "MESH":
            continue
        objs += 1
        me = ob.data
        tris += sum(len(p.vertices) - 2 for p in me.polygons)
        for v in me.vertices:
            w = ob.matrix_world @ v.co
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
        for m in me.materials:
            if m:
                mats[m.name] = getattr(m, "surface_render_method", getattr(m, "blend_method", ""))
    size = hi - lo
    print(f"PH {d.name:24s} objs {objs:3d} tris {tris:9d} size {size.x:6.2f} x {size.y:6.2f} x {size.z:6.2f}  mats {mats}")
