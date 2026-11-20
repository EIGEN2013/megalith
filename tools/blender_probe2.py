"""Check the Blender APIs the flora script relies on (skin layer, normals, loop triangles)."""
import bpy
import numpy as np

me = bpy.data.meshes.new("skel")
me.from_pydata([(0, 0, 0), (0, 0, 1), (0.5, 0, 1.5)], [(0, 1), (1, 2)], [])
obj = bpy.data.objects.new("skel", me)
bpy.context.collection.objects.link(obj)
print("PROBE skin layers before", len(me.skin_vertices))
skin = obj.modifiers.new("skin", "SKIN")
print("PROBE skin layers after", len(me.skin_vertices))
if len(me.skin_vertices):
    for i, d in enumerate(me.skin_vertices[0].data):
        d.radius = (0.2 - i * 0.05, 0.2 - i * 0.05)
    me.skin_vertices[0].data[0].use_root = True
print("PROBE skin props", [p for p in skin.bl_rna.properties.keys() if p not in ("rna_type",)])
sub = obj.modifiers.new("sub", "SUBSURF")
sub.levels = 1
dg = bpy.context.evaluated_depsgraph_get()
ev = obj.evaluated_get(dg)
m2 = ev.to_mesh()
m2.calc_loop_triangles()
n = np.empty(len(m2.vertices) * 3)
m2.vertex_normals.foreach_get("vector", n)
print("PROBE evaluated", len(m2.vertices), len(m2.loop_triangles))
ev.to_mesh_clear()
