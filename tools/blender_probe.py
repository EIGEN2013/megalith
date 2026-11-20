"""Print what this Blender build offers for the asset pipeline (run with --background --python)."""
import bpy

print("PROBE version", bpy.app.version_string)
try:
    import numpy
    print("PROBE numpy", numpy.__version__)
except Exception as exc:  # noqa: BLE001
    print("PROBE numpy missing", exc)
mods = [m.identifier for m in bpy.types.Modifier.bl_rna.properties["type"].enum_items]
print("PROBE modifiers", [m for m in mods if m in ("REMESH", "SMOOTH", "LAPLACIANSMOOTH", "DECIMATE", "DISPLACE", "SOLIDIFY", "SKIN", "SUBSURF")])
props = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
print("PROBE gltf", [p for p in props if any(k in p for k in ("format", "image", "jpeg", "yup", "apply", "selection", "tangent", "material", "normals", "texcoord", "animation", "extras"))])
node_types = [n for n in dir(bpy.types) if n.startswith("ShaderNode") and any(k in n for k in ("Principled", "TexImage", "NormalMap", "Emission"))]
print("PROBE nodes", node_types)
m = bpy.data.materials.new("probe")
m.use_nodes = True
bsdf = [n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"]
print("PROBE principled inputs", [i.name for i in bsdf[0].inputs] if bsdf else "none")
