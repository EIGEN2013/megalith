"""Pack the villagers' wardrobe into one GLB the game loads alongside Robin:

    skin_<villager>   a one-triangle mesh whose material carries that villager's texture
    prop_<name>       each prop from tools/blender_npc_props.py, in Robin's model space

Writes live_src/npcs.glb.
Usage: python build_npcs.py
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from glb import Glb  # noqa: E402

ROOT = HERE.parent
NPC = ROOT / "assets_src" / "npc"
PROP_MATERIALS = {"metal": (0.45, 0.6), "wood": (0.85, 0.0), "straw": (0.9, 0.0), "cloth": (0.95, 0.0), "hair": (0.75, 0.0), "plant": (0.8, 0.0)}


def main():
    out = Glb()
    J = out.json
    J["asset"]["generator"] = "build_npcs.py"
    J.update({"materials": [], "meshes": [], "nodes": [], "images": [], "textures": [],
              "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]})
    villagers = json.loads((NPC / "villagers.json").read_text(encoding="utf-8"))
    tri = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], dtype=np.float32)
    for name in villagers:
        for kind, path in (("skin", NPC / f"{name}.jpg"), ("face", NPC / f"face_{name}.jpg")):
            if not path.exists():
                continue
            J["images"].append({"bufferView": out.append_view(path.read_bytes()), "mimeType": "image/jpeg"})
            J["textures"].append({"source": len(J["images"]) - 1, "sampler": 0})
            J["materials"].append({"name": f"{kind}_{name}", "doubleSided": True,
                                   "pbrMetallicRoughness": {"baseColorTexture": {"index": len(J["textures"]) - 1}, "metallicFactor": 0.0, "roughnessFactor": 0.85}})
            J["meshes"].append({"name": f"{kind}_{name}", "primitives": [{"attributes": {
                "POSITION": out.add_accessor(tri, "VEC3", minmax=True), "TEXCOORD_0": out.add_accessor(tri[:, :2], "VEC2")},
                "material": len(J["materials"]) - 1}]})
            J["nodes"].append({"name": f"{kind}_{name}", "mesh": len(J["meshes"]) - 1})
    mat_index = {}
    for path in sorted((NPC / "props").glob("*.glb")):
        src = Glb(str(path))
        prims = []
        for p in src.json["meshes"][0]["primitives"]:
            mname = src.json["materials"][p["material"]]["name"]
            if mname not in mat_index:
                rough, metal = PROP_MATERIALS.get(mname, (0.85, 0.0))
                J["materials"].append({"name": "prop_" + mname, "doubleSided": True,
                                       "pbrMetallicRoughness": {"metallicFactor": metal, "roughnessFactor": rough}})
                mat_index[mname] = len(J["materials"]) - 1
            a = p["attributes"]
            prims.append({"attributes": {
                "POSITION": out.add_accessor(src.read(a["POSITION"]).astype(np.float32), "VEC3", minmax=True, target=34962),
                "NORMAL": out.add_accessor(src.read(a["NORMAL"]).astype(np.float32), "VEC3", target=34962),
                "COLOR_0": out.add_accessor(src.read(a["COLOR_0"]), "VEC4", target=34962, normalized=True)},
                "indices": out.add_accessor(src.read(p["indices"]).reshape(-1), "SCALAR", target=34963), "material": mat_index[mname]})
        J["meshes"].append({"name": "prop_" + path.stem, "primitives": prims})
        J["nodes"].append({"name": "prop_" + path.stem, "mesh": len(J["meshes"]) - 1})
    J["scenes"] = [{"nodes": list(range(len(J["nodes"])))}]
    J["scene"] = 0
    out.save(str(ROOT / "live_src" / "npcs.glb"))
    print("npcs.glb", round(len(out.bin) / 1e6, 2), "MB:", [n["name"] for n in J["nodes"]])


if __name__ == "__main__":
    main()
