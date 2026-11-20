"""Print a summary of a GLB file: nodes, meshes, skins, animations, bounds."""
import json
import struct
import sys


def read_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, length = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67, "not a GLB"
    off = 12
    gltf = None
    bin_chunk = b""
    while off < length:
        clen, ctype = struct.unpack_from("<II", data, off)
        chunk = data[off + 8: off + 8 + clen]
        if ctype == 0x4E4F534A:
            gltf = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:
            bin_chunk = chunk
        off += 8 + clen
    return gltf, bin_chunk


def main():
    path = sys.argv[1]
    g, b = read_glb(path)
    print("asset:", g.get("asset"))
    print("bin bytes:", len(b))
    print("scenes:", g.get("scenes"))
    nodes = g.get("nodes", [])
    print("nodes:", len(nodes))
    for i, n in enumerate(nodes):
        keys = {k: n[k] for k in n if k not in ("children",)}
        print(f"  [{i}] {json.dumps(keys)[:240]} children={n.get('children')}")
    for i, m in enumerate(g.get("meshes", [])):
        for p in m["primitives"]:
            attrs = p["attributes"]
            pos = g["accessors"][attrs["POSITION"]]
            idx = g["accessors"][p["indices"]]["count"] if "indices" in p else None
            print(f"mesh[{i}] {m.get('name')} attrs={list(attrs)} verts={pos['count']} "
                  f"tris={idx // 3 if idx else None} min={pos.get('min')} max={pos.get('max')} mat={p.get('material')}")
    print("materials:", json.dumps(g.get("materials"))[:800])
    for i, im in enumerate(g.get("images", [])):
        bv = g["bufferViews"][im["bufferView"]] if "bufferView" in im else None
        print(f"image[{i}] {im.get('mimeType')} bytes={bv['byteLength'] if bv else im.get('uri', '')[:40]}")
    for i, s in enumerate(g.get("skins", [])):
        print(f"skin[{i}] joints={len(s['joints'])} skeleton={s.get('skeleton')}")
        print("   ", [nodes[j].get("name") for j in s["joints"]])
    for i, a in enumerate(g.get("animations", [])):
        tmax = 0
        for s in a["samplers"]:
            tmax = max(tmax, g["accessors"][s["input"]].get("max", [0])[0])
        print(f"anim[{i}] {a.get('name')} channels={len(a['channels'])} duration={tmax:.3f}")
    print("extensions:", g.get("extensionsUsed"))


if __name__ == "__main__":
    main()
