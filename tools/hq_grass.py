"""Grass-tuft cards from the scanned grass clumps (Poly Haven grass_medium_01): the four fullest clumps
are stood up facing the camera and rendered (colour+alpha, normal) into a 2x2 atlas, the same way as
the tree clusters. The game scatters thousands of crossed cards with this atlas around the player.

Writes assets_src/hq/trees_tex/leaf_grass_{c,n}.png
Run: blender --background --factory-startup --python tools/hq_grass.py
"""
import pathlib
import sys

import bpy
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hq_trees2 as T2  # noqa: E402


def main():
    src = T2.import_gltf(next((T2.PH / "grass_medium_01").glob("*.gltf")))
    # split back into clumps by x position (the scan lays them out in a row)
    me = src.data
    nf = len(me.polygons)
    cen, mi = T2.face_data(src)
    nv = len(me.vertices)
    pos = np.empty(nv * 3)
    me.vertices.foreach_get("co", pos)
    pos = pos.reshape(-1, 3)
    tri = np.empty(nf * 3, dtype=np.int64)
    me.loops.foreach_get("vertex_index", tri)
    tri = tri.reshape(-1, 3)
    uv = np.empty(nf * 6, dtype=np.float32)
    me.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 3, 2)
    cn = np.empty(nf * 9, dtype=np.float32)
    me.corner_normals.foreach_get("vector", cn)
    cn = cn.reshape(-1, 3, 3)
    # connected clumps: label propagation over shared vertices
    lab = np.arange(nv)
    for _ in range(200):
        fl = lab[tri].min(axis=1)
        before = lab.copy()
        np.minimum.at(lab, tri.ravel(), np.repeat(fl, 3))
        if np.array_equal(before, lab):
            break
    flab = lab[tri].min(axis=1)
    ids, counts = np.unique(flab, return_counts=True)
    # group islands into clumps by their x centre (2 cm bins merge blades of one clump)
    xs = np.array([cen[flab == i, 0].mean() for i in ids])
    order = np.argsort(xs)
    groups, cur, last = [], [], None
    for i in order:
        if last is not None and xs[i] - last > 0.12:
            groups.append(cur)
            cur = []
        cur.append(ids[i])
        last = xs[i]
    groups.append(cur)
    sizes = [sum(counts[np.searchsorted(ids, g)]) for g in groups]
    best = [groups[i] for i in np.argsort(sizes)[::-1][:4]]
    src.hide_render = True
    for m in me.materials:
        b = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        diff = next((n for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image and "diff" in n.image.name), None)
        if b is not None and diff is not None:
            for l in list(b.inputs["Base Color"].links):
                m.node_tree.links.remove(l)
            m.node_tree.links.new(diff.outputs["Color"], b.inputs["Base Color"])
    normal_mats = [T2.normal_variant(m, None) for m in me.materials]
    T2.render_setup(0.3)
    tiles_c, tiles_n = [], []
    R = np.array([[1, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])      # stand the clump up: +Z (up) -> image up, -Y -> camera
    for k, g in enumerate(best):
        keep = np.nonzero(np.isin(flab, g))[0]
        P = pos[tri[keep]].reshape(-1, 3)
        base = np.array([P[:, 0].mean(), P[:, 1].mean(), P[:, 2].min()])
        P = (P - base) @ R.T
        top = P[:, 1].max()
        P[:, 1] -= 0.148                                        # card spans y from -0.15 (ground) upward
        cm = bpy.data.meshes.new("clump")
        cm.from_pydata([tuple(p) for p in P], [], [tuple(f) for f in np.arange(len(P)).reshape(-1, 3)])
        cm.update()
        for m in me.materials:
            cm.materials.append(m)
        cm.polygons.foreach_set("material_index", mi[keep].astype(np.int32))
        cm.uv_layers.new(name="UVMap").data.foreach_set("uv", uv[keep].reshape(-1))
        cm.polygons.foreach_set("use_smooth", np.ones(len(keep), dtype=bool))
        cm.normals_split_custom_set([tuple(n) for n in (cn[keep].reshape(-1, 3) @ R.T)])
        ob = bpy.data.objects.new("clump", cm)
        bpy.context.scene.collection.objects.link(ob)
        print(f"[clump] {k}: {len(keep)} faces, height {top:.2f} m")
        for passname, samples in (("c", 48), ("n", 8)):
            for i in range(len(cm.materials)):
                cm.materials[i] = me.materials[i] if passname == "c" else normal_mats[i]
            px = T2.render_pass(T2.TEX / f"_grass_{k}_{passname}.png", passname == "c", None, samples)
            (tiles_c if passname == "c" else tiles_n).append(px)
        bpy.data.objects.remove(ob, do_unlink=True)
    T2.pack_atlas("grass", tiles_c, tiles_n)


main()
