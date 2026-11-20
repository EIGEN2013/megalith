"""Photoreal trees from Poly Haven scans.

1. Leaf-cluster cards: a scanned tree (real leaf meshes, scanned textures) is imported, clusters of
   whole leaves and their twigs are cut out around points on the outer crown, turned to face the
   camera, and rendered in Cycles under an even white sky (so leaves shade each other naturally):
   colour+alpha and a matching normal map, packed 2x2 into an atlas per species.
2. Trees: branching skeletons (tools/hq_nature.Tree) clad in scanned, tiling bark (colour, normal,
   AO/roughness), crowned with hundreds of crossed cluster cards facing outward.

Writes assets_src/hq/trees.glb (root nodes tree_oak, tree_oak_b, tree_ash, tree_fir, tree_fir_b)
Run: blender --background --factory-startup --python tools/hq_trees2.py
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
import hq_nature as HN  # noqa: E402

ROOT = HERE.parent
PH = ROOT / "ph"
OUT = ROOT / "assets_src" / "hq"
TEX = OUT / "trees_tex"
TEX.mkdir(parents=True, exist_ok=True)
CELL = 1024


def import_gltf(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path))
    obs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for o in obs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]
    if len(obs) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return ob


def face_data(ob):
    me = ob.data
    n = len(me.polygons)
    c = np.empty(n * 3)
    me.polygons.foreach_get("center", c)
    mi = np.empty(n, dtype=np.int32)
    me.polygons.foreach_get("material_index", mi)
    return c.reshape(-1, 3), mi


def render_setup(scale):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.use_denoising = False
    sc.cycles.transparent_max_bounces = 512      # hundreds of overlapping alpha-cut needles
    sc.cycles.max_bounces = 16
    sc.render.film_transparent = True
    sc.render.resolution_x = sc.render.resolution_y = CELL
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.filter_size = 1.0
    w = bpy.data.worlds.new("w")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = scale
    cam.location = (0, 0, 20)
    cam.data.clip_end = 100
    bpy.context.scene.collection.objects.link(cam)
    sc.camera = cam
    # normal-pass material: facing world normal -> 0..1
    nm = bpy.data.materials.new("normal_pass")
    nm.use_nodes = True
    nt = nm.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sgn = nt.nodes.new("ShaderNodeMath")
    sgn.operation = "MULTIPLY_ADD"
    nt.links.new(geo.outputs["Backfacing"], sgn.inputs[0])
    sgn.inputs[1].default_value = -2.0
    sgn.inputs[2].default_value = 1.0
    sc_ = nt.nodes.new("ShaderNodeVectorMath")
    sc_.operation = "SCALE"
    nt.links.new(geo.outputs["Normal"], sc_.inputs[0])
    nt.links.new(sgn.outputs[0], sc_.inputs["Scale"])
    half = nt.nodes.new("ShaderNodeVectorMath")
    half.operation = "MULTIPLY_ADD"
    nt.links.new(sc_.outputs[0], half.inputs[0])
    half.inputs[1].default_value = (0.5, 0.5, 0.5)
    half.inputs[2].default_value = (0.5, 0.5, 0.5)
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(half.outputs[0], em.inputs["Color"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return nm


def render_pass(path, colour, normal_mat, samples):
    sc = bpy.context.scene
    vl = bpy.context.view_layer
    sc.cycles.samples = samples
    raw = True
    vl.material_override = None
    if colour:
        sc.view_settings.view_transform = "Standard"
    else:
        try:
            sc.view_settings.view_transform = "Raw"
        except TypeError:
            sc.view_settings.view_transform = "Standard"
            raw = False
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    im = bpy.data.images.load(str(path))
    px = np.empty(CELL * CELL * 4, dtype=np.float32)
    im.pixels.foreach_get(px)
    bpy.data.images.remove(im)
    px = px.reshape(CELL, CELL, 4)
    if not raw:
        px[..., :3] = px[..., :3] ** 2.2
    return px


def add_alpha(mat, alpha_path):
    """Drive the material's alpha from a separate mask image (on the same UVs)."""
    nt = mat.node_tree
    b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    a = nt.nodes.new("ShaderNodeTexImage")
    a.image = bpy.data.images.load(str(alpha_path))
    a.image.colorspace_settings.name = "Non-Color"
    nt.links.new(a.outputs["Color"], b.inputs["Alpha"])
    return a.image


def normal_variant(mat, alpha_img):
    nm = bpy.data.materials.new(mat.name + "_n")
    nm.use_nodes = True
    nt = nm.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sgn = nt.nodes.new("ShaderNodeMath")
    sgn.operation = "MULTIPLY_ADD"
    nt.links.new(geo.outputs["Backfacing"], sgn.inputs[0])
    sgn.inputs[1].default_value = -2.0
    sgn.inputs[2].default_value = 1.0
    sc_ = nt.nodes.new("ShaderNodeVectorMath")
    sc_.operation = "SCALE"
    nt.links.new(geo.outputs["Normal"], sc_.inputs[0])
    nt.links.new(sgn.outputs[0], sc_.inputs["Scale"])
    half = nt.nodes.new("ShaderNodeVectorMath")
    half.operation = "MULTIPLY_ADD"
    nt.links.new(sc_.outputs[0], half.inputs[0])
    half.inputs[1].default_value = (0.5, 0.5, 0.5)
    half.inputs[2].default_value = (0.5, 0.5, 0.5)
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(half.outputs[0], em.inputs["Color"])
    if alpha_img is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = alpha_img
        mix = nt.nodes.new("ShaderNodeMixShader")
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        nt.links.new(tex.outputs["Color"], mix.inputs[0])
        nt.links.new(tr.outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    else:
        nt.links.new(em.outputs[0], out.inputs["Surface"])
    return nm


def cluster_atlas(name, gltf, leaf_keys, branch_keys, radius, seed, up_bias=0.5, alpha=None, branch_reach=0.7):
    """Cut 4 clusters from a scanned tree and render them into a 2x2 colour/normal atlas."""
    src = import_gltf(gltf)
    mats = [m.name if m else "" for m in src.data.materials]
    is_leaf = np.array([any(k in m for k in leaf_keys) for m in mats])
    is_branch = np.array([any(k in m for k in branch_keys) for m in mats])
    alpha_imgs = {}
    for m in src.data.materials:
        if not m or not m.node_tree:
            continue
        # base colour straight from the scanned diffuse (the importer may multiply in wind/AO vertex colours)
        b = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        diff = next((n for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image and "diff" in n.image.name), None)
        if b is not None and diff is not None:
            for l in list(b.inputs["Base Color"].links):
                m.node_tree.links.remove(l)
            m.node_tree.links.new(diff.outputs["Color"], b.inputs["Base Color"])
            print(f"[mat] {m.name}: base colour <- {diff.image.name}")
        if alpha and any(k in m.name for k in leaf_keys):
            alpha_imgs[m.name] = add_alpha(m, alpha)
    normal_mats = [normal_variant(m, alpha_imgs.get(m.name)) if m else None for m in src.data.materials]
    centres, mi = face_data(src)
    leaf = is_leaf[mi]
    crown = centres[leaf].mean(axis=0)
    rng = np.random.default_rng(seed)
    rel = centres[leaf] - crown
    reach = np.linalg.norm(rel * [1, 1, up_bias], axis=1)
    outer = centres[leaf][reach > np.quantile(reach, 0.5)]
    # candidates on the outer crown, ranked by how many leaf faces surround them (dense, full clusters)
    cands = outer[rng.choice(len(outer), size=min(300, len(outer)), replace=False)]
    lc = centres[leaf]
    sub = lc[rng.choice(len(lc), size=min(60000, len(lc)), replace=False)]
    dens = np.array([np.sum(np.linalg.norm(sub - c, axis=1) < radius) for c in cands])
    chosen = []
    for i in np.argsort(-dens):
        c = cands[i]
        if all(np.linalg.norm(c - q) > radius * 1.5 for q in chosen):
            chosen.append(c)
        if len(chosen) == 4:
            break
    render_setup(radius * 2.5)
    # whole-mesh arrays (the glTF import is all triangles)
    me = src.data
    nv, nf = len(me.vertices), len(me.polygons)
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
    src.hide_render = True
    tiles_c, tiles_n = [], []
    for k, c in enumerate(chosen):
        d = np.linalg.norm(centres - c, axis=1)
        # whole leaves: label leaf islands near the point by their connected vertices
        cand = np.nonzero((d < radius * 1.7) & leaf)[0]
        F = tri[cand]
        lab = np.arange(nv)
        for _ in range(80):
            fl = lab[F].min(axis=1)
            before = lab[F].copy()
            np.minimum.at(lab, F.ravel(), np.repeat(fl, 3))
            if np.array_equal(before, lab[F]):
                break
        face_lab = lab[F].min(axis=1)
        good = np.unique(face_lab[d[cand] < radius])
        keep = cand[np.isin(face_lab, good)]
        keep = np.concatenate([keep, np.nonzero((d < radius * branch_reach) & is_branch[mi])[0]])
        # face the camera: outward -> +Z, world up -> +Y
        o = (c - crown) * np.array([1, 1, 0.35])
        o /= np.linalg.norm(o)
        side = np.cross([0, 0, 1.0], o)
        side /= np.linalg.norm(side) + 1e-9
        up = np.cross(o, side)
        R = np.stack([side, up, o])
        P = (pos[tri[keep]].reshape(-1, 3) - c) @ R.T
        cm = bpy.data.meshes.new("cluster")
        cm.from_pydata([tuple(p) for p in P], [], [tuple(f) for f in np.arange(len(P)).reshape(-1, 3)])
        cm.update()
        for m in src.data.materials:
            cm.materials.append(m)
        cm.polygons.foreach_set("material_index", mi[keep].astype(np.int32))
        cm.uv_layers.new(name="UVMap").data.foreach_set("uv", uv[keep].reshape(-1))
        cm.polygons.foreach_set("use_smooth", np.ones(len(keep), dtype=bool))
        cm.normals_split_custom_set([tuple(n) for n in (cn[keep].reshape(-1, 3) @ R.T)])
        ob = bpy.data.objects.new("cluster", cm)
        bpy.context.scene.collection.objects.link(ob)
        print(f"[cluster] {name} {k}: {len(keep)} faces")
        for passname, samples in (("c", 64), ("n", 8)):
            for i, m in enumerate(cm.materials):
                cm.materials[i] = src.data.materials[i] if passname == "c" else normal_mats[i]
            px = render_pass(TEX / f"_{name}_{k}_{passname}.png", passname == "c", None, samples)
            (tiles_c if passname == "c" else tiles_n).append(px)
        bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.meshes.remove(cm)
    return pack_atlas(name, tiles_c, tiles_n)


def pack_atlas(name, tiles_c, tiles_n):
    S = CELL * 2
    ac = np.zeros((S, S, 4), dtype=np.float32)
    an = np.zeros((S, S, 4), dtype=np.float32)
    for i in range(4):
        y0, x0 = (i // 2) * CELL, (i % 2) * CELL
        ac[y0:y0 + CELL, x0:x0 + CELL] = tiles_c[i]
        an[y0:y0 + CELL, x0:x0 + CELL] = tiles_n[i]
    a = ac[..., 3].copy()
    filled = a > 0.02
    rgb = ac[..., :3] * filled[..., None]
    nrm = an[..., :3] * filled[..., None]
    for _ in range(32):               # bleed colours outward so mip-maps never fringe
        acc = np.zeros_like(rgb)
        accn = np.zeros_like(nrm)
        cnt = np.zeros(a.shape, dtype=np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.roll(filled, (dy, dx), (0, 1))
            acc += np.roll(rgb, (dy, dx), (0, 1)) * sh[..., None]
            accn += np.roll(nrm, (dy, dx), (0, 1)) * sh[..., None]
            cnt += sh
        grow = (~filled) & (cnt > 0)
        rgb[grow] = acc[grow] / cnt[grow][:, None]
        nrm[grow] = accn[grow] / cnt[grow][:, None]
        filled |= grow
    rgb[~filled] = rgb[filled].mean(axis=0)
    nrm[~filled] = (0.5, 0.5, 1.0)
    ac[..., :3] = rgb
    an[..., :3] = nrm
    an[..., 3] = 1
    paths = {}
    for key, arr in (("c", ac), ("n", an)):
        im = bpy.data.images.new(f"{name}_{key}", S, S, alpha=(key == "c"))
        im.colorspace_settings.name = "sRGB" if key == "c" else "Non-Color"
        im.pixels.foreach_set(arr.ravel())
        path = TEX / f"leaf_{name}_{key}.png"
        im.filepath_raw = str(path)
        im.file_format = "PNG"
        im.save()
        paths[key] = path
    print(f"[cards] {name}")
    return paths


# ----------------------------------------------------------------------------- trees
def cluster_cards(T, size_range, count, seed, droop=0.0):
    """Crossed cards centred on twig tips, facing out of the crown, image-up = world-up."""
    r = np.random.default_rng(seed)
    tips = np.array(T.tips)
    centre = tips.mean(axis=0)
    cards = []
    order = np.concatenate([r.permutation(len(tips)) for _ in range(1 + count // max(len(tips), 1))])[:count]
    for n_, i in enumerate(order):
        p = tips[i] + (r.normal(0, 0.25, 3) if n_ >= len(tips) else 0)
        o = (p - centre) * np.array([1, 1, 0.45])
        o /= np.linalg.norm(o) + 1e-9
        o = o + r.normal(0, 0.25, 3)
        o /= np.linalg.norm(o)
        cards.append((p, o, r.uniform(*size_range)))
    T.card_list = cards
    T.centre = centre


def card_mesh(T, seed):
    r = np.random.default_rng(seed)
    v, f, uv, nn = [], [], [], []
    for p, o, s in T.card_list:
        cell = r.integers(4)
        cu, cv = (cell % 2) * 0.5, (cell // 2) * 0.5
        side = np.cross([0, 0, 1.0], o)
        side /= np.linalg.norm(side) + 1e-9
        up = np.cross(o, side)
        for twist in (0.0, r.uniform(0.7, 1.1) * r.choice([-1, 1])):
            sd = side * math.cos(twist) + o * math.sin(twist)
            fn = np.cross(sd, up)
            k0 = len(v)
            for (x, y) in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
                q = p + sd * x * s + up * y * s
                v.append(q)
                uv.append((cu + (x + 0.5) * 0.5, cv + (y + 0.5) * 0.5))
                out = q - T.centre
                out /= np.linalg.norm(out) + 1e-9
                bent = out * 0.7 + o * 0.3 + np.array([0, 0, 0.2])
                nn.append(bent / np.linalg.norm(bent))
            f.append((k0, k0 + 1, k0 + 2, k0 + 3))
    return v, f, uv, nn


def ph_tex(asset, keys=("diff", "nor_gl", "arm")):
    d = PH / asset
    out = {}
    for k in keys:
        hits = list(d.rglob(f"*_{k}_2k.jpg"))
        out[k] = hits[0]
    return out


def bark_material(name, asset):
    t = ph_tex(asset)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    b = nt.nodes.get("Principled BSDF")

    def img(path, cs):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(str(path))
        n.image.colorspace_settings.name = cs
        if n.image.size[0] > 1024:
            n.image.scale(1024, 1024)
        return n
    nt.links.new(img(t["diff"], "sRGB").outputs["Color"], b.inputs["Base Color"])
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(img(t["nor_gl"], "Non-Color").outputs["Color"], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    sp = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(img(t["arm"], "Non-Color").outputs["Color"], sp.inputs[0])
    nt.links.new(sp.outputs[1], b.inputs["Roughness"])
    nt.links.new(sp.outputs[2], b.inputs["Metallic"])
    return mat


def leaf_material(name, tex):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    b = nt.nodes.get("Principled BSDF")
    c = nt.nodes.new("ShaderNodeTexImage")
    c.image = bpy.data.images.load(str(tex["c"]))
    nt.links.new(c.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(c.outputs["Alpha"], b.inputs["Alpha"])
    n = nt.nodes.new("ShaderNodeTexImage")
    n.image = bpy.data.images.load(str(tex["n"]))
    n.image.colorspace_settings.name = "Non-Color"
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(n.outputs["Color"], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    b.inputs["Roughness"].default_value = 0.6
    return mat


def bark_uv_scale(T, tile=1.4):
    """hq_nature.Tree writes u in 0.5 m tiles round the trunk and v in metres; rescale to the scan's tile."""
    uv = np.array(T.bark["uv"], dtype=np.float64)
    uv[:, 0] *= 0.5 / tile * 2.0
    uv[:, 1] /= tile
    T.bark["uv"] = [tuple(x) for x in uv]


BROAD = {
    "tree_oak": dict(kind="oak", trunk=4.2, r=0.55, depth=4, kids=[5, 4, 3, 3], start=[0.78, 0.3, 0.25, 0.3], angle=[(35, 55), (35, 60), (35, 60), (30, 55)],
                     ratio=[1.15, 0.62, 0.55, 0.55], rratio=0.6, up=[0.2, 0.12, 0.1, 0.1], wiggle=0.13, photo=0.03, droop=0.05, taper=0.45,
                     cards=1700, card=(1.5, 2.3), leaf="oak", seed=1, bark="bark_brown_02"),
    "tree_oak_b": dict(kind="oak", trunk=3.4, r=0.46, depth=4, kids=[4, 4, 3, 3], start=[0.75, 0.3, 0.25, 0.3], angle=[(28, 48), (35, 60), (35, 60), (30, 55)],
                       ratio=[1.1, 0.62, 0.55, 0.55], rratio=0.6, up=[0.35, 0.15, 0.1, 0.1], wiggle=0.15, photo=0.05, droop=0.04, taper=0.45,
                       cards=1400, card=(1.4, 2.1), leaf="oak", seed=2, bark="bark_brown_02"),
    "tree_ash": dict(kind="ash", trunk=5.0, r=0.32, depth=4, kids=[4, 3, 3, 3], start=[0.6, 0.3, 0.3, 0.3], angle=[(25, 40), (30, 50), (30, 55), (30, 55)],
                     ratio=[0.75, 0.6, 0.55, 0.5], rratio=0.6, up=[0.45, 0.2, 0.1, 0.1], wiggle=0.1, photo=0.06, droop=0.03, taper=0.5,
                     cards=1100, card=(1.2, 1.8), leaf="ash", seed=3, bark="bark_brown_02", leader=True),
}
FIRS = {"tree_fir": dict(height=17.0, r=0.42, spread=3.6, card=1.6, seed=8), "tree_fir_b": dict(height=12.5, r=0.32, spread=3.0, card=1.45, seed=9)}


def main():
    import sys as _s
    only = [a for a in _s.argv[_s.argv.index("--") + 1:]] if "--" in _s.argv else []
    leaf_tex = {}
    for key, args, kw in (("oak", ("island_tree_02", ["leaves"], ["branches"], 0.75, 4), {}),
                          ("ash", ("tree_small_02", ["leaves"], ["branch"], 0.6, 5), {"branch_reach": 0.45}),
                          ("fir", ("fir_tree_01", ["twig"], ["dead_branches"], 0.85, 6), {"up_bias": 0.3, "alpha": PH / "fir_alpha.png", "branch_reach": 0.3})):
        cached = TEX / f"leaf_{key}_c.png"
        if only and key not in only and cached.exists():
            leaf_tex[key] = {"c": cached, "n": TEX / f"leaf_{key}_n.png"}
            continue
        leaf_tex[key] = cluster_atlas(key, next((PH / args[0]).glob("*.gltf")), *args[1:], **kw)
    HN.reset()
    leaf_mats = {k: leaf_material(f"leaf_{k}", t) for k, t in leaf_tex.items()}
    bark_mats = {"bark_brown_02": bark_material("bark_oak", "bark_brown_02"), "pine_bark": bark_material("bark_pine", "pine_bark")}
    roots = []
    for name, spec in BROAD.items():
        T = HN.build_species(dict(spec, card=spec["card"][0], cards=10), spec["seed"])
        cluster_cards(T, spec["card"], spec["cards"], spec["seed"])
        roots.extend(emit(name, T, bark_mats[spec["bark"]], leaf_mats[spec["leaf"]]))
    for name, spec in FIRS.items():
        T = HN.pine_tree(spec, spec["seed"])
        tips = [c[0] for c in T.cards]
        T.tips = tips
        cards = []
        r = np.random.default_rng(spec["seed"])
        for p, d, s in T.cards:
            o = d * np.array([1, 1, 0.3])
            o /= np.linalg.norm(o) + 1e-9
            cards.append((p + d * s * 0.3, o, s * r.uniform(0.9, 1.2)))
        T.card_list = cards
        roots.extend(emit(name, T, bark_mats["pine_bark"], leaf_mats["fir"]))
    from glb import Glb
    for part, pick in (("trees_broad", lambda n: "fir" not in n), ("trees_fir", lambda n: "fir" in n)):
        bpy.ops.object.select_all(action="DESELECT")
        for r_ in roots:
            if pick(r_.name):
                r_.select_set(True)
                for c in r_.children:
                    c.select_set(True)
        out = OUT / f"{part}.glb"
        bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB", use_selection=True, export_yup=True, export_apply=False,
                                  export_animations=False, export_image_format="WEBP", export_normals=True, export_texcoords=True,
                                  export_materials="EXPORT", export_extras=False)
        g = Glb(str(out))
        for m in g.json.get("materials", []):
            if m["name"].startswith("leaf_"):
                m["alphaMode"] = "MASK"
                m["alphaCutoff"] = 0.4
                m["doubleSided"] = True
        g.save(str(out))
        print("[export]", out, round(out.stat().st_size / 1e6, 1), "MB")


def emit(name, T, bark_mat, leaf_mat):
    root = HN.link(bpy.data.objects.new(name, None))
    bark_uv_scale(T)
    b = T.bark
    bo = HN.mesh_from(name + "_bark", b["v"], b["f"], uvs=b["uv"], mat=bark_mat)
    v, f, uv, nn = card_mesh(T, 7)
    lo = HN.mesh_from(name + "_leaves", v, f, uvs=uv, normals=nn, mat=leaf_mat)
    bo.parent = root
    lo.parent = root
    # far version: every fourth cluster, enlarged, and the bark reduced to its main limbs
    full = T.card_list
    T.card_list = [(p, o, s * 1.75) for i, (p, o, s) in enumerate(full) if i % 4 == 0]
    v, f, uv, nn = card_mesh(T, 11)
    T.card_list = full
    lf = HN.mesh_from(name + "_leaves_far", v, f, uvs=uv, normals=nn, mat=leaf_mat)
    bf = bo.copy()
    bf.data = bo.data.copy()
    bf.name = name + "_bark_far"
    HN.link(bf)
    m = bf.modifiers.new("d", "DECIMATE")
    m.ratio = 0.12
    bpy.context.view_layer.objects.active = bf
    bpy.ops.object.select_all(action="DESELECT")
    bf.select_set(True)
    bpy.ops.object.modifier_apply(modifier="d")
    far = HN.link(bpy.data.objects.new(name + "_far", None))
    lf.parent = far
    bf.parent = far
    print(f"[tree] {name}: bark {len(b['f']) * 2} tris, {len(full)} cards; far {len(f) * 2} card tris")
    return root, far


if __name__ == "__main__":
    main()
