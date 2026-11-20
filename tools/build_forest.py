"""Rebuild the ruins (the first area) to the village's standard.

Kept exactly as they are: every stone, the far peaks and the moon (copied node for node from
the live world.glb), and the colliders and points of interest in world.json.
New: a finer, hand-painted terrain, and the forest from tools/blender_flora.py placed as
instances - the same trees on the same spots as before, a carpet of grass, flowers, ferns,
curling plants, mushrooms and pebbles.

Writes, in the format the village already uses:
    live_src/forest.glb        stones, peaks, moon, terrain, and one proto_<name> mesh per plant
    live_src/forest_inst.f32   15 floats per instance: 3x4 transform (row-major) + RGB tint
    live_src/forest.json       which instances belong to which prototype, and how to draw them
    live_src/world.json        the same world data with the height map at the terrain's 0.5 m step

Usage: python build_forest.py
"""
import base64
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from glb import Glb  # noqa: E402
import build_world as W  # noqa: E402  (height(), paths and the noise the first build used)

ROOT = HERE.parent
LIVE = ROOT / "live_src"
FLORA = ROOT / "assets_src" / "flora"
rng = np.random.default_rng(2026)

KEEP = ("trilithon", "door_stone", "altar", "rune_stone", "small_stone", "monolith", "rune_pillar", "rock_spire", "rock_crag",
        "mountains", "moon")
STEP = 0.5
HALF = 80.0
PLAY_R = 72.0                      # plants grow out to here; the fog takes over beyond


# ------------------------------------------------------------------ copy part of a GLB
class Copier:
    """Copies nodes (and everything they use) from one GLB into another."""

    def __init__(self, src, dst):
        self.src, self.dst = src, dst
        self.maps = {k: {} for k in ("accessors", "meshes", "materials", "textures", "images", "samplers")}
        for key in ("meshes", "materials", "textures", "images", "samplers", "nodes", "accessors", "bufferViews"):
            dst.json.setdefault(key, [])

    def _view(self, index):
        bv = self.src.json["bufferViews"][index]
        return self.dst.append_view(self.src.view_bytes(index), bv.get("target"))

    def accessor(self, i):
        if i not in self.maps["accessors"]:
            acc = dict(self.src.json["accessors"][i])
            raw_view = self.src.json["bufferViews"][acc["bufferView"]]
            if raw_view.get("byteStride"):
                raise ValueError("strided buffer views are not copied")
            acc["bufferView"] = self._view(acc["bufferView"])
            self.dst.json["accessors"].append(acc)
            self.maps["accessors"][i] = len(self.dst.json["accessors"]) - 1
        return self.maps["accessors"][i]

    def _simple(self, kind, i, fix):
        if i not in self.maps[kind]:
            item = json.loads(json.dumps(self.src.json[kind][i]))
            fix(item)
            self.dst.json[kind].append(item)
            self.maps[kind][i] = len(self.dst.json[kind]) - 1
        return self.maps[kind][i]

    def image(self, i):
        def fix(img):
            img["bufferView"] = self._view(img["bufferView"])
        return self._simple("images", i, fix)

    def sampler(self, i):
        return self._simple("samplers", i, lambda s: None)

    def texture(self, i):
        def fix(t):
            t["source"] = self.image(t["source"])
            if "sampler" in t:
                t["sampler"] = self.sampler(t["sampler"])
        return self._simple("textures", i, fix)

    def material(self, i):
        def fix(m):
            def walk(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k == "index" and "texCoord" not in obj or (k == "index" and isinstance(v, int)):
                            obj[k] = self.texture(v)
                        else:
                            walk(v)
                elif isinstance(obj, list):
                    for v in obj:
                        walk(v)
            walk(m)
        return self._simple("materials", i, fix)

    def mesh(self, i):
        def fix(m):
            for p in m["primitives"]:
                p["attributes"] = {k: self.accessor(v) for k, v in p["attributes"].items()}
                if "indices" in p:
                    p["indices"] = self.accessor(p["indices"])
                if "material" in p:
                    p["material"] = self.material(p["material"])
        return self._simple("meshes", i, fix)

    def node(self, i):
        node = {k: v for k, v in self.src.json["nodes"][i].items() if k != "children"}
        if "mesh" in node:
            node["mesh"] = self.mesh(node["mesh"])
        self.dst.json["nodes"].append(node)
        return len(self.dst.json["nodes"]) - 1


# ------------------------------------------------------------------ prototypes
def read_asset(name):
    g = Glb(str(FLORA / f"{name}.glb"))
    prims = []
    for p in g.json["meshes"][0]["primitives"]:
        a = p["attributes"]
        prims.append({"material": g.json["materials"][p["material"]]["name"],
                      "pos": g.read(a["POSITION"]).astype(np.float32), "nor": g.read(a["NORMAL"]).astype(np.float32),
                      "col": g.read(a["COLOR_0"]), "idx": g.read(p["indices"]).reshape(-1)})
    return prims


MATERIALS = {   # roughness, double sided, emissive colour (linear) for the glow family
    "bark": (0.92, False, None), "leaf": (0.7, True, None), "plant": (0.8, True, None), "grass": (0.85, True, None),
    "mush": (0.6, False, None), "rock": (0.9, False, None), "terrain": (0.95, False, None),
    "glow_teal": (0.45, False, (0.10, 0.75, 0.62)), "glow_violet": (0.45, False, (0.45, 0.20, 0.95)), "glow_gold": (0.45, False, (0.85, 0.62, 0.16)),
}


# ------------------------------------------------------------------ terrain
def terrain_mesh():
    n = int(2 * HALF / STEP) + 1
    xs = np.linspace(-HALF, HALF, n)
    X, Z = np.meshgrid(xs, xs)
    H = W.height(X, Z)
    r = np.hypot(X, Z)
    p = np.stack([X, np.zeros_like(X), Z], axis=-1)
    patch = W.vnoise(p * 0.11, 21.0)
    fine = W.vnoise(p * 1.3, 5.0)
    # the meadow floor under the grass: deep green, mossier and bluer toward the forest
    meadow = W.blend([0.12, 0.22, 0.07], [0.24, 0.34, 0.11], patch)
    moss = W.blend([0.07, 0.17, 0.10], [0.13, 0.26, 0.15], patch)
    col = meadow + (moss - meadow) * W.sstep(18, 36, r)[..., None]
    col = col * (0.88 + 0.24 * fine[..., None])
    # earth paths with worn edges and scattered grit
    edge = W.path_distance(X, Z) + (W.vnoise(p * 0.7, 2.0) - 0.5) * 1.0
    on_path = 1 - W.sstep(0.6, 1.6, edge)
    dirt = W.blend([0.33, 0.25, 0.17], [0.50, 0.40, 0.28], W.vnoise(p * 2.2, 8.0))
    dirt = dirt * (0.85 + 0.3 * W.vnoise(p * 6.0, 11.0)[..., None])
    col = col + (dirt - col) * on_path[..., None]
    # bare trampled ground round the altar and the door
    trampled = (1 - W.sstep(2.0, 5.5, np.hypot(X, Z + 3.5))) * 0.55
    col = col + (W.blend([0.30, 0.27, 0.18], [0.40, 0.36, 0.24], fine) - col) * trampled[..., None]
    col = col + (np.array([0.10, 0.18, 0.12]) - col) * W.sstep(62, 76, r)[..., None]   # the rim fades into dark woodland
    verts = np.stack([X, H, Z], axis=-1).reshape(-1, 3)
    # normals straight from the height field
    gz, gx = np.gradient(H, STEP)
    nor = np.stack([-gx, np.ones_like(gx), -gz], axis=-1).reshape(-1, 3)
    nor /= np.linalg.norm(nor, axis=1, keepdims=True)
    faces = []
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    faces = np.concatenate([np.stack([a, c, b], axis=1), np.stack([a, d, c], axis=1)])
    return verts, nor, np.clip(col.reshape(-1, 3), 0, 1), faces, H, on_path


# ------------------------------------------------------------------ placement helpers
class Placer:
    def __init__(self, colliders, on_path_grid):
        self.boxes = colliders
        self.on_path = on_path_grid
        self.n = on_path_grid.shape[0]

    def path_amount(self, x, z):
        i = np.clip(np.round((z + HALF) / STEP).astype(int), 0, self.n - 1)
        j = np.clip(np.round((x + HALF) / STEP).astype(int), 0, self.n - 1)
        return self.on_path[i, j]

    def inside_box(self, x, z, pad=0.15):
        hit = np.zeros(np.shape(x), dtype=bool)
        for b in self.boxes:
            if "a" not in b and b["hx"] < 0.25:          # thin tree colliders still leave room at the foot
                pad_b = pad
            else:
                pad_b = pad
            c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
            dx, dz = x - b["x"], z - b["z"]
            lx, lz = dx * c - dz * s, dx * s + dz * c
            hit |= (np.abs(lx) < b["hx"] + pad_b) & (np.abs(lz) < b["hz"] + pad_b)
        return hit


def matrix(x, y, z, yaw, scale, tilt=(0.0, 0.0)):
    """3x4 row-major transform: scale, then lean (around x then z), then turn, then move."""
    sx, sy, sz = (scale, scale, scale) if np.isscalar(scale) else scale
    cy, sny = math.cos(yaw), math.sin(yaw)
    ax, az = tilt
    cx, snx, cz, snz = math.cos(ax), math.sin(ax), math.cos(az), math.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -snx], [0, snx, cx]])
    rz = np.array([[cz, -snz, 0], [snz, cz, 0], [0, 0, 1]])
    ry = np.array([[cy, 0, sny], [0, 1, 0], [-sny, 0, cy]])
    m = ry @ rz @ rx @ np.diag([sx, sy, sz])
    return np.concatenate([np.hstack([m, [[x], [y], [z]]]).ravel()])


# ------------------------------------------------------------------ main
def main():
    world = json.loads((LIVE / "world.json").read_text(encoding="utf-8"))
    src = Glb(str(LIVE / "world.glb"))
    out = Glb()
    out.json["asset"]["generator"] = "build_forest.py"
    out.json["extensionsUsed"] = ["KHR_materials_emissive_strength"]
    copier = Copier(src, out)
    kept = []
    for i, node in enumerate(src.json["nodes"]):
        if node.get("name", "").startswith(KEEP):
            kept.append(copier.node(i))
    print(f"kept {len(kept)} nodes from the live world (stones, peaks, moon)")

    # materials shared by the new geometry
    mat_index = {}

    def material(name):
        if name not in mat_index:
            rough, double, glow = MATERIALS[name]
            m = {"name": name, "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": rough}, "doubleSided": double}
            if glow:
                m["emissiveFactor"] = list(glow)
                m["extensions"] = {"KHR_materials_emissive_strength": {"emissiveStrength": 1.0}}
            out.json["materials"].append(m)
            mat_index[name] = len(out.json["materials"]) - 1
        return mat_index[name]

    def add_mesh(name, prims):
        prim_json = []
        for p in prims:
            col = p["col"]
            if col.dtype != np.uint8:      # 8-bit colour is plenty for painted vertices
                col = np.round(np.clip(col.astype(np.float64) / (65535 if col.dtype == np.uint16 else 1), 0, 1) * 255).astype(np.uint8)
            if col.shape[1] == 3:
                col = np.concatenate([col, np.full((len(col), 1), 255, np.uint8)], axis=1)
            idx = p["idx"]
            prim_json.append({"attributes": {
                "POSITION": out.add_accessor(p["pos"].astype(np.float32), "VEC3", minmax=True, target=34962),
                "NORMAL": out.add_accessor(p["nor"].astype(np.float32), "VEC3", target=34962),
                "COLOR_0": out.add_accessor(col, "VEC4", target=34962, normalized=True)},
                "indices": out.add_accessor(idx.astype(np.uint16 if len(p["pos"]) < 65536 else np.uint32), "SCALAR", target=34963),
                "material": material(p["material"])})
        out.json["meshes"].append({"name": name, "primitives": prim_json})
        out.json["nodes"].append({"name": name, "mesh": len(out.json["meshes"]) - 1})

    # terrain
    tv, tn, tc, tf, H, on_path = terrain_mesh()
    linear = np.where(tc <= 0.04045, tc / 12.92, ((tc + 0.055) / 1.055) ** 2.4)
    add_mesh("terrain-col", [{"material": "terrain", "pos": tv, "nor": tn, "col": linear.astype(np.float32), "idx": tf.reshape(-1)}])
    print(f"terrain: {len(tv)} verts, {len(tf)} tris")

    # prototypes
    names = sorted(p.stem for p in FLORA.glob("*.glb"))
    for name in names:
        add_mesh("proto_" + name, read_asset(name))

    placer = Placer(world["colliders"], on_path)
    groups = {}          # mesh -> list of 15-float rows

    def put(proto, x, z, yaw, scale, tint=(1, 1, 1), sink=0.02, tilt=(0.0, 0.0), y=None):
        y = float(W.height(x, z)) - sink if y is None else y
        groups.setdefault(proto, []).append(np.concatenate([matrix(x, y, z, yaw, scale, tilt), tint]))

    def scatter(count, r_lo, r_hi, ok, tries=None):
        """Random points in the ring r_lo..r_hi that pass ok(x, z) (vectorised)."""
        tries = tries or count * 6
        r = np.sqrt(rng.uniform(r_lo ** 2, r_hi ** 2, tries))
        a = rng.uniform(0, 2 * math.pi, tries)
        x, z = r * np.cos(a), r * np.sin(a)
        keep = ok(x, z)
        return x[keep][:count], z[keep][:count]

    glades = np.array(W.MEMORY_SPOTS[1:])

    def near_glade(x, z, radius):
        return np.min(np.hypot(x[:, None] - glades[:, 0], z[:, None] - glades[:, 1]), axis=1) < radius

    # trees: the same trunks, on the same spots, as before
    trees = [c for c in world["colliders"] if "a" not in c and c["hx"] > 0.2]
    painted = ["tree_teal", "tree_violet", "tree_magenta"]
    green = ["tree_green", "tree_green_b"]
    trunk_r = {"tree_teal": 0.42, "tree_violet": 0.40, "tree_magenta": 0.38, "tree_green": 0.40, "tree_green_b": 0.36,
               "tree_giant": 0.62, "tree_giant_b": 0.58}
    tree_xz = []
    for k, t in enumerate(trees):
        r = math.hypot(t["x"], t["z"])
        radius = t["hx"] / 0.95
        if k < 4:
            kind = "tree_giant" if k % 2 == 0 else "tree_giant_b"
        else:
            family = painted if rng.random() < (0.8 if r < 38 else 0.45) else green   # colour near the henge, green further out
            kind = family[int(rng.integers(0, len(family)))]
        scale = float(np.clip(radius / trunk_r[kind], 0.75, 1.4))
        put(kind, t["x"], t["z"], rng.uniform(0, 2 * math.pi), scale, sink=0.05)
        tree_xz.append((t["x"], t["z"]))
    # a ring of woodland on the rim, beyond where Robin can walk, so the valley ends in trees
    rim_x, rim_z = scatter(130, 67, 79, lambda x, z: np.ones(len(x), dtype=bool), tries=3000)
    taken = [np.array(p) for p in tree_xz]
    for x, z in zip(rim_x, rim_z):
        if any(np.hypot(x - p[0], z - p[1]) < 5.5 for p in taken):
            continue
        taken.append(np.array([x, z]))
        kind = (green + green + painted)[int(rng.integers(0, 7))]
        put(kind, x, z, rng.uniform(0, 2 * math.pi), rng.uniform(1.0, 1.5), sink=0.1)
    tree_xz = np.array(tree_xz)

    def near_tree(x, z, lo, hi):
        d = np.min(np.hypot(x[:, None] - tree_xz[:, 0], z[:, None] - tree_xz[:, 1]), axis=1)
        return (d > lo) & (d < hi)

    def open_ground(x, z, path_ok=0.15, pad=0.2):
        return (placer.path_amount(x, z) < path_ok) & ~placer.inside_box(x, z, pad)

    # grass: a carpet everywhere but the paths and the stones' feet, thinning out at the rim
    gx, gz = scatter(40000, 0.0, PLAY_R, lambda x, z: open_ground(x, z, 0.35, 0.05), tries=80000)
    for x, z in zip(gx, gz):
        r = math.hypot(x, z)
        proto = ("grass_a", "grass_b", "grass_c", "grass_a", "grass_teal" if r > 20 else "grass_b")[int(rng.integers(0, 5))]
        tone = rng.uniform(0.82, 1.15)
        tint = np.array([tone * rng.uniform(0.92, 1.06), tone, tone * rng.uniform(0.88, 1.02)])
        put(proto, x, z, rng.uniform(0, 2 * math.pi), rng.uniform(0.8, 1.35), tint, sink=0.03)
    # grass leaning into the path edges
    ex, ez = scatter(2600, 0.0, PLAY_R, lambda x, z: (placer.path_amount(x, z) > 0.2) & (placer.path_amount(x, z) < 0.6) & ~placer.inside_box(x, z, 0.05), tries=40000)
    for x, z in zip(ex, ez):
        put("grass_b", x, z, rng.uniform(0, 2 * math.pi), rng.uniform(0.6, 1.0), (0.95, 0.95, 0.85), sink=0.03)
    print(f"grass: {len(gx) + len(ex)} tufts")

    # flowers in drifts: pick drift centres, then scatter around them
    centres_x, centres_z = scatter(70, 6, PLAY_R - 4, lambda x, z: open_ground(x, z), tries=2000)
    flowers = ["flower_pink", "flower_violet", "flower_gold"]
    for cx, cz in zip(centres_x, centres_z):
        kind = flowers[int(rng.integers(0, 3))]
        n = int(rng.integers(6, 14))
        fx = cx + rng.normal(0, 1.6, n)
        fz = cz + rng.normal(0, 1.6, n)
        ok = open_ground(fx, fz)
        for x, z in zip(fx[ok], fz[ok]):
            put(kind, x, z, rng.uniform(0, 2 * math.pi), rng.uniform(0.8, 1.35))
    # ferns, curls and blade clumps under and around the trees
    for kind, count, lo, hi, size in (("fern_teal", 260, 1.0, 6.5, (0.8, 1.5)), ("fern_green", 220, 1.0, 7.0, (0.8, 1.5)),
                                       ("curl_violet", 140, 1.2, 5.0, (0.8, 1.4)), ("curl_teal", 110, 1.2, 5.0, (0.8, 1.4)),
                                       ("blades_teal", 160, 1.5, 8.0, (0.7, 1.3)), ("blades_violet", 140, 1.5, 8.0, (0.7, 1.3))):
        x, z = scatter(count, 12, PLAY_R, lambda x, z: open_ground(x, z) & near_tree(x, z, lo, hi), tries=count * 30)
        for xx, zz in zip(x, z):
            put(kind, xx, zz, rng.uniform(0, 2 * math.pi), rng.uniform(*size), sink=0.03)
    # a few curls and blades round the henge itself, like the foreground of the painting
    x, z = scatter(60, 6, 16, lambda x, z: open_ground(x, z, 0.1, 0.4), tries=3000)
    for xx, zz in zip(x, z):
        put(("curl_violet", "blades_teal", "fern_teal", "blades_violet")[int(rng.integers(0, 4))], xx, zz, rng.uniform(0, 6.28), rng.uniform(0.7, 1.1))

    # mushrooms: clusters at tree feet, glowing ones in the shade and in the glades
    for kind, count, lo, hi, size in (("mush_tan", 120, 0.6, 2.6, (0.8, 1.6)), ("mush_orange", 90, 0.6, 2.6, (0.8, 1.6)),
                                       ("mush_glow_teal", 110, 0.8, 4.0, (0.8, 1.7)), ("mush_glow_small", 140, 0.5, 3.0, (0.9, 1.8)),
                                       ("mush_gold", 40, 1.0, 5.0, (0.8, 1.3))):
        x, z = scatter(count, 10, PLAY_R, lambda x, z: open_ground(x, z) & near_tree(x, z, lo, hi), tries=count * 40)
        for xx, zz in zip(x, z):
            put(kind, xx, zz, rng.uniform(0, 2 * math.pi), rng.uniform(*size), sink=0.02)
    for gxz in glades:          # glowing rings round the memory stones' glades
        for k in range(14):
            a = k / 14 * 2 * math.pi + rng.uniform(-0.15, 0.15)
            x, z = gxz[0] + math.cos(a) * rng.uniform(4.2, 6.0), gxz[1] + math.sin(a) * rng.uniform(4.2, 6.0)
            put(("mush_glow_teal", "mush_glow_small", "mush_gold")[k % 3], x, z, rng.uniform(0, 6.28), rng.uniform(1.0, 1.6))
    # giant violet mushrooms stand like lanterns along the paths out of the circle
    x, z = scatter(22, 16, PLAY_R - 3, lambda x, z: (placer.path_amount(x, z) < 0.05) & (W.path_distance(x, z) < 4.5) & ~placer.inside_box(x, z, 0.8), tries=6000)
    giants = []
    for xx, zz in zip(x, z):
        s = rng.uniform(1.0, 1.7)
        put("mush_glow_violet", xx, zz, rng.uniform(0, 6.28), s, sink=0.05)
        giants.append({"x": round(float(xx), 3), "z": round(float(zz), 3), "yaw": 0.0, "hx": round(0.1 * s, 3), "hz": round(0.1 * s, 3),
                       "y0": round(float(W.height(xx, zz)) - 0.3, 3), "y1": round(float(W.height(xx, zz)) + 1.1 * s, 3), "a": 1})

    # pebbles along path edges and among the stones
    x, z = scatter(260, 0, PLAY_R, lambda x, z: (placer.path_amount(x, z) > 0.25) & (placer.path_amount(x, z) < 0.75) & ~placer.inside_box(x, z, 0.1), tries=60000)
    for xx, zz in zip(x, z):
        put(f"pebble_{int(rng.integers(0, 3))}", xx, zz, rng.uniform(0, 6.28), rng.uniform(0.3, 0.75), sink=0.06,
            tilt=(rng.normal(0, 0.15), rng.normal(0, 0.15)))
    x, z = scatter(260, 4, PLAY_R, lambda x, z: open_ground(x, z) & near_tree(x, z, 1.0, 9.0), tries=20000)
    for xx, zz in zip(x, z):
        put(f"pebble_{int(rng.integers(1, 4))}", xx, zz, rng.uniform(0, 6.28), rng.uniform(0.7, 1.6), sink=0.08)

    # ---------------------------------------------------------- write
    out.json["scenes"] = [{"name": "Forest", "nodes": list(range(len(out.json["nodes"])))}]
    out.json["scene"] = 0
    out.save(str(LIVE / "forest.glb"))
    rows, listing, offset = [], [], 0
    for proto in sorted(groups):
        block = np.array(groups[proto], dtype=np.float32)
        kind = proto.split("_")[0]
        listing.append({"set": kind, "mesh": "proto_" + proto, "offset": offset, "count": len(block),
                        "draw": "each" if kind == "tree" else "chunks",
                        "far": {"grass": 32, "flower": 38, "fern": 48, "curl": 36, "blades": 46, "pebble": 28, "mush": 44}.get(kind, 200)})
        rows.append(block)
        offset += len(block)
    inst = np.concatenate(rows)
    (LIVE / "forest_inst.f32").write_bytes(inst.astype("<f4").tobytes())
    (LIVE / "forest.json").write_text(json.dumps({"instances": listing, "chunk": 24.0}, indent=1), encoding="utf-8")

    world["height"] = {"half": HALF, "size": H.shape[0], "step": STEP, "unit": 0.01,
                       "int16_le_base64": base64.b64encode(np.round(H * 100).astype("<i2").tobytes()).decode("ascii")}
    world["colliders"] = [c for c in world["colliders"]] + giants
    (LIVE / "world.json").write_text(json.dumps(world, separators=(",", ":")), encoding="utf-8")
    total_tris = 0
    for item in listing:
        mesh = next(m for m in out.json["meshes"] if m["name"] == item["mesh"])
        tris = sum(out.json["accessors"][p["indices"]]["count"] // 3 for p in mesh["primitives"])
        total_tris += tris * item["count"]
        print(f"  {item['mesh']:24s} x{item['count']:6d}  ({tris} tris each)")
    print(f"forest.glb {len(out.bin) / 1e6:.1f} MB, {len(inst)} instances, {total_tris / 1e6:.1f} M tris if all drawn")


if __name__ == "__main__":
    main()
