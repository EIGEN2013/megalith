"""Assemble the Megalith world: a rune-carved stone circle on a mossy mound, ringed by a misty
fantasy forest.

Where the pieces come from
    stones and rocks   assets_src/models/*.glb, modelled in Blender from the reference painting
                       (tools/prep_stone_textures.py, then tools/blender_stones.py)
    rune pillar        assets_src/rune_pillar.glb, generated with Higgsfield
    bark, leaves, ground textures   tools/make_nature_textures.py
    terrain, trees, mushrooms, plants, far peaks, moon   generated here

Writes
    hero_game/models/world.glb   everything visible (nodes ending in -col get collision in Godot)
    hero_game/world.json         height map, box colliders and points of interest

Axes: +Y up, north is -Z (the great trilithon), the path leaves the circle to the south (+Z).
Usage: python build_world.py
"""
import base64
import io
import json
import math
import pathlib

import numpy as np
from PIL import Image

from glb import Glb

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "assets_src"
TEX = SRC / "tex"
rng = np.random.default_rng(11)


# ------------------------------------------------------------------ noise
def _hash(ix, iy, iz, seed):
    v = np.sin(ix * 127.1 + iy * 311.7 + iz * 74.7 + seed * 19.19) * 43758.5453
    return v - np.floor(v)


def vnoise(p, seed=0.0):
    p = np.asarray(p, dtype=np.float64)
    i = np.floor(p)
    f = p - i
    u = f * f * (3 - 2 * f)
    x0, y0, z0 = i[..., 0], i[..., 1], i[..., 2]
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]

    def h(dx, dy, dz):
        return _hash(x0 + dx, y0 + dy, z0 + dz, seed)
    a = h(0, 0, 0) * (1 - ux) + h(1, 0, 0) * ux
    b = h(0, 1, 0) * (1 - ux) + h(1, 1, 0) * ux
    c = h(0, 0, 1) * (1 - ux) + h(1, 0, 1) * ux
    d = h(0, 1, 1) * (1 - ux) + h(1, 1, 1) * ux
    return (a * (1 - uy) + b * uy) * (1 - uz) + (c * (1 - uy) + d * uy) * uz


def fbm(p, seed=0.0, octaves=4):
    """Fractal noise, roughly -1..1."""
    p = np.asarray(p, dtype=np.float64)
    total, amp, freq = 0.0, 0.5, 1.0
    for o in range(octaves):
        total = total + amp * (vnoise(p * freq, seed + o * 7.31) * 2 - 1)
        amp *= 0.5
        freq *= 2.0
    return total


def sstep(lo, hi, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def blend(a, b, t):
    return np.asarray(a) + (np.asarray(b) - np.asarray(a)) * np.asarray(t)[..., None]


# ------------------------------------------------------------------ mesh builder
class Mesh:
    """Collects vertex-coloured triangles; optionally with texture coordinates or given normals."""

    def __init__(self, flat=False):
        self.flat = flat
        self.v, self.c, self.f, self.uv, self.n = [], [], [], [], []
        self.count = 0

    def add(self, verts, colors, faces, inner=None, uv=None, normals=None):
        """inner: per-vertex points inside the solid; faces are wound to face away from them."""
        verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        colors = np.broadcast_to(np.asarray(colors, dtype=np.float64), verts.shape).copy()
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3).copy()
        if inner is not None:
            inner = np.broadcast_to(np.asarray(inner, dtype=np.float64), verts.shape)
            tri = verts[faces]
            normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            outward = tri.mean(axis=1) - inner[faces].mean(axis=1)
            flip = np.einsum("ij,ij->i", normal, outward) < 0
            faces[flip] = faces[flip][:, ::-1]
        self.v.append(verts)
        self.c.append(colors)
        self.f.append(faces + self.count)
        self.uv.append(None if uv is None else np.asarray(uv, dtype=np.float64).reshape(-1, 2))
        self.n.append(None if normals is None else np.broadcast_to(np.asarray(normals, dtype=np.float64), verts.shape).copy())
        self.count += len(verts)

    def arrays(self):
        v = np.concatenate(self.v)
        c = np.clip(np.concatenate(self.c), 0, 1)
        f = np.concatenate(self.f)
        uv = np.concatenate(self.uv) if all(u is not None for u in self.uv) else None
        tri = v[f]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        if self.flat:
            v, c = v[f.reshape(-1)], c[f.reshape(-1)]
            uv = uv[f.reshape(-1)] if uv is not None else None
            f = np.arange(len(v)).reshape(-1, 3)
            n = np.repeat(fn, 3, axis=0)
        else:
            n = np.zeros_like(v)
            for k in range(3):
                np.add.at(n, f[:, k], fn)
            start = 0
            for chunk, given in zip(self.v, self.n):
                if given is not None:
                    n[start:start + len(chunk)] = given
                start += len(chunk)
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)   # sRGB -> linear
        return v.astype(np.float32), n.astype(np.float32), linear, f, None if uv is None else uv.astype(np.float32)


def grid_faces(nu, nv):
    """Triangles for an (nv rows) x (nu columns) vertex grid laid out row by row."""
    faces = []
    for j in range(nv - 1):
        for i in range(nu - 1):
            a = j * nu + i
            faces += [[a, a + 1, a + nu + 1], [a, a + nu + 1, a + nu]]
    return np.array(faces)


def icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, dtype=np.float64) / np.linalg.norm(p) for p in v]
    for _ in range(level):
        cache, nf = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[key] = len(v) - 1
            return cache[key]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v), np.array(f)


ICO = {lvl: icosphere(lvl) for lvl in (0, 1, 2)}


def rot_y(points, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.asarray(points, dtype=np.float64)
    return np.stack([p[..., 0] * c + p[..., 2] * s, p[..., 1], -p[..., 0] * s + p[..., 2] * c], axis=-1)


def tube(mesh, points, radii, sides, color_fn, cap=True, wrap=None, tile=2.2):
    """Sweep a ring along a polyline. color_fn(t 0..1 along the tube, ring verts) -> colours.
    wrap: how many times a texture goes round the tube (None = no texture coordinates);
    tile: metres of tube per texture repeat."""
    points = np.asarray(points, dtype=np.float64)
    k = len(points)
    tangents = np.gradient(points, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    ref = np.array([1.0, 0, 0]) if abs(tangents[0][0]) < 0.9 else np.array([0, 0, 1.0])
    n1 = np.cross(tangents[0], ref)
    n1 /= np.linalg.norm(n1)
    ring_n = sides + 1                       # the seam vertex is doubled so the texture can wrap
    ang = np.arange(ring_n) / sides * 2 * math.pi
    along = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]) / tile
    verts, cols, inner, uvs = [], [], [], []
    for i in range(k):
        n1 = n1 - tangents[i] * np.dot(n1, tangents[i])
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(tangents[i], n1)
        ring = points[i] + radii[i] * (np.cos(ang)[:, None] * n1 + np.sin(ang)[:, None] * n2)
        verts.append(ring)
        cols.append(color_fn(i / (k - 1), ring))
        inner.append(np.repeat(points[i][None], ring_n, axis=0))
        uvs.append(np.stack([np.arange(ring_n) / sides * (wrap or 1), np.full(ring_n, along[i])], axis=1))
    faces = []
    for i in range(k - 1):
        for s in range(sides):
            a = i * ring_n + s
            faces += [[a, a + 1, a + 1 + ring_n], [a, a + 1 + ring_n, a + ring_n]]
    verts, cols, inner, uvs = np.concatenate(verts), np.concatenate(cols), np.concatenate(inner), np.concatenate(uvs)
    if cap:
        tip = len(verts)
        verts = np.vstack([verts, points[-1] + tangents[-1] * radii[-1] * 0.6])
        cols = np.vstack([cols, cols[-1]])
        inner = np.vstack([inner, points[-1] - tangents[-1] * 0.01])
        uvs = np.vstack([uvs, [(wrap or 1) * 0.5, along[-1] + 0.05]])
        base = (k - 1) * ring_n
        faces += [[base + s, base + s + 1, tip] for s in range(sides)]
    mesh.add(verts, cols, faces, inner, uv=uvs if wrap else None)


def blob(mesh, center, radii, level, color_fn, noise=0.15, seed=0.0, freq=1.6, yaw=0.0, tilt=0.0):
    sv, sf = ICO[level]
    wobble = 1 + noise * fbm(sv * freq + seed, seed, 3)
    p = sv * wobble[:, None] * np.asarray(radii, dtype=np.float64)
    if tilt:
        c, s = math.cos(tilt), math.sin(tilt)
        p = np.stack([p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c, p[:, 2]], axis=1)
    p = rot_y(p, yaw) + np.asarray(center, dtype=np.float64)
    mesh.add(p, color_fn(sv, p), sf, np.asarray(center, dtype=np.float64))


# ------------------------------------------------------------------ terrain
HALF, STEP = 80, 1.0
GRID = int(2 * HALF / STEP) + 1
PLATEAU_Y = 3.0
GROUND_TILE = 4.0        # metres per repeat of the grass texture
PATHS = [
    [(0, 1.5), (0.6, 6), (-1.4, 12), (-0.4, 18), (2.0, 25), (0.8, 33)],                      # down the mound
    [(0.8, 33), (-9, 39), (-22, 38), (-32, 33), (-38, 28)],                                  # west branch
    [(0.8, 33), (11, 39), (25, 34), (36, 24), (42, 13), (44, 8)],                            # east branch
]
MEMORY_SPOTS = [(-2.9, 5.6), (-39.5, 26.5), (45.5, 6.5)]   # henge, west glade, east glade
MOSS = np.array([0.22, 0.36, 0.16])


def path_distance(x, z):
    x, z = np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64)
    best = np.full(np.broadcast(x, z).shape, 1e9)
    for line in PATHS:
        for (ax, az), (bx, bz) in zip(line, line[1:]):
            dx, dz = bx - ax, bz - az
            t = np.clip(((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz), 0, 1)
            best = np.minimum(best, np.hypot(x - (ax + t * dx), z - (az + t * dz)))
    return best


def height(x, z):
    x, z = np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64)
    r = np.hypot(x, z)
    p = np.stack([x, np.zeros_like(x), z], axis=-1)
    mound = PLATEAU_Y * (1 - sstep(11.5, 30, r))
    rolling = fbm(p * 0.045, 3.0, 3) * 2.2 * sstep(16, 38, r)
    detail = fbm(p * 0.33, 9.0, 3) * 0.16 * (0.35 + 0.65 * sstep(9, 16, r))
    rim = np.maximum(0, r - 62) ** 2 * 0.03
    on_path = 1 - sstep(0.6, 2.2, path_distance(x, z))
    glade = np.zeros_like(r)
    for gx, gz in MEMORY_SPOTS[1:]:
        glade = np.maximum(glade, 1 - sstep(3, 9, np.hypot(x - gx, z - gz)))
    rolling = rolling * (1 - 0.5 * glade)
    return mound + rolling + detail * (1 - 0.7 * on_path) - 0.07 * on_path + rim


def build_terrain():
    xs = np.linspace(-HALF, HALF, GRID)
    X, Z = np.meshgrid(xs, xs)            # row = z index, column = x index
    H = height(X, Z)
    r = np.hypot(X, Z)
    p = np.stack([X, np.zeros_like(X), Z], axis=-1)
    patch = vnoise(p * 0.12, 21.0)
    speck = vnoise(p * 0.9, 5.0)
    grass = blend([0.17, 0.31, 0.13], [0.33, 0.46, 0.19], patch)
    moss = blend([0.08, 0.22, 0.17], [0.15, 0.31, 0.21], patch)
    col = grass + (moss - grass) * sstep(17, 34, r)[..., None]
    col = col * (0.9 + 0.2 * speck[..., None])
    dirt = blend([0.36, 0.29, 0.20], [0.50, 0.41, 0.28], speck)
    edge = path_distance(X, Z) + (vnoise(p * 0.8, 2.0) - 0.5) * 0.9
    col = col + (dirt - col) * (1 - sstep(0.7, 1.7, edge))[..., None]
    col = col + (np.array([0.33, 0.36, 0.40]) - col) * sstep(64, 74, r)[..., None]
    col = col * 1.3                          # the grass texture it is multiplied with averages ~0.78
    verts = np.stack([X, H, Z], axis=-1).reshape(-1, 3)
    mesh = Mesh()
    mesh.add(verts, col.reshape(-1, 3), grid_faces(GRID, GRID), verts - np.array([0, 1.0, 0]),
             uv=np.stack([X / GROUND_TILE, Z / GROUND_TILE], axis=-1).reshape(-1, 2))
    return mesh, H


# ------------------------------------------------------------------ modelled assets
colliders = []
placements = []
# asset: (file, material name(s) may override per placement, has runes that glow)
ASSETS = {
    "trilithon": "models/trilithon.glb", "door_stone": "models/door_stone.glb", "altar": "models/altar.glb",
    "monolith": "models/monolith.glb", "rune_stone": "models/rune_stone.glb", "small_stone": "models/small_stone.glb",
    "rock_spire": "models/rock_spire.glb", "rock_crag": "models/rock_crag.glb", "rune_pillar": "rune_pillar.glb",
}
GLOWING = {"trilithon", "door_stone", "altar", "rune_stone", "small_stone"}
_loaded = {}


def asset(name):
    """Read a modelled GLB, normalised: base on y=0, centred, one unit tall."""
    if name in _loaded:
        return _loaded[name]
    path = SRC / ASSETS[name]
    if not path.exists():
        raise SystemExit(f"missing {path}: run tools/prep_stone_textures.py and tools/blender_stones.py first")
    src = Glb(str(path))
    prim = src.json["meshes"][0]["primitives"][0]
    pos = src.read(prim["attributes"]["POSITION"]).astype(np.float64)
    uv = src.read(prim["attributes"]["TEXCOORD_0"]).astype(np.float32)
    faces = src.read(prim["indices"]).reshape(-1, 3).astype(np.int64)
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    pos = (pos - np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])) / (hi[1] - lo[1])
    if "NORMAL" in prim["attributes"]:
        normals = src.read(prim["attributes"]["NORMAL"]).astype(np.float64)
    else:                                   # the Higgsfield mesh ships without normals
        _, pid = np.unique(np.round(pos, 5), axis=0, return_inverse=True)
        pid = pid.reshape(-1)
        fn = np.cross(pos[faces[:, 1]] - pos[faces[:, 0]], pos[faces[:, 2]] - pos[faces[:, 0]])
        acc = np.zeros((pid.max() + 1, 3))
        for k in range(3):
            np.add.at(acc, pid[faces[:, k]], fn)
        normals = acc[pid]
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    textures = {}
    for kind in ("color", "normal", "emit"):
        file = TEX / f"{name}_{kind}.jpg"
        if file.exists() and (kind != "emit" or name in GLOWING):
            textures[kind] = file.read_bytes()
    if "color" not in textures:             # fall back to the picture packed inside the GLB
        image = Image.open(io.BytesIO(src.view_bytes(src.json["images"][0]["bufferView"]))).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=88)
        textures["color"] = buf.getvalue()
    size = pos.max(axis=0) - pos.min(axis=0)
    _loaded[name] = {"pos": pos.astype(np.float32), "normals": normals.astype(np.float32), "uv": uv, "faces": faces,
                     "tex": textures, "width": float(size[0]), "depth": float(size[2])}
    return _loaded[name]


def box(x, z, yaw, hx, hz, y0, y1, solid=True):
    """A box collider. solid=True also asks the Godot build to create a body for it (meshes
    whose node name ends in -col bring their own)."""
    entry = {"x": round(float(x), 3), "z": round(float(z), 3), "yaw": round(float(yaw), 4), "hx": round(float(hx), 3),
             "hz": round(float(hz), 3), "y0": round(float(y0), 3), "y1": round(float(y1), 3)}
    if solid:
        entry["a"] = 1
    colliders.append(entry)


def place(name, material, x, z, yaw, tall=None, fit=None, sink=0.25, base=None, collide=True):
    """Stand an asset at (x, z). tall: uniform scale to that height; fit: (width, height, depth)."""
    a = asset(name)
    scale = (tall, tall, tall) if fit is None else (fit[0] / a["width"], fit[1], fit[2] / a["depth"])
    base = float(height(x, z)) - sink if base is None else base
    placements.append({"asset": name, "material": material, "x": float(x), "y": base, "z": float(z), "yaw": float(yaw), "scale": scale})
    if collide:
        box(x, z, yaw, a["width"] * scale[0] * 0.4, a["depth"] * scale[2] * 0.4, base, base + scale[1] * 0.93)
    return base


def toward_centre(x, z):
    return math.atan2(-x, -z)


def build_henge():
    def ring_point(angle_deg, radius):
        a = math.radians(angle_deg)
        return radius * math.sin(a), -radius * math.cos(a)

    # Five trilithons in a horseshoe that opens to the south, the tallest in the north.
    for angle, radius, width, tall, deep in ((0, 9.2, 7.0, 7.4, 1.55), (-66, 8.8, 5.8, 5.9, 1.3), (66, 8.8, 5.7, 5.6, 1.3),
                                             (-128, 8.6, 5.3, 5.0, 1.15), (128, 8.6, 5.2, 4.7, 1.15)):
        x, z = ring_point(angle, radius)
        yaw = -math.radians(angle)
        base = place("trilithon", "rune_trilithon", x, z, yaw, fit=(width, tall, deep), base=PLATEAU_Y - 0.3, collide=False)
        for side in (-1, 1):                 # the uprights block the way; the gap between them does not
            off = rot_y(np.array([side * 0.295 * width, 0, 0]), yaw)
            box(x + off[0], z + off[2], yaw, 0.115 * width, deep * 0.42, base, base + tall * 0.85)

    # The door of light and the altar in front of it.
    place("door_stone", "portal", 0.0, -5.6, 0.0, tall=4.5, base=PLATEAU_Y - 0.3)
    altar_base = place("altar", "rune_altar", 0.0, -2.9, 0.0, tall=1.02, base=PLATEAU_Y - 0.1, collide=False)
    box(0.0, -2.9, 0.0, 0.95, 0.5, altar_base, altar_base + 0.92)

    # Rune stones inside the circle, and a looser outer ring of monoliths and capped pillars.
    for x, z, tall, kind in ((2.6, 6.4, 2.4, "rune_stone"), (-4.9, 1.6, 1.8, "small_stone"), (4.6, 2.2, 3.1, "rune_stone"),
                             (-4.2, -3.6, 2.7, "rune_stone"), (4.3, -4.2, 2.2, "small_stone")):
        place(kind, "rune_stone" if kind == "rune_stone" else "rune_small", x, z, toward_centre(x, z) + rng.uniform(-0.3, 0.3), tall=tall)
    for i, angle in enumerate((-158, -98, -34, 31, 97, 160, 202)):
        x, z = ring_point(angle + rng.uniform(-6, 6), rng.uniform(12.6, 14.2))
        if i % 2:
            place("rune_pillar", "stone_pillar", x, z, toward_centre(x, z), tall=rng.uniform(3.2, 4.2), sink=0.15, collide=False)
            box(x, z, 0.0, 0.34, 0.34, height(x, z) - 0.5, height(x, z) + 3.5)
        else:
            place("monolith", "stone_monolith", x, z, toward_centre(x, z) + rng.uniform(-0.4, 0.4), tall=rng.uniform(3.6, 4.8))

    # The three memory stones: one beside the path, two out in forest glades.
    for i, (x, z) in enumerate(MEMORY_SPOTS):
        place("rune_stone", f"memory_{i + 1}", x, z, toward_centre(x, z) if i else 0.35, tall=2.9)
        if i:                                # a ring of knee-high stones marks each glade
            for k in range(7):
                a = k / 7 * 2 * math.pi + rng.uniform(-0.2, 0.2)
                sx, sz = x + 3.4 * math.cos(a), z + 3.4 * math.sin(a)
                place("small_stone", "rune_small", sx, sz, math.atan2(x - sx, z - sz), tall=rng.uniform(0.6, 1.2), sink=0.1)


# ------------------------------------------------------------------ forest
PALETTES = [  # (shadow, light) per leaf colour; index = which cluster of the leaf sheet
    (np.array([0.05, 0.36, 0.40]), np.array([0.25, 0.78, 0.70])),   # teal
    (np.array([0.30, 0.16, 0.50]), np.array([0.62, 0.42, 0.86])),   # violet
    (np.array([0.50, 0.14, 0.42]), np.array([0.86, 0.40, 0.70])),   # magenta
    (np.array([0.10, 0.30, 0.14]), np.array([0.36, 0.60, 0.24])),   # leaf green
]


def bark_tint(t, ring):
    """The bark texture does the work; this only varies it and greens the foot of the trunk."""
    tone = vnoise(ring * np.array([1.5, 0.5, 1.5]), 3.0)
    col = np.array([1.0, 0.95, 0.9]) * (0.72 + 0.4 * tone)[:, None]
    mossy = np.clip(1 - t * 5, 0, 1) * (0.35 + 0.5 * vnoise(ring * 2.0, 9.0))
    return col + (MOSS * 1.9 - col) * mossy[:, None]


def leaf_pad(leaves, centre, radius, palette):
    """A crown of leaves: a handful of tilted cards, each showing one cluster of the leaf sheet."""
    u0, v0 = (palette % 2) * 0.5, (palette // 2) * 0.5
    tint = 0.55 + 0.45 * PALETTES[palette][1] / PALETTES[palette][1].max()
    for _ in range(int(rng.integers(6, 9))):
        half = radius * rng.uniform(0.75, 1.05)
        corners = np.array([[-half, 0, -half], [half, 0, -half], [half, 0, half], [-half, 0, half]])
        tilt, roll = rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4)
        ct, st, cr, sr = math.cos(tilt), math.sin(tilt), math.cos(roll), math.sin(roll)
        corners = np.stack([corners[:, 0], corners[:, 1] * ct - corners[:, 2] * st, corners[:, 1] * st + corners[:, 2] * ct], axis=1)
        corners = np.stack([corners[:, 0] * cr - corners[:, 1] * sr, corners[:, 0] * sr + corners[:, 1] * cr, corners[:, 2]], axis=1)
        offset = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2), rng.uniform(-0.3, 0.3)]) * radius
        corners = rot_y(corners, rng.uniform(0, 2 * math.pi)) + centre + offset
        # Light the leaves as if they faced the sky, so both sides of a card stay bright.
        normals = np.array([0, 1.0, 0]) + (corners - centre) * (0.35 / half)
        uv = np.array([[u0 + 0.01, v0 + 0.01], [u0 + 0.49, v0 + 0.01], [u0 + 0.49, v0 + 0.49], [u0 + 0.01, v0 + 0.49]])
        leaves.add(corners, tint * rng.uniform(0.82, 1.08), [[0, 1, 2], [0, 2, 3]], uv=uv, normals=normals)


def build_tree(trunks, leaves, x, z, scale, palette):
    ground = float(height(x, z))
    tall = rng.uniform(5.5, 9.0) * scale
    r0 = rng.uniform(0.32, 0.5) * scale
    lean = rng.uniform(0, 2 * math.pi)
    lean_v = np.array([math.cos(lean), 0, math.sin(lean)]) * rng.uniform(0.1, 0.32)
    perp = np.array([-lean_v[2], 0, lean_v[0]]) / np.linalg.norm(lean_v)
    phase, swing = rng.uniform(0, 6.28), rng.uniform(0.3, 0.75) * scale
    ts = np.linspace(0, 1, 11)
    path = np.array([[x, ground - 0.4, z]]) + np.outer(ts * tall, [0, 1, 0]) + np.outer(ts * tall, lean_v) \
        + np.outer(np.sin(ts * math.pi * 1.6 + phase) * swing * ts, perp)
    radii = r0 * ((1 - ts) ** 0.8 * 0.85 + 0.15) * (1 + 0.9 * np.exp(-ts * 9))
    tube(trunks, path, radii, 9, bark_tint, wrap=2)
    box(x, z, 0.0, r0 * 0.95, r0 * 0.95, ground - 0.5, ground + tall * 0.6, solid=False)
    second = (palette + int(rng.integers(1, 4))) % 4
    tips = [(path[-1], 1.25)]
    for _ in range(int(rng.integers(2, 5))):
        start = path[int(rng.uniform(0.5, 0.88) * 10)]
        a = rng.uniform(0, 2 * math.pi)
        out = np.array([math.cos(a), 0, math.sin(a)])
        length = rng.uniform(1.8, 3.6) * scale
        bs = np.linspace(0, 1, 6)
        bpath = start + np.outer(bs * length, out) + np.outer(np.sin(bs * math.pi * 0.5) * length * rng.uniform(0.25, 0.6), [0, 1, 0])
        tube(trunks, bpath, r0 * 0.34 * (1 - bs * 0.7), 6, bark_tint, wrap=1)
        tips.append((bpath[-1], rng.uniform(0.75, 1.05)))
    for tip, size in tips:                   # flat, layered crowns like the painting's
        for layer in range(int(rng.integers(1, 3))):
            radius = rng.uniform(1.6, 2.4) * scale * size * (1 - 0.25 * layer)
            centre = tip + np.array([rng.uniform(-0.5, 0.5), 0.2 + layer * 0.6 * scale, rng.uniform(-0.5, 0.5)])
            leaf_pad(leaves, centre, radius, palette if rng.random() < 0.7 else second)


def build_mushroom(stems, caps, x, z, tall, cap_r, colour, glow):
    ground = float(height(x, z))
    bend = np.array([rng.uniform(-0.2, 0.2), 0, rng.uniform(-0.2, 0.2)]) * tall
    pts = np.array([[x, ground - 0.05, z], [x, ground + tall * 0.5, z], [x, ground + tall, z]]) + np.outer([0, 0.5, 1], bend)
    pale = np.array([0.80, 0.76, 0.66])
    tube(stems, pts, np.array([0.22, 0.16, 0.13]) * cap_r * 1.4 + 0.012, 6, lambda t, ring: np.repeat((pale * (0.7 + 0.3 * t))[None], len(ring), 0), cap=False)

    def cap_colour(unit, p):
        up = np.clip(unit[:, 1], 0, 1)
        spots = sstep(0.62, 0.7, vnoise(p * (9 / max(cap_r, 0.1)), 6.0))
        base = colour * (0.55 + 0.45 * up)[:, None]
        return base + (np.array([1, 1, 0.95]) - base) * (spots * (0.0 if glow else 0.7))[:, None]
    blob(caps, pts[-1] + np.array([0, cap_r * 0.12, 0]), (cap_r, cap_r * 0.5, cap_r), 2 if cap_r > 0.3 else 1, cap_colour, noise=0.06, seed=rng.uniform(0, 9))


def build_tuft(mesh, x, z, colour, size):
    ground = float(height(x, z))
    verts, cols, faces = [], [], []
    blades = int(rng.integers(6, 10))
    for b in range(blades):
        a = b / blades * 2 * math.pi + rng.uniform(-0.3, 0.3)
        out = np.array([math.cos(a), 0, math.sin(a)])
        side = np.array([-out[2], 0, out[0]]) * 0.05 * size
        reach, tall = rng.uniform(0.15, 0.4) * size, rng.uniform(0.3, 0.75) * size
        base = np.array([x, ground - 0.03, z]) + out * 0.03
        i = len(verts)
        verts += [base - side, base + side, base + out * reach * 0.35 + np.array([0, tall * 0.5, 0]) - side * 0.7,
                  base + out * reach * 0.35 + np.array([0, tall * 0.5, 0]) + side * 0.7, base + out * reach + np.array([0, tall, 0])]
        cols += [colour * 0.45, colour * 0.45, colour * 0.8, colour * 0.8, np.minimum(colour * 1.25 + 0.05, 1)]
        faces += [[i, i + 1, i + 3], [i, i + 3, i + 2], [i + 2, i + 3, i + 4]]
    mesh.add(np.array(verts), np.array(cols), faces, normals=np.array([0, 1.0, 0]))


def build_pebble(mesh, x, z, size):
    seed = rng.uniform(0, 100)

    def colour(unit, p):
        tone = vnoise(p * 1.1, seed)
        return blend([0.15, 0.16, 0.20], [0.36, 0.37, 0.42], tone)
    blob(mesh, (x, float(height(x, z)) + size * 0.25, z), (size, size * rng.uniform(0.6, 1.0), size * rng.uniform(0.7, 1.0)), 0,
         colour, noise=0.3, seed=seed, freq=1.1, yaw=rng.uniform(0, 3.1))


def scatter(count, r_lo, r_hi, taken, spacing, path_gap=2.6, tries=4000):
    """Rejection-sample points in an annulus, clear of paths, glades and each other."""
    points = []
    for _ in range(tries):
        if len(points) >= count:
            break
        r, a = math.sqrt(rng.uniform(r_lo ** 2, r_hi ** 2)), rng.uniform(0, 2 * math.pi)
        x, z = r * math.cos(a), r * math.sin(a)
        if path_distance(x, z) < path_gap:
            continue
        if any(math.hypot(x - gx, z - gz) < 6.5 for gx, gz in MEMORY_SPOTS[1:]):
            continue
        if any(math.hypot(x - px, z - pz) < spacing for px, pz in taken + points):
            continue
        points.append((x, z))
    return points


# ------------------------------------------------------------------ assemble
def main():
    terrain, heights = build_terrain()
    trunks, leaves, flora = Mesh(), Mesh(), Mesh()
    stems, caps, pebbles = Mesh(), Mesh(), Mesh(flat=True)
    glow_teal, glow_violet = Mesh(), Mesh()
    mountains, moon = Mesh(flat=True), Mesh()

    build_henge()

    # Trees: giants frame the southern approach, the rest fill the valley.
    tree_points = [(-17.5, 15.0), (18.5, 12.5), (-21.0, -9.0), (20.0, -13.0)]
    for i, (x, z) in enumerate(tree_points):
        build_tree(trunks, leaves, x, z, 1.45 if i < 2 else 1.2, (1, 0, 2, 0)[i])
    forest = scatter(78, 20, 76, tree_points, 6.0)
    for x, z in forest:
        build_tree(trunks, leaves, x, z, rng.uniform(0.8, 1.3), int(rng.choice([0, 0, 1, 1, 2, 3, 3])))
    tree_points += forest

    # Rocks: the two modelled crags, in many sizes; big ones flank the path up the mound.
    rock_points = [(-6.5, 16.5), (7.5, 14.0), (-11.5, 9.5), (12.5, 6.0)] + scatter(30, 13, 70, tree_points, 3.0, path_gap=2.2)
    for i, (x, z) in enumerate(rock_points):
        tall = rng.uniform(2.4, 3.6) if i < 4 else rng.uniform(0.8, 3.0)
        kind = "rock_spire" if i % 2 == 0 else "rock_crag"
        place(kind, "stone_" + kind[5:], x, z, rng.uniform(0, 2 * math.pi), tall=tall, sink=0.12 * tall, collide=False)
        if tall > 1.1:
            a = asset(kind)
            box(x, z, 0.0, a["width"] * tall * 0.3, a["width"] * tall * 0.3, height(x, z) - 0.5, height(x, z) + tall * 0.7)
    for x, z in scatter(60, 4, 70, [], 1.5, path_gap=1.2):
        build_pebble(pebbles, x, z, rng.uniform(0.12, 0.35))

    tan, orange = np.array([0.80, 0.62, 0.36]), np.array([0.86, 0.45, 0.20])
    teal, violet = np.array([0.35, 0.95, 0.85]), np.array([0.72, 0.45, 1.0])
    spots = scatter(70, 5, 70, [], 2.2, path_gap=1.0) + [(x + rng.uniform(-3, 3), z + rng.uniform(-3, 3)) for x, z in MEMORY_SPOTS[1:] for _ in range(5)]
    for x, z in spots:
        kind = rng.choice(["tan", "orange", "teal", "violet"], p=[0.25, 0.2, 0.3, 0.25])
        giant = rng.random() < 0.12
        for _ in range(int(rng.integers(2, 5))):
            mx, mz = x + rng.uniform(-0.5, 0.5), z + rng.uniform(-0.5, 0.5)
            tall = rng.uniform(0.9, 1.7) if giant else rng.uniform(0.14, 0.45)
            cap_r = tall * rng.uniform(0.38, 0.6)
            if kind in ("teal", "violet"):
                build_mushroom(stems, glow_teal if kind == "teal" else glow_violet, mx, mz, tall, cap_r, teal if kind == "teal" else violet, True)
            else:
                build_mushroom(stems, caps, mx, mz, tall, cap_r, tan if kind == "tan" else orange, False)

    tuft_colours = [np.array([0.18, 0.62, 0.55]), np.array([0.55, 0.32, 0.75]), np.array([0.30, 0.60, 0.22]),
                    np.array([0.78, 0.36, 0.60]), np.array([0.16, 0.45, 0.50])]
    for x, z in scatter(520, 3.5, 72, [], 0.9, path_gap=0.9, tries=24000):
        build_tuft(flora, x, z, tuft_colours[int(rng.integers(0, 5))], rng.uniform(0.7, 1.6))

    # Far spires and the daytime moon.
    for k in range(20):
        a = k / 20 * 2 * math.pi + rng.uniform(-0.12, 0.12)
        dist, tall, width = rng.uniform(115, 165), rng.uniform(45, 95), rng.uniform(14, 26)
        cx, cz = dist * math.cos(a), dist * math.sin(a)
        ts = np.linspace(0, 1, 7)
        spine = np.array([[cx, -6, cz]]) + np.outer(ts * tall, [0, 1, 0]) + np.outer(np.sin(ts * 3 + k) * width * 0.25 * ts, [math.cos(a + 1.5), 0, math.sin(a + 1.5)])
        seed = rng.uniform(0, 50)

        def spire_colour(t, ring, seed=seed):
            tone = vnoise(ring * 0.08, seed)
            return np.array([0.34, 0.40, 0.50])[None] * (0.8 + 0.35 * tone[:, None]) + np.array([0.12, 0.14, 0.16]) * t
        tube(mountains, spine, width * (1 - ts) ** 0.9 * (1 + 0.25 * np.sin(ts * 9 + k)) + 0.4, 7, spire_colour)
    direction = np.array([0.22, 0.44, -0.87])
    direction /= np.linalg.norm(direction)
    centre = direction * 300
    right = np.cross(direction, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, direction)
    ring = [centre + 26 * (math.cos(t) * right + math.sin(t) * up) for t in np.linspace(0, 2 * math.pi, 40, endpoint=False)]
    moon.add(np.vstack([centre] + ring), np.array([0.93, 0.96, 1.0]), [[0, 1 + i, 1 + (i + 1) % 40] for i in range(40)], centre + direction * 5)

    # ---------------------------------------------------------- write the GLB
    out = Glb()
    O = out.json
    O["asset"]["generator"] = "build_world.py"
    O["extensionsUsed"] = ["KHR_materials_emissive_strength"]
    O["scene"] = 0
    O["nodes"], O["meshes"], O["materials"], O["images"], O["textures"] = [], [], [], [], []
    O["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
    texture_cache = {}

    def texture(data, mime):
        key = hash(data)
        if key not in texture_cache:
            O["images"].append({"bufferView": out.append_view(data), "mimeType": mime})
            O["textures"].append({"source": len(O["images"]) - 1, "sampler": 0})
            texture_cache[key] = len(O["textures"]) - 1
        return {"index": texture_cache[key]}

    def material(name, base=None, normal=None, emit=None, emissive=None, strength=1.0, double_sided=False, cutout=None, roughness=0.95):
        m = {"name": name, "doubleSided": double_sided,
             "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": roughness}}
        if base is not None:
            m["pbrMetallicRoughness"]["baseColorTexture"] = base
        if normal is not None:
            m["normalTexture"] = normal
        if cutout is not None:
            m["alphaMode"], m["alphaCutoff"] = "MASK", cutout
        if emit is not None or emissive is not None:
            if emit is not None:
                m["emissiveTexture"] = emit
            else:
                m["pbrMetallicRoughness"]["baseColorFactor"] = [0.05, 0.05, 0.05, 1]
            m["emissiveFactor"] = [float(v) for v in (emissive or (1, 1, 1))]
            m["extensions"] = {"KHR_materials_emissive_strength": {"emissiveStrength": strength}}
        O["materials"].append(m)
        return len(O["materials"]) - 1

    def index_accessor(faces, vertex_count):
        return out.add_accessor(faces.reshape(-1).astype(np.uint16 if vertex_count < 65536 else np.uint32), "SCALAR", target=34963)

    def node(name, mesh, mat):
        if mesh.count == 0:
            return
        v, n, c, f, uv = mesh.arrays()
        colour = np.concatenate([c, np.ones((len(c), 1))], axis=1)
        attributes = {
            "POSITION": out.add_accessor(v, "VEC3", minmax=True, target=34962),
            "NORMAL": out.add_accessor(n, "VEC3", target=34962),
            "COLOR_0": out.add_accessor(np.round(np.clip(colour, 0, 1) * 65535).astype(np.uint16), "VEC4", target=34962, normalized=True),
        }
        if uv is not None:
            attributes["TEXCOORD_0"] = out.add_accessor(uv, "VEC2", target=34962)
        O["meshes"].append({"name": name.replace("-col", ""), "primitives": [
            {"attributes": attributes, "indices": index_accessor(f, len(v)), "material": mat}]})
        O["nodes"].append({"name": name, "mesh": len(O["meshes"]) - 1})
        print(f"  {name:16s} {len(v):6d} verts {len(f):6d} tris")

    def jpg(name):
        return texture((TEX / name).read_bytes(), "image/jpeg")

    matte = material("matte")
    node("terrain-col", terrain, material("ground", base=jpg("ground.jpg")))
    node("trunks-col", trunks, material("bark", base=jpg("bark_color.jpg"), normal=jpg("bark_normal.jpg"), roughness=0.9))
    node("leaves", leaves, material("leaves", base=texture((TEX / "leaves.png").read_bytes(), "image/png"), double_sided=True, cutout=0.45, roughness=0.8))
    node("pebbles", pebbles, matte)
    node("flora", flora, material("leaf", double_sided=True))
    node("mushroom_stems", stems, matte)
    node("mushroom_caps", caps, matte)
    node("mountains", mountains, material("far"))
    node("moon", moon, material("moon", emissive=(0.93, 0.96, 1.0), strength=1.0))
    node("glow_teal", glow_teal, material("glow_teal", emissive=(0.25, 0.95, 0.80), strength=1.6))
    node("glow_violet", glow_violet, material("glow_violet", emissive=(0.62, 0.38, 1.0), strength=1.6))

    # Modelled assets: one mesh per asset and material, placed as many times as the layout asks.
    geometry, meshes, counts = {}, {}, {}
    for spot in placements:
        name, mat_name = spot["asset"], spot["material"]
        a = asset(name)
        if name not in geometry:
            geometry[name] = {"attributes": {"POSITION": out.add_accessor(a["pos"], "VEC3", minmax=True, target=34962),
                                             "NORMAL": out.add_accessor(a["normals"], "VEC3", target=34962),
                                             "TEXCOORD_0": out.add_accessor(a["uv"], "VEC2", target=34962)},
                              "indices": index_accessor(a["faces"], len(a["pos"]))}
        if (name, mat_name) not in meshes:
            tex = a["tex"]
            mat = material(mat_name, base=texture(tex["color"], "image/jpeg"),
                           normal=texture(tex["normal"], "image/jpeg") if "normal" in tex else None,
                           emit=texture(tex["emit"], "image/jpeg") if "emit" in tex else None, strength=2.0)
            O["meshes"].append({"name": f"{name}_{mat_name}", "primitives": [dict(geometry[name], material=mat)]})
            meshes[(name, mat_name)] = len(O["meshes"]) - 1
        counts[name] = counts.get(name, 0) + 1
        O["nodes"].append({"name": f"{name}_{counts[name]}", "mesh": meshes[(name, mat_name)],
                           "translation": [spot["x"], spot["y"], spot["z"]],
                           "rotation": [0.0, math.sin(spot["yaw"] / 2), 0.0, math.cos(spot["yaw"] / 2)],
                           "scale": [float(s) for s in spot["scale"]]})
    for name, count in counts.items():
        print(f"  {name:16s} {len(asset(name)['pos']):6d} verts {len(asset(name)['faces']):6d} tris x {count}")

    O["scenes"] = [{"name": "World", "nodes": list(range(len(O["nodes"])))}]
    target = ROOT / "hero_game" / "models" / "world.glb"
    out.save(str(target))
    print("saved", target, f"{len(out.bin) / 1e6:.2f} MB")

    # ---------------------------------------------------------- world.json
    centimetres = np.round(heights * 100).astype("<i2")
    interact = [
        {"id": "altar", "x": 0.0, "z": -2.9, "radius": 2.4},
        {"id": "door", "x": 0.0, "z": -5.6, "radius": 2.3},
    ] + [{"id": f"memory_{i + 1}", "x": round(x, 2), "z": round(z, 2), "radius": 2.4} for i, (x, z) in enumerate(MEMORY_SPOTS)]
    for item in interact:
        item["y"] = round(float(height(item["x"], item["z"])), 2)
    world = {
        "spawn": {"x": 0.0, "z": 1.8, "heading": 0.0},
        "bounds": 69.0,
        "plateau_radius": 15.0,
        "height": {"half": HALF, "size": GRID, "step": STEP, "unit": 0.01, "int16_le_base64": base64.b64encode(centimetres.tobytes()).decode("ascii")},
        "colliders": colliders,
        "interact": interact,
    }
    (ROOT / "hero_game" / "world.json").write_text(json.dumps(world, separators=(",", ":")), encoding="utf-8")
    print(f"world.json: {len(colliders)} colliders, {len(interact)} points of interest")


if __name__ == "__main__":
    main()
