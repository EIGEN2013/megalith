"""The enlarged village: seven new buildings, two gatehouses and the low town wall, modelled piece by piece
(see village_kit.py / village_roof.py) and written as

  assets_src/village_ext/village_ext.glb    quantised meshes in the village's own materials (plaster, wood, stone,
                                            iron, glass, thatch, cloth, produce); kit_* meshes are wall pieces
  assets_src/village_ext/village_ext.json   instances (new roofs laid with the village's slate pieces, wall
                                            segments), colliders, the wall outline, gates and building names

Usage: python tools/hq_village.py [building-id ...]     (ids limit the run, for previews)
"""
import base64
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geo_kit as G  # noqa: E402
from geo_kit import T, R, frame, box, prism, quad, lathe, tube, patch, blob, Acc  # noqa: E402
import village_plan as PLAN  # noqa: E402
from village_kit import (Kit, WOOD, WOOD_GREY, WOOD_RAW, IRON, STONE, STONE_GREY, MORTAR, TERRA, THATCH, LEAF, PLASTER,  # noqa: E402
                         PAINT, FLOWERS, TW)
from village_roof import Roof, Instances, C_GAME  # noqa: E402
from village_interior import Room, Furnish, OAK, OAK_DARK, BRASS  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "assets_src" / "village_ext"
OUT.mkdir(parents=True, exist_ok=True)

LAND = json.loads((ROOT / "assets_src" / "hq" / "land.json").read_text())
NEAR = LAND["near"]
NH = np.frombuffer(base64.b64decode(NEAR["int16_le_base64"]), "<i2").astype(float).reshape(NEAR["size"], NEAR["size"]) * NEAR["unit"]


def ground(x, z):
    """Terrain height at game (x, z)."""
    fx = np.clip((np.asarray(x, float) + NEAR["half"]) / NEAR["step"], 0, NEAR["size"] - 1.001)
    fz = np.clip((np.asarray(z, float) + NEAR["half"]) / NEAR["step"], 0, NEAR["size"] - 1.001)
    ix, iz = np.floor(fx).astype(int), np.floor(fz).astype(int)
    tx, tz = fx - ix, fz - iz
    a = NH[iz, ix] * (1 - tx) + NH[iz, ix + 1] * tx
    b = NH[iz + 1, ix] * (1 - tx) + NH[iz + 1, ix + 1] * tx
    return a * (1 - tz) + b * tz


INST = Instances()
COLLIDERS = []


class Site:
    """A building's place in the world: game centre, facing yaw, floor height; local <-> world helpers."""

    def __init__(self, x, z, face, W, D, extra=0.0):
        self.x, self.z = x, z
        self.yaw = PLAN.yaw_of(face)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.c, self.s = c, s
        xs = np.linspace(-W / 2 - extra, W / 2 + extra, 7)
        ys = np.linspace(-D / 2 - extra, D / 2 + extra, 7)
        hs = [self.gz_world(lx, ly) for lx in xs for ly in ys]
        self.floor = max(hs) + 0.12
        self.low = min(hs)
        self.Mb = T(x, -z, self.floor) @ R("z", self.yaw)

    def world(self, lx, ly):
        bx = self.x + lx * self.c - ly * self.s
        by = -self.z + lx * self.s + ly * self.c
        return bx, -by           # game x, z

    def gz_world(self, lx, ly):
        gx, gz = self.world(lx, ly)
        return float(ground(gx, gz))

    def gz(self, lx, ly):
        """Terrain height in local z (relative to the floor)."""
        return self.gz_world(lx, ly) - self.floor

    def collider(self, lx, ly, hx, hy, y0=None, y1=3.0, yaw_extra=0.0):
        gx, gz = self.world(lx, ly)
        COLLIDERS.append({"x": round(gx, 3), "z": round(gz, 3), "yaw": round(self.yaw + yaw_extra, 4), "hx": round(hx, 3), "hz": round(hy, 3),
                          "y0": round(self.floor + (y0 if y0 is not None else -2.0), 3), "y1": round(self.floor + y1, 3)})


# ---------------------------------------------------------------- a generic house
INTERIORS = []


def hollow(site, kit, W, D, H, t, door_side, door, lights, name, bid):
    """A walkable ground floor: wall colliders round a doorway, a floor to stand on, stone steps up to the
    threshold, and the room's record for the game (camera bounds, lamp light, name)."""
    x0, x1, y0, y1 = -W / 2, W / 2, -D / 2, D / 2
    du0, du1 = door["u"] - door["w"] / 2, door["u"] + door["w"] / 2
    walls = {"front": ((x0, x1), y0 + t / 2, "x"), "back": ((x0, x1), y1 - t / 2, "x"), "left": ((y0, y1), x0 + t / 2, "y"), "right": ((y0, y1), x1 - t / 2, "y")}
    for side, ((a, b), c, ax) in walls.items():
        spans = [(a, b)]
        if side == door_side:
            if side == "front":
                g0, g1 = x0 + du0, x0 + du1
            elif side == "back":
                g0, g1 = x1 - du1, x1 - du0
            elif side == "left":
                g0, g1 = y1 - du1, y1 - du0
            else:
                g0, g1 = y0 + du0, y0 + du1
            spans = [(a, g0), (g1, b)]
        for s0, s1 in spans:
            if s1 - s0 < 0.05:
                continue
            m = (s0 + s1) / 2
            if ax == "x":
                site.collider(m, c, (s1 - s0) / 2, t / 2 + 0.04, y1=H)
            else:
                site.collider(c, m, t / 2 + 0.04, (s1 - s0) / 2, y1=H)
    site.collider(0, 0, W / 2 - 0.01, D / 2 - 0.01, y0=-2.0, y1=0.0)      # the floor itself
    # steps from the street up to the threshold
    Mw, L = kit.wall_frame(door_side, x0, x1, y0, y1, 0)
    out = Mw @ np.array([door["u"], -0.45, 0, 1.0])
    gz = site.gz(out[0], out[1])
    rise = 0.17
    n = int(math.ceil(max(0.0, -gz - 0.06) / rise))
    for i in range(n):
        top = -(i + 1) * rise + 0.02
        d = 0.34
        yc = -0.25 - d * (i + 0.5) - 0.06
        hgt = top - (gz - 0.3)
        box(kit.acc, "stone", Mw @ T(door["u"], yc, top - hgt / 2), door["w"] / 2 + 0.35 + 0.05 * i, d / 2 + 0.01, hgt / 2,
            kit.tone(STONE_GREY, 0.07), 0.02)
        p = Mw @ np.array([door["u"], yc, 0, 1.0])
        site.collider(p[0], p[1], door["w"] / 2 + 0.35, d / 2, y0=-3.0, y1=top, yaw_extra=0.0 if door_side in ("front", "back") else math.pi / 2)
    gx, gz_ = site.world(0, 0)
    rec = {"id": bid, "name": name, "x": round(gx, 3), "z": round(gz_, 3), "yaw": round(site.yaw, 4), "hx": round(W / 2 - t, 3), "hz": round(D / 2 - t, 3),
           "floor": round(site.floor, 3), "ceil": round(site.floor + H, 3), "lights": []}
    for (lx, ly, lz, col, inten, dist) in lights:
        wx, wz = site.world(lx, ly)
        rec["lights"].append({"x": round(wx, 3), "y": round(site.floor + lz, 3), "z": round(wz, 3), "color": col, "intensity": inten, "distance": dist})
    INTERIORS.append(rec)


def house(kit, roof, site, W, D, storeys, rspec, pcol, wcol=WOOD, scol=STONE, solid=True):
    """Storeys bottom-up; each {'h', 'kind': 'stone'|'timber', 'jetty', 'open': {side: [openings]}, 'style'}."""
    acc = kit.acc
    x0, x1, y0, y1 = -W / 2, W / 2, -D / 2, D / 2
    # plinth down to the ground all round
    low = min(site.gz(x, y) for x in np.linspace(x0, x1, 6) for y in np.linspace(y0, y1, 6)) - 0.35
    ph = -low
    pc = kit.courses(ph + 0.02, coursed=False)
    for i, side in enumerate(("front", "right", "back", "left")):
        Mw, L = kit.wall_frame(side, x0 - 0.06, x1 + 0.06, y0 - 0.06, y1 + 0.06, low)
        if storeys[0]["kind"] == "stone":
            Mw, L = kit.wall_frame(side, x0, x1, y0, y1, low)
        kit.stone_wall(Mw, L, ph + (0.02 if storeys[0]["kind"] == "timber" else 0.0), [], col=scol, courses=pc, parity=i % 2,
                       quoins=True, proud=0.06 if storeys[0]["kind"] == "timber" else 0.05)
    if storeys[0]["kind"] == "timber":   # the plinth's top: a run of flat capping stones
        for side in ("front", "right", "back", "left"):
            Mw, L = kit.wall_frame(side, x0 - 0.08, x1 + 0.08, y0 - 0.08, y1 + 0.08, 0)
            u = 0.0
            while u < L - 0.05:
                bl = min(L - u, kit.rng.uniform(0.5, 0.9))
                box(acc, "stone", Mw @ T(u + bl / 2, 0.06, 0.0), bl / 2 - 0.01, 0.11, 0.035, kit.J(STONE_GREY, 0.07), 0.012)
                u += bl
    z = 0.0
    courses = None
    for si, st in enumerate(storeys):
        j = st.get("jetty", 0.0)
        if si > 0 and j > 0:
            Mw, L = kit.wall_frame("front", x0, x1, y0, y1, 0)
            br = st.get("brackets", [0.2, L - 0.2])
            kit.jetty(Mw, L, j, z, wcol, br)
            y0 -= j
            jb = st.get("jetty_back", 0.0)
            if jb:
                Mw, L = kit.wall_frame("back", x0, x1, y0, y1, 0)
                kit.jetty(Mw, L, jb, z, wcol, [0.2, L - 0.2])
                y1 += jb
        for i, side in enumerate(("front", "right", "back", "left")):
            Mw, L = kit.wall_frame(side, x0, x1, y0, y1, z)
            ops = st.get("open", {}).get(side, [])
            if st["kind"] == "stone":
                if courses is None or st.get("new_courses"):
                    courses = kit.courses(st["h"], coursed=st.get("coursed", False))
                kit.stone_wall(Mw, L, st["h"], ops, col=scol, coursed=st.get("coursed", False), courses=courses, parity=i % 2)
            else:
                kit.timber_wall(Mw, L, st["h"], ops, pcol, wcol, style=st.get("style", "cross"))
            for lt in st.get("lanterns", {}).get(side, []):
                kit.lantern(Mw, lt[0], lt[1])
            for sg in st.get("signs", {}).get(side, []):
                kit.sign(Mw, sg[0], sg[1], sg[2])
        if st["kind"] == "timber" and si > 0:   # a floor beam line between storeys
            for side in ("front", "right", "back", "left"):
                Mw, L = kit.wall_frame(side, x0, x1, y0, y1, z)
                box(acc, "wood", Mw @ T(L / 2, -0.01, -0.02), L / 2 + 0.02, 0.07, 0.06, kit.J(wcol, 0.04), 0.012)
        z += st["h"]
    zt = z
    # roof
    pitch = math.radians(rspec["pitch"])
    if rspec["axis"] == "x":
        Mr = T(0, (y0 + y1) / 2, 0)
        Lr, Sp = x1 - x0, y1 - y0
        gables = (("left", x0, x1, y0, y1), ("right", x0, x1, y0, y1))
    else:
        Mr = frame(((x0 + x1) / 2, (y0 + y1) / 2, 0), (0, 1, 0), (-1, 0, 0), (0, 0, 1))
        Lr, Sp = y1 - y0, x1 - x0
        gables = (("front", x0, x1, y0, y1), ("back", x0, x1, y0, y1))
    excl = []
    info_pre = None
    if rspec.get("dormers"):
        # the dormers need the roof's geometry first: compute it without slates, then lay the slates around them
        tp = math.tan(pitch)
        sh = Sp / 2
        info_pre = {"zr": zt + 0.2 + sh * tp, "sides": {}}
        for sgn in (-1, 1):
            U = np.array([0.0, -sgn * math.cos(pitch), math.sin(pitch)])
            n = np.array([0.0, sgn * math.sin(pitch), math.cos(pitch)])
            A = np.array([1.0, 0, 0]) * (1 if sgn < 0 else -1)
            eave = np.array([0.0, sgn * (sh + rspec.get("ov_e", 0.45)), zt + 0.2 - rspec.get("ov_e", 0.45) * tp])
            info_pre["sides"][sgn] = (eave, U, n, A, (sh + rspec.get("ov_e", 0.45)) / math.cos(pitch))
        for (a_c, b_d) in rspec["dormers"]:
            excl.append(roof.dormer(Mr, info_pre, a_c, b_d, rspec.get("dormer_w", 1.3), rspec.get("dormer_h", 1.35), pitch, pcol, wcol,
                                    shutters=rspec.get("dormer_shutters")))
    for (cx, cy, top_extra) in rspec.get("chimneys", []):
        # chimney positions are building-local; mark them clear of slates (canonical coordinates)
        p_can = np.linalg.inv(Mr) @ np.array([cx, cy, 0, 1.0])
        a, s = p_can[0], p_can[1]
        sgn = -1 if s < 0 else 1
        tp = math.tan(pitch)
        ov_e = rspec.get("ov_e", 0.45)
        b = (Sp / 2 + ov_e - abs(s)) / math.cos(pitch)
        if sgn < 0:
            excl.append((a - 0.55, a + 0.55, b - 0.6, b + 0.6))
        else:
            excl.append((-a - 0.55, -a + 0.55, b - 0.6, b + 0.6))
    info = roof.gable(Mr, Lr, Sp, zt, pitch, ov_e=rspec.get("ov_e", 0.45), ov_g=rspec.get("ov_g", 0.35), kind=rspec.get("kind", "slate"),
                      wcol=wcol, excl=excl, finial=rspec.get("finial", False), thick=rspec.get("thick", 0.32))
    zr = info["zr"]
    # gable ends
    tp = math.tan(pitch)
    for gi, (side, gx0, gx1, gy0, gy1) in enumerate(gables):
        Mw, L = kit.wall_frame(side, gx0, gx1, gy0, gy1, zt)
        hmax = zr - zt - 0.12
        clip = lambda u, L=L, hmax=hmax: min(hmax, (L / 2 - abs(u - L / 2)) * tp + 0.12)
        gstyle = rspec.get("gable", "timber" if storeys[-1]["kind"] == "timber" else "stone")
        gwin = rspec.get("gable_windows", {}).get(side, [])
        if gstyle == "stone":
            kit.stone_wall(Mw, L, hmax, gwin, col=scol, coursed=storeys[-1].get("coursed", False), quoins=False, clip=clip)
        else:
            timber_gable(kit, Mw, L, hmax, tp, gwin, pcol, wcol)
    for (cx, cy, top_extra) in rspec.get("chimneys", []):
        roof.chimney(cx, cy, zt - 0.6, zr + top_extra, col=rspec.get("chimney_col", scol))
    if solid:
        site.collider(0, 0, W / 2, D / 2, y1=zt)
    else:            # the storeys above the walkable room are still solid
        site.collider(0, 0, W / 2, D / 2, y0=storeys[0]["h"] - 0.05, y1=zt)
    return {"zt": zt, "zr": zr, "y0": y0, "y1": y1, "Mr": Mr, "info": info}


def timber_gable(kit, Mw, L, H, tp, wins, pcol, wcol):
    """The gable triangle: plaster in pieces round any window, a tie beam, king post, studs and raking braces."""
    acc = kit.acc

    def piece(ua, ub, va, vb):
        pts = [(max(ua, va / tp), va), (min(ub, L - va / tp), va), (min(ub, L - vb / tp), vb), (max(ua, vb / tp), vb)]
        if pts[1][0] - pts[0][0] < 0.02 and pts[2][0] - pts[3][0] < 0.02:
            return
        pts = [p for i, p in enumerate(pts) if i == 0 or np.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]) > 1e-4]
        if len(pts) < 3:
            return
        Mp = Mw @ frame((0, 0.01, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        prism(acc, "plaster", Mp, pts, 0.02, kit.J(pcol, 0.03), cap_back=False)

    top = L / 2 * tp
    if wins:
        o = wins[0]
        u0, u1, v0, v1 = kit.hole(o)
        piece(0, L, 0, v0)
        piece(0, u0, v0, v1)
        piece(u1, L, v0, v1)
        piece(0, L, v1, top)
        kit.opening(Mw, o, "plaster", pcol)
        for u in (u0 - TW / 2, u1 + TW / 2):
            kit.timber(Mw, (u, TW), (u, min(H, (L / 2 - abs(u - L / 2)) * tp) - 0.05), TW, wcol)
        kit.timber(Mw, (u0 - TW, v0 - TW / 2), (u1 + TW, v0 - TW / 2), TW, wcol)
        kit.timber(Mw, (u0 - TW, v1 + TW / 2), (u1 + TW, v1 + TW / 2), TW, wcol)
    else:
        piece(0, L, 0, top)
        kit.timber(Mw, (L / 2, TW), (L / 2, H - 0.05), TW, wcol)
    kit.timber(Mw, (0, TW / 2), (L, TW / 2), TW * 1.1, wcol)
    for f in (0.22, 0.78):
        u = L * f
        h = (L / 2 - abs(u - L / 2)) * tp
        if h > 0.6 and not (wins and kit.hole(wins[0])[0] - 0.3 < u < kit.hole(wins[0])[1] + 0.3):
            kit.timber(Mw, (u, TW), (u, h - 0.05), TW * 0.85, wcol)
    # raking braces under the roof line
    for sgn in (-1, 1):
        a = (0.35, TW) if sgn < 0 else (L - 0.35, TW)
        b = (L / 2 + sgn * -0.5 * 0, H * 0.55)
        kit.timber(Mw, a, (L / 2 - sgn * 0.35, min(H * 0.6, (L / 2 - 0.35) * tp * 0.8)), TW * 0.8, wcol)


BLOCK_PROTOS = 3


def block_kit(glb):
    """Unit stones bevelled round their faces, in three degrees of wear, for the instanced masonry."""
    for k, b in enumerate((0.07, 0.12, 0.18)):
        acc = Acc()
        G.fbox(acc, "stone", np.eye(4), 0.5, 0.5, 0.5, (1.0, 1.0, 1.0), b)
        glb.add_acc(f"kit_block{k}", acc)


ROOM_REACH = (0.15, 0.3, 0.6, 1.0, 1.6, 2.4, 3.3)


def blocks_out(kit, acc, Mb, ground_z, grime_z0, rng, inner=None):
    """Shade every stone (AO from the whole building, grime near the ground) and record it as an instance.
    `inner` (a room's geometry) is occluded with a longer reach, so a closed room is dim away from its windows."""
    every = acc
    if inner is not None:
        every = Acc()
        every.merge(acc)
        every.merge(inner)
    if not kit.blocks:
        grid = G.occupancy(every)
        G.bake_ao(acc, ground=ground_z, grid=grid)
        if inner is not None:
            G.bake_ao(inner, ground=None, grid=grid, reach=ROOM_REACH, strength=0.82)
        return grid, 0
    Ms = np.array([b[0] for b in kit.blocks])
    cols = np.array([b[1] for b in kit.blocks])
    probe = np.array([[0, -0.5, 0, 1], [0.32, -0.5, 0.32, 1], [-0.32, -0.5, 0.32, 1], [0.32, -0.5, -0.32, 1], [-0.32, -0.5, -0.32, 1], [0, 0, 0, 1.0]])
    extra = np.einsum("nij,kj->nki", Ms, probe)[..., :3].reshape(-1, 3)
    grid = G.occupancy(every, extra)
    G.bake_ao(acc, ground=ground_z, grid=grid)
    if inner is not None:
        G.bake_ao(inner, ground=None, grid=grid, reach=ROOM_REACH, strength=0.82)
    P = np.einsum("nij,j->ni", Ms, np.array([0, -0.5, 0, 1.0]))[:, :3]
    Nf = -Ms[:, :3, 1]
    Nf /= np.linalg.norm(Nf, axis=1, keepdims=True)
    ao = G.ao_at(grid, P + Nf * 0.01, Nf, ground=ground_z)
    gr = 1 - np.clip(1 - (P[:, 2] - grime_z0) / 1.0, 0, 1) ** 1.6 * 0.3
    lin = (cols ** 2.2) * ((0.25 + 0.75 * ao) * gr)[:, None]
    for M, c in zip(Ms, lin):
        W = Mb @ M
        G3 = C_GAME @ W[:3, :3] @ C_GAME.T
        t = C_GAME @ W[:3, 3]
        INST.d.setdefault(f"kit_block{rng.integers(BLOCK_PROTOS)}_stone", []).append(
            [G3[0, 0], G3[0, 1], G3[0, 2], t[0], G3[1, 0], G3[1, 1], G3[1, 2], t[1], G3[2, 0], G3[2, 1], G3[2, 2], t[2], c[0], c[1], c[2]])
    return grid, len(Ms)


def place(acc, site, glb, name, kit=None, inner=None):
    """AO and grime in local space, then into the world and the GLB; the stones become instances."""
    gz = site.low - site.floor - 0.02
    if kit is not None:
        _, n = blocks_out(kit, acc, site.Mb, gz, site.low - site.floor, kit.rng, inner)
        print(f"  {name}: {n} stones")
    else:
        G.bake_ao(acc, ground=gz)
    G.grime(acc, ("plaster", "stone"), z0=site.low - site.floor, height=1.0, amount=0.3)
    if inner is not None:
        acc.merge(inner)
    world = Acc()
    world.merge(acc, site.Mb)
    return glb.add_acc(name, world)


# ---------------------------------------------------------------- the buildings
def tavern(glb):
    rng = np.random.default_rng(11)
    b = next(b for b in PLAN.BUILDINGS if b["id"] == "tavern")
    W, D = 12.0, 8.6
    site = Site(b["x"], b["z"], b["face"], W, D)
    acc = Acc(rng)
    kit = Kit(acc, rng, instanced=True)
    roof = Roof(kit, INST, site.Mb)
    pcol = PLASTER["cream"]
    win = lambda u, w=1.05, h=1.2, s=0.9, sh="green", fl=False: {"kind": "win", "u": u, "w": w, "h": h, "sill": s, "shutters": sh, "flowers": fl}
    gwin = lambda u, w=1.05: {"kind": "win", "u": u, "w": w, "h": 1.2, "sill": 0.9, "depth": 0.5, "inside": True}
    storeys = [
        {"h": 3.1, "kind": "stone", "open": {
            "front": [{"kind": "door", "u": W / 2, "w": 1.5, "h": 2.0, "sill": 0, "arch": "round", "double": True, "open": 98, "depth": 0.5},
                      gwin(2.0), gwin(3.9), gwin(8.1), gwin(10.0)],
            "left": [gwin(6.3, 0.9)], "right": [gwin(6.2, 1.0)],
            "back": [{"kind": "door", "u": 2.2, "w": 1.0, "h": 2.05, "sill": 0, "depth": 0.5}, gwin(6.0, 1.0), gwin(9.5, 1.0)]},
         "lanterns": {"front": [(4.95, 2.45), (7.05, 2.45)]}, "signs": {"front": [(0.55, 2.75, "gull")]}},
        {"h": 2.75, "kind": "timber", "jetty": 0.45, "style": "cross", "brackets": [0.25, 4.0, 8.0, 11.75], "open": {
            "front": [win(1.6, 0.95, 1.15, 0.75, fl=True), win(4.3, 0.95, 1.15, 0.75), win(7.7, 0.95, 1.15, 0.75), win(10.4, 0.95, 1.15, 0.75, fl=True)],
            "left": [win(2.6, 0.9, 1.1, 0.75), win(6.4, 0.9, 1.1, 0.75)], "right": [win(2.6, 0.9, 1.1, 0.75), win(6.4, 0.9, 1.1, 0.75, fl=True)],
            "back": [win(2.0, 0.9, 1.1, 0.75), win(6.0, 0.9, 1.1, 0.75), win(10.0, 0.9, 1.1, 0.75)]}},
        {"h": 2.5, "kind": "timber", "jetty": 0.35, "style": "cross", "brackets": [0.25, 6.0, 11.75], "open": {
            "front": [win(2.3, 0.85, 1.0, 0.7, fl=True), win(6.0, 0.85, 1.0, 0.7), win(9.7, 0.85, 1.0, 0.7, fl=True)],
            "left": [win(4.5, 0.8, 0.95, 0.7)], "right": [win(4.5, 0.8, 0.95, 0.7)],
            "back": [win(3.0, 0.85, 1.0, 0.7), win(9.0, 0.85, 1.0, 0.7)]}},
    ]
    rspec = {"axis": "x", "pitch": 52, "ov_e": 0.45, "ov_g": 0.38, "dormers": [(-3.2, 1.25), (3.2, 1.25)], "dormer_shutters": "green",
             "chimneys": [(-4.6, 1.6, 0.9), (4.2, 1.1, 0.8)], "finial": True,
             "gable_windows": {"left": [win(5.3, 0.75, 0.9, 0.75, None)], "right": [win(5.3, 0.75, 0.9, 0.75, None)]}}
    h = house(kit, roof, site, W, D, storeys, rspec, pcol, solid=False)
    inner = tavern_inside(kit, site, W, D, storeys[0]["open"])
    # the yard: trestle tables, a barrel stack, crates, a bench along the wall
    for (tx, ty) in ((-3.2, -4.2), (3.4, -4.4)):
        gz = site.gz(tx, ty)
        kit.table(T(tx, ty, gz) @ R("z", rng.normal(0, 0.06)), 2.0)
        site.collider(tx, ty, 1.05, 0.95, y0=gz - 1, y1=gz + 0.8)
    for k, (bx, by) in enumerate(((-5.4, -5.1), (-4.75, -5.15), (-5.1, -5.1))):
        gz = site.gz(bx, by)
        kit.barrel(T(bx, by - 0.65 if k == 2 else by, gz + (0.0 if k < 2 else 0.0)) @ R("z", rng.uniform(0, 6)))
    kit.barrel(T(-5.1, -5.75, site.gz(-5.1, -5.75)))
    kit.barrel(T(-5.05, -5.45, site.gz(-5.05, -5.45) + 0.85) @ R("z", 0.4), r=0.27, h=0.75)
    site.collider(-5.1, -5.4, 0.75, 0.7, y0=-2, y1=1.6)
    for (cx, cy, s) in ((5.4, -5.0, 0.55), (5.5, -5.5, 0.45), (5.35, -5.2, 0.4)):
        kit.crate(T(cx, cy, site.gz(cx, cy) + (0.55 if s == 0.4 else 0)) @ R("z", rng.normal(0, 0.2)), s)
    kit.bench(T(-2.0, -D / 2 - 0.35, site.gz(-2.0, -D / 2 - 0.35)), 1.6)
    kit.bench(T(2.6, -D / 2 - 0.35, site.gz(2.6, -D / 2 - 0.35)), 1.6)
    for (sx, sy) in ((5.6, 3.0), (5.7, 2.4)):
        kit.sack(T(sx + 0.4, sy + 1.8, site.gz(sx + 0.4, sy + 1.8)))
    kit.log_pile(T(-W / 2 - 0.55, 1.5, site.gz(-W / 2 - 0.55, 1.5)) @ R("z", math.pi / 2), 1.8, 4)
    return place(acc, site, glb, "tavern", kit, inner)


def smithy(glb):
    rng = np.random.default_rng(12)
    b = next(b for b in PLAN.BUILDINGS if b["id"] == "smithy")
    W, D = 9.0, 7.0
    site = Site(b["x"], b["z"], b["face"], W + 6, D)
    acc = Acc(rng)
    kit = Kit(acc, rng, instanced=True)
    roof = Roof(kit, INST, site.Mb)
    pcol = PLASTER["lime"]
    win = lambda u, w=1.0, h=1.1, s=0.95, sh="red": {"kind": "win", "u": u, "w": w, "h": h, "sill": s, "shutters": sh}
    gwin = lambda u: {"kind": "win", "u": u, "w": 1.0, "h": 1.1, "sill": 0.95, "shutters": "red", "depth": 0.5, "inside": True}
    storeys = [
        {"h": 3.0, "kind": "stone", "open": {
            "front": [{"kind": "door", "u": 4.6, "w": 2.0, "h": 2.4, "sill": 0, "double": True, "col": (0.16, 0.12, 0.09), "open": 92, "depth": 0.5},
                      gwin(1.6), gwin(7.5)],
            "right": [gwin(1.2)], "back": [gwin(4.5)], "left": [{"kind": "door", "u": 2.2, "w": 1.0, "h": 2.1, "sill": 0, "step": False, "depth": 0.5}]},
         "signs": {"front": [(8.45, 2.65, "anvil")]}},
        {"h": 2.5, "kind": "timber", "style": "tall", "open": {
            "front": [win(2.4, 0.9, 1.0, 0.8), win(6.6, 0.9, 1.0, 0.8)], "back": [win(4.5, 0.9, 1.0, 0.8)],
            "left": [win(3.5, 0.8, 0.9, 0.8)], "right": [win(3.5, 0.8, 0.9, 0.8)]}},
    ]
    rspec = {"axis": "x", "pitch": 46, "ov_e": 0.4, "ov_g": 0.3, "chimneys": [(3.4, 0.6, 1.4)], "chimney_col": (0.42, 0.39, 0.35)}
    house(kit, roof, site, W, D, storeys, rspec, pcol, solid=False)
    inner = smithy_inside(kit, site, W, D, storeys[0]["open"])
    # the open forge under a lean-to on the left
    lx0 = -W / 2
    for py in (-3.2, 0.0, 3.2):
        gz = site.gz(lx0 - 3.0, py)
        box(acc, "wood", T(lx0 - 3.0, py, (gz + 2.25) / 2), 0.11, 0.11, (2.25 - gz) / 2, kit.J(WOOD, 0.06), 0.015)
        box(acc, "stone", T(lx0 - 3.0, py, gz + 0.08), 0.2, 0.2, 0.12, kit.J(STONE_GREY, 0.05), 0.02)
        kit.timber(frame((lx0 - 3.0, py, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), (0.0, 1.75), (2.0, 2.6), 0.12, WOOD) if False else None
    tube(acc, "wood", np.eye(4), [(lx0 - 3.0, -3.5, 2.3), (lx0 - 3.0, 3.5, 2.3)], 0.1, 8, kit.J(WOOD, 0.05))
    Mr = frame((lx0, 0, 0), (0, 1, 0), (-1, 0, 0), (0, 0, 1))
    roof.shed(Mr, D, 3.0, 2.95, math.radians(15), ov_e=0.35, ov_g=0.25)
    gzf = site.gz(lx0 - 1.5, 0)
    # paved floor under the lean-to
    for i in range(8):
        for j in range(18):
            px, py = lx0 - 0.2 - i * 0.37 - 0.18, -3.3 + j * 0.37 + 0.18
            box(acc, "stone", T(px, py, site.gz(px, py) + 0.02), 0.175, 0.175, 0.05, kit.J(STONE_GREY, 0.1), 0.02)
    # hearth with its hood and flue
    hx, hy = lx0 - 1.0, 1.9
    hz = site.gz(hx, hy)
    Mh = T(hx, hy, hz)
    for i, side in enumerate(("front", "right", "back", "left")):
        Mw, L = kit.wall_frame(side, -0.75, 0.75, -0.6, 0.6, 0)
        kit.stone_wall(Mh @ Mw, L, 0.82, [], col=(0.42, 0.39, 0.35), coursed=True, quoins=False, parity=i % 2, proud=0.02)
    box(acc, "stone", Mh @ T(0, 0, 0.86), 0.8, 0.65, 0.04, kit.J(STONE_GREY, 0.05), 0.015)
    for _ in range(26):   # coals and clinker in the hearth
        blob(acc, "produce", Mh @ T(rng.uniform(-0.45, 0.45), rng.uniform(-0.35, 0.3), 0.92), rng.uniform(0.04, 0.07),
             kit.J((0.08, 0.07, 0.07), 0.2) if rng.random() < 0.7 else kit.J((0.75, 0.28, 0.06), 0.1), seg=5, rings=3, squash=0.6, rng=rng)
    prism(acc, "stone", Mh @ frame((0, 0, 1.6), (1, 0, 0), (0, 1, 0), (0, 0, 1)), [(-0.85, -0.5), (0.85, -0.5), (0.3, 0.45), (-0.3, 0.45)], 1.2,
          kit.J((0.4, 0.37, 0.34), 0.05))
    box(acc, "stone", Mh @ T(0, 0.1, 2.9), 0.3, 0.3, 0.9, kit.J((0.4, 0.37, 0.34), 0.05), 0.02)
    lathe(acc, "wood", Mh @ T(-0.95, 0.1, 0.55) @ R("y", math.pi / 2), [(0.0, -0.3), (0.22, -0.2), (0.25, 0.1), (0.12, 0.3), (0.04, 0.45)], 10,
          kit.J((0.32, 0.2, 0.13), 0.05), smooth=True)   # bellows
    # anvil on a stump, quench tub, grindstone, a rack of tools
    ax_, ay_ = lx0 - 1.3, -0.6
    az = site.gz(ax_, ay_)
    lathe(acc, "wood", T(ax_, ay_, az), [(0.34, 0), (0.33, 0.55), (0.0, 0.56)], 12, kit.J(WOOD_GREY, 0.05), smooth=False)
    Ma = T(ax_, ay_, az + 0.56)
    box(acc, "iron", Ma @ T(0, 0, 0.12), 0.14, 0.11, 0.12, IRON, 0.01)
    box(acc, "iron", Ma @ T(0, 0, 0.3), 0.3, 0.09, 0.06, IRON, 0.01)
    prism(acc, "iron", Ma @ frame((0.3, 0, 0.3), (1, 0, 0), (0, 1, 0), (0, 0, 1)), [(0, -0.06), (0.22, 0.03), (0, 0.06)], 0.14, IRON)
    box(acc, "iron", Ma @ T(0, 0, 0.02), 0.2, 0.14, 0.03, IRON, 0.008)
    tube(acc, "iron", Ma, [(-0.1, 0.05, 0.37), (0.15, 0.05, 0.37)], 0.015, 6, IRON)
    box(acc, "iron", Ma @ T(0.2, 0.05, 0.39), 0.04, 0.03, 0.03, IRON, 0.005)
    kit.barrel(T(lx0 - 2.4, -2.5, site.gz(lx0 - 2.4, -2.5)), r=0.36, h=0.6, col=WOOD_GREY)
    lathe(acc, "glass", T(lx0 - 2.4, -2.5, site.gz(lx0 - 2.4, -2.5) + 0.52), [(0.0, 0), (0.33, 0)], 16, (0.05, 0.06, 0.06), smooth=False)
    gx, gy = lx0 - 2.2, 1.4
    g0 = site.gz(gx, gy)
    lathe(acc, "stone", T(gx, gy, g0 + 0.62) @ R("x", math.pi / 2) @ T(0, 0, -0.07), [(0.0, 0), (0.42, 0.0), (0.42, 0.14), (0.0, 0.14)], 18,
          kit.J((0.6, 0.56, 0.48), 0.04), smooth=False)
    for sy in (-0.12, 0.12):
        box(acc, "wood", T(gx, gy + sy, g0 + 0.33), 0.5, 0.03, 0.035, WOOD_GREY, 0.006)
        box(acc, "wood", T(gx - 0.45, gy + sy, g0 + 0.3), 0.035, 0.03, 0.3, WOOD_GREY, 0.006)
        box(acc, "wood", T(gx + 0.45, gy + sy, g0 + 0.3), 0.035, 0.03, 0.3, WOOD_GREY, 0.006)
    tube(acc, "iron", np.eye(4), [(gx, gy - 0.2, g0 + 0.62), (gx, gy + 0.32, g0 + 0.62), (gx, gy + 0.32, g0 + 0.42)], 0.012, 5, IRON)
    Mw, L = kit.wall_frame("left", -W / 2, W / 2, -D / 2, D / 2, 0)
    box(acc, "wood", Mw @ T(4.6, -0.03, 1.75), 1.0, 0.025, 0.08, kit.J(WOOD, 0.05), 0.006)
    for k in range(9):    # tongs, hammers and horseshoes on pegs
        u = 3.8 + k * 0.2
        if k % 3 == 2:
            tube(acc, "iron", Mw, [(u + 0.07 * math.cos(t), -0.06, 1.55 + 0.07 * math.sin(t)) for t in np.linspace(-0.4, math.pi + 0.4, 9)], 0.009, 4, IRON)
        else:
            tube(acc, "iron", Mw, [(u, -0.06, 1.72), (u + rng.normal(0, 0.02), -0.06, 1.2 + rng.uniform(0, 0.2))], 0.01, 4, IRON)
            box(acc, "iron", Mw @ T(u, -0.06, 1.18), 0.03, 0.03, 0.05, IRON, 0.005) if k % 3 == 1 else None
    for _ in range(40):   # a heap of charcoal
        a, r = rng.uniform(0, 6.28), rng.uniform(0, 0.55) ** 0.8
        px, py = lx0 - 2.3 + math.cos(a) * r, 2.6 + math.sin(a) * r * 0.6
        blob(acc, "produce", T(px, py, site.gz(px, py) + (0.55 - r) * 0.5), rng.uniform(0.06, 0.1), kit.J((0.07, 0.065, 0.06), 0.25), seg=5, rings=3,
             squash=0.7, rng=rng)
    kit.log_pile(T(W / 2 + 0.5, 0.8, site.gz(W / 2 + 0.5, 0.8)) @ R("z", math.pi / 2), 2.2, 4)
    site.collider(lx0 - 1.0, 1.9, 0.8, 0.65, y0=-2, y1=1.0)
    site.collider(lx0 - 1.3, -0.6, 0.35, 0.35, y0=-2, y1=1.0)
    site.collider(lx0 - 3.0, 0, 0.15, 3.4, y0=-2, y1=2.3)
    return place(acc, site, glb, "smithy", kit, inner)


def chapel(glb):
    rng = np.random.default_rng(13)
    b = next(b for b in PLAN.BUILDINGS if b["id"] == "chapel")
    W, D = 6.6, 12.5
    site = Site(b["x"], b["z"], b["face"], W, D)
    acc = Acc(rng)
    kit = Kit(acc, rng, instanced=True)
    roof = Roof(kit, INST, site.Mb)
    scol = (0.55, 0.52, 0.46)
    lan = lambda u: {"kind": "win", "u": u, "w": 0.72, "h": 1.9, "sill": 1.55, "arch": "lancet", "depth": 0.45, "inside": True}
    storeys = [{"h": 5.2, "kind": "stone", "coursed": True, "open": {
        "front": [{"kind": "door", "u": W / 2, "w": 1.5, "h": 2.25, "sill": 0, "arch": "lancet", "double": True, "col": (0.2, 0.13, 0.08),
                   "open": 96, "depth": 0.45}],
        "left": [lan(2.6), lan(6.25), lan(9.9)], "right": [lan(2.6), lan(6.25), lan(9.9)],
        "back": [{"kind": "win", "u": W / 2, "w": 0.9, "h": 2.2, "sill": 1.6, "arch": "lancet", "depth": 0.45, "inside": True}]},
        "lanterns": {"front": [(W / 2 - 1.35, 2.5), (W / 2 + 1.35, 2.5)]}}]
    rspec = {"axis": "y", "pitch": 56, "ov_e": 0.35, "ov_g": 0.25, "gable": "stone",
             "gable_windows": {"front": [{"kind": "win", "u": W / 2, "w": 0.55, "h": 0.8, "sill": 0.7, "arch": "lancet", "depth": 0.45, "inside": True}]}}
    h = house(kit, roof, site, W, D, storeys, rspec, PLASTER["white"], scol=scol, solid=False)
    zr = h["zr"]
    inner = chapel_inside(kit, site, W, D, storeys[0]["open"], zr, math.radians(56))
    # buttresses along the sides
    for side, xs in (("left", -W / 2), ("right", W / 2)):
        for y in (-D / 2 + 0.35, -1.75 + 0.0, 2.0, D / 2 - 0.35):
            sg = -1 if side == "left" else 1
            gz = site.gz(xs + sg * 0.4, y)
            Mb_ = T(xs, y, gz - 0.2)
            M = Mb_ @ frame((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)) if sg > 0 else Mb_ @ frame((0, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1))
            # laid in courses, stepping back twice, each step shed by a sloping weathering stone
            steps = ((0.0, 1.45, 0.78), (1.45, 3.0, 0.52), (3.0, 3.85, 0.3))
            Mrot = R("z", math.pi / 2)          # the stones' open backs face into the wall
            for (z0, z1, p) in steps:
                for k, z in enumerate(np.arange(z0, z1 - 0.05, 0.3)):
                    ch = min(0.3, z1 - z)
                    if k % 2 == 0 or p < 0.4:
                        parts = [(0.0, p)]
                    else:
                        parts = [(0.0, p * 0.45), (p * 0.45, p)]
                    for (a, b_) in parts:
                        c = kit.tone(scol, 0.09)
                        kit.blocks.append((M @ T((a + b_) / 2, 0, z + ch / 2) @ Mrot @ np.diag([0.56, b_ - a - 0.015, ch - 0.015, 1.0]), c))
                p_next = dict(((0.0, 0.52), (1.45, 0.3), (3.0, 0.0)))[z0]
                cap = [(p_next - 0.02, z1 - 0.06), (p + 0.03, z1 - 0.06), (p + 0.03, z1 - 0.02), (p_next - 0.02, z1 + 0.22)]
                if p_next == 0.0:
                    cap = [(0.0, z1), (p + 0.03, z1), (p + 0.03, z1 + 0.03), (0.0, z1 + 0.42)]
                prism(acc, "stone", M @ frame((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), cap, 0.6, kit.tone(scol, 0.04, light=True))
    # bell-cote on the front gable: two piers, an arch, a small roof, the bell
    bc = T(0, -D / 2 + 0.05, zr - 0.15)
    for sx in (-0.48, 0.48):
        box(acc, "stone", bc @ T(sx, 0, 0.55), 0.13, 0.2, 0.55, kit.J(scol, 0.05), 0.02)
    prism(acc, "stone", bc @ frame((0, 0, 1.1), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
          [(-0.62, 0), (0.62, 0), (0.62, 0.18)] + [(0.35 * math.cos(t), 0.18 + 0.0 + 0.35 * math.sin(t) * 0.0) for t in ()] + [(-0.62, 0.18)], 0.42,
          kit.J(scol, 0.04))
    prism(acc, "stone", bc @ frame((0, 0, 1.28), (1, 0, 0), (0, 1, 0), (0, 0, 1)), [(-0.7, 0), (0.7, 0), (0.0, 0.55)], 0.5, kit.J(scol, 0.04))
    lathe(acc, "stone", bc @ T(0, 0, 1.83), [(0.05, 0), (0.05, 0.3), (0.0, 0.31)], 6, kit.J(scol, 0.03), smooth=False)
    box(acc, "stone", bc @ T(0, 0, 2.0), 0.14, 0.04, 0.04, kit.J(scol, 0.03), 0.01)
    bell = bc @ T(0, 0, 0.42)
    lathe(acc, "iron", bell, [(0.0, 0.42), (0.12, 0.4), (0.16, 0.3), (0.18, 0.12), (0.25, 0.0), (0.23, -0.02), (0.15, 0.05), (0.0, 0.06)], 16,
          (0.30, 0.21, 0.10), smooth=True)
    box(acc, "wood", bell @ T(0, 0, 0.5), 0.36, 0.05, 0.05, WOOD, 0.008)
    # a cross on the back gable
    cb = T(0, D / 2 + 0.1, zr + 0.05)
    box(acc, "stone", cb @ T(0, 0, 0.45), 0.06, 0.06, 0.45, kit.J(scol, 0.03), 0.015)
    box(acc, "stone", cb @ T(0, 0, 0.62), 0.24, 0.06, 0.06, kit.J(scol, 0.03), 0.015)
    # the churchyard: a few old stones leaning in the grass
    for k, (gx, gy) in enumerate(((-W / 2 - 2.0, -3.0), (-W / 2 - 2.2, -0.8), (-W / 2 - 1.9, 1.5), (-W / 2 - 3.6, -2.0), (-W / 2 - 3.8, 0.5), (-W / 2 - 3.5, 3.0))):
        g0 = site.gz(gx, gy)
        M = T(gx, gy, g0 - 0.2) @ R("z", math.pi / 2 + rng.normal(0, 0.08)) @ R("x", rng.normal(0, 0.06)) @ R("y", rng.normal(0, 0.06))
        w, hh = rng.uniform(0.25, 0.34), rng.uniform(0.7, 1.0)
        poly = [(-w, 0), (w, 0), (w, hh)] + [(w * math.cos(t), hh + w * math.sin(t) * 0.7) for t in np.linspace(0, math.pi, 9)[1:-1]] + [(-w, hh)]
        prism(acc, "stone", M, poly, 0.1, kit.J((0.5, 0.5, 0.46), 0.1))
        site.collider(gx, gy, 0.15, w, y0=-2, y1=0.9)
    kit.bench(T(W / 2 + 0.6, -D / 2 + 2.4, site.gz(W / 2 + 0.6, -D / 2 + 2.4)) @ R("z", math.pi / 2), 1.6)
    return place(acc, site, glb, "chapel", kit, inner)


def tallhouse(glb):
    rng = np.random.default_rng(14)
    b = next(b for b in PLAN.BUILDINGS if b["id"] == "tallhouse")
    W, D = 6.6, 8.5
    site = Site(b["x"], b["z"], b["face"], W, D)
    acc = Acc(rng)
    kit = Kit(acc, rng, instanced=True)
    roof = Roof(kit, INST, site.Mb)
    pcol = PLASTER["ochre"]
    win = lambda u, w=0.9, h=1.1, s=0.75, sh="blue", fl=False: {"kind": "win", "u": u, "w": w, "h": h, "sill": s, "shutters": sh, "flowers": fl}
    storeys = [
        {"h": 2.9, "kind": "timber", "style": "cross", "open": {
            "front": [{"kind": "door", "u": 1.3, "w": 1.0, "h": 2.1, "sill": 0, "col": PAINT["blue"], "open": 100, "depth": 0.22},
                      dict(win(4.4, 2.0, 1.35, 0.85, None), depth=0.22, inside=True)],
            "left": [dict(win(4.2, 0.9, 1.1, 0.9), depth=0.22, inside=True)], "right": [dict(win(3.0, 0.9, 1.1, 0.9), depth=0.22, inside=True)],
            "back": [dict(win(3.3, 0.9, 1.1, 0.9), depth=0.22, inside=True)]},
         "lanterns": {"front": [(2.15, 2.3)]}},
        {"h": 2.65, "kind": "timber", "jetty": 0.45, "style": "cross", "brackets": [0.25, 3.3, 6.35], "open": {
            "front": [win(1.6, fl=True), win(5.0, fl=True)], "left": [win(3.0), win(6.0)], "right": [win(3.0), win(6.0)], "back": [win(3.3)]}},
        {"h": 2.5, "kind": "timber", "jetty": 0.4, "style": "cross", "brackets": [0.25, 6.35], "open": {
            "front": [win(1.7, 0.85, 1.0, 0.7), win(4.9, 0.85, 1.0, 0.7)], "left": [win(4.5, 0.85, 1.0, 0.7)], "right": [win(4.5, 0.85, 1.0, 0.7)],
            "back": [win(3.3, 0.85, 1.0, 0.7)]}},
    ]
    rspec = {"axis": "y", "pitch": 58, "ov_e": 0.4, "ov_g": 0.45, "finial": True, "chimneys": [(2.4, 2.0, 0.9)],
             "gable_windows": {"front": [win(3.3, 0.8, 1.0, 0.9, "blue")], "back": [win(3.3, 0.7, 0.9, 0.9, None)]}}
    house(kit, roof, site, W, D, storeys, rspec, pcol, solid=False)
    inner = weaver_inside(kit, site, W, D, storeys[0]["open"])
    for k in range(3):
        kit.crate(T(W / 2 + 0.5, -2.5 + k * 0.6, site.gz(W / 2 + 0.5, -2.5 + k * 0.6)) @ R("z", rng.normal(0, 0.15)), 0.5)
    kit.barrel(T(W / 2 + 0.55, -0.9, site.gz(W / 2 + 0.55, -0.9)), col=WOOD_GREY)
    return place(acc, site, glb, "tallhouse", kit, inner)


def cottage(glb):
    rng = np.random.default_rng(15)
    b = next(b for b in PLAN.BUILDINGS if b["id"] == "cottage")
    W, D = 9.0, 5.6
    site = Site(b["x"], b["z"], b["face"], W, D)
    acc = Acc(rng)
    kit = Kit(acc, rng, instanced=True)
    roof = Roof(kit, INST, site.Mb)
    scol = (0.56, 0.53, 0.47)
    win = lambda u, w=0.8, h=0.9, s=0.85, fl=True: {"kind": "win", "u": u, "w": w, "h": h, "sill": s, "shutters": "green", "flowers": fl, "depth": 0.45, "inside": True}
    storeys = [{"h": 2.45, "kind": "stone", "open": {
        "front": [win(1.4), {"kind": "door", "u": 3.4, "w": 1.0, "h": 1.98, "sill": 0, "col": PAINT["green"], "open": 100, "depth": 0.45}, win(5.6), win(7.6, fl=False)],
        "back": [win(3.0, fl=False), win(6.5, fl=False)], "left": [], "right": []}}]
    rspec = {"axis": "x", "pitch": 50, "ov_e": 0.55, "ov_g": 0.42, "kind": "thatch", "thick": 0.36, "gable": "stone",
             "chimneys": [(3.9, 0.0, 0.7)], "chimney_col": scol}
    house(kit, roof, site, W, D, storeys, rspec, PLASTER["white"], scol=scol, solid=False)
    inner = farm_inside(kit, site, W, D, storeys[0]["open"])
    # porch hood over the door
    Mw, L = kit.wall_frame("front", -W / 2, W / 2, -D / 2, D / 2, 0)
    for sx in (-0.65, 0.65):
        tube(acc, "wood", Mw, [(3.4 + sx, 0.0, 2.05), (3.4 + sx, -0.5, 2.35)], 0.05, 6, kit.J(WOOD, 0.05))
    Mh = Mw @ frame((3.4, -0.3, 2.45), (1, 0, 0), (0, 0, 1), (0, -1, 0))
    for sgn in (-1, 1):
        box(acc, "thatch", Mw @ T(3.4 + sgn * 0.42, -0.38, 2.43) @ R("y", sgn * 0.55), 0.48, 0.42, 0.06, kit.J(THATCH, 0.05), 0.03)
    # garden: a picket fence, cabbage rows, a water butt, a bench, firewood
    fx0, fx1, fy = -W / 2 + 0.3, W / 2 - 0.3, -D / 2 - 2.6
    for x in np.arange(fx0, fx1, 0.16):
        if abs(x - 3.4) < 0.6:
            continue
        g0 = site.gz(x, fy)
        prism(acc, "wood", T(x, fy, g0 - 0.1), [(-0.04, 0), (0.04, 0), (0.04, 0.9), (0.0, 0.98), (-0.04, 0.9)], 0.02, kit.J(WOOD_GREY, 0.08))
    for z_ in (0.3, 0.7):
        box(acc, "wood", T((fx0 + 2.8) / 2, fy + 0.03, site.gz(0, fy) + z_), (2.8 - fx0) / 2, 0.02, 0.035, kit.J(WOOD_GREY, 0.05), 0.006)
        box(acc, "wood", T((4.0 + fx1) / 2, fy + 0.03, site.gz(0, fy) + z_), (fx1 - 4.0) / 2, 0.02, 0.035, kit.J(WOOD_GREY, 0.05), 0.006)
    for row in range(3):
        for k in range(8):
            px, py = 4.4 + k * 0.42, -D / 2 - 0.8 - row * 0.55
            g0 = site.gz(px, py)
            blob(acc, "produce", T(px, py, g0 + 0.1), rng.uniform(0.14, 0.18), kit.J((0.28, 0.42, 0.2), 0.1), seg=8, rings=5, squash=0.7, rng=rng)
    for row in range(3):
        for k in range(7):
            px, py = -3.9 + k * 0.4, -D / 2 - 0.9 - row * 0.5
            for _ in range(3):
                blob(acc, "produce", T(px + rng.normal(0, 0.05), py + rng.normal(0, 0.05), site.gz(px, py) + 0.1), 0.06,
                     kit.J((0.22, 0.38, 0.12), 0.12), seg=5, rings=3, squash=1.4, rng=rng)
    kit.barrel(T(-W / 2 + 0.45, -D / 2 - 0.45, site.gz(-W / 2 + 0.45, -D / 2 - 0.45)), r=0.34, h=0.9, col=WOOD_GREY)
    kit.bench(T(1.9, -D / 2 - 0.4, site.gz(1.9, -D / 2 - 0.4)), 1.3)
    kit.log_pile(T(-W / 2 - 0.5, 0.3, site.gz(-W / 2 - 0.5, 0.3)) @ R("z", math.pi / 2), 2.6, 5)
    site.collider(0, fy, (fx1 - fx0) / 2, 0.05, y0=-2, y1=1.0)
    return place(acc, site, glb, "cottage", kit, inner)


def bakery(glb):
    rng = np.random.default_rng(16)
    b = next(b for b in PLAN.BUILDINGS if b["id"] == "bakery")
    W, D = 8.0, 8.4
    site = Site(b["x"], b["z"], b["face"], W + 2.5, D)
    acc = Acc(rng)
    kit = Kit(acc, rng, instanced=True)
    roof = Roof(kit, INST, site.Mb)
    pcol = PLASTER["pink"]
    win = lambda u, w=0.9, h=1.1, s=0.75, sh="ochre", fl=True: {"kind": "win", "u": u, "w": w, "h": h, "sill": s, "shutters": sh, "flowers": fl}
    storeys = [
        {"h": 3.0, "kind": "timber", "style": "cross", "open": {
            "front": [{"kind": "door", "u": 1.5, "w": 1.0, "h": 2.15, "sill": 0, "col": PAINT["blue"], "open": 100, "depth": 0.22},
                      dict(win(5.1, 2.4, 1.25, 0.95, None, False), depth=0.22, inside=True)],
            "left": [dict(win(4.2, 0.9, 1.1, 0.9, "ochre", False), depth=0.22, inside=True)],
            "back": [dict(win(2.5, 0.9, 1.1, 0.9, None, False), depth=0.22, inside=True), dict(win(5.8, 0.9, 1.1, 0.9, None, False), depth=0.22, inside=True)],
            "right": [dict(win(2.0, 0.8, 1.0, 1.0, None, False), depth=0.22, inside=True),
                      {"kind": "blind", "u": 1.6 + D / 2, "w": 1.06, "h": 0.55, "sill": 0.67, "arch": "round"}]},      # the oven's mouth, inside
         "signs": {"front": [(7.45, 2.75, "pretzel")]}},
        {"h": 2.6, "kind": "timber", "jetty": 0.4, "style": "cross", "brackets": [0.25, 4.0, 7.75], "open": {
            "front": [win(1.5), win(4.0), win(6.5)], "left": [win(4.2, fl=False)], "right": [win(4.2, fl=False)], "back": [win(4.0, fl=False)]}},
    ]
    rspec = {"axis": "x", "pitch": 50, "ov_e": 0.45, "ov_g": 0.35, "chimneys": [(2.9, 2.6, 1.0)], "dormers": [(-1.2, 1.2)], "dormer_shutters": "ochre",
             "gable_windows": {"left": [win(4.2, 0.7, 0.85, 0.8, None, False)], "right": [win(4.2, 0.7, 0.85, 0.8, None, False)]}}
    house(kit, roof, site, W, D, storeys, rspec, pcol, solid=False)
    inner = bakery_inside(kit, site, W, D, storeys[0]["open"])
    Mw, L = kit.wall_frame("front", -W / 2, W / 2, -D / 2, D / 2, 0)
    # the counter: a board let down from the shop window, bread on it, an awning above
    box(acc, "wood", Mw @ T(5.1, -0.32, 0.92), 1.3, 0.32, 0.035, kit.J(WOOD_RAW, 0.05), 0.01)
    for sx in (-1.1, 1.1):
        tube(acc, "iron", Mw, [(5.1 + sx, -0.6, 0.9), (5.1 + sx, -0.02, 0.45)], 0.012, 5, IRON)
    for k in range(9):    # loaves, rolls and a pretzel basket
        px = 4.0 + k * 0.25
        blob(acc, "produce", Mw @ T(px, -0.32 + rng.normal(0, 0.05), 0.99) @ R("z", rng.normal(0, 0.4)), 0.11, kit.J((0.62, 0.40, 0.18), 0.08),
             seg=8, rings=5, squash=0.55, rng=rng, rough=0.05)
    lathe(acc, "wood", Mw @ T(6.05, -0.33, 0.955), [(0.0, 0), (0.16, 0.0), (0.2, 0.1), (0.19, 0.11), (0.15, 0.02)], 12, kit.J((0.45, 0.33, 0.18), 0.05))
    stripes = lambda U, V: np.where((np.floor(U / 0.3) % 2)[:, None] == 0, np.array([[0.62, 0.15, 0.12]]), np.array([[0.88, 0.84, 0.74]]))
    Ma = Mw @ T(3.6, -0.02, 2.55) @ R("x", -0.5)
    patch(acc, "cloth", Ma, 3.0, 1.05, 10, 4, (0.8, 0.8, 0.8), lambda U, V: 0.04 * np.sin(np.pi * V / 1.05), stripes)
    for k in range(10):
        Mv = Ma @ T(0.15 + k * 0.3, 0, 0)
        prism(acc, "cloth", Mv @ frame((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)) @ R("x", 0.5), [(-0.15, 0), (0.15, 0), (0.0, -0.16)], 0.01,
              (0.62, 0.15, 0.12) if k % 2 == 0 else (0.88, 0.84, 0.74)) if False else None
    for sx in (3.6, 6.6):
        tube(acc, "iron", Mw, [(sx, 0.0, 2.55), (sx, -0.92, 2.05)], 0.012, 5, IRON)
    # the bread oven, built out from the right gable: a domed brick oven with a little flue
    ox, oy = W / 2 + 1.05, 1.6
    g0 = site.gz(ox, oy)
    for i, side in enumerate(("front", "right", "back", "left")):
        Mo, Lo = kit.wall_frame(side, ox - 1.05, ox + 1.05, oy - 1.1, oy + 1.1, g0 - 0.3)
        kit.stone_wall(Mo, Lo, 1.2 - g0 + g0, [], col=(0.5, 0.45, 0.38), coursed=True, quoins=True, parity=i % 2, proud=0.03)
    prof = [(1.0 * math.cos(t), 0.9 * math.sin(t)) for t in np.linspace(0, math.pi / 2, 10)]
    lathe(acc, "stone", T(ox, oy, g0 + 0.9), [(p[0], p[1]) for p in prof], 20, kit.J((0.56, 0.32, 0.22), 0.05), smooth=True)
    for t in np.linspace(0.1, 1.45, 8):    # brick courses on the dome
        r_, z_ = 1.0 * math.cos(t) + 0.01, 0.9 * math.sin(t)
        lathe(acc, "stone", T(ox, oy, g0 + 0.9), [(r_, z_ - 0.012), (r_ + 0.008, z_), (r_, z_ + 0.012)], 20, (0.38, 0.22, 0.15), smooth=True)
    lathe(acc, "stone", T(ox + 0.25, oy + 0.3, g0 + 1.65), [(0.13, 0), (0.12, 0.8), (0.16, 0.85), (0.16, 0.92), (0.1, 0.92)], 10, kit.J(TERRA, 0.05))
    site.collider(ox, oy, 1.1, 1.15, y0=-2, y1=1.8)
    for k, (sx, sy) in enumerate(((-W / 2 + 0.3, -D / 2 - 0.4), (-W / 2 + 0.7, -D / 2 - 0.45), (-W / 2 + 0.5, -D / 2 - 0.9))):
        kit.sack(T(sx, sy, site.gz(sx, sy)) @ R("z", rng.uniform(0, 6)), 0.26)
    kit.crate(T(W / 2 - 0.4, -D / 2 - 0.5, site.gz(W / 2 - 0.4, -D / 2 - 0.5)), 0.5)
    kit.log_pile(T(W / 2 + 1.05, -1.4, site.gz(W / 2 + 1.05, -1.4)), 1.3, 5)
    return place(acc, site, glb, "bakery", kit, inner)


def fisher(glb):
    rng = np.random.default_rng(17)
    b = next(b for b in PLAN.BUILDINGS if b["id"] == "fisher")
    W, D = 6.5, 8.6
    site = Site(b["x"], b["z"], b["face"], W + 4, D)
    acc = Acc(rng)
    kit = Kit(acc, rng, instanced=True)
    roof = Roof(kit, INST, site.Mb)
    scol = (0.5, 0.48, 0.44)
    win = lambda u, w=0.75, h=0.85, s=0.9: {"kind": "win", "u": u, "w": w, "h": h, "sill": s, "shutters": "blue", "depth": 0.45, "inside": True}
    storeys = [{"h": 2.6, "kind": "stone", "open": {
        "front": [{"kind": "door", "u": W / 2, "w": 1.0, "h": 1.98, "sill": 0, "col": (0.24, 0.34, 0.4), "open": 100, "depth": 0.45}],
        "left": [win(2.4), win(6.2)], "right": [win(2.4), win(6.2)], "back": []},
        "lanterns": {"front": [(W / 2 + 0.85, 2.2)]}}]
    rspec = {"axis": "y", "pitch": 50, "ov_e": 0.5, "ov_g": 0.45, "kind": "thatch", "thick": 0.34, "gable": "stone",
             "chimneys": [(0.0, D / 2 - 0.5, 0.6)], "chimney_col": scol,
             "gable_windows": {"front": [{"kind": "win", "u": W / 2, "w": 0.6, "h": 0.7, "sill": 0.6, "shutters": "blue"}]}}
    house(kit, roof, site, W, D, storeys, rspec, PLASTER["white"], scol=scol, solid=False)
    inner = fisher_inside(kit, site, W, D, storeys[0]["open"])
    # net-drying rack along the south side, nets sagging between its rails, cork floats along the top
    rx = W / 2 + 1.9
    for py in (-2.6, 0.0, 2.6):
        g0 = site.gz(rx, py)
        box(acc, "wood", T(rx, py, g0 + 1.0), 0.06, 0.06, 1.15, kit.J(WOOD_GREY, 0.08), 0.01)
    gm = site.gz(rx, 0)
    tube(acc, "wood", np.eye(4), [(rx, -2.8, gm + 2.05), (rx, 2.8, gm + 2.05)], 0.04, 6, kit.J(WOOD_GREY, 0.05))
    for seg_ in range(2):
        ya, yb = -2.55 + seg_ * 2.6, -0.05 + seg_ * 2.6
        NU, NV = 24, 14
        for i in range(NU + 1):
            y = ya + (yb - ya) * i / NU
            pts = []
            for j in range(NV + 1):
                t = j / NV
                sag = 0.25 * math.sin(math.pi * i / NU) * t
                pts.append((rx + 0.04 + 0.05 * math.sin(t * 3 + i), y, gm + 2.0 - t * 1.6 - sag * 0.5))
            tube(acc, "cloth", np.eye(4), pts, 0.006, 3, kit.J((0.22, 0.24, 0.18), 0.06), cap=False)
        for j in range(0, NV + 1, 2):
            t = j / NV
            pts = [(rx + 0.04, ya + (yb - ya) * i / NU, gm + 2.0 - t * 1.6 - 0.25 * math.sin(math.pi * i / NU) * t * 0.5) for i in range(NU + 1)]
            tube(acc, "cloth", np.eye(4), pts, 0.005, 3, kit.J((0.22, 0.24, 0.18), 0.06), cap=False)
        for i in range(0, NU + 1, 3):
            y = ya + (yb - ya) * i / NU
            blob(acc, "produce", T(rx + 0.05, y, gm + 1.98), 0.05, kit.J((0.62, 0.45, 0.25), 0.06), seg=6, rings=3, squash=0.8, rng=rng)
    site.collider(rx, 0, 0.1, 2.8, y0=-2, y1=2.1)
    # an upturned boat: clinker strakes over a hull, keel up
    bx, by = W / 2 + 3.6, 1.0
    g0 = site.gz(bx, by)
    Mbt = T(bx, by, g0 - 0.05) @ R("z", 0.15)
    Lb, Bb, Hb = 4.6, 0.75, 0.6

    def hull(U, V):          # U across the bottom (0..1 = one gunwale to the other), V stem to stern
        ang = U * math.pi
        t = V * 2 - 1
        r = np.sqrt(np.clip(1 - t ** 2 * 0.98, 0, 1)) * (1 - 0.15 * np.abs(t) ** 3)
        return Bb * r * np.cos(ang) * 0.985, Lb / 2 * t, Hb * r * np.sin(ang) * 0.985

    paint = lambda Uu, Vv: np.where((np.abs(np.cos(Uu * math.pi)) > 0.55)[:, None], np.array([[0.20, 0.33, 0.37]]), np.array([[0.32, 0.25, 0.17]])) * \
        (0.85 + 0.15 * np.sin(Vv * 40)[:, None])
    G.surface(acc, "wood", Mbt, hull, 16, 30, (0.3, 0.25, 0.18), paint, flip=True)
    for k in range(7):
        ang = (k + 0.5) / 7 * math.pi / 2
        pts = []
        for t in np.linspace(-1, 1, 17):
            r = math.sqrt(max(0.0, 1 - t ** 2 * 0.98))
            pts.append((Bb * r * math.cos(ang), Lb / 2 * t, Hb * r * math.sin(ang)))
        for sgn in (-1, 1):
            p2 = [(sgn * p[0], p[1], p[2]) for p in pts]
            tube(acc, "wood", Mbt, p2, [0.045 * (0.5 + 0.5 * math.sqrt(max(0, 1 - (2 * i / 16 - 1) ** 2))) for i in range(17)], 5,
                 kit.J((0.18, 0.3, 0.34) if k < 3 else (0.3, 0.24, 0.17), 0.08))
    lathe(acc, "wood", Mbt @ R("x", -math.pi / 2) @ S_y(1.0), [(0.0, -Lb / 2), (0.1, -Lb / 2 + 0.2), (0.12, Lb / 2 - 0.2), (0.0, Lb / 2)], 4,
          kit.J(WOOD_GREY, 0.05), smooth=False) if False else None
    box(acc, "wood", Mbt @ T(0, 0, Hb + 0.03), 0.05, Lb / 2 - 0.1, 0.05, kit.J(WOOD_GREY, 0.05), 0.01)
    site.collider(bx, by, 0.85, 2.35, y0=-2, y1=0.7, yaw_extra=-0.15)
    # oars against the wall, pots, barrels, a coil of rope, an anchor
    Mw, L = kit.wall_frame("right", -W / 2, W / 2, -D / 2, D / 2, 0)
    for k, u in enumerate((6.9, 7.3)):
        tube(acc, "wood", Mw, [(u, -0.15, 0.0), (u + 0.25, -0.06, 2.3)], 0.03, 6, kit.J(WOOD_RAW, 0.06))
        box(acc, "wood", Mw @ T(u + 0.03, -0.15, 0.32) @ R("y", -0.1), 0.08, 0.012, 0.32, kit.J(WOOD_RAW, 0.06), 0.006)
    for k in range(4):     # lobster pots: half-round cages of withies
        px, py = W / 2 + 1.0 + (k % 2) * 0.62, -3.0 - (k // 2) * 0.62
        g0 = site.gz(px, py) + (0.0 if k < 2 else 0.0)
        Mp = T(px, py, g0)
        box(acc, "wood", Mp @ T(0, 0, 0.02), 0.28, 0.24, 0.02, kit.J(WOOD_GREY, 0.08), 0.005)
        for t in np.linspace(-0.22, 0.22, 4):
            tube(acc, "wood", Mp, [(0.26 * math.cos(a), t, 0.04 + 0.32 * math.sin(a)) for a in np.linspace(0, math.pi, 9)], 0.009, 4,
                 kit.J((0.36, 0.28, 0.18), 0.08), cap=False)
        for a in np.linspace(0.3, math.pi - 0.3, 5):
            tube(acc, "wood", Mp, [(0.26 * math.cos(a), -0.24, 0.04 + 0.32 * math.sin(a)), (0.26 * math.cos(a), 0.24, 0.04 + 0.32 * math.sin(a))], 0.007, 3,
                 kit.J((0.36, 0.28, 0.18), 0.08), cap=False)
    kit.barrel(T(-W / 2 - 0.5, -D / 2 + 0.6, site.gz(-W / 2 - 0.5, -D / 2 + 0.6)), col=WOOD_GREY)
    kit.barrel(T(-W / 2 - 0.55, -D / 2 + 1.3, site.gz(-W / 2 - 0.55, -D / 2 + 1.3)), r=0.27, h=0.7)
    pts = []
    for t in np.linspace(0, 5 * 2 * math.pi, 160):
        r = 0.12 + t * 0.008
        pts.append((r * math.cos(t), r * math.sin(t), 0.03 + 0.03 * (t // (2 * math.pi)) * 0.5))
    cx_, cy_ = W / 2 + 1.0, 3.2
    tube(acc, "cloth", T(cx_, cy_, site.gz(cx_, cy_)), pts, 0.02, 5, kit.J((0.55, 0.48, 0.34), 0.05), cap=False)
    Man = T(-W / 2 - 0.35, 2.2, site.gz(-W / 2 - 0.35, 2.2)) @ R("y", 0.2) @ R("z", math.pi / 2)
    tube(acc, "iron", Man, [(0, 0, 0.1), (0, 0, 1.3)], 0.035, 6, IRON)
    tube(acc, "iron", Man, [(-0.45 * math.cos(a), 0, 0.42 - 0.35 * math.sin(a)) for a in np.linspace(-0.2, math.pi + 0.2, 13)], 0.03, 6, IRON)
    tube(acc, "iron", Man, [(-0.25, 0, 1.15), (0.25, 0, 1.15)], 0.025, 6, IRON)
    lathe(acc, "iron", Man @ T(0, 0, 1.3), [(0.0, 0), (0.07, 0.03), (0.07, 0.07), (0.0, 0.1)], 8, IRON) if False else tube(
        acc, "iron", Man, [(0.07 * math.cos(a), 0, 1.38 + 0.07 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 11)], 0.014, 5, IRON, cap=False)
    return place(acc, site, glb, "fisher", kit, inner)


def S_y(k):
    return np.diag([1.0, k, 1.0, 1.0])


# ---------------------------------------------------------------- the town wall and its gates
SEG_LEN = 4.0


def wall_kit(glb, rng):
    """Three straight wall segments (4 m), a buttress pier and an end pier, as instanceable kit meshes."""
    names = []
    for v in range(3):
        acc = Acc(rng)
        kit = Kit(acc, rng)
        H = PLAN.WALL_HEIGHT + 0.7 - 0.26
        hw = PLAN.WALL_THICK / 2
        courses = kit.courses(H)
        for side, Mw in (("out", frame((-SEG_LEN / 2, -hw, -0.7), (1, 0, 0), (0, 1, 0), (0, 0, 1))),
                         ("in", frame((SEG_LEN / 2, hw, -0.7), (-1, 0, 0), (0, -1, 0), (0, 0, 1)))):
            kit.stone_wall(Mw, SEG_LEN, H, [], col=(0.5, 0.47, 0.41), courses=courses, quoins=False, proud=0.05)
        box(acc, "stone", T(0, 0, H - 0.7 - 0.02), SEG_LEN / 2, hw - 0.02, 0.05, MORTAR, 0.0)
        # coping: ridged capstones
        x = -SEG_LEN / 2
        while x < SEG_LEN / 2 - 0.05:
            l = min(SEG_LEN / 2 - x, rng.uniform(0.42, 0.62))
            prof = [(-hw - 0.08, 0), (hw + 0.08, 0), (hw + 0.06, 0.1), (0.0, 0.26), (-hw - 0.06, 0.1)]
            M = T(x + l / 2, 0, H - 0.7) @ R("z", rng.normal(0, 0.01)) @ frame((0, 0, 0), (0, 1, 0), (-1, 0, 0), (0, 0, 1))
            prism(acc, "stone", M, prof, l - 0.02, kit.J((0.56, 0.54, 0.49), 0.07))
            x += l
        G.bake_ao(acc, ground=-0.02, strength=0.6)
        G.grime(acc, ("stone",), z0=0.0, height=0.8, amount=0.35, tint=(0.55, 0.58, 0.42))
        glb.add_acc(f"kit_wall{v}", acc)
        names.append(f"kit_wall{v}")
    # buttress pier, standing out on the field side
    acc = Acc(rng)
    kit = Kit(acc, rng)
    hw = PLAN.WALL_THICK / 2
    Hp = PLAN.WALL_HEIGHT + 0.7 + 0.25
    for i, side in enumerate(("front", "right", "back", "left")):
        Mw, L = kit.wall_frame(side, -0.55, 0.55, -hw - 0.55, hw + 0.08, -0.7)
        kit.stone_wall(Mw, L, Hp - 0.3, [], col=(0.52, 0.49, 0.43), coursed=True, parity=i % 2, proud=0.04)
    prism(acc, "stone", T(0, -0.25, Hp - 1.0) @ frame((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)) if False else
          T(0, (-hw - 0.55 + hw + 0.08) / 2, Hp - 1.0) @ frame((0, 0, 0), (0, 1, 0), (-1, 0, 0), (0, 0, 1)),
          [(-(hw + 0.6 + hw + 0.12) / 2, 0), ((hw + 0.6 + hw + 0.12) / 2, 0), ((hw + 0.6 + hw + 0.12) / 2, 0.12), (0.0, 0.42), (-(hw + 0.6 + hw + 0.12) / 2, 0.12)],
          1.22, kit.J((0.57, 0.55, 0.5), 0.05))
    G.bake_ao(acc, ground=-0.02, strength=0.6)
    G.grime(acc, ("stone",), z0=0.0, height=0.8, amount=0.35, tint=(0.55, 0.58, 0.42))
    glb.add_acc("kit_pier", acc)
    return names


def gatehouse(glb, gate, direction):
    """Two square towers with a parapet, a round arch between them, open oak gates and banners."""
    rng = np.random.default_rng(sum(map(ord, gate["id"])))
    acc = Acc(rng)
    kit = Kit(acc, rng, instanced=True)
    gx, gz_ = gate["x"], gate["z"]
    yaw = math.atan2(direction[0], direction[1])     # local x along the wall
    # front (-y) faces out of the village: pick the sign
    out = np.array([gx, gz_]) - np.array([0.0, 20.0])
    nrm = np.array([direction[1], -direction[0]])
    ang = math.atan2(-direction[1], direction[0])
    Mb_yaw = math.atan2(direction[1] * -1, direction[0]) * -1
    # Blender rotation so local +x runs along `direction` (game) and local -y points away from the village
    bx_dir = np.array([direction[0], -direction[1]])
    a = math.atan2(bx_dir[1], bx_dir[0])
    local_front = np.array([math.sin(a), -math.cos(a)])          # Blender xy of local -y
    if np.dot(np.array([local_front[0], -local_front[1]]), out) < 0:
        a += math.pi
    floor = float(max(ground(gx + dx, gz_ + dz) for dx in (-6, 0, 6) for dz in (-2, 0, 2))) + 0.05
    low = float(min(ground(gx + dx, gz_ + dz) for dx in (-6, 0, 6) for dz in (-2, 0, 2)))
    Mb = T(gx, -gz_, floor) @ R("z", a)
    scol = (0.54, 0.51, 0.45)
    tw, th, pw = 3.0, 5.4, 3.4
    for sx in (-1, 1):
        cx = sx * (pw / 2 + tw / 2)
        courses = kit.courses(th - (low - floor) + 0.0, coursed=True)
        for i, side in enumerate(("front", "right", "back", "left")):
            Mw, L = kit.wall_frame(side, cx - tw / 2, cx + tw / 2, -tw / 2, tw / 2, low - floor - 0.3)
            ops = []
            if side == "front":
                ops = [{"kind": "win", "u": L / 2, "w": 0.22, "h": 0.9, "sill": th - (low - floor) - 1.6}]
            kit.stone_wall(Mw, L, th - (low - floor - 0.3), ops, col=scol, coursed=True, courses=courses, parity=i % 2)
        # a string course and the parapet with merlons
        box(acc, "stone", T(cx, 0, th + 0.06), tw / 2 + 0.12, tw / 2 + 0.12, 0.08, kit.J(scol, 0.04) * 1.05, 0.02)
        for k in range(4):
            for side in range(4):
                t = -tw / 2 + 0.3 + k * (tw - 0.6) / 3
                p = [(t, -tw / 2 + 0.06), (tw / 2 - 0.06, t), (-t, tw / 2 - 0.06), (-tw / 2 + 0.06, -t)][side]
                rot = [0, math.pi / 2, math.pi, -math.pi / 2][side]
                box(acc, "stone", T(cx + p[0], p[1], th + 0.45) @ R("z", rot), 0.24, 0.16, 0.33, kit.J(scol, 0.06), 0.02)
        for i_ in range(5):           # flagstones on the tower top
            for j_ in range(5):
                fx_, fy_ = -tw / 2 + 0.2 + (i_ + 0.5) * (tw - 0.4) / 5, -tw / 2 + 0.2 + (j_ + 0.5) * (tw - 0.4) / 5
                box(acc, "stone", T(cx + fx_, fy_, th + 0.16), (tw - 0.4) / 10 - 0.01, (tw - 0.4) / 10 - 0.01, 0.03, kit.tone((0.6, 0.58, 0.53), 0.08), 0.01)
        COLL = (cx, tw / 2, tw / 2)
        gxw = gx + (cx * math.cos(a)) * 1
        gzw = gz_ - (cx * math.sin(a))
        COLLIDERS.append({"x": round(gxw, 3), "z": round(gzw, 3), "yaw": round(a, 4), "hx": tw / 2, "hz": tw / 2, "y0": round(low - 1, 3),
                          "y1": round(floor + th + 0.8, 3)})
    # the arch and the wall over it
    sp = 3.0
    r = pw / 2
    for side, ysgn in (("front", -1), ("back", 1)):
        Mw = frame((-pw / 2, ysgn * 0.6, 0), (1, 0, 0) if ysgn < 0 else (1, 0, 0), (0, -ysgn, 0) if False else (0, 1, 0), (0, 0, 1))
        if ysgn > 0:
            Mw = frame((pw / 2, 0.6, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1))
        else:
            Mw = frame((-pw / 2, -0.6, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        top = 5.0
        clip = lambda u: top if (u < 0.02 or u > pw - 0.02) else (top if True else 0)
        # stones above the arch only
        arch_top = lambda u: sp + math.sqrt(max(0.0, r * r - (u - r) ** 2))
        for v, ch in kit.courses(top, coursed=True):
            u = 0.0
            while u < pw - 0.05:
                bl = min(pw - u, rng.uniform(0.45, 0.85))
                cu = u + bl / 2
                if v > max(arch_top(u), arch_top(u + bl)) + 0.25:
                    kit._block(Mw, u, u + bl, v, ch, scol, 0.022, 0.05)
                u += bl
        bed = [(0.0, sp), (0.0, top), (pw, top), (pw, sp)] + [(r + r * math.cos(t), sp + r * math.sin(t)) for t in np.linspace(0.0, math.pi, 17)[1:-1]]
        prism(acc, "stone", Mw @ frame((0, 0.05, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), bed, 0.04, MORTAR, cap_back=False)
        n = 13
        pts = [(r + r * math.cos(t), sp + r * math.sin(t)) for t in np.linspace(math.pi, 0, n + 1)]
        for k in range(n):
            p0, p1 = pts[k], pts[k + 1]
            m = ((p0[0] + p1[0]) / 2 - r, (p0[1] + p1[1]) / 2 - sp)
            d = np.array(m) / (np.linalg.norm(m) + 1e-9)
            o_ = 0.42
            poly = [p0, p1, (p1[0] + d[0] * o_, p1[1] + d[1] * o_), (p0[0] + d[0] * o_, p0[1] + d[1] * o_)]
            prism(acc, "stone", Mw @ frame((0, -0.03 - (0.04 if k == n // 2 else 0), 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), poly, 0.16,
                  np.clip(kit.J(scol, 0.06) * 1.08, 0, 1))
    # soffit of the arch (barrel vault through the wall)
    pts = [(r * math.cos(t), sp + r * math.sin(t)) for t in np.linspace(math.pi, 0, 17)]
    for p0, p1 in zip(pts[:-1], pts[1:]):
        quad(acc, "stone", np.eye(4), [(p0[0], -0.6, p0[1]), (p1[0], -0.6, p1[1]), (p1[0], 0.6, p1[1]), (p0[0], 0.6, p0[1])], np.asarray(scol) * 0.7)
    box(acc, "stone", T(0, 0, 5.05), pw / 2 + 0.02, 0.62, 0.06, kit.J(scol, 0.04), 0.015)
    for k in range(5):
        box(acc, "stone", T(-pw / 2 + 0.35 + k * (pw - 0.7) / 4, -0.45, 5.4), 0.22, 0.14, 0.3, kit.J(scol, 0.06), 0.02)
    # gate leaves, standing open against the passage walls
    for sx in (-1, 1):
        ang_ = sx * math.radians(98)
        Mg = T(sx * (r - 0.05), 0.35, 0) @ R("z", ang_ if sx < 0 else -ang_ + math.pi) if False else T(sx * (r - 0.06), 0.4, 0) @ R("z", math.radians(95) * (-sx))
        lw = r - 0.06
        for k in range(8):
            px_ = -sx * (lw / 8 * (k + 0.5))
            ht = min(sp + math.sqrt(max(0, r * r - (abs(px_) - 0.0) ** 2)) - 0.1 if False else sp + 0.9, 3.9)
            box(acc, "wood", Mg @ T(px_, 0, ht / 2), lw / 16 - 0.004, 0.04, ht / 2, kit.J((0.22, 0.15, 0.1), 0.07), 0.008)
        for z in (0.5, 1.9, 3.2):
            box(acc, "iron", Mg @ T(-sx * lw / 2, -0.05, z), lw / 2, 0.006, 0.04, IRON, 0.003)
            for t in np.linspace(-lw / 2 + 0.08, lw / 2 - 0.08, 7):
                box(acc, "iron", Mg @ T(-sx * lw / 2 + t, -0.057, z), 0.012, 0.006, 0.012, IRON, 0.003)
    # banners down the towers' outer faces
    for sx in (-1, 1):
        cx = sx * (pw / 2 + tw / 2)
        Mbn = T(cx - 0.45, -tw / 2 - 0.08, th - 0.2 - 2.4)
        box(acc, "wood", Mbn @ T(0.45, 0, 2.42), 0.55, 0.025, 0.025, WOOD, 0.005)
        pole = lambda U, V: np.where(((V > 0.35) & (V < 0.55))[:, None], np.array([[0.78, 0.6, 0.2]]), np.array([[0.13, 0.2, 0.36]]))
        patch(acc, "cloth", Mbn, 0.9, 2.4, 6, 12, (0.13, 0.2, 0.36), lambda U, V: 0.03 * np.sin(V * 2.2 + U * 2), pole)
        prism(acc, "cloth", Mbn @ frame((0.0, -0.01, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), [(0, 0), (0.45, 0.35), (0.9, 0), (0.9, -0.01), (0, -0.01)], 0.01,
              (0.13, 0.2, 0.36))
        kit.lantern(frame((cx - sx * 1.0, -tw / 2, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), 0.0, 3.0) if False else None
    blocks_out(kit, acc, Mb, low - floor - 0.02, low - floor, rng)
    G.grime(acc, ("stone",), z0=low - floor, height=1.0, amount=0.3)
    world = Acc()
    world.merge(acc, Mb)
    return glb.add_acc(gate["id"], world), a, floor


def wall_path():
    """Chords along the wall, skipping the gatehouses: [(x0, z0, x1, z1, pier_at_start)]"""
    out = []
    pts = PLAN.WALL
    gates = [(g["x"], g["z"]) for g in PLAN.GATES]
    for i, ((ax, az), (bx, bz)) in enumerate(zip(pts[:-1], pts[1:])):
        L = math.hypot(bx - ax, bz - az)
        dx, dz = (bx - ax) / L, (bz - az) / L
        s0, s1 = 0.0, L
        for gx, gz_ in gates:
            if math.hypot(gx - ax, gz_ - az) < 0.5:
                s0 = 5.0
            if math.hypot(gx - bx, gz_ - bz) < 0.5:
                s1 = L - 5.0
        n = max(1, int(round((s1 - s0) / SEG_LEN)))
        for k in range(n):
            t0 = s0 + (s1 - s0) * k / n
            t1 = s0 + (s1 - s0) * (k + 1) / n
            out.append((ax + dx * t0, az + dz * t0, ax + dx * t1, az + dz * t1, k == 0, k % 3 == 2))
    return out


def town_wall(glb, rng):
    kits = wall_kit(glb, rng)
    segs = wall_path()
    for i, (x0, z0, x1, z1, first, every3) in enumerate(segs):
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        L = math.hypot(x1 - x0, z1 - z0)
        d = np.array([x1 - x0, z1 - z0]) / L
        # Blender rotation: local x along the chord; local -y (the pier's side) away from the village
        a = math.atan2(-(z1 - z0), x1 - x0)
        out = np.array([cx, cz]) - np.array([3.0, 22.0])
        front = np.array([math.sin(a), -math.cos(a)])        # Blender xy of local -y
        if np.dot(np.array([front[0], -front[1]]), out) < 0:
            a += math.pi
        g = float(min(ground(x0, z0), ground(x1, z1), ground(cx, cz)))
        g = min(g + 0.06, float(ground(cx, cz)) + 0.02)
        sx = L / SEG_LEN
        Mb = T(cx, -cz, g) @ R("z", a) @ np.diag([sx, 1, 1, 1])
        G3 = C_GAME @ Mb[:3, :3] @ C_GAME.T
        t = C_GAME @ Mb[:3, 3]
        rgb = np.clip(rng.normal(1.0, 0.04, 3) * rng.normal(1.0, 0.05), 0.8, 1.1)
        name = kits[rng.integers(len(kits))]
        for mat in ("stone",):
            INST.d.setdefault(f"{name}_{mat}", []).append([G3[0, 0], G3[0, 1], G3[0, 2], t[0], G3[1, 0], G3[1, 1], G3[1, 2], t[1],
                                                           G3[2, 0], G3[2, 1], G3[2, 2], t[2], *rgb])
        COLLIDERS.append({"x": round(cx, 3), "z": round(cz, 3), "yaw": round(a, 4), "hx": round(L / 2, 3), "hz": PLAN.WALL_THICK / 2 + 0.05,
                          "y0": round(g - 1.0, 3), "y1": round(g + PLAN.WALL_HEIGHT, 3)})
        if first or every3:
            px, pz = (x0, z0) if first else (x1, z1)
            gp = float(ground(px, pz)) + 0.02
            Mp = T(px, -pz, gp) @ R("z", a)
            G3 = C_GAME @ Mp[:3, :3] @ C_GAME.T
            t = C_GAME @ Mp[:3, 3]
            INST.d.setdefault("kit_pier_stone", []).append([G3[0, 0], G3[0, 1], G3[0, 2], t[0], G3[1, 0], G3[1, 1], G3[1, 2], t[1],
                                                           G3[2, 0], G3[2, 1], G3[2, 2], t[2], *np.clip(rng.normal(1.0, 0.04, 3), 0.85, 1.1)])
    # the far ends
    for (px, pz), (qx, qz) in ((PLAN.WALL[0], PLAN.WALL[1]), (PLAN.WALL[-1], PLAN.WALL[-2])):
        a = math.atan2(-(qz - pz), qx - px)
        gp = float(ground(px, pz)) + 0.02
        Mp = T(px, -pz, gp) @ R("z", a)
        G3 = C_GAME @ Mp[:3, :3] @ C_GAME.T
        t = C_GAME @ Mp[:3, 3]
        INST.d.setdefault("kit_pier_stone", []).append([G3[0, 0], G3[0, 1], G3[0, 2], t[0], G3[1, 0], G3[1, 1], G3[1, 2], t[1],
                                                       G3[2, 0], G3[2, 1], G3[2, 2], t[2], 1.0, 1.0, 1.0])
    return len(segs)


# ---------------------------------------------------------------- the rooms you can walk into
def tavern_inside(kit, site, W, D, ops):
    """The Gull's taproom: a fire on the gable wall, a panelled bar with shelves of bottles, casks, round tables,
    a stair to the rooms above, an iron candle-ring over it all."""
    rng = kit.rng
    acc_i = Acc(rng)
    ki = Kit(acc_i, rng)
    ki.blocks = kit.blocks
    t, H = 0.5, 2.95
    room = Room(ki, -W / 2 + t, W / 2 - t, -D / 2 + t, D / 2 - t, H, t, ops, plaster=(0.85, 0.79, 0.66))
    room.walls(-W / 2, W / 2, -D / 2, D / 2, wainscot=1.05)
    room.floor_boards()
    room.ceiling(joist_dir="y")
    F = Furnish(ki)
    _, Mi_l, _ = room.inner_frame("left", -W / 2, W / 2, -D / 2, D / 2)
    F.fireplace(Mi_l, 1.6 + D / 2, w=1.9, h=1.5, depth=0.75, opening=1.15)
    # a ship's wheel over the mantel
    Mwheel = Mi_l @ T(1.6 + D / 2, -0.06, 2.3) @ R("x", math.pi / 2)
    tube(acc_i, "wood", Mwheel, [(0.32 * math.cos(a), 0.32 * math.sin(a), 0) for a in np.linspace(0, 2 * math.pi, 25)], 0.03, 6, ki.J(OAK, 0.04), cap=False)
    for a in np.linspace(0, 2 * math.pi, 8, endpoint=False):
        tube(acc_i, "wood", Mwheel, [(0.0, 0.0, 0), (0.46 * math.cos(a), 0.46 * math.sin(a), 0)], 0.018, 5, ki.J(OAK, 0.04))
    lathe(acc_i, "iron", Mwheel @ T(0, 0, -0.04), [(0.0, 0), (0.07, 0), (0.07, 0.08), (0.0, 0.08)], 10, BRASS)
    # the bar
    bx0, bx1, by = 0.6, 3.6, 2.6
    box(acc_i, "wood", T((bx0 + bx1) / 2, by, 0.52), (bx1 - bx0) / 2, 0.26, 0.52, ki.J(OAK_DARK, 0.04), 0.01)
    for x in np.arange(bx0 + 0.05, bx1 - 0.3, 0.5):
        box(acc_i, "wood", T(x + 0.25, by - 0.27, 0.52), 0.21, 0.012, 0.36, ki.J(OAK, 0.05), 0.012)
        box(acc_i, "wood", T(x + 0.25, by - 0.285, 0.52), 0.15, 0.008, 0.28, ki.J(OAK, 0.06) * 1.05, 0.01)
    box(acc_i, "wood", T((bx0 + bx1) / 2, by - 0.04, 1.075), (bx1 - bx0) / 2 + 0.08, 0.36, 0.035, ki.J(OAK, 0.04), 0.01)
    tube(acc_i, "iron", np.eye(4), [(bx0, by - 0.42, 0.16), (bx1, by - 0.42, 0.16)], 0.02, 6, BRASS)
    for x in np.arange(bx0 + 0.3, bx1, 0.9):
        tube(acc_i, "iron", np.eye(4), [(x, by - 0.42, 0.16), (x, by - 0.27, 0.06)], 0.012, 5, BRASS)
    for k, x in enumerate((0.9, 1.25, 2.4, 3.1)):
        F.tankard(T(x, by - 0.1, 1.11) @ R("z", rng.uniform(0, 6)))
    F.bottle(T(1.9, by + 0.05, 1.11), (0.12, 0.22, 0.1))
    F.candle(T(2.8, by - 0.05, 1.11))
    _, Mi_b, _ = room.inner_frame("back", -W / 2, W / 2, -D / 2, D / 2)
    bottles = [(0.1, 0.2, 0.08), (0.22, 0.12, 0.05), (0.08, 0.14, 0.12), (0.3, 0.2, 0.06)]
    for z in (1.45, 1.9, 2.35):
        F.shelf(Mi_b, bx0 + W / 2 + 0.2, 3.0 + W / 2, z)
        for x in np.arange(bx0 + 0.35, 3.0, 0.2):
            if rng.random() < 0.8:
                if rng.random() < 0.6:
                    F.bottle(T(x + rng.normal(0, 0.02), D / 2 - t - 0.13, z + 0.02), ki.J(bottles[rng.integers(4)], 0.1))
                else:
                    F.tankard(T(x, D / 2 - t - 0.13, z + 0.02) @ R("z", rng.uniform(0, 6)))
    for (cx, cy) in ((5.05, 3.3), (5.05, 2.6)):
        F.barrel(T(cx, cy, 0), r=0.32, h=0.85, col=WOOD_RAW)
        tube(acc_i, "wood", np.eye(4), [(cx - 0.3, cy, 0.25), (cx - 0.42, cy, 0.25), (cx - 0.42, cy, 0.19)], 0.018, 5, ki.J(OAK, 0.05))
    F.barrel(T(4.9, 3.0, 0.86) @ R("z", 0.6), r=0.26, h=0.7, col=WOOD_RAW)
    # tables, stools, chairs by the fire
    for (tx, ty) in ((-2.9, -2.0), (-1.2, -0.1)):
        F.table_round(T(tx, ty, 0))
        for a in np.linspace(0, 2 * math.pi, 4, endpoint=False) + rng.uniform(0, 1):
            F.stool(T(tx + 0.85 * math.cos(a), ty + 0.85 * math.sin(a), 0))
        for k in range(rng.integers(2, 4)):
            a = rng.uniform(0, 6.28)
            F.tankard(T(tx + 0.3 * math.cos(a), ty + 0.3 * math.sin(a), 0.785) @ R("z", rng.uniform(0, 6)))
        F.plate(T(tx + 0.1, ty - 0.15, 0.785))
        F.candle(T(tx, ty + 0.05, 0.785))
        site.collider(tx, ty, 0.68, 0.68, y0=-2, y1=0.8)
    F.table_long(T(2.6, -2.55, 0), 1.8, 0.8)
    F.bench(T(2.6, -3.2, 0), 1.7)
    F.bench(T(2.6, -1.9, 0), 1.7)
    for x in (2.0, 2.9, 3.3):
        F.tankard(T(x, -2.5 + rng.normal(0, 0.1), 0.785) @ R("z", rng.uniform(0, 6)))
    F.plate(T(2.4, -2.6, 0.785))
    blob(acc_i, "produce", T(2.4, -2.6, 0.83), 0.09, (0.62, 0.42, 0.2), seg=8, rings=5, squash=0.55, rng=rng)
    site.collider(2.6, -2.55, 0.95, 0.9, y0=-2, y1=0.8)
    for cy in (0.9, 2.3):
        F.chair(T(-3.55, cy, 0) @ R("z", -math.pi / 2 + rng.normal(0, 0.15)))
    F.table_round(T(-3.5, 1.6, 0) @ np.diag([0.6, 0.6, 1.0, 1.0]), r=0.6)
    F.tankard(T(-3.5, 1.55, 0.785))
    F.rug(T(-4.0, 1.6, 0) @ R("z", math.pi / 2), 1.9, 1.1, (0.48, 0.16, 0.12), (0.75, 0.62, 0.3))
    site.collider(-W / 2 + t + 0.4, 1.6, 0.42, 0.98, y0=-2, y1=2.6)
    site.collider(-3.55, 1.6, 0.35, 1.0, y0=-2, y1=0.9)
    site.collider((bx0 + bx1) / 2, by, (bx1 - bx0) / 2 + 0.08, 0.32, y0=-2, y1=1.12)
    site.collider(5.05, 2.95, 0.4, 0.75, y0=-2, y1=1.6)
    # stairs up the right-hand wall to the rooms above, and the hatch they reach
    F.stairs(T(5.02, -3.75, 0) @ R("z", math.pi / 2), L=3.1, H=H, w=0.85, steps=13, rail=1)
    quad(acc_i, "wood", np.eye(4), [(4.6, -0.66, H - 0.01), (4.6, 0.3, H - 0.01), (5.5, 0.3, H - 0.01), (5.5, -0.66, H - 0.01)], (0.02, 0.015, 0.01))
    site.collider(5.05, -2.2, 0.45, 1.55, y0=-2, y1=H)
    F.chandelier(T(-0.6, 0.45, H - 1.15))
    door = ops["front"][0]
    hollow(site, ki, W, D, H, t, "front", door,
           [(-2.6, 1.0, 1.5, [1.0, 0.6, 0.32], 2.6, 12.0)], "갈매기 여관", "tavern")
    return acc_i


def chapel_inside(kit, site, W, D, ops, zr, pitch):
    """The nave: flagstones, whitewashed walls with a painted band, open oak trusses under the roof, pews either
    side of the aisle, a raised chancel with the altar, candles and a lectern."""
    rng = kit.rng
    acc_i = Acc(rng)
    ki = Kit(acc_i, rng)
    ki.blocks = kit.blocks
    t, H = 0.45, 5.2
    tp = math.tan(pitch)
    room = Room(ki, -W / 2 + t, W / 2 - t, -D / 2 + t, D / 2 - t, H, t, ops, plaster=(0.88, 0.86, 0.79))
    room.walls(-W / 2, W / 2, -D / 2, D / 2, wainscot=0)
    room.flagstones((0.55, 0.53, 0.49))
    F = Furnish(ki)
    for side in ("front", "right", "back", "left"):
        Mw, Mi, L = room.inner_frame(side, -W / 2, W / 2, -D / 2, D / 2)
        holes = []
        for o in ops.get(side, []):
            u0, u1, v0, v1 = Kit.hole(o)
            holes.append((L - u1 - t, L - u0 - t, v0, v1))
        for (a, b) in room._spans(L - 2 * t, holes, 0.4):
            box(acc_i, "stone", Mi @ T(t + (a + b) / 2, -0.04, 0.17), (b - a) / 2, 0.05, 0.17, ki.tone((0.56, 0.54, 0.5), 0.04), 0.012)
        for (z, col) in ((4.55, (0.52, 0.22, 0.16)), (4.68, (0.2, 0.26, 0.36))):
            box(acc_i, "plaster", Mi @ T(L / 2, -0.004, z), L / 2 - t, 0.004, 0.05, col, 0)
    # the gable walls' inner faces, up under the roof boards
    deck = lambda x: 5.18 + (W / 2 - abs(x)) * tp
    apex = deck(0.0) - 0.04
    hw = lambda z: min(W / 2 - t, W / 2 - (z - 5.18) / tp)
    for yface, facing, hole in ((-D / 2 + t, 1, (-0.32, 0.32, H + 0.65, H + 0.7 + 0.8 + 0.35)), (D / 2 - t, -1, None)):
        zs = [H] + ([hole[2], hole[3]] if hole else []) + [apex]
        M = frame((0, yface, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1)) if facing > 0 else frame((0, yface, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        for za, zb in zip(zs[:-1], zs[1:]):
            inhole = hole and za >= hole[2] - 1e-6 and zb <= hole[3] + 1e-6
            parts = [(-9, hole[0]), (hole[1], 9)] if inhole else [(-9, 9)]
            for xa, xb in parts:
                pts = [(max(xa, -hw(za)), za), (min(xb, hw(za)), za), (min(xb, hw(zb)), zb), (max(xa, -hw(zb)), zb)]
                clean = [p for i, p in enumerate(pts) if i == 0 or abs(p[0] - pts[i - 1][0]) + abs(p[1] - pts[i - 1][1]) > 1e-4]
                if len(clean) >= 3 and (clean[1][0] - clean[0][0]) > 0.01:
                    prism(acc_i, "plaster", M, clean, 0.02, (0.86, 0.84, 0.77), cap_back=False)
    # oak trusses, purlins, ridge and common rafters
    rz = lambda x: deck(x) - 0.1
    for y in (-3.2, 0.0, 3.2):
        box(acc_i, "wood", T(0, y, 5.32), W / 2 - 0.1, 0.12, 0.13, ki.J(OAK_DARK, 0.04), 0.02)
        for s in (-1, 1):
            G.beam(acc_i, "wood", (s * (W / 2 - 0.35), y, rz(W / 2 - 0.35) - 0.08), (0, y, apex - 0.12), 0.18, 0.2, ki.J(OAK_DARK, 0.04), up=(0, 1, 0))
            G.beam(acc_i, "wood", (0, y, 5.95), (s * 1.25, y, rz(1.25) - 0.2), 0.12, 0.14, ki.J(OAK_DARK, 0.04), up=(0, 1, 0))
        box(acc_i, "wood", T(0, y, (5.45 + apex - 0.12) / 2), 0.1, 0.1, (apex - 0.12 - 5.45) / 2, ki.J(OAK_DARK, 0.04), 0.015)
        lathe(acc_i, "wood", T(0, y, 5.13), [(0.0, 0), (0.09, 0.08), (0.0, 0.2)], 8, ki.J(OAK_DARK, 0.04))
    for s in (-1, 1):
        box(acc_i, "wood", T(s * 1.3, 0, rz(1.3) - 0.14), 0.08, D / 2 - t, 0.1, ki.J(OAK_DARK, 0.04), 0.015)
        box(acc_i, "wood", T(s * (W / 2 - t - 0.06), 0, 5.12), 0.1, D / 2 - t, 0.08, ki.J(OAK_DARK, 0.04), 0.015)
        for y in np.arange(-D / 2 + t + 0.3, D / 2 - t, 0.6):
            G.beam(acc_i, "wood", (s * (W / 2 - t), y, rz(W / 2 - t) + 0.02), (0, y, apex - 0.05), 0.09, 0.12, ki.J(OAK, 0.05), up=(0, 1, 0))
    box(acc_i, "wood", T(0, 0, apex - 0.1), 0.08, D / 2 - t, 0.1, ki.J(OAK_DARK, 0.04), 0.015)
    # pews either side of the aisle, all facing the altar
    for x in (-1.6, 1.6):
        for y in np.arange(-4.4, 1.7, 1.0):
            F.pew(T(x, y, 0) @ R("z", math.pi), 2.2)
        site.collider(x, -1.35, 1.15, 3.4, y0=-2, y1=1.05)
    # the chancel: a step up, the altar dressed in linen, candles, a brass cross; a lectern below
    cy0 = 3.3
    box(acc_i, "stone", T(0, (cy0 + D / 2 - t) / 2, 0.09), W / 2 - t, (D / 2 - t - cy0) / 2, 0.09, ki.tone((0.6, 0.58, 0.53), 0.03), 0.015)
    box(acc_i, "stone", T(0, cy0 - 0.02, 0.17), W / 2 - t, 0.03, 0.02, ki.tone((0.62, 0.6, 0.55), 0.03), 0.008)
    site.collider(0, (cy0 + D / 2 - t) / 2, W / 2 - t, (D / 2 - t - cy0) / 2, y0=-2, y1=0.18)
    ay = 5.05
    box(acc_i, "stone", T(0, ay, 0.18 + 0.5), 0.82, 0.36, 0.5, ki.tone((0.62, 0.6, 0.55), 0.03), 0.02)
    box(acc_i, "cloth", T(0, ay, 1.2), 0.86, 0.4, 0.012, (0.9, 0.88, 0.82), 0.004)
    patch(acc_i, "cloth", T(-0.86, ay - 0.405, 0.82), 1.72, 0.38, 12, 3, (0.9, 0.88, 0.82), lambda U, V: 0.006 * np.sin(U * 25))
    patch(acc_i, "cloth", T(-0.2, ay - 0.42, 0.55), 0.4, 0.66, 3, 4, (0.55, 0.12, 0.1))
    box(acc_i, "cloth", T(0, ay, 1.215), 0.2, 0.41, 0.004, (0.55, 0.12, 0.1), 0)
    for sx in (-0.6, 0.6):
        lathe(acc_i, "iron", T(sx, ay, 1.21), [(0.0, 0), (0.08, 0.0), (0.08, 0.02), (0.025, 0.05), (0.02, 0.3), (0.04, 0.33), (0.045, 0.36), (0.0, 0.36)], 10, BRASS)
        F.candle(T(sx, ay, 1.57), h=0.22, holder=False)
    tube(acc_i, "iron", np.eye(4), [(0, ay + 0.1, 1.21), (0, ay + 0.1, 1.78)], 0.022, 6, BRASS)
    tube(acc_i, "iron", np.eye(4), [(-0.17, ay + 0.1, 1.6), (0.17, ay + 0.1, 1.6)], 0.022, 6, BRASS)
    for sx in (-1.5, 1.5):
        M = T(sx, 4.1, 0.18)
        lathe(acc_i, "iron", M, [(0.0, 0), (0.2, 0), (0.05, 0.08), (0.03, 1.3), (0.1, 1.36), (0.0, 1.38)], 8, IRON)
        F.candle(M @ T(0, 0, 1.38), h=0.3, holder=False)
    site.collider(0, ay, 0.86, 0.4, y0=-2, y1=1.25)
    M = T(1.7, 2.6, 0)
    lathe(acc_i, "wood", M, [(0.22, 0), (0.18, 0.06), (0.05, 0.12), (0.05, 1.0), (0.0, 1.0)], 10, ki.J(OAK, 0.04))
    box(acc_i, "wood", M @ T(0, 0, 1.08) @ R("x", -0.45), 0.25, 0.2, 0.02, ki.J(OAK, 0.04), 0.006)
    box(acc_i, "cloth", M @ T(0, 0.01, 1.11) @ R("x", -0.45), 0.2, 0.15, 0.025, (0.45, 0.1, 0.08), 0.008)
    site.collider(1.7, 2.6, 0.25, 0.25, y0=-2, y1=1.1)
    door = ops["front"][0]
    hollow(site, ki, W, D, H, t, "front", door, [(0.0, 4.4, 2.0, [1.0, 0.8, 0.55], 1.6, 12.0)], "바닷가 예배당", "chapel")
    return acc_i


def fisher_inside(kit, site, W, D, ops):
    """Bren's one room: a hearth with a kettle, a box bed under a patchwork quilt, a sea chest, a dresser of
    plates with a model ship on top, a table laid with fish and bread, nets hung from the joists."""
    rng = kit.rng
    acc_i = Acc(rng)
    ki = Kit(acc_i, rng)
    ki.blocks = kit.blocks
    t, H = 0.45, 2.45
    room = Room(ki, -W / 2 + t, W / 2 - t, -D / 2 + t, D / 2 - t, H, t, ops, plaster=(0.8, 0.76, 0.66))
    room.walls(-W / 2, W / 2, -D / 2, D / 2, wainscot=0)
    room.flagstones((0.5, 0.48, 0.44))
    room.ceiling(col=(0.22, 0.16, 0.11), joist_dir="x", spacing=0.55)
    F = Furnish(ki)
    _, Mi_b, _ = room.inner_frame("back", -W / 2, W / 2, -D / 2, D / 2)
    F.fireplace(Mi_b, W / 2, w=1.7, h=1.35, depth=0.6, opening=1.0, stone=(0.5, 0.48, 0.44))
    site.collider(0, D / 2 - t - 0.3, 0.9, 0.35, y0=-2, y1=H)
    F.bed(T(-1.88, D / 2 - t - 0.6, 0), L=1.85, Wd=1.1)
    site.collider(-1.88, D / 2 - t - 0.6, 0.95, 0.58, y0=-2, y1=1.0)
    F.chest(T(-W / 2 + t + 0.3, 1.2, 0) @ R("z", math.pi / 2), col=(0.2, 0.28, 0.32))
    site.collider(-W / 2 + t + 0.3, 1.2, 0.28, 0.47, y0=-2, y1=0.62)
    F.table_long(T(0.9, -0.6, 0) @ R("z", math.pi / 2), 1.25, 0.72)
    for (sx, sy) in ((0.25, -0.95), (0.25, -0.2), (1.6, -0.6)):
        F.stool(T(sx, sy, 0))
    F.plate(T(0.85, -0.75, 0.785))
    for k in range(2):     # a pair of fish on the plate
        y0_ = -0.82 + k * 0.1
        pts = [(0.72 + i * 0.045, y0_, 0.815) for i in range(7)]
        tube(acc_i, "produce", np.eye(4), pts, [0.012, 0.024, 0.03, 0.03, 0.026, 0.016, 0.006], 7, (0.62, 0.66, 0.68))
        prism(acc_i, "produce", T(0.72 + 0.3, y0_, 0.815) @ frame((0, 0, 0), (1, 0, 0), (0, 0, -1), (0, 1, 0)), [(0, 0), (0.06, 0.03), (0.06, -0.03)], 0.004,
              (0.55, 0.58, 0.6))
    blob(acc_i, "produce", T(1.0, -0.35, 0.82), 0.08, (0.6, 0.42, 0.2), seg=8, rings=5, squash=0.6, rng=rng)
    F.tankard(T(0.95, -0.15, 0.785))
    F.candle(T(0.75, -0.3, 0.785))
    site.collider(0.9, -0.6, 0.8, 0.8, y0=-2, y1=0.8)
    F.dresser(T(W / 2 - t - 0.27, 0.6, 0) @ R("z", -math.pi / 2), 1.3)
    site.collider(W / 2 - t - 0.27, 0.6, 0.3, 0.68, y0=-2, y1=2.1)
    # a model ship on the dresser
    Ms = T(W / 2 - t - 0.2, 0.6, 2.04) @ R("z", math.pi / 2)
    lathe(acc_i, "wood", Ms @ R("y", math.pi / 2) @ np.diag([1.0, 0.55, 1.0, 1.0]), [(0.0, -0.22), (0.05, -0.15), (0.06, 0.0), (0.05, 0.16), (0.0, 0.24)], 8,
          ki.J((0.3, 0.2, 0.12), 0.05))
    tube(acc_i, "wood", Ms, [(0, 0, 0.02), (0, 0, 0.36)], 0.006, 4, (0.25, 0.18, 0.1))
    patch(acc_i, "cloth", Ms @ T(-0.1, 0.0, 0.1), 0.2, 0.22, 3, 3, (0.85, 0.8, 0.68), lambda U, V: 0.02 * np.sin(np.pi * U / 0.2))
    F.hanging_net(T(-W / 2 + t + 0.15, -1.3, H - 0.08) @ R("z", math.pi / 2), 2.2, 1.0)
    Mw, _, _ = room.inner_frame("front", -W / 2, W / 2, -D / 2, D / 2)
    for k, x in enumerate((-2.55, -2.3)):
        tube(acc_i, "wood", np.eye(4), [(x, -D / 2 + t + 0.12, 0.0), (x + 0.15, -D / 2 + t + 0.06, 2.1)], 0.028, 6, ki.J(WOOD_RAW, 0.06))
        box(acc_i, "wood", T(x + 0.02, -D / 2 + t + 0.12, 0.3), 0.07, 0.012, 0.3, ki.J(WOOD_RAW, 0.06), 0.005)
    F.barrel(T(2.35, -D / 2 + t + 0.4, 0), r=0.3, h=0.8, col=WOOD_GREY)
    site.collider(2.35, -D / 2 + t + 0.4, 0.32, 0.32, y0=-2, y1=0.85)
    pts = [((0.1 + a * 0.012) * math.cos(a), (0.1 + a * 0.012) * math.sin(a), 0.025 + 0.02 * (a // 6.28)) for a in np.linspace(0, 6 * 2 * math.pi, 180)]
    tube(acc_i, "cloth", T(-1.6, -D / 2 + t + 0.45, 0), pts, 0.018, 5, ki.J((0.55, 0.48, 0.34), 0.05), cap=False)
    F.rug(T(0.3, 1.2, 0), 1.5, 0.95, (0.22, 0.3, 0.42), (0.62, 0.55, 0.4))
    door = ops["front"][0]
    hollow(site, ki, W, D, H, t, "front", door, [(0.0, D / 2 - t - 1.0, 0.7, [1.0, 0.52, 0.22], 2.0, 8.0)], "브렌의 집", "fisher")
    return acc_i


def smithy_inside(kit, site, W, D, ops):
    """The smith's shop: a brick forge with hood and bellows against the gable wall, an anvil on its stump,
    a stone quench trough, a workbench with a leg vise, a board of tongs and hammers, a rack of finished swords and
    axes, painted shields on the wall, a bin of coal and a cask of bar iron."""
    rng = kit.rng
    acc_i = Acc(rng)
    ki = Kit(acc_i, rng)
    ki.blocks = kit.blocks
    t, H = 0.5, 2.85
    room = Room(ki, -W / 2 + t, W / 2 - t, -D / 2 + t, D / 2 - t, H, t, ops, plaster=(0.72, 0.68, 0.6))
    room.walls(-W / 2, W / 2, -D / 2, D / 2, wainscot=0)
    room.flagstones((0.42, 0.40, 0.37))
    room.ceiling(col=(0.2, 0.15, 0.1))
    F = Furnish(ki)
    _, Mi_r, _ = room.inner_frame("right", -W / 2, W / 2, -D / 2, D / 2)
    F.forge(Mi_r, 2.9, w=1.6, d=0.95)
    site.collider(W / 2 - t - 0.48, 0.6, 0.5, 0.85, y0=-2, y1=2.6)
    site.collider(W / 2 - t - 0.45, -0.75, 0.42, 0.38, y0=-2, y1=1.2)
    # the anvil
    ax_, ay_ = 2.2, 0.6
    lathe(acc_i, "wood", T(ax_, ay_, 0), [(0.34, 0), (0.33, 0.55), (0.0, 0.56)], 12, ki.J(WOOD_GREY, 0.05), smooth=False)
    Ma = T(ax_, ay_, 0.56) @ R("z", math.pi / 2)
    box(acc_i, "iron", Ma @ T(0, 0, 0.12), 0.14, 0.11, 0.12, IRON, 0.01)
    box(acc_i, "iron", Ma @ T(0, 0, 0.3), 0.3, 0.09, 0.06, IRON, 0.01)
    prism(acc_i, "iron", Ma @ frame((0.3, 0, 0.3), (1, 0, 0), (0, 1, 0), (0, 0, 1)), [(0, -0.06), (0.22, 0.03), (0, 0.06)], 0.14, IRON)
    box(acc_i, "iron", Ma @ T(0, 0, 0.02), 0.2, 0.14, 0.03, IRON, 0.008)
    tube(acc_i, "iron", Ma, [(-0.1, 0.03, 0.37), (0.12, 0.03, 0.37)], 0.016, 6, (0.3, 0.2, 0.12))
    box(acc_i, "iron", Ma @ T(0.16, 0.03, 0.39), 0.045, 0.03, 0.03, IRON, 0.005)
    prism(acc_i, "produce", Ma @ frame((-0.15, -0.02, 0.365), (1, 0, 0), (0, 0, 1), (0, -1, 0)), [(-0.12, -0.012), (0.08, -0.012), (0.14, 0.0), (0.08, 0.012), (-0.12, 0.012)],
          0.01, (1.0, 0.45, 0.12))     # a bar still glowing from the fire
    site.collider(ax_, ay_, 0.36, 0.36, y0=-2, y1=1.0)
    F.trough(T(2.6, 2.3, 0), 1.2, 0.55, 0.6)
    site.collider(2.6, 2.3, 0.62, 0.3, y0=-2, y1=0.62)
    F.workbench(T(-2.2, 2.55, 0) @ R("z", math.pi), 2.6, 0.7)
    site.collider(-2.2, 2.55, 1.32, 0.37, y0=-2, y1=0.95)
    F.weapon_rack(T(-2.6, -0.6, 0), 6)
    site.collider(-2.6, -0.6, 0.86, 0.3, y0=-2, y1=1.4)
    F.coal_bin(T(3.35, -1.95, 0), 0.9, 0.6)
    site.collider(3.35, -1.95, 0.48, 0.33, y0=-2, y1=0.62)
    F.barrel(T(1.1, 2.55, 0), r=0.28, h=0.75, col=WOOD_GREY)
    for i in range(7):       # bar iron standing in the cask
        a = rng.uniform(0, 6.28)
        tube(acc_i, "iron", np.eye(4), [(1.1 + 0.12 * math.cos(a), 2.55 + 0.12 * math.sin(a), 0.4), (1.1 + 0.18 * math.cos(a), 2.55 + 0.18 * math.sin(a), 1.35 + rng.uniform(0, 0.25))],
             0.012, 4, (0.2, 0.2, 0.21))
    site.collider(1.1, 2.55, 0.3, 0.3, y0=-2, y1=0.9)
    # a board of tools on the left wall, shields on the back wall
    _, Mi_l, _ = room.inner_frame("left", -W / 2, W / 2, -D / 2, D / 2)
    box(acc_i, "wood", Mi_l @ T(2.3, -0.025, 1.55), 0.85, 0.025, 0.45, ki.J(OAK_DARK, 0.05), 0.008)
    for i in range(11):
        u = 1.6 + i * 0.14
        if i % 4 == 3:
            tube(acc_i, "iron", Mi_l, [(u + 0.06 * math.cos(a), -0.06, 1.75 + 0.06 * math.sin(a)) for a in np.linspace(-0.4, math.pi + 0.4, 9)], 0.009, 4, IRON)
        elif i % 4 == 1:
            tube(acc_i, "wood", Mi_l, [(u, -0.07, 1.85), (u, -0.07, 1.4)], 0.013, 5, ki.J(WOOD_RAW, 0.05))
            box(acc_i, "iron", Mi_l @ T(u, -0.07, 1.86), 0.05, 0.025, 0.025, IRON, 0.005)
        else:
            tube(acc_i, "iron", Mi_l, [(u - 0.02, -0.06, 1.85), (u + 0.01, -0.06, 1.3)], 0.008, 4, IRON)
            tube(acc_i, "iron", Mi_l, [(u + 0.02, -0.06, 1.85), (u - 0.01, -0.06, 1.3)], 0.008, 4, IRON)
    _, Mi_b, _ = room.inner_frame("back", -W / 2, W / 2, -D / 2, D / 2)
    for k, (u, col) in enumerate(((5.7, (0.48, 0.14, 0.12)), (6.6, (0.16, 0.24, 0.38)), (7.5, (0.2, 0.32, 0.2)))):
        F.shield(Mi_b @ T(u, -0.05, 1.8) @ R("x", math.pi / 2) @ R("z", rng.normal(0, 0.1)), 0.3, col)
    door = ops["front"][0]
    hollow(site, ki, W, D, H, t, "front", door, [(2.6, 0.6, 1.2, [1.0, 0.45, 0.18], 2.6, 9.0)], "대장간", "smithy")
    return acc_i


def bakery_inside(kit, site, W, D, ops):
    """The bake-house: a counter behind the shop window piled with baskets of bread, racks of loaves, a kneading
    table with dough rising under a cloth, flour sacks, the bread oven's arched mouth glowing in the gable wall with
    its peel beside it, and a small hearth for the kettle."""
    rng = kit.rng
    acc_i = Acc(rng)
    ki = Kit(acc_i, rng)
    ki.blocks = kit.blocks
    t, H = 0.22, 2.85
    oven_u = D - (1.6 + D / 2)       # inner u' on the right wall (L - (y - y0)); the oven stands outside at y = 1.6
    ov_w, ov_sill, ov_h = 0.66, 0.82, 0.38
    room = Room(ki, -W / 2 + t, W / 2 - t, -D / 2 + t, D / 2 - t, H, t, ops, plaster=(0.88, 0.82, 0.7))
    room.walls(-W / 2, W / 2, -D / 2, D / 2, wainscot=0.9)
    room.floor_boards((0.42, 0.34, 0.24))
    room.ceiling()
    F = Furnish(ki)
    # the oven mouth: a brick arch, a soot-black throat, embers and loaves inside, the iron door set aside
    _, Mi_r, _ = room.inner_frame("right", -W / 2, W / 2, -D / 2, D / 2)
    Mo = Mi_r @ T(oven_u, 0, 0)
    r = ov_w / 2
    a0, a1 = ov_sill - 0.15, ov_sill + ov_h + r + 0.25
    face = [(-r - 0.2, a0), (r + 0.2, a0), (r + 0.2, a1), (-r - 0.2, a1)]
    brick = (0.56, 0.33, 0.24)
    hole = [(r, ov_sill), (r, ov_sill + ov_h)] + [(r * math.cos(t_), ov_sill + ov_h + r * math.sin(t_)) for t_ in np.linspace(0.15, math.pi - 0.15, 9)] + [(-r, ov_sill + ov_h), (-r, ov_sill)]
    for (p0, p1) in zip(hole[:-1], hole[1:]):          # brick voussoirs and jambs round the opening
        m = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 - (ov_sill + ov_h))
        dv = np.array(m) / (np.linalg.norm(m) + 1e-9) if m[1] > 0 else np.array([np.sign(m[0]), 0.0])
        poly = [p0, p1, (p1[0] + dv[0] * 0.2, p1[1] + dv[1] * 0.2), (p0[0] + dv[0] * 0.2, p0[1] + dv[1] * 0.2)]
        prism(acc_i, "stone", Mo @ frame((0, -0.06, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), poly, 0.1, ki.tone(brick, 0.08))
    R0, top_ = r + 0.22, ov_sill + ov_h
    Mface = Mo @ frame((0, -0.02, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    arch_pts = [(r * math.cos(t_), top_ + r * math.sin(t_)) for t_ in np.linspace(math.pi, 0, 11)]
    for poly in ([(-R0, top_), (-r, top_)] + arch_pts[1:-1] + [(r, top_), (R0, top_), (R0, a1), (-R0, a1)],
                 [(-R0, a0), (-r, a0), (-r, top_), (-R0, top_)], [(r, a0), (R0, a0), (R0, top_), (r, top_)], [(-r, a0), (r, a0), (r, ov_sill), (-r, ov_sill)]):
        prism(acc_i, "stone", Mface, poly, 0.04, ki.tone((0.5, 0.3, 0.22), 0.06))
    box(acc_i, "stone", Mo @ T(0, -0.1, ov_sill - 0.07), r + 0.28, 0.12, 0.07, ki.tone((0.55, 0.52, 0.46), 0.04), 0.012)
    deep = 0.9
    soot = (0.05, 0.04, 0.035)
    quad(acc_i, "stone", Mo, [(-r, deep, ov_sill), (r, deep, ov_sill), (r, deep, ov_sill + ov_h + r), (-r, deep, ov_sill + ov_h + r)], (0.35, 0.12, 0.04))
    quad(acc_i, "stone", Mo, [(-r, 0.0, ov_sill), (-r, deep, ov_sill), (-r, deep, ov_sill + ov_h + r), (-r, 0.0, ov_sill + ov_h + r)], soot)
    quad(acc_i, "stone", Mo, [(r, 0.0, ov_sill), (r, 0.0, ov_sill + ov_h + r), (r, deep, ov_sill + ov_h + r), (r, deep, ov_sill)], soot)
    quad(acc_i, "stone", Mo, [(-r, 0.0, ov_sill + ov_h + r * 0.85), (-r, deep, ov_sill + ov_h + r * 0.85), (r, deep, ov_sill + ov_h + r * 0.85), (r, 0.0, ov_sill + ov_h + r * 0.85)], soot)
    quad(acc_i, "stone", Mo, [(-r, 0.0, ov_sill + 0.005), (r, 0.0, ov_sill + 0.005), (r, deep, ov_sill + 0.005), (-r, deep, ov_sill + 0.005)], (0.6, 0.22, 0.06))
    for _ in range(18):
        blob(acc_i, "produce", Mo @ T(rng.uniform(-r + 0.05, r - 0.05), rng.uniform(0.55, deep - 0.05), ov_sill + 0.02), rng.uniform(0.02, 0.035),
             (1.0, rng.uniform(0.35, 0.6), 0.08), seg=5, rings=3, squash=0.6, rng=rng)
    for i in range(3):
        F.loaf(Mo @ T(-0.16 + i * 0.16, 0.3 + 0.08 * (i % 2), ov_sill + 0.01) @ np.diag([0.8, 0.8, 0.8, 1.0]), 0)
    box(acc_i, "iron", Mo @ T(r + 0.42, -0.05, 0.45) @ R("y", 0.18), 0.34, 0.015, 0.3, (0.08, 0.08, 0.08), 0.008)
    tube(acc_i, "iron", Mo, [(r + 0.42, -0.08, 0.5), (r + 0.42, -0.12, 0.5), (r + 0.42, -0.12, 0.42), (r + 0.42, -0.08, 0.42)], 0.008, 4, IRON)
    for k in range(2):
        F.peel(Mo @ T(-r - 0.45 - k * 0.22, -0.18, 0) @ R("x", -0.12) @ R("y", 0.06 * (k - 0.5)))
    site.collider(W / 2 - t - 0.25, 1.6, 0.25, 0.9, y0=-2, y1=1.0)
    # the shop counter behind the window
    cx0, cx1, cy = -0.45, 2.65, -D / 2 + t + 0.32
    box(acc_i, "wood", T((cx0 + cx1) / 2, cy, 0.47), (cx1 - cx0) / 2, 0.3, 0.47, ki.J(OAK, 0.05), 0.01)
    for x in np.arange(cx0 + 0.05, cx1 - 0.3, 0.52):
        box(acc_i, "wood", T(x + 0.26, cy + 0.31, 0.47), 0.22, 0.012, 0.34, ki.J(OAK, 0.06) * 1.05, 0.01)
    box(acc_i, "wood", T((cx0 + cx1) / 2, cy, 0.96), (cx1 - cx0) / 2 + 0.05, 0.34, 0.03, ki.J(OAK, 0.04), 0.008)
    for i, x in enumerate((-0.1, 0.65, 1.45, 2.25)):
        F.basket(T(x, cy - 0.02, 0.99), 0.2, 0.12, 3 + i % 2)
    # a balance scale
    Ms = T(1.05, cy + 0.2, 0.99)
    tube(acc_i, "iron", Ms, [(0, 0, 0), (0, 0, 0.35)], 0.01, 5, BRASS)
    tube(acc_i, "iron", Ms, [(-0.16, 0, 0.34), (0.16, 0, 0.34)], 0.007, 4, BRASS)
    for s in (-0.16, 0.16):
        tube(acc_i, "iron", Ms, [(s, 0, 0.34), (s, 0, 0.18)], 0.002, 3, BRASS)
        lathe(acc_i, "iron", Ms @ T(s, 0, 0.16), [(0.0, 0), (0.07, 0.0), (0.08, 0.02), (0.0, 0.01)], 10, BRASS)
    lathe(acc_i, "iron", Ms, [(0.0, 0), (0.06, 0), (0.06, 0.015), (0.0, 0.015)], 10, BRASS)
    site.collider((cx0 + cx1) / 2, cy, (cx1 - cx0) / 2 + 0.05, 0.34, y0=-2, y1=1.0)
    # racks of bread, a kneading table, flour
    F.bread_rack(T(-2.6, 2.3, 0), 1.5, 0.45, 4)
    site.collider(-2.6, 2.3, 0.78, 0.26, y0=-2, y1=1.9)
    F.bread_rack(T(-W / 2 + t + 0.3, -2.0, 0) @ R("z", math.pi / 2), 1.5, 0.45, 4)
    site.collider(-W / 2 + t + 0.3, -2.0, 0.26, 0.78, y0=-2, y1=1.9)
    F.kneading_table(T(0.5, 0.4, 0), 1.8, 0.8)
    F.stool(T(0.5, 1.2, 0))
    site.collider(0.5, 0.4, 0.95, 0.45, y0=-2, y1=0.85)
    for k, (sx, sy) in enumerate(((-3.2, 3.4), (-2.75, 3.5), (-3.0, 3.05), (-3.05, 3.35))):
        ki.sack(T(sx, sy, 0.0 if k < 3 else 0.42) @ R("z", rng.uniform(0, 6)), 0.25)
    site.collider(-3.0, 3.3, 0.45, 0.45, y0=-2, y1=0.9)
    F.barrel(T(-0.5, D / 2 - t - 0.35, 0), r=0.3, h=0.8, col=WOOD_RAW)
    patch(acc_i, "cloth", T(-0.8, D / 2 - t - 0.05, 0.81), 0.6, 0.6, 3, 3, (0.93, 0.92, 0.88))
    site.collider(-0.5, D / 2 - t - 0.35, 0.3, 0.3, y0=-2, y1=0.85)
    # a small hearth for the kettle, on the back wall
    _, Mi_b, _ = room.inner_frame("back", -W / 2, W / 2, -D / 2, D / 2)
    F.fireplace(Mi_b, 2.9 + W / 2, w=1.2, h=1.25, depth=0.5, opening=0.7, stone=(0.5, 0.46, 0.4))
    site.collider(2.9, D / 2 - t - 0.25, 0.62, 0.28, y0=-2, y1=2.5)
    F.chandelier(T(0.5, 0.4, H - 1.1), r=0.4, n=6)
    door = ops["front"][0]
    hollow(site, ki, W, D, H, t, "front", door, [(1.6, 0.9, 1.9, [1.0, 0.7, 0.4], 2.0, 10.0)], "빵집", "bakery")
    return acc_i


def weaver_inside(kit, site, W, D, ops):
    """The tall house's ground floor is a weaver's shop: a floor loom before the shop window, a spinning wheel,
    shelves of cloth bolts, a cutting table, baskets of wool, a small fire."""
    rng = kit.rng
    acc_i = Acc(rng)
    ki = Kit(acc_i, rng)
    ki.blocks = kit.blocks
    t, H = 0.22, 2.75
    room = Room(ki, -W / 2 + t, W / 2 - t, -D / 2 + t, D / 2 - t, H, t, ops, plaster=(0.86, 0.78, 0.6))
    room.walls(-W / 2, W / 2, -D / 2, D / 2, wainscot=0.95)
    room.floor_boards((0.34, 0.25, 0.17))
    room.ceiling()
    F = Furnish(ki)
    _, Mi_r, _ = room.inner_frame("right", -W / 2, W / 2, -D / 2, D / 2)
    F.fireplace(Mi_r, 2.25, w=1.3, h=1.3, depth=0.55, opening=0.8, stone=(0.5, 0.46, 0.4))
    site.collider(W / 2 - t - 0.3, 2.0, 0.32, 0.7, y0=-2, y1=2.5)
    F.loom(T(0.75, -1.35, 0), 1.4)
    site.collider(0.75, -1.6, 0.78, 1.1, y0=-2, y1=1.8)
    F.spinning_wheel(T(-1.9, 0.5, 0) @ R("z", 0.4))
    site.collider(-1.9, 0.5, 0.45, 0.35, y0=-2, y1=1.1)
    _, Mi_l, _ = room.inner_frame("left", -W / 2, W / 2, -D / 2, D / 2)
    F.cloth_bolts(Mi_l, 5.4, 7.9, (0.35, 0.95, 1.55))
    site.collider(-W / 2 + t + 0.21, 2.4, 0.22, 1.25, y0=-2, y1=2.0)
    F.table_long(T(0.5, 2.5, 0), 1.6, 0.8)
    patch(acc_i, "cloth", T(-0.2, 2.75, 0.79), 1.2, 0.5, 6, 3, (0.55, 0.15, 0.12), lambda U, V: 0.004 * np.sin(U * 11))
    for s in (-1, 1):
        tube(acc_i, "iron", T(0.9, 2.3, 0.8) @ R("z", s * 0.25), [(0, 0, 0), (0.16, 0, 0)], 0.006, 4, IRON)
    F.stool(T(0.5, 1.7, 0))
    site.collider(0.5, 2.5, 0.85, 0.45, y0=-2, y1=0.85)
    for (bx, by) in ((-2.5, -0.9), (-2.5, -1.55)):
        lathe(acc_i, "wood", T(bx, by, 0), [(0.18, 0.0), (0.24, 0.35), (0.25, 0.36), (0.0, 0.01)], 14, ki.J((0.5, 0.36, 0.18), 0.05), smooth=False)
        for _ in range(5):
            blob(acc_i, "cloth", T(bx + rng.normal(0, 0.08), by + rng.normal(0, 0.08), 0.36 + rng.uniform(0, 0.06)), 0.08,
                 ki.J([(0.85, 0.82, 0.74), (0.55, 0.15, 0.12), (0.18, 0.28, 0.45)][rng.integers(3)], 0.05), seg=7, rings=5, rng=rng, rough=0.06)
    site.collider(-2.5, -1.2, 0.3, 0.6, y0=-2, y1=0.5)
    F.chandelier(T(0.0, 0.4, H - 1.0), r=0.38, n=6)
    hollow(site, ki, W, D, H, t, "front", ops["front"][0], [(0.2, 0.4, 1.9, [1.0, 0.72, 0.42], 2.0, 10.0)], "높은 집 · 베 짜는 집", "tallhouse")
    return acc_i


def farm_inside(kit, site, W, D, ops):
    """The thatched cottage's kitchen and living room: a wide hearth with a pot, a scrubbed table with benches, the
    dresser of plates, a bed and a cradle, a butter churn, onions and herbs hung from the beams."""
    rng = kit.rng
    acc_i = Acc(rng)
    ki = Kit(acc_i, rng)
    ki.blocks = kit.blocks
    t, H = 0.45, 2.3
    room = Room(ki, -W / 2 + t, W / 2 - t, -D / 2 + t, D / 2 - t, H, t, ops, plaster=(0.86, 0.84, 0.76))
    room.walls(-W / 2, W / 2, -D / 2, D / 2, wainscot=0)
    room.flagstones((0.52, 0.5, 0.45))
    room.ceiling(col=(0.24, 0.17, 0.11), joist_dir="x", spacing=0.55)
    F = Furnish(ki)
    _, Mi_r, _ = room.inner_frame("right", -W / 2, W / 2, -D / 2, D / 2)
    F.fireplace(Mi_r, 2.8, w=1.6, h=1.3, depth=0.6, opening=1.0, stone=(0.52, 0.49, 0.44))
    site.collider(W / 2 - t - 0.3, 0.0, 0.32, 0.85, y0=-2, y1=2.3)
    F.table_long(T(0.6, 0.25, 0), 1.8, 0.8)
    F.bench(T(0.6, -0.4, 0), 1.7)
    F.bench(T(0.6, 0.9, 0), 1.7)
    F.plate(T(0.2, 0.15, 0.785))
    F.plate(T(1.0, 0.35, 0.785))
    blob(acc_i, "produce", T(0.6, 0.25, 0.84), 0.1, (0.62, 0.42, 0.2), seg=8, rings=5, squash=0.6, rng=rng)
    F.tankard(T(1.25, 0.1, 0.785))
    F.candle(T(0.55, 0.0, 0.785))
    site.collider(0.6, 0.25, 0.95, 0.85, y0=-2, y1=0.85)
    F.dresser(T(-0.25, D / 2 - t - 0.27, 0), 1.2)
    site.collider(-0.25, D / 2 - t - 0.27, 0.65, 0.3, y0=-2, y1=2.1)
    F.cot(T(-W / 2 + t + 0.5, 0.3, 0) @ R("z", math.pi / 2), 1.9, 0.85, (0.48, 0.16, 0.12))
    site.collider(-W / 2 + t + 0.5, 0.3, 0.45, 0.98, y0=-2, y1=0.6)
    F.cradle(T(-2.6, -1.55, 0))
    site.collider(-2.6, -1.55, 0.47, 0.3, y0=-2, y1=0.6)
    M = T(3.0, -1.6, 0)          # a butter churn
    lathe(acc_i, "wood", M, [(0.2, 0.0), (0.17, 0.7), (0.15, 0.72), (0.0, 0.72)], 12, ki.J(WOOD_RAW, 0.05))
    for z in (0.1, 0.6):
        lathe(acc_i, "iron", M, [(0.2 - z * 0.04 + 0.006, z - 0.02), (0.2 - z * 0.04 + 0.006, z + 0.02)], 12, IRON)
    tube(acc_i, "wood", M, [(0, 0, 0.72), (0, 0, 1.15)], 0.018, 5, ki.J(WOOD_RAW, 0.05))
    site.collider(3.0, -1.6, 0.22, 0.22, y0=-2, y1=1.0)
    for i, x in enumerate(np.arange(-1.4, 2.6, 0.55)):
        if i % 2:
            F.onions(T(x, -1.7, H - 0.2), 8)
        else:
            F.herb_bunch(T(x, -1.75, H - 0.2), [(0.55, 0.3, 0.7), (0.85, 0.75, 0.3), (0.35, 0.45, 0.25)][i % 3])
    for (bx, by) in ((2.2, -1.9),):
        F.basket(T(bx, by, 0), 0.24, 0.16, 0)
        for _ in range(6):
            blob(acc_i, "produce", T(bx + rng.normal(0, 0.06), by + rng.normal(0, 0.06), 0.18), 0.06, ki.J((0.35, 0.5, 0.22), 0.1), seg=7, rings=4, rng=rng)
    F.rug(T(-1.6, 0.4, 0), 1.3, 0.9, (0.48, 0.16, 0.12), (0.75, 0.62, 0.3))
    hollow(site, ki, W, D, H, t, "front", ops["front"][1], [(1.8, 0.0, 1.3, [1.0, 0.6, 0.3], 2.0, 9.0)], "초가집", "cottage")
    return acc_i


# ---------------------------------------------------------------- rooms behind the old houses' doors
ROOM_DEPTH = -60.0
OLD_HOUSES = [(0, "house_a", "엘드릭의 집", "study"), (1, "house_b", "마르타의 집", "store"), (2, "house_c", "리나의 집", "herbs"),
              (3, "house_d", "토비네 집", "family"), (4, "house_e", "오스윈의 집", "guard"), (5, "house_f", "목공소", "carpenter")]


class HiddenSite(Site):
    """A room kept out of sight below the village: same footprint and turn as its house, no terrain."""

    def __init__(self, x, z, yaw, floor):
        self.x, self.z, self.yaw = x, z, yaw
        self.c, self.s = math.cos(yaw), math.sin(yaw)
        self.floor = self.low = floor
        self.Mb = T(x, -z, floor) @ R("z", yaw)

    def gz(self, lx, ly):
        return 0.0


def old_house_rooms(glb):
    V = json.loads((ROOT / "raw" / "live" / "village.json").read_text(encoding="utf-8"))
    out = 0
    for (ci, hid, name, theme) in OLD_HOUSES:
        c = V["colliders"][ci]
        rng = np.random.default_rng(100 + ci)
        # which face looks at the plaza: that wall has the door; turn the room so its door is at the canonical front
        to_plaza = np.array([0.0 - c["x"], 2.0 - c["z"]])
        best = None
        for k_, theta in enumerate((0.0, math.pi / 2, math.pi, -math.pi / 2)):
            yaw = c["yaw"] + theta
            front = np.array([math.sin(yaw), math.cos(yaw)])          # canonical front (local -y) in game x, z
            d = float(front @ to_plaza)
            if best is None or d > best[0]:
                best = (d, theta)
        theta = best[1]
        swap = abs(math.sin(theta)) > 0.5
        W, D = (2 * c["hz"], 2 * c["hx"]) if swap else (2 * c["hx"], 2 * c["hz"])
        site = HiddenSite(c["x"], c["z"], c["yaw"] + theta, ROOM_DEPTH)
        acc = Acc(rng)
        kit = Kit(acc, rng, instanced=True)
        t, H = 0.35, 2.8
        win = lambda u, w=0.9: {"kind": "win", "u": u, "w": w, "h": 1.15, "sill": 0.9, "depth": t, "inside": True, "shutters": rng.choice(["green", "red", "blue"])}
        ops = {"front": [{"kind": "door", "u": W / 2, "w": 1.05, "h": 2.1, "sill": 0, "depth": t, "step": False}, win(W * 0.2)],
               "back": [win(W * 0.3), win(W * 0.72)], "left": [win(D * 0.5)], "right": []}
        x0, x1, y0, y1 = -W / 2, W / 2, -D / 2, D / 2
        for side, lst in ops.items():
            Mw, L = Kit.wall_frame(side, x0, x1, y0, y1, 0)
            for o in lst:
                kit.opening(Mw, o, "plaster", (0.84, 0.8, 0.7))
        room = Room(kit, x0 + t, x1 - t, y0 + t, y1 - t, H, t, ops, plaster=(0.85, 0.8, 0.69))
        F = Furnish(kit)
        lamp = THEMES[theme](kit, room, F, site, W, D, t, H, rng)
        for (sx, sy, hx_, hy_) in ((0, y0 + t / 2, W / 2, t / 2), (0, y1 - t / 2, W / 2, t / 2), (x0 + t / 2, 0, t / 2, D / 2), (x1 - t / 2, 0, t / 2, D / 2)):
            site.collider(sx, sy, hx_, hy_ + 0.04, y1=H)
        site.collider(0, 0, W / 2, D / 2, y0=-2.0, y1=0.0)
        blocks_out(kit, Acc(), site.Mb, -100.0, -100.0, rng, acc)
        # daylight behind every window (the room lies out of sight; this is what its windows look out on)
        day = Acc()
        for side, lst in ops.items():
            Mw, L = Kit.wall_frame(side, x0, x1, y0, y1, 0)
            for o in lst:
                if o["kind"] != "win":
                    continue
                a, b = o["u"] - o["w"] / 2 - 0.7, o["u"] + o["w"] / 2 + 0.7
                lo_, hi_ = o["sill"] - 0.6, o["sill"] + o["h"] + 0.7
                quad(day, "cloth", Mw, [(a, -0.55, lo_), (a, -0.55, hi_), (b, -0.55, hi_), (b, -0.55, lo_)], (0.93, 0.96, 1.0))
        acc.merge(day)
        world = Acc()
        world.merge(acc, site.Mb)
        out += glb.add_acc("room_" + hid, world)
        gx, gz_ = site.world(0, 0)
        fx, fz = math.sin(site.yaw), math.cos(site.yaw)       # the front's outward direction (game)
        door_out = (c["x"] + fx * (D / 2 + 1.1), c["z"] + fz * (D / 2 + 1.1))
        door_in = (c["x"] + fx * (D / 2 - t - 0.9), c["z"] + fz * (D / 2 - t - 0.9))
        wx, wz = site.world(lamp[0], lamp[1])
        INTERIORS.append({"id": hid, "name": name, "x": round(gx, 3), "z": round(gz_, 3), "yaw": round(site.yaw, 4), "hx": round(W / 2 - t, 3), "hz": round(D / 2 - t, 3),
                          "floor": ROOM_DEPTH, "ceil": ROOM_DEPTH + H, "hidden": True,
                          "lights": [{"x": round(wx, 3), "y": ROOM_DEPTH + lamp[2], "z": round(wz, 3), "color": lamp[3], "intensity": lamp[4], "distance": lamp[5]}],
                          "door_out": {"x": round(door_out[0], 3), "z": round(door_out[1], 3), "heading": round(math.atan2(-fx, -fz), 4)},
                          "door_in": {"x": round(door_in[0], 3), "y": ROOM_DEPTH, "z": round(door_in[1], 3), "heading": round(math.atan2(-fx, -fz), 4)}})
        print(f"  {hid} ({name}): {W:.1f} x {D:.1f}")
    return out


def _shell(kit, room, W, D, floor="boards", ceiling=True, wainscot=0.95):
    room.walls(-W / 2, W / 2, -D / 2, D / 2, wainscot=wainscot)
    if floor == "boards":
        room.floor_boards()
    else:
        room.flagstones((0.5, 0.48, 0.44))
    if ceiling:
        room.ceiling()


def theme_study(kit, room, F, site, W, D, t, H, rng):
    _shell(kit, room, W, D)
    _, Mi_b, Lb = room.inner_frame("back", -W / 2, W / 2, -D / 2, D / 2)
    _, Mi_r, Lr = room.inner_frame("right", -W / 2, W / 2, -D / 2, D / 2)
    F.fireplace(Mi_r, Lr / 2, w=1.7, h=1.4, depth=0.6, opening=1.0)
    site.collider(W / 2 - t - 0.3, 0, 0.32, 0.9, y0=-2, y1=2.6)
    for k_, x in enumerate((-W / 2 + t + 0.75, -W / 2 + t + 2.25)):
        F.bookshelf(T(x, D / 2 - t - 0.17, 0), 1.4, 2.1)
    site.collider(-W / 2 + t + 1.5, D / 2 - t - 0.17, 1.5, 0.18, y0=-2, y1=2.2)
    F.desk(T(-0.6, -0.2, 0), 1.5, 0.75)
    F.chair(T(-0.6, 0.45, 0) @ R("z", math.pi))
    site.collider(-0.6, -0.2, 0.8, 0.42, y0=-2, y1=0.85)
    F.armchair(T(W / 2 - t - 1.5, 1.0, 0) @ R("z", -math.pi / 2 - 0.4))
    F.rug(T(W / 2 - t - 1.4, 0.0, 0) @ R("z", math.pi / 2), 2.0, 1.2, (0.22, 0.3, 0.42), (0.75, 0.62, 0.3))
    F.chest(T(-W / 2 + t + 0.35, -0.8, 0) @ R("z", math.pi / 2))
    site.collider(-W / 2 + t + 0.35, -0.8, 0.28, 0.47, y0=-2, y1=0.6)
    _, Mi_l, Ll = room.inner_frame("left", -W / 2, W / 2, -D / 2, D / 2)
    patch(kit.acc, "cloth", Mi_l @ T(Ll * 0.15, -0.01, 1.2), 1.1, 0.8, 6, 4, (0.8, 0.72, 0.52),
          lambda U, V: 0.01 * np.sin(U * 4) * np.sin(V * 5), lambda U, V: np.clip(np.array([[0.8, 0.72, 0.52]]) - (np.abs(np.sin(U * 5 + np.sin(V * 4) * 2)) < 0.08)[:, None] * 0.3, 0, 1))
    F.chandelier(T(0, 0, H - 1.1), r=0.4, n=6)
    return (0.6, 0.2, 1.8, [1.0, 0.68, 0.4], 2.2, 10.0)


def theme_store(kit, room, F, site, W, D, t, H, rng):
    _shell(kit, room, W, D, wainscot=0)
    for i in range(6):
        x = -W / 2 + t + 0.4 + (i % 3) * 0.6
        y = D / 2 - t - 0.4 - (i // 3) * 0.6
        kit.crate(T(x, y, 0) @ R("z", rng.normal(0, 0.1)), 0.55)
        if i % 2 == 0:
            kit.crate(T(x, y, 0.55) @ R("z", rng.normal(0, 0.15)), 0.45)
    site.collider(-W / 2 + t + 1.0, D / 2 - t - 0.7, 0.95, 0.65, y0=-2, y1=1.1)
    for i in range(5):
        kit.sack(T(W / 2 - t - 0.35 - (i % 3) * 0.45, D / 2 - t - 0.35 - (i // 3) * 0.4, 0) @ R("z", rng.uniform(0, 6)), 0.26)
    site.collider(W / 2 - t - 0.8, D / 2 - t - 0.6, 0.75, 0.55, y0=-2, y1=0.8)
    for (bx, by) in ((W / 2 - t - 0.4, -0.6), (W / 2 - t - 0.4, 0.15)):
        F.barrel(T(bx, by, 0), 0.32, 0.85)
    site.collider(W / 2 - t - 0.4, -0.22, 0.35, 0.75, y0=-2, y1=0.9)
    _, Mi_l, Ll = room.inner_frame("left", -W / 2, W / 2, -D / 2, D / 2)
    F.jar_shelf(Mi_l, Ll * 0.62, Ll - t - 0.2, (0.5, 1.1, 1.7))
    F.table_long(T(0.0, -0.2, 0), 1.6, 0.8)
    for i in range(3):
        F.basket(T(-0.5 + i * 0.5, -0.2, 0.785), 0.18, 0.11, 0)
        for _ in range(5):
            blob(kit.acc, "produce", T(-0.5 + i * 0.5 + rng.normal(0, 0.05), -0.2 + rng.normal(0, 0.05), 0.9), 0.045,
                 kit.J([(0.75, 0.12, 0.1), (0.8, 0.65, 0.2), (0.4, 0.6, 0.2)][i], 0.08), seg=7, rings=4, rng=rng)
    for i in range(12):        # a little pile of coins and the ledger
        lathe(kit.acc, "iron", T(0.55 + rng.normal(0, 0.03), 0.05 + rng.normal(0, 0.03), 0.785 + 0.006 * i), [(0.0, 0), (0.018, 0), (0.018, 0.005), (0.0, 0.005)], 8,
              (0.7, 0.55, 0.2))
    box(kit.acc, "cloth", T(0.3, 0.1, 0.8), 0.14, 0.1, 0.02, (0.35, 0.2, 0.12), 0.004)
    site.collider(0.0, -0.2, 0.85, 0.45, y0=-2, y1=0.85)
    F.chandelier(T(0, 0, H - 1.1), r=0.35, n=5)
    return (0.0, 0.0, 1.9, [1.0, 0.72, 0.42], 2.0, 10.0)


def theme_herbs(kit, room, F, site, W, D, t, H, rng):
    _shell(kit, room, W, D, floor="flags", wainscot=0)
    _, Mi_r, Lr = room.inner_frame("right", -W / 2, W / 2, -D / 2, D / 2)
    Mf = F.fireplace(Mi_r, Lr / 2, w=1.6, h=1.35, depth=0.6, opening=1.0)
    site.collider(W / 2 - t - 0.3, 0, 0.32, 0.85, y0=-2, y1=2.6)
    F.cauldron(T(W / 2 - t - 1.15, 0.0, 0), 0.3)
    site.collider(W / 2 - t - 1.15, 0, 0.33, 0.33, y0=-2, y1=0.55)
    _, Mi_b, Lb = room.inner_frame("back", -W / 2, W / 2, -D / 2, D / 2)
    F.jar_shelf(Mi_b, t + 0.2, Lb * 0.24, (0.6, 1.15, 1.7))
    F.jar_shelf(Mi_b, Lb * 0.52, Lb * 0.68, (0.6, 1.15, 1.7))
    F.table_long(T(-0.5, 0.1, 0), 1.6, 0.75)
    F.mortar(T(-0.9, 0.0, 0.785))
    for i in range(6):
        blob(kit.acc, "produce", T(-0.3 + rng.normal(0, 0.15), 0.15 + rng.normal(0, 0.12), 0.8), 0.03, kit.J([(0.55, 0.3, 0.7), (0.85, 0.75, 0.3), (0.3, 0.5, 0.25)][i % 3], 0.1),
             seg=5, rings=3, rng=rng)
    F.jar(T(0.1, 0.25, 0.785), 0.14, (0.35, 0.42, 0.3))
    F.candle(T(-0.1, -0.15, 0.785))
    site.collider(-0.5, 0.1, 0.85, 0.42, y0=-2, y1=0.85)
    for i, x in enumerate(np.arange(-W / 2 + t + 0.4, W / 2 - t - 1.2, 0.42)):     # bunches drying from every joist
        for y in (-0.9, 0.9):
            F.herb_bunch(T(x + rng.normal(0, 0.05), y, H - 0.2), [(0.55, 0.3, 0.7), (0.85, 0.75, 0.3), (0.35, 0.45, 0.25), (0.8, 0.25, 0.25)][(i + int(y > 0)) % 4])
    F.cot(T(-W / 2 + t + 0.5, -D / 2 + t + 1.4, 0) @ R("z", math.pi / 2), 1.8, 0.8, (0.3, 0.38, 0.24))
    site.collider(-W / 2 + t + 0.5, -D / 2 + t + 1.4, 0.43, 0.95, y0=-2, y1=0.6)
    return (-0.3, 0.2, 1.7, [1.0, 0.7, 0.42], 2.0, 9.0)


def theme_family(kit, room, F, site, W, D, t, H, rng):
    _shell(kit, room, W, D, wainscot=0.9)
    _, Mi_r, Lr = room.inner_frame("right", -W / 2, W / 2, -D / 2, D / 2)
    F.fireplace(Mi_r, Lr / 2, w=1.7, h=1.35, depth=0.6, opening=1.05)
    site.collider(W / 2 - t - 0.3, 0, 0.32, 0.9, y0=-2, y1=2.6)
    F.table_long(T(-0.2, -0.3, 0), 1.9, 0.85)
    F.bench(T(-0.2, -1.0, 0), 1.8)
    F.bench(T(-0.2, 0.4, 0), 1.8)
    for x in (-0.8, -0.2, 0.4):
        F.plate(T(x, -0.3, 0.785))
    F.candle(T(-0.2, -0.2, 0.785))
    site.collider(-0.2, -0.3, 1.0, 0.9, y0=-2, y1=0.85)
    for k_, y in enumerate((D / 2 - t - 0.5, D / 2 - t - 1.5)):
        F.cot(T(-W / 2 + t + 1.05, y, 0), 1.9, 0.85, [(0.48, 0.16, 0.12), (0.22, 0.3, 0.42)][k_])
    site.collider(-W / 2 + t + 1.05, D / 2 - t - 1.0, 1.0, 0.95, y0=-2, y1=0.6)
    F.cradle(T(0.9, D / 2 - t - 0.4, 0))
    F.toy_horse(T(1.4, 1.0, 0) @ R("z", 0.7))
    F.sword(T(-W / 2 + t + 0.2, -D / 2 + t + 0.6, 0.0) @ R("y", 0.25) @ np.diag([0.6, 0.6, 0.6, 1.0]))
    F.dresser(T(W / 2 - t - 1.6, D / 2 - t - 0.27, 0), 1.2)
    site.collider(W / 2 - t - 1.6, D / 2 - t - 0.27, 0.65, 0.3, y0=-2, y1=2.1)
    for x in np.arange(-1.6, 1.8, 0.7):
        F.onions(T(x, 1.6, H - 0.2), 7)
    F.rug(T(1.6, -0.6, 0), 1.4, 1.0, (0.3, 0.38, 0.24), (0.75, 0.62, 0.3))
    return (0.5, -0.2, 1.6, [1.0, 0.62, 0.34], 2.1, 10.0)


def theme_guard(kit, room, F, site, W, D, t, H, rng):
    _shell(kit, room, W, D, floor="flags", wainscot=0)
    _, Mi_r, Lr = room.inner_frame("right", -W / 2, W / 2, -D / 2, D / 2)
    F.fireplace(Mi_r, Lr / 2, w=1.5, h=1.3, depth=0.55, opening=0.95)
    site.collider(W / 2 - t - 0.28, 0, 0.3, 0.8, y0=-2, y1=2.5)
    F.armour_stand(T(-W / 2 + t + 0.5, D / 2 - t - 0.5, 0) @ R("z", 0.6))
    site.collider(-W / 2 + t + 0.5, D / 2 - t - 0.5, 0.3, 0.3, y0=-2, y1=1.8)
    F.weapon_rack(T(-0.2, D / 2 - t - 0.3, 0), 5)
    site.collider(-0.2, D / 2 - t - 0.3, 0.75, 0.3, y0=-2, y1=1.4)
    _, Mi_l, Ll = room.inner_frame("left", -W / 2, W / 2, -D / 2, D / 2)
    F.shield(Mi_l @ T(Ll * 0.25, -0.05, 1.7) @ R("x", math.pi / 2), 0.32, (0.16, 0.24, 0.38))
    F.cot(T(W / 2 - t - 1.6, D / 2 - t - 0.48, 0), 1.9, 0.85, (0.3, 0.3, 0.32))
    site.collider(W / 2 - t - 1.6, D / 2 - t - 0.48, 1.0, 0.48, y0=-2, y1=0.6)
    F.table_round(T(-1.0, -0.6, 0), 0.55)
    for a in (0.5, 2.6):
        F.stool(T(-1.0 + 0.8 * math.cos(a), -0.6 + 0.8 * math.sin(a), 0))
    F.tankard(T(-0.9, -0.5, 0.785))
    for i in range(3):
        box(kit.acc, "plaster", T(-1.15 + 0.06 * i, -0.75, 0.8) @ R("z", rng.uniform(0, 1)), 0.014, 0.014, 0.014, (0.9, 0.88, 0.82), 0.003)
    F.candle(T(-1.1, -0.45, 0.785))
    site.collider(-1.0, -0.6, 0.6, 0.6, y0=-2, y1=0.8)
    F.chest(T(W / 2 - t - 0.6, -D / 2 + t + 0.4, 0), col=(0.3, 0.2, 0.12))
    return (0.0, 0.0, 1.8, [1.0, 0.66, 0.38], 2.0, 10.0)


def theme_carpenter(kit, room, F, site, W, D, t, H, rng):
    _shell(kit, room, W, D, wainscot=0)
    F.workbench(T(0.0, D / 2 - t - 0.4, 0) @ R("z", math.pi), 2.6, 0.7)
    site.collider(0.0, D / 2 - t - 0.4, 1.32, 0.37, y0=-2, y1=0.95)
    F.planks(T(-W / 2 + t + 0.25, 0.0, 0.06) @ R("z", math.pi / 2), 9, 2.6)
    site.collider(-W / 2 + t + 0.25, 0.0, 0.18, 1.32, y0=-2, y1=0.55)
    for k_, (x, y) in enumerate(((1.3, -0.3), (2.2, 0.6))):     # casks being made, hoops waiting
        M = T(x, y, 0)
        n = 18
        for i in range(n if k_ else n - 5):
            a = i / n * 2 * math.pi
            pts = [(0.3 * (0.86 + 0.14 * math.sin(math.pi * s)) * math.cos(a), 0.3 * (0.86 + 0.14 * math.sin(math.pi * s)) * math.sin(a), 0.85 * s) for s in np.linspace(0, 1, 6)]
            tube(kit.acc, "wood", M, pts, 0.03, 4, kit.J(WOOD_RAW, 0.08), cap=False)
        lathe(kit.acc, "iron", M, [(0.265, 0.1), (0.27, 0.14)], 16, IRON)
        site.collider(x, y, 0.33, 0.33, y0=-2, y1=0.9)
    for i in range(3):
        tube(kit.acc, "iron", T(W / 2 - t - 0.3, -D / 2 + t + 0.4, 0.05 + i * 0.03), [(0.3 * math.cos(a), 0.3 * math.sin(a), 0) for a in np.linspace(0, 2 * math.pi, 21)],
             0.012, 4, IRON, cap=False)
    for sx in (-1.4, -0.4):     # a saw horse
        M = T(sx, -0.9, 0)
        box(kit.acc, "wood", M @ T(0, 0, 0.6), 0.35, 0.05, 0.05, kit.J(OAK, 0.05), 0.006)
        for a in (-0.3, 0.3):
            for s in (-1, 1):
                tube(kit.acc, "wood", M, [(s * 0.3, 0, 0.6), (s * 0.3, a * 1.2, 0.0)], 0.025, 5, kit.J(OAK, 0.05))
    box(kit.acc, "wood", T(-0.9, -0.9, 0.67), 0.9, 0.12, 0.02, kit.J(WOOD_RAW, 0.06), 0.004)
    site.collider(-0.9, -0.9, 0.95, 0.35, y0=-2, y1=0.7)
    for _ in range(80):        # shavings
        x, y = rng.uniform(-1.5, 1.5), rng.uniform(0.6, D / 2 - t - 0.9)
        blob(kit.acc, "produce", T(x, y, 0.005), rng.uniform(0.02, 0.04), kit.J((0.78, 0.62, 0.4), 0.1), seg=4, rings=2, squash=0.3, rng=rng)
    _, Mi_l, Ll = room.inner_frame("left", -W / 2, W / 2, -D / 2, D / 2)
    box(kit.acc, "wood", Mi_l @ T(Ll * 0.75, -0.025, 1.6), 0.8, 0.025, 0.4, kit.J(OAK_DARK, 0.05), 0.008)
    for i in range(9):
        u = Ll * 0.75 - 0.65 + i * 0.16
        tube(kit.acc, "wood", Mi_l, [(u, -0.06, 1.85), (u, -0.06, 1.45)], 0.012, 5, kit.J(WOOD_RAW, 0.05))
        box(kit.acc, "iron", Mi_l @ T(u, -0.07, 1.42 if i % 2 else 1.86), 0.04, 0.02, 0.03, IRON, 0.004)
    F.chandelier(T(0, 0, H - 1.1), r=0.35, n=5)
    return (0.0, 0.2, 1.9, [1.0, 0.74, 0.46], 2.0, 10.0)


THEMES = {"study": theme_study, "store": theme_store, "herbs": theme_herbs, "family": theme_family, "guard": theme_guard, "carpenter": theme_carpenter}


BUILDERS = {"tavern": tavern, "smithy": smithy, "chapel": chapel, "tallhouse": tallhouse, "cottage": cottage, "bakery": bakery, "fisher": fisher}


def main():
    only = set(sys.argv[1:])
    glb = G.GlbOut()
    rng = np.random.default_rng(5)
    report = {}
    block_kit(glb)
    for bid, fn in BUILDERS.items():
        if only and bid not in only:
            continue
        report[bid] = fn(glb)
        print(f"{bid:10s} {report[bid]:7d} tris")
    if not only or "rooms" in only:
        print("old-house rooms", old_house_rooms(glb), "tris")
    if not only or "wall" in only:
        n = town_wall(glb, rng)
        print("wall segments", n)
        for g in PLAN.GATES:
            i = PLAN.WALL.index(min(PLAN.WALL, key=lambda p: math.hypot(p[0] - g["x"], p[1] - g["z"])))
            a, b = PLAN.WALL[max(0, i - 1)], PLAN.WALL[min(len(PLAN.WALL) - 1, i + 1)]
            d = np.array([b[0] - a[0], b[1] - a[1]])
            d /= np.linalg.norm(d)
            tris, yaw, floor = gatehouse(glb, g, d)
            g["yaw"], g["floor"] = round(yaw, 4), round(floor, 3)
            print(f"{g['id']:10s} {tris:7d} tris")
    if only:   # previews: the slates as plain geometry, so a render shows the roofs
        pv = Acc()
        C4 = np.eye(4)
        C4[:3, :3] = C_GAME.T
        for proto, rows in INST.d.items():
            if not proto.startswith(("proto_slate", "proto_ridge", "kit_block")):
                continue
            for r_ in rows:
                G4 = np.eye(4)
                G4[:3, :] = np.array(r_[:12]).reshape(3, 4)
                col = np.clip(np.array(r_[12:15]) ** (1 / 2.2) * 1.3, 0, 1)
                if proto.startswith("kit_block"):
                    P_, N_, I_ = G.fbox_geom(0.5, 0.5, 0.5, 0.07 + 0.05 * int(proto[9]))
                    # proto is in glTF axes; preview geometry in Blender axes: rotate the unit block by C^T first
                    Mg = C4 @ G4 @ np.linalg.inv(C4)
                    pv.put("stone", Mg, P_, N_, np.clip(np.array(r_[12:15]) ** (1 / 2.2), 0, 1), I_)
                    continue
                if proto.startswith("proto_slate"):
                    box(pv, "slate", C4 @ G4 @ T(0, 0, -0.5), 0.5, 0.008, 0.5, col)
                else:
                    box(pv, "slate", C4 @ G4, 0.5, 0.07, 0.16, col)
        glb.add_acc("preview_slates", pv)
    path = OUT / ("village_ext.glb" if not only else "village_ext_preview.glb")
    glb.save(path)
    inst = [{"proto": k, "count": len(v), "f32": base64.b64encode(np.asarray(v, "<f4").tobytes()).decode("ascii")} for k, v in INST.d.items()]
    data = {"instances": inst, "colliders": COLLIDERS, "wall": [list(p) for p in PLAN.wall_polygon()], "gates": PLAN.GATES, "interiors": INTERIORS,
            "buildings": [{"id": b["id"], "name": b["name"], "x": b["x"], "z": b["z"]} for b in PLAN.BUILDINGS]}
    (OUT / ("village_ext.json" if not only else "village_ext_preview.json")).write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print({k: len(v) for k, v in INST.d.items()})
    print("glb", round(path.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
