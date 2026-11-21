"""Rooms you can walk into: inner walls, floors, ceilings and furniture, in building-local axes (x along the
front, front at -y, z up from the floor). Built into their own Acc so their ambient occlusion can reach further
than an exterior's (a closed room should be dim where the lamp does not reach).
"""
import math

import numpy as np

from geo_kit import T, R, frame, box, fbox, prism, quad, lathe, tube, patch, blob, surface
from village_kit import Kit, WOOD, WOOD_GREY, WOOD_RAW, IRON, STONE, STONE_GREY, MORTAR, TERRA, LEAF, FLOWERS

OAK = (0.26, 0.17, 0.10)
OAK_DARK = (0.15, 0.10, 0.065)
PEWTER = (0.42, 0.43, 0.45)
BRASS = (0.55, 0.40, 0.17)
WAX = (0.90, 0.85, 0.70)
FLAME = (1.0, 0.72, 0.30)
LINEN = (0.86, 0.83, 0.76)
WOOL = [(0.48, 0.16, 0.12), (0.22, 0.30, 0.42), (0.56, 0.46, 0.24), (0.30, 0.38, 0.24)]


class Room:
    """One rectangular room: x0..x1, y0..y1 (inner faces), floor at z = 0, ceiling at z = H."""

    def __init__(self, kit, x0, x1, y0, y1, H, t, walls_open, plaster=(0.84, 0.80, 0.70)):
        self.k, self.acc, self.rng = kit, kit.acc, kit.rng
        self.x0, self.x1, self.y0, self.y1, self.H, self.t = x0, x1, y0, y1, H, t
        self.open = walls_open            # side -> openings (as given to the outer wall, in its u coordinates)
        self.plaster = plaster

    def inner_frame(self, side, X0, X1, Y0, Y1):
        """Inner face of an outer wall: matrix whose patches face into the room, u running as outside."""
        Mw, L = Kit.wall_frame(side, X0, X1, Y0, Y1, 0)
        Mi = Mw @ frame((L, self.t, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1))
        return Mw, Mi, L

    def walls(self, X0, X1, Y0, Y1, wainscot=1.0, dado=True, beams=True, extra=None):
        k, t = self.k, self.t
        for side in ("front", "right", "back", "left"):
            Mw, Mi, L = self.inner_frame(side, X0, X1, Y0, Y1)
            ops = self.open.get(side, [])
            holes = []
            for o in ops:
                u0, u1, v0, v1 = Kit.hole(o)
                holes.append((L - u1 - t, L - u0 - t, v0, v1))
            for (a, b, v0, v1) in (extra or {}).get(side, []):      # inner-only openings (inner u coordinates)
                holes.append((a - t, b - t, v0, v1))
            Lin = L - 2 * t
            M = Mi @ T(t, 0, 0)
            k.plaster_cells(M, Lin, self.H, holes, self.plaster, bulge=0.008, cell=0.3, stains=False)
            if wainscot:
                # boarded dado: vertical boards below the rail, a moulded rail along the top
                for i, (a, b) in enumerate(self._spans(Lin, holes, wainscot)):
                    x = a
                    while x < b - 0.02:
                        w = min(0.17, b - x)
                        box(self.acc, "wood", M @ T(x + w / 2, -0.03, wainscot / 2), w / 2 - 0.004, 0.015, wainscot / 2, k.J(OAK, 0.07), 0.004)
                        x += w
                    if dado:
                        box(self.acc, "wood", M @ T((a + b) / 2, -0.05, wainscot + 0.02), (b - a) / 2, 0.035, 0.035, k.J(OAK_DARK, 0.04), 0.01)
                    box(self.acc, "wood", M @ T((a + b) / 2, -0.04, 0.06), (b - a) / 2, 0.025, 0.06, k.J(OAK_DARK, 0.04), 0.006)
            if beams:   # posts at the corners of the room
                pass

    def _spans(self, L, holes, top):
        """Stretches of a wall below `top` that no opening interrupts."""
        cuts = sorted((h[0], h[1]) for h in holes if h[2] < top)
        spans, a = [], 0.0
        for c0, c1 in cuts:
            if c0 - a > 0.05:
                spans.append((a, c0))
            a = max(a, c1)
        if L - a > 0.05:
            spans.append((a, L))
        return spans

    def floor_boards(self, col=OAK, width=0.21):
        k, rng = self.k, self.rng
        y = self.y0
        while y < self.y1 - 0.02:
            w = min(width * rng.uniform(0.85, 1.15), self.y1 - y)
            x = self.x0 + rng.uniform(-1.0, 0.0)
            while x < self.x1:
                L = rng.uniform(1.4, 3.2)
                a, b = max(x, self.x0), min(x + L, self.x1)
                if b - a > 0.05:
                    c = k.J(col, 0.09) * rng.uniform(0.82, 1.08)
                    box(self.acc, "wood", T((a + b) / 2, y + w / 2, -0.03), (b - a) / 2 - 0.003, w / 2 - 0.004, 0.03, c, 0.004)
                x += L
            y += w

    def flagstones(self, col=(0.52, 0.50, 0.46)):
        k, rng = self.k, self.rng
        y = self.y0
        while y < self.y1 - 0.02:
            h = min(rng.uniform(0.45, 0.7), self.y1 - y)
            x = self.x0
            while x < self.x1 - 0.02:
                w = min(rng.uniform(0.45, 0.85), self.x1 - x)
                box(self.acc, "stone", T(x + w / 2, y + h / 2, -0.04 + rng.normal(0, 0.004)), w / 2 - 0.012, h / 2 - 0.012, 0.04, k.tone(col, 0.1), 0.015)
                x += w
            y += h
        quad(self.acc, "stone", np.eye(4), [(self.x0, self.y0, -0.07), (self.x1, self.y0, -0.07), (self.x1, self.y1, -0.07), (self.x0, self.y1, -0.07)], MORTAR)

    def ceiling(self, col=OAK, joist_dir="y", spacing=0.6):
        """Boards overhead on joists, carried by a heavy beam across the room."""
        k = self.k
        H = self.H
        Mc = frame((self.x0, self.y1, H), (1, 0, 0), (0, 0, 1), (0, -1, 0))
        boards = lambda U, V: np.clip(np.asarray(col)[None] * 0.8 * (0.85 + 0.15 * np.sin(np.floor(U / 0.19) * 12.9898))[:, None], 0, 1)
        patch(self.acc, "wood", Mc, self.x1 - self.x0, self.y1 - self.y0, int((self.x1 - self.x0) / 0.19), 3, col, None, boards)
        if joist_dir == "y":
            for x in np.arange(self.x0 + spacing / 2, self.x1, spacing):
                box(self.acc, "wood", T(x, (self.y0 + self.y1) / 2, H - 0.09), 0.065, (self.y1 - self.y0) / 2, 0.09, k.J(OAK_DARK, 0.06), 0.012)
            box(self.acc, "wood", T((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2, H - 0.3), (self.x1 - self.x0) / 2, 0.16, 0.13, k.J(OAK_DARK, 0.04), 0.02)
        else:
            for y in np.arange(self.y0 + spacing / 2, self.y1, spacing):
                box(self.acc, "wood", T((self.x0 + self.x1) / 2, y, H - 0.09), (self.x1 - self.x0) / 2, 0.065, 0.09, k.J(OAK_DARK, 0.06), 0.012)
            box(self.acc, "wood", T((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2, H - 0.3), 0.16, (self.y1 - self.y0) / 2, 0.13, k.J(OAK_DARK, 0.04), 0.02)


# ---------------------------------------------------------------- furniture
class Furnish:
    def __init__(self, kit):
        self.k, self.acc, self.rng = kit, kit.acc, kit.rng

    def stool(self, M):
        k = self.k
        c = k.J(OAK, 0.08)
        lathe(self.acc, "wood", M @ T(0, 0, 0.44), [(0.0, 0), (0.17, 0.0), (0.18, 0.02), (0.17, 0.045), (0.0, 0.05)], 12, c, smooth=True)
        for a in (0.4, 2.5, 4.6):
            p0 = (0.12 * math.cos(a), 0.12 * math.sin(a), 0.45)
            p1 = (0.19 * math.cos(a), 0.19 * math.sin(a), 0.0)
            tube(self.acc, "wood", M, [p0, p1], 0.022, 6, c * 0.9)
        tube(self.acc, "wood", M, [(0.155 * math.cos(a), 0.155 * math.sin(a), 0.17) for a in np.linspace(0, 2 * math.pi, 13)], 0.012, 4, c * 0.85, cap=False)

    def chair(self, M):
        k = self.k
        c = k.J(OAK, 0.07)
        box(self.acc, "wood", M @ T(0, 0, 0.45), 0.22, 0.21, 0.025, c, 0.008)
        for sx in (-0.19, 0.19):
            for sy in (-0.18, 0.18):
                h = 0.45 if sy < 0 else 1.0
                box(self.acc, "wood", M @ T(sx, sy, h / 2), 0.022, 0.022, h / 2, c * 0.9, 0.006)
        for z in (0.68, 0.9):
            box(self.acc, "wood", M @ T(0, 0.18, z), 0.19, 0.016, 0.04, c, 0.006)
        box(self.acc, "wood", M @ T(0, 0.18, 0.79), 0.06, 0.012, 0.1, c * 1.05, 0.004)

    def table_round(self, M, r=0.6):
        k = self.k
        c = k.J(OAK, 0.06)
        lathe(self.acc, "wood", M @ T(0, 0, 0.74), [(0.0, 0), (r, 0.0), (r + 0.01, 0.02), (r, 0.045), (0.0, 0.045)], 24, c, smooth=False)
        lathe(self.acc, "wood", M, [(0.32, 0.0), (0.3, 0.04), (0.08, 0.1), (0.06, 0.3), (0.09, 0.45), (0.06, 0.6), (0.1, 0.74), (0.0, 0.74)], 12, c * 0.85, smooth=True)

    def table_long(self, M, L=1.8, Wd=0.8):
        k = self.k
        c = k.J(OAK, 0.06)
        for i in range(4):
            box(self.acc, "wood", M @ T(0, -Wd / 2 + Wd / 8 + i * Wd / 4, 0.755), L / 2, Wd / 8 - 0.004, 0.03, k.J(c, 0.05), 0.006)
        for sx in (-L / 2 + 0.12, L / 2 - 0.12):
            for sy in (-Wd / 2 + 0.08, Wd / 2 - 0.08):
                lathe(self.acc, "wood", M @ T(sx, sy, 0), [(0.045, 0.0), (0.04, 0.2), (0.055, 0.32), (0.035, 0.45), (0.045, 0.72), (0.0, 0.725)], 8, c * 0.85)
        box(self.acc, "wood", M @ T(0, 0, 0.12), L / 2 - 0.12, 0.025, 0.04, c * 0.8, 0.006)

    def bench(self, M, L=1.6):
        k = self.k
        c = k.J(OAK, 0.06)
        box(self.acc, "wood", M @ T(0, 0, 0.44), L / 2, 0.16, 0.035, c, 0.01)
        for sx in (-L / 2 + 0.15, L / 2 - 0.15):
            prism(self.acc, "wood", M @ frame((sx, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)), [(-0.15, 0), (-0.08, 0.0), (0, 0.12), (0.08, 0.0), (0.15, 0), (0.12, 0.41), (-0.12, 0.41)], 0.04, c * 0.9)

    def tankard(self, M, col=PEWTER):
        lathe(self.acc, "iron", M, [(0.0, 0), (0.045, 0.0), (0.042, 0.12), (0.046, 0.13), (0.0, 0.131)], 10, col, smooth=True)
        tube(self.acc, "iron", M, [(0.04, 0, 0.1), (0.075, 0, 0.09), (0.075, 0, 0.04), (0.04, 0, 0.03)], 0.008, 4, col)

    def plate(self, M, col=(0.62, 0.55, 0.44)):
        lathe(self.acc, "stone", M, [(0.0, 0.0), (0.1, 0.0), (0.13, 0.018), (0.12, 0.02), (0.08, 0.008), (0.0, 0.008)], 14, col, smooth=True)

    def bottle(self, M, col):
        lathe(self.acc, "glass", M, [(0.0, 0), (0.045, 0.0), (0.048, 0.02), (0.046, 0.17), (0.02, 0.23), (0.014, 0.29), (0.017, 0.3), (0.0, 0.3)], 10, col, smooth=True)
        lathe(self.acc, "wood", M @ T(0, 0, 0.3), [(0.0, 0), (0.012, 0), (0.012, 0.025), (0.0, 0.025)], 6, (0.5, 0.38, 0.22))

    def candle(self, M, h=0.16, holder=True):
        if holder:
            lathe(self.acc, "iron", M, [(0.0, 0), (0.06, 0.0), (0.06, 0.012), (0.02, 0.02), (0.015, 0.07), (0.035, 0.08), (0.0, 0.085)], 10, BRASS)
            M = M @ T(0, 0, 0.085)
        lathe(self.acc, "produce", M, [(0.0, 0), (0.017, 0), (0.017, h), (0.0, h + 0.004)], 8, WAX)
        lathe(self.acc, "produce", M @ T(0, 0, h + 0.004), [(0.0, 0), (0.009, 0.02), (0.0, 0.05)], 6, FLAME)

    def barrel(self, M, r=0.3, h=0.85, col=WOOD_RAW):
        self.k.barrel(M, r, h, col)

    def fireplace(self, Mw, u, w=1.9, h=1.45, depth=0.75, opening=1.1, stone=STONE):
        """A stone fireplace built against a wall (Mw: inner wall frame, faces +... into the room at y=0)."""
        k = self.k
        # chimney breast out of the wall
        for i, (side, a, b) in enumerate((("front", 0, 0),)):
            pass
        x0 = u - w / 2
        courses = k.courses(h + 0.9, coursed=True)
        # the breast's face and returns (stones as instances via kit._block, which builds into the wall's -y side)
        Mf = Mw @ T(0, -depth, 0)
        holes = [(u - opening / 2, u + opening / 2, 0.0, h - 0.35)]
        for v, ch in courses:
            x = x0
            while x < x0 + w - 0.05:
                bl = min(x0 + w - x, k.rng.uniform(0.3, 0.6))
                cu = x + bl / 2
                if not (holes[0][0] - 0.02 < cu < holes[0][1] + 0.02 and v < holes[0][3]):
                    k._block(Mf, x, x + bl, v, ch, stone, 0.02, 0.03)
                x += bl
        for side_u, sgn in ((x0, -1), (x0 + w, 1)):
            Mr = Mw @ frame((side_u, -depth, 0), (0, 1, 0) if sgn > 0 else (0, -1, 0), (-sgn, 0, 0), (0, 0, 1)) if False else None
        # returns of the breast
        box(self.acc, "stone", Mw @ T(x0 + 0.04, -depth / 2, (h + 0.9) / 2), 0.04, depth / 2, (h + 0.9) / 2, k.tone(stone, 0.05), 0.0)
        box(self.acc, "stone", Mw @ T(x0 + w - 0.04, -depth / 2, (h + 0.9) / 2), 0.04, depth / 2, (h + 0.9) / 2, k.tone(stone, 0.05), 0.0)
        # firebox: soot-black back and cheeks, hearthstone, mantel beam
        soot = (0.06, 0.05, 0.045)
        box(self.acc, "stone", Mw @ T(u, -0.05, (h - 0.35) / 2), opening / 2, 0.04, (h - 0.35) / 2, soot, 0)
        for s in (-1, 1):
            box(self.acc, "stone", Mw @ T(u + s * (opening / 2 + 0.02), -depth / 2, (h - 0.35) / 2), 0.03, depth / 2, (h - 0.35) / 2, soot, 0)
        box(self.acc, "stone", Mw @ T(u, -depth / 2, h - 0.33), opening / 2 + 0.03, depth / 2, 0.03, soot, 0)
        box(self.acc, "stone", Mw @ T(u, -depth - 0.3, 0.02), w / 2 + 0.1, 0.35, 0.03, k.tone((0.46, 0.45, 0.42), 0.05), 0.012)
        box(self.acc, "wood", Mw @ T(u, -depth - 0.06, h - 0.2), w / 2 + 0.08, 0.13, 0.12, k.J(OAK_DARK, 0.04), 0.02)
        box(self.acc, "wood", Mw @ T(u, -depth - 0.1, h - 0.06), w / 2 + 0.14, 0.17, 0.03, k.J(OAK_DARK, 0.04), 0.008)
        # fire dogs, logs, embers, a kettle on a crane
        for s in (-0.3, 0.3):
            tube(self.acc, "iron", Mw, [(u + s, -0.15, 0.03), (u + s, -depth + 0.05, 0.12), (u + s, -depth + 0.02, 0.28)], 0.02, 5, IRON)
        for i, (dz, rr) in enumerate(((0.12, 0.07), (0.12, 0.065), (0.24, 0.06))):
            Ml = Mw @ T(u + (i - 1) * 0.12, -depth * 0.55 + i * 0.05, dz) @ R("y", math.pi / 2) @ R("x", (i - 1) * 0.15)
            lathe(self.acc, "wood", Ml @ T(0, 0, -0.42), [(rr, 0), (rr, 0.84)], 9, k.J((0.18, 0.12, 0.08), 0.1))
            lathe(self.acc, "produce", Ml @ T(0, 0, -0.42), [(0.0, -0.001), (rr * 0.95, 0.0)], 9, (0.85, 0.35, 0.08))
        for _ in range(22):
            blob(self.acc, "produce", Mw @ T(u + k.rng.uniform(-0.4, 0.4), -depth * 0.5 + k.rng.uniform(-0.2, 0.2), 0.04), k.rng.uniform(0.025, 0.045),
                 (1.0, k.rng.uniform(0.35, 0.6), 0.1) if k.rng.random() < 0.6 else (0.1, 0.08, 0.07), seg=5, rings=3, squash=0.6, rng=k.rng)
        for f in range(5):     # flames: twisted translucent-looking tongues (bright, unlit-looking colours)
            a = k.rng.uniform(-0.3, 0.3)
            pts = [(u + a + 0.03 * math.sin(t * 6 + f), -depth * 0.52 + 0.02 * math.cos(t * 5), 0.12 + t * 0.45) for t in np.linspace(0, 1, 7)]
            tube(self.acc, "produce", Mw, pts, [0.06 * (1 - t) + 0.006 for t in np.linspace(0, 1, 7)], 6, (1.0, 0.62 + 0.1 * f / 5, 0.18))
        tube(self.acc, "iron", Mw, [(u - opening / 2 + 0.1, -0.1, 0.2), (u - opening / 2 + 0.1, -0.1, h - 0.45), (u, -depth * 0.5, h - 0.45)], 0.015, 5, IRON)
        tube(self.acc, "iron", Mw, [(u, -depth * 0.5, h - 0.45), (u, -depth * 0.5, 0.75)], 0.006, 4, IRON)
        lathe(self.acc, "iron", Mw @ T(u, -depth * 0.5, 0.48), [(0.0, 0), (0.12, 0.02), (0.16, 0.12), (0.14, 0.24), (0.1, 0.27), (0.0, 0.27)], 12, (0.09, 0.09, 0.09))
        return (u, -depth * 0.5, 0.5)

    def shelf(self, Mw, u0, u1, z, d=0.26, items=None):
        k = self.k
        box(self.acc, "wood", Mw @ T((u0 + u1) / 2, -d / 2, z), (u1 - u0) / 2, d / 2, 0.02, k.J(OAK, 0.05), 0.006)
        for u in (u0 + 0.08, u1 - 0.08):
            prism(self.acc, "wood", Mw @ frame((u, 0, z - 0.02), (0, -1, 0), (1, 0, 0), (0, 0, 1)), [(0, 0), (d - 0.04, 0), (0, -0.2)], 0.03, k.J(OAK_DARK, 0.05))

    def hanging_net(self, M, w=2.2, h=1.2):
        k = self.k
        NU, NV = 22, 10
        for i in range(NU + 1):
            pts = [(-w / 2 + w * i / NU, 0.02 * math.sin(j + i), -h * j / NV - 0.25 * math.sin(math.pi * i / NU) * (j / NV)) for j in range(NV + 1)]
            tube(self.acc, "cloth", M, pts, 0.005, 3, k.J((0.26, 0.26, 0.2), 0.06), cap=False)
        for j in range(0, NV + 1, 2):
            pts = [(-w / 2 + w * i / NU, 0.0, -h * j / NV - 0.25 * math.sin(math.pi * i / NU) * (j / NV)) for i in range(NU + 1)]
            tube(self.acc, "cloth", M, pts, 0.004, 3, k.J((0.26, 0.26, 0.2), 0.06), cap=False)
        for i in range(0, NU + 1, 3):
            blob(self.acc, "produce", M @ T(-w / 2 + w * i / NU, 0, -0.02), 0.045, k.J((0.62, 0.45, 0.25), 0.06), seg=6, rings=3, squash=0.8, rng=k.rng)

    def rug(self, M, w, h, col_a, col_b):
        stripes = lambda U, V: np.where((((np.floor(U / 0.12) % 5) == 0) | (V < 0.08) | (V > h - 0.08))[:, None], np.array([col_b]), np.array([col_a]))
        patch(self.acc, "cloth", M @ frame((-w / 2, h / 2, 0.006), (1, 0, 0), (0, 0, 1), (0, -1, 0)) @ R("x", 0), w, h, 12, 8, col_a,
              lambda U, V: 0.004 * np.sin(U * 9) * np.sin(V * 7), stripes)

    def chandelier(self, M, r=0.55, n=8):
        k = self.k
        for a in np.linspace(0, 2 * math.pi, 4, endpoint=False):
            tube(self.acc, "iron", M, [(0, 0, 1.1), (r * 0.98 * math.cos(a), r * 0.98 * math.sin(a), 0.03)], 0.007, 4, IRON)
        tube(self.acc, "iron", M, [(0, 0, 1.1), (0, 0, 2.0)], 0.01, 4, IRON)
        tube(self.acc, "wood", M, [(r * math.cos(a), r * math.sin(a), 0) for a in np.linspace(0, 2 * math.pi, 25)], 0.035, 6, k.J(OAK_DARK, 0.04), cap=False)
        for a in np.linspace(0, 2 * math.pi, n, endpoint=False):
            self.candle(M @ T(r * math.cos(a), r * math.sin(a), 0.03), h=0.12, holder=False)

    def stairs(self, M, L=3.2, H=2.9, w=0.9, steps=13, rail=-1):
        k = self.k
        c = k.J(OAK, 0.05)
        run = L / steps
        for i in range(steps):
            box(self.acc, "wood", M @ T(run * (i + 0.5), 0, H / steps * (i + 1) - 0.025), run / 2 + 0.02, w / 2, 0.025, k.J(c, 0.05), 0.006)
            box(self.acc, "wood", M @ T(run * i + 0.01, 0, H / steps * (i + 0.5)), 0.01, w / 2 - 0.02, H / steps / 2, c * 0.8, 0)
        for sy in (-w / 2, w / 2):
            prism(self.acc, "wood", M @ frame((0, sy, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), [(0, 0), (0.25, 0), (L, H - 0.2), (L, H), (L - 0.25, H), (0, 0.22)], 0.06,
                  c * 0.85)
        sy = rail * w / 2
        for i in range(0, steps, 2):
            x = run * (i + 0.5)
            z = H / steps * (i + 1)
            lathe(self.acc, "wood", M @ T(x, sy, z), [(0.025, 0), (0.03, 0.1), (0.018, 0.5), (0.028, 0.8), (0.0, 0.82)], 6, c * 0.9)
        tube(self.acc, "wood", M, [(0, sy, 1.0), (L, sy, H + 0.9)], 0.035, 6, c)
        lathe(self.acc, "wood", M @ T(0, sy, 0), [(0.06, 0), (0.06, 1.05), (0.08, 1.1), (0.04, 1.2), (0.0, 1.22)], 8, c)

    def pew(self, M, L=2.4):
        k = self.k
        c = k.J(OAK, 0.05)
        box(self.acc, "wood", M @ T(0, 0, 0.45), L / 2, 0.2, 0.03, c, 0.01)
        box(self.acc, "wood", M @ T(0, 0.2, 0.75), L / 2, 0.025, 0.28, c, 0.008)
        box(self.acc, "wood", M @ T(0, 0.22, 1.04), L / 2 + 0.02, 0.04, 0.025, c * 1.05, 0.008)
        box(self.acc, "wood", M @ T(0, -0.32, 0.12), L / 2 - 0.05, 0.07, 0.025, c * 0.9, 0.008)   # kneeler
        for sx in (-L / 2, L / 2):
            pts = [(-0.22, 0), (0.24, 0), (0.24, 1.05)] + [(0.24 - 0.12 * (1 - math.cos(t)), 1.05 + 0.12 * math.sin(t)) for t in np.linspace(0, math.pi / 2, 5)][1:] + \
                  [(-0.08, 1.12), (-0.08, 0.5), (-0.22, 0.48)]
            prism(self.acc, "wood", M @ frame((sx, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)) if sx > 0 else M @ frame((sx, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)), pts, 0.05, c * 0.92)

    def bed(self, M, L=1.95, Wd=1.1):
        k = self.k
        c = k.J(OAK, 0.05)
        for sx in (-L / 2, L / 2):
            for sy in (-Wd / 2, Wd / 2):
                h = 1.05 if sx < 0 else 0.7
                box(self.acc, "wood", M @ T(sx, sy, h / 2), 0.05, 0.05, h / 2, c, 0.01)
                lathe(self.acc, "wood", M @ T(sx, sy, h), [(0.04, 0), (0.06, 0.04), (0.0, 0.1)], 8, c)
        box(self.acc, "wood", M @ T(-L / 2, 0, 0.75), 0.03, Wd / 2, 0.25, c * 0.9, 0.008)
        box(self.acc, "wood", M @ T(L / 2, 0, 0.5), 0.03, Wd / 2, 0.15, c * 0.9, 0.008)
        for sy in (-Wd / 2, Wd / 2):
            box(self.acc, "wood", M @ T(0, sy, 0.32), L / 2, 0.03, 0.1, c * 0.9, 0.008)
        box(self.acc, "cloth", M @ T(0, 0, 0.47), L / 2 - 0.04, Wd / 2 - 0.04, 0.09, (0.7, 0.62, 0.45), 0.06)
        blob(self.acc, "cloth", M @ T(-L / 2 + 0.3, 0, 0.6), 0.26, LINEN, seg=10, rings=6, squash=0.35, rng=k.rng, rough=0.04)
        quilt = lambda U, V: np.where(((np.floor(U / 0.22) + np.floor(V / 0.22)) % 2 == 0)[:, None], np.array([WOOL[0]]), np.array([WOOL[2]]))
        patch(self.acc, "cloth", M @ frame((-L / 2 + 0.55, Wd / 2 + 0.05, 0.57), (1, 0, 0), (0, 0, 1), (0, -1, 0)), L - 0.6, Wd + 0.1, 10, 8, WOOL[0],
              lambda U, V: 0.03 * np.sin(U * 6) * np.sin(V * 5) - 0.18 * (np.clip(np.abs(V - (Wd + 0.1) / 2) - Wd / 2 + 0.02, 0, 1) * 8), quilt)

    def chest(self, M, L=0.9, Wd=0.5, H=0.5, col=OAK):
        k = self.k
        c = k.J(col, 0.05)
        box(self.acc, "wood", M @ T(0, 0, H * 0.4), L / 2, Wd / 2, H * 0.4, c, 0.012)
        prism(self.acc, "wood", M @ frame((0, 0, H * 0.8), (0, 1, 0), (-1, 0, 0), (0, 0, 1)) @ R("z", 0),
              [(-Wd / 2, 0)] + [(-Wd / 2 * math.cos(t), 0.12 * math.sin(t)) for t in np.linspace(0.3, math.pi - 0.3, 6)] + [(Wd / 2, 0)], L, c * 1.05)
        for sx in (-L / 2 + 0.12, L / 2 - 0.12):
            box(self.acc, "iron", M @ T(sx, -Wd / 2 - 0.004, H * 0.45), 0.025, 0.004, H * 0.45, IRON, 0)
        box(self.acc, "iron", M @ T(0, -Wd / 2 - 0.008, H * 0.7), 0.05, 0.008, 0.06, BRASS, 0.005)

    # ------------------------------------------------------------ the smithy
    def forge(self, Mw, u, w=1.6, d=0.95, h=0.8, stone=(0.42, 0.39, 0.35)):
        """A waist-high brick hearth against the wall (Mw: inner wall frame), a glowing coal bed, a hood and flue,
        the bellows beside it on a frame."""
        k, rng = self.k, self.rng
        Mf = Mw @ T(u, -d / 2, 0)
        for i, side in enumerate(("front", "right", "back", "left")):
            Mws, L = Kit.wall_frame(side, -w / 2, w / 2, -d / 2, d / 2, 0)
            k.stone_wall(Mf @ Mws, L, h, [], col=stone, coursed=True, quoins=False, parity=i % 2, proud=0.02)
        box(self.acc, "stone", Mf @ T(0, 0, h + 0.04), w / 2 + 0.06, d / 2 + 0.06, 0.04, k.tone(stone, 0.04) * 1.05, 0.015)
        box(self.acc, "stone", Mf @ T(0, 0.05, h + 0.06), w / 2 - 0.25, d / 2 - 0.25, 0.03, (0.05, 0.045, 0.04), 0)
        for _ in range(60):     # the fire: glowing coals in the middle, black at the edges
            x, y = rng.uniform(-w / 2 + 0.3, w / 2 - 0.3), rng.uniform(-d / 2 + 0.3, d / 2 - 0.2)
            hot = math.exp(-(x * x + y * y) / 0.09)
            col = (1.0, 0.35 + 0.35 * hot, 0.08) if rng.random() < 0.25 + 0.7 * hot else (0.07, 0.065, 0.06)
            blob(self.acc, "produce", Mf @ T(x, y, h + 0.1), rng.uniform(0.03, 0.05), col, seg=5, rings=3, squash=0.6, rng=rng)
        for f in range(4):
            a = rng.uniform(-0.15, 0.15)
            pts = [(a + 0.02 * math.sin(t * 7 + f), 0.05 + 0.02 * math.cos(t * 5), h + 0.12 + t * 0.3) for t in np.linspace(0, 1, 6)]
            tube(self.acc, "produce", Mf, pts, [0.05 * (1 - t) + 0.005 for t in np.linspace(0, 1, 6)], 6, (1.0, 0.6, 0.15))
        # the hood: a pyramid of plastered brick narrowing into the flue
        hood = [(-w / 2 - 0.05, 0), (w / 2 + 0.05, 0), (0.25, 0.75), (-0.25, 0.75)]
        prism(self.acc, "stone", Mf @ frame((0, 0.08, h + 0.95), (1, 0, 0), (0, 1, 0), (0, 0, 1)), hood, d - 0.12, k.tone((0.36, 0.33, 0.3), 0.04))
        box(self.acc, "stone", Mf @ T(0, 0.2, h + 1.7 + 0.5), 0.27, 0.27, 0.5, k.tone((0.36, 0.33, 0.3), 0.04), 0.02)
        box(self.acc, "iron", Mf @ T(0, -d / 2 + 0.02, h + 0.97), w / 2 + 0.06, 0.02, 0.03, IRON, 0)
        # bellows on a frame at the side, with the blast pipe into the fire
        Mb = Mf @ T(w / 2 + 0.55, 0.1, 0)
        for sx in (-0.25, 0.25):
            box(self.acc, "wood", Mb @ T(sx, 0, 0.4), 0.04, 0.3, 0.4, k.J(OAK_DARK, 0.05), 0.01)
        lathe(self.acc, "cloth", Mb @ T(0, 0.1, 0.95) @ R("x", math.pi / 2) @ np.diag([1.0, 0.45, 1.0, 1.0]),
              [(0.0, -0.45), (0.22, -0.35), (0.3, -0.05), (0.26, 0.2), (0.08, 0.38), (0.0, 0.45)], 14, (0.28, 0.18, 0.11))
        box(self.acc, "wood", Mb @ T(0, 0.1, 1.08), 0.33, 0.46, 0.02, k.J(OAK, 0.05), 0.006)
        tube(self.acc, "iron", Mf, [(w / 2 + 0.55, -0.35, h + 0.95), (w / 2 + 0.1, -0.1, h + 0.25), (w / 2 - 0.3, 0.05, h + 0.12)], 0.04, 7, IRON)
        tube(self.acc, "wood", Mb, [(0, 0.5, 1.12), (0, 0.95, 1.35)], 0.025, 6, k.J(OAK, 0.05))
        return Mf

    def workbench(self, M, L=2.6, Wd=0.7):
        k, rng = self.k, self.rng
        c = k.J(OAK, 0.06)
        box(self.acc, "wood", M @ T(0, 0, 0.86), L / 2, Wd / 2, 0.05, c, 0.012)
        for sx in (-L / 2 + 0.12, L / 2 - 0.12):
            for sy in (-Wd / 2 + 0.08, Wd / 2 - 0.08):
                box(self.acc, "wood", M @ T(sx, sy, 0.42), 0.06, 0.06, 0.42, c * 0.9, 0.01)
        box(self.acc, "wood", M @ T(0, 0, 0.18), L / 2 - 0.1, Wd / 2 - 0.05, 0.02, c * 0.85, 0.006)
        # a leg vise at one end
        V = M @ T(-L / 2 + 0.4, -Wd / 2 - 0.05, 0.91)
        box(self.acc, "iron", V @ T(0, 0, 0.06), 0.09, 0.06, 0.06, IRON, 0.01)
        box(self.acc, "iron", V @ T(0, -0.12, 0.06), 0.09, 0.03, 0.08, IRON, 0.01)
        tube(self.acc, "iron", V, [(-0.15, -0.1, 0.02), (0.15, -0.1, 0.02)], 0.012, 5, IRON)
        tube(self.acc, "iron", V, [(0, -0.12, 0.02), (0, -0.12, -0.85)], 0.025, 6, IRON)
        # tools and work strewn over the top
        for i in range(7):
            x = -L / 2 + 0.75 + i * 0.25 + rng.normal(0, 0.03)
            a = rng.uniform(-0.6, 0.6)
            Mt = M @ T(x, rng.uniform(-0.15, 0.15), 0.915) @ R("z", a)
            if i % 3 == 0:     # a hammer
                tube(self.acc, "wood", Mt, [(-0.16, 0, 0.02), (0.12, 0, 0.02)], 0.014, 6, k.J(WOOD_RAW, 0.05))
                box(self.acc, "iron", Mt @ T(0.14, 0, 0.03), 0.025, 0.07, 0.025, IRON, 0.006)
            elif i % 3 == 1:   # tongs
                tube(self.acc, "iron", Mt, [(-0.25, -0.02, 0.012), (0.12, 0.0, 0.012), (0.2, 0.02, 0.012)], 0.009, 4, IRON)
                tube(self.acc, "iron", Mt, [(-0.25, 0.02, 0.012), (0.12, 0.0, 0.012), (0.2, -0.02, 0.012)], 0.009, 4, IRON)
            else:              # a bar of iron, half-drawn into a blade
                prism(self.acc, "iron", Mt @ frame((0, 0, 0.012), (1, 0, 0), (0, 0, 1), (0, -1, 0)), [(-0.22, -0.02), (0.15, -0.02), (0.25, 0.0), (0.15, 0.02), (-0.22, 0.02)], 0.012, (0.18, 0.18, 0.19))

    def sword(self, M, L=0.95):
        """Upright, point up: blade, crossguard, grip, pommel."""
        prism(self.acc, "iron", M @ frame((0, 0, 0.2), (1, 0, 0), (0, 1, 0), (0, 0, 1)), [(-0.025, 0), (0.025, 0), (0.02, L - 0.12), (0.0, L), (-0.02, L - 0.12)], 0.008,
              (0.55, 0.56, 0.58))
        box(self.acc, "iron", M @ T(0, 0, 0.19), 0.1, 0.015, 0.015, BRASS, 0.005)
        tube(self.acc, "cloth", M, [(0, 0, 0.04), (0, 0, 0.18)], 0.016, 6, (0.25, 0.14, 0.08))
        lathe(self.acc, "iron", M @ T(0, 0, 0.0), [(0.0, 0), (0.028, 0.01), (0.028, 0.04), (0.0, 0.05)], 8, BRASS)

    def axe(self, M):
        tube(self.acc, "wood", M, [(0, 0, 0.0), (0, 0, 0.85)], 0.017, 6, self.k.J(WOOD_RAW, 0.05))
        prism(self.acc, "iron", M @ frame((0, 0, 0.72), (1, 0, 0), (0, 1, 0), (0, 0, 1)), [(-0.03, -0.04), (0.06, -0.06), (0.16, -0.1), (0.17, 0.12), (0.06, 0.07), (-0.03, 0.05)],
              0.03, (0.3, 0.3, 0.31))

    def weapon_rack(self, M, n=6):
        """A standing rack: a slotted rail at the top, a trough at the bottom, swords and axes between."""
        k = self.k
        c = k.J(OAK_DARK, 0.05)
        L = 0.24 * n + 0.2
        for sx in (-L / 2, L / 2):
            box(self.acc, "wood", M @ T(sx, 0, 0.7), 0.04, 0.2, 0.7, c, 0.01)
            box(self.acc, "wood", M @ T(sx, 0, 0.03), 0.06, 0.28, 0.03, c, 0.008)
        box(self.acc, "wood", M @ T(0, 0, 0.12), L / 2, 0.12, 0.03, c, 0.006)
        box(self.acc, "wood", M @ T(0, 0, 0.95), L / 2, 0.06, 0.03, c, 0.006)
        for i in range(n):
            x = -L / 2 + 0.2 + i * 0.24
            if i % 3 == 2:
                self.axe(M @ T(x, 0, 0.14) @ R("z", 0.2))
            else:
                self.sword(M @ T(x, 0, 0.14))

    def shield(self, M, r=0.32, col=(0.48, 0.14, 0.12)):
        k = self.k
        prof = [(0.0, 0.05), (r * 0.5, 0.045), (r, 0.0), (r + 0.01, -0.02), (0.0, -0.02)]
        lathe(self.acc, "wood", M, prof, 20, col, smooth=True)
        tube(self.acc, "iron", M, [(r * math.cos(a), r * math.sin(a), 0.0) for a in np.linspace(0, 2 * math.pi, 21)], 0.012, 5, IRON, cap=False)
        lathe(self.acc, "iron", M @ T(0, 0, 0.04), [(0.0, 0.07), (0.07, 0.04), (0.09, 0.0), (0.0, 0.0)], 12, (0.32, 0.32, 0.33))
        for a in np.linspace(0, 2 * math.pi, 4, endpoint=False):
            box(self.acc, "plaster", M @ R("z", a) @ T(r * 0.55, 0, 0.042), r * 0.4, 0.04, 0.004, (0.85, 0.72, 0.3), 0)

    def trough(self, M, L=1.2, Wd=0.55, H=0.6):
        k = self.k
        c = k.tone((0.5, 0.48, 0.44), 0.05)
        for s in (-1, 1):
            box(self.acc, "stone", M @ T(s * (L / 2 - 0.06), 0, H / 2), 0.06, Wd / 2, H / 2, c, 0.02)
            box(self.acc, "stone", M @ T(0, s * (Wd / 2 - 0.06), H / 2), L / 2, 0.06, H / 2, c, 0.02)
        box(self.acc, "stone", M @ T(0, 0, 0.06), L / 2, Wd / 2, 0.06, c, 0.01)
        quad(self.acc, "glass", M, [(-L / 2 + 0.12, -Wd / 2 + 0.12, H - 0.08), (L / 2 - 0.12, -Wd / 2 + 0.12, H - 0.08), (L / 2 - 0.12, Wd / 2 - 0.12, H - 0.08),
                                    (-L / 2 + 0.12, Wd / 2 - 0.12, H - 0.08)], (0.04, 0.06, 0.06))

    def coal_bin(self, M, L=0.9, Wd=0.6):
        k, rng = self.k, self.rng
        c = k.J(OAK_DARK, 0.05)
        for s in (-1, 1):
            box(self.acc, "wood", M @ T(s * L / 2, 0, 0.3), 0.025, Wd / 2, 0.3, c, 0.006)
            box(self.acc, "wood", M @ T(0, s * Wd / 2, 0.3), L / 2, 0.025, 0.3, c, 0.006)
        for _ in range(45):
            x, y = rng.uniform(-L / 2 + 0.08, L / 2 - 0.08), rng.uniform(-Wd / 2 + 0.08, Wd / 2 - 0.08)
            blob(self.acc, "produce", M @ T(x, y, 0.5 + 0.08 * (1 - abs(x) / L)), rng.uniform(0.04, 0.07), k.J((0.06, 0.055, 0.05), 0.25), seg=5, rings=3, squash=0.7, rng=rng)
        tube(self.acc, "iron", M, [(0.1, 0.0, 0.62), (0.35, 0.1, 1.05)], 0.012, 5, IRON)
        box(self.acc, "iron", M @ T(0.07, -0.02, 0.6) @ R("y", 0.5), 0.1, 0.08, 0.008, IRON, 0.004)

    # ------------------------------------------------------------ the bakery
    def loaf(self, M, kind=0):
        rng = self.rng
        crust = self.k.J((0.62, 0.40, 0.18), 0.08) * rng.uniform(0.85, 1.05)
        if kind == 0:      # a round cob with a cut cross
            blob(self.acc, "produce", M @ T(0, 0, 0.06), 0.11, crust, seg=10, rings=6, squash=0.6, rng=rng, rough=0.04)
            for a in (0, math.pi / 2):
                box(self.acc, "produce", M @ T(0, 0, 0.125) @ R("z", a), 0.07, 0.006, 0.008, np.clip(crust * 1.35, 0, 1), 0)
        elif kind == 1:    # a long loaf with slashes
            blob(self.acc, "produce", M @ T(0, 0, 0.05) @ np.diag([2.1, 1.0, 1.0, 1.0]), 0.075, crust, seg=10, rings=6, squash=0.75, rng=rng, rough=0.03)
            for i in range(3):
                box(self.acc, "produce", M @ T(-0.08 + i * 0.08, 0, 0.11) @ R("z", 0.6), 0.03, 0.006, 0.006, np.clip(crust * 1.35, 0, 1), 0)
        else:              # a plaited ring
            for j in range(3):
                pts = [((0.08 + 0.012 * math.sin(a * 3 + j * 2.1)) * math.cos(a), (0.08 + 0.012 * math.sin(a * 3 + j * 2.1)) * math.sin(a),
                        0.03 + 0.012 * math.cos(a * 3 + j * 2.1)) for a in np.linspace(0, 2 * math.pi, 25)]
                tube(self.acc, "produce", M, pts, 0.022, 6, crust, cap=False)

    def basket(self, M, r=0.22, h=0.14, loaves=4):
        k, rng = self.k, self.rng
        lathe(self.acc, "wood", M, [(r * 0.8, 0.0), (r, h), (r + 0.01, h + 0.01), (r * 0.95, h + 0.01), (r * 0.77, 0.01), (0.0, 0.01)], 16,
              k.J((0.5, 0.36, 0.18), 0.05), smooth=False)
        for t in np.linspace(0.02, h, 4):
            rr = r * 0.8 + (r * 0.2) * t / h + 0.004
            tube(self.acc, "wood", M, [(rr * math.cos(a), rr * math.sin(a), t) for a in np.linspace(0, 2 * math.pi, 17)], 0.005, 3, k.J((0.42, 0.3, 0.15), 0.05), cap=False)
        for i in range(loaves):
            a = i * 2 * math.pi / max(1, loaves) + rng.uniform(0, 0.4)
            self.loaf(M @ T(r * 0.45 * math.cos(a), r * 0.45 * math.sin(a), h * 0.4) @ R("z", rng.uniform(0, 6)) @ np.diag([0.75, 0.75, 0.75, 1.0]), rng.integers(3))

    def bread_rack(self, M, L=1.5, Wd=0.45, levels=4):
        k, rng = self.k, self.rng
        c = k.J(OAK, 0.06)
        for sx in (-L / 2, L / 2):
            for sy in (-Wd / 2, Wd / 2):
                box(self.acc, "wood", M @ T(sx, sy, 0.9), 0.03, 0.03, 0.9, c * 0.9, 0.006)
        for i in range(levels):
            z = 0.25 + i * 0.42
            for j in range(4):
                box(self.acc, "wood", M @ T(0, -Wd / 2 + Wd / 8 + j * Wd / 4, z), L / 2, Wd / 8 - 0.008, 0.012, k.J(c, 0.05), 0.003)
            x = -L / 2 + 0.15
            while x < L / 2 - 0.12:
                kind = rng.integers(3)
                self.loaf(M @ T(x, rng.normal(0, 0.06), z + 0.012) @ R("z", rng.normal(0, 0.3)), kind)
                x += 0.22 if kind != 1 else 0.3

    def kneading_table(self, M, L=1.8, Wd=0.8):
        k, rng = self.k, self.rng
        self.table_long(M, L, Wd)
        patch(self.acc, "cloth", M @ frame((-L / 2 + 0.2, Wd / 2 - 0.1, 0.788), (1, 0, 0), (0, 0, 1), (0, -1, 0)), L - 0.5, Wd - 0.2, 6, 4, (0.92, 0.9, 0.84),
              lambda U, V: 0.002 * np.sin(U * 13))
        for (x, y, r) in ((-0.3, 0.05, 0.16), (0.25, -0.1, 0.12)):
            blob(self.acc, "produce", M @ T(x, y, 0.81), r, (0.93, 0.86, 0.68), seg=10, rings=6, squash=0.45, rng=rng, rough=0.03)
        tube(self.acc, "wood", M, [(0.55, 0.15, 0.82), (0.55, -0.25, 0.82)], 0.03, 8, k.J(WOOD_RAW, 0.04))
        for y in (0.15, -0.25):
            tube(self.acc, "wood", M, [(0.55, y, 0.82), (0.55, y + (0.07 if y > 0 else -0.07), 0.82)], 0.014, 6, k.J(WOOD_RAW, 0.04))
        lathe(self.acc, "stone", M @ T(-0.65, -0.15, 0.785), [(0.0, 0), (0.1, 0), (0.17, 0.11), (0.165, 0.12), (0.09, 0.015), (0.0, 0.015)], 14, (0.6, 0.38, 0.24))

    def peel(self, M):
        tube(self.acc, "wood", M, [(0, 0, 0.3), (0, 0, 2.1)], 0.018, 6, self.k.J(WOOD_RAW, 0.05))
        box(self.acc, "wood", M @ T(0, 0, 0.15), 0.17, 0.01, 0.16, self.k.J(WOOD_RAW, 0.05), 0.004)

    # ------------------------------------------------------------ homes and workshops
    def bookshelf(self, M, L=1.4, H=2.1, d=0.32):
        """A tall case of shelves crowded with books (spines of many colours), a few lying flat, a candlestick."""
        k, rng = self.k, self.rng
        c = k.J(OAK_DARK, 0.05)
        for sx in (-L / 2, L / 2):
            box(self.acc, "wood", M @ T(sx, 0, H / 2), 0.025, d / 2, H / 2, c, 0.006)
        box(self.acc, "wood", M @ T(0, d / 2 - 0.01, H / 2), L / 2, 0.01, H / 2, c * 0.8, 0)
        box(self.acc, "wood", M @ T(0, 0, H + 0.03), L / 2 + 0.05, d / 2 + 0.03, 0.03, c, 0.008)
        spines = [(0.42, 0.12, 0.1), (0.15, 0.22, 0.35), (0.2, 0.32, 0.18), (0.45, 0.32, 0.12), (0.3, 0.2, 0.12), (0.55, 0.48, 0.32), (0.12, 0.12, 0.12)]
        for z in np.linspace(0.06, H - 0.4, 5):
            box(self.acc, "wood", M @ T(0, 0, z), L / 2, d / 2, 0.015, c, 0.004)
            x = -L / 2 + 0.04
            while x < L / 2 - 0.08:
                if rng.random() < 0.1:
                    x += rng.uniform(0.08, 0.2)
                    continue
                bw = rng.uniform(0.025, 0.06)
                bh = rng.uniform(0.2, 0.34)
                lean = rng.normal(0, 0.03) if rng.random() < 0.8 else 0.25
                col = k.J(spines[rng.integers(len(spines))], 0.1)
                box(self.acc, "cloth", M @ T(x + bw / 2, -0.01, z + 0.015 + bh / 2) @ R("y", lean), bw / 2, d / 2 - 0.04, bh / 2, col, 0.006)
                box(self.acc, "plaster", M @ T(x + bw / 2, -d / 2 + 0.035, z + 0.015 + bh * 0.75) @ R("y", lean), bw / 2 + 0.001, 0.002, 0.01, (0.75, 0.6, 0.25), 0)
                x += bw + 0.004
        for i in range(3):
            box(self.acc, "cloth", M @ T(0.2, 0, H - 0.37 + 0.015 + 0.03 * i + 0.015), 0.14, 0.1, 0.014, k.J(spines[i], 0.1), 0.004)

    def desk(self, M, L=1.5, Wd=0.75):
        """A writing desk with a map spread out, an inkwell and quill, a brass compass and a candle."""
        k, rng = self.k, self.rng
        self.table_long(M, L, Wd)
        patch(self.acc, "cloth", M @ frame((-0.45, 0.25, 0.79), (1, 0, 0), (0, 0, 1), (0, -1, 0)), 0.8, 0.55, 6, 4, (0.82, 0.74, 0.55),
              lambda U, V: 0.006 * np.sin(U * 9) * np.sin(V * 7),
              lambda U, V: np.clip(np.array([[0.82, 0.74, 0.55]]) * (0.9 + 0.1 * np.sin(U * 31 + V * 17))[:, None] - (np.abs(np.sin(U * 6 + np.sin(V * 5))) < 0.06)[:, None] * 0.25, 0, 1))
        lathe(self.acc, "glass", M @ T(0.5, 0.15, 0.785), [(0.0, 0), (0.035, 0), (0.04, 0.04), (0.02, 0.06), (0.0, 0.06)], 10, (0.05, 0.05, 0.06))
        tube(self.acc, "cloth", M, [(0.5, 0.15, 0.83), (0.42, 0.2, 1.05)], [0.004, 0.02], 5, (0.92, 0.9, 0.85))
        lathe(self.acc, "iron", M @ T(0.3, -0.2, 0.785), [(0.0, 0), (0.08, 0), (0.085, 0.02), (0.08, 0.03), (0.0, 0.03)], 16, BRASS)
        box(self.acc, "iron", M @ T(0.3, -0.2, 0.818) @ R("z", 0.6), 0.06, 0.006, 0.002, (0.5, 0.1, 0.08), 0)
        self.candle(M @ T(-0.6, -0.25, 0.785))
        for i in range(3):
            box(self.acc, "cloth", M @ T(0.55, -0.22, 0.8 + 0.035 * i), 0.13, 0.09, 0.016, k.J([(0.42, 0.12, 0.1), (0.15, 0.22, 0.35), (0.3, 0.2, 0.12)][i], 0.1), 0.004)

    def armchair(self, M):
        k = self.k
        c = k.J(OAK_DARK, 0.05)
        box(self.acc, "wood", M @ T(0, 0, 0.42), 0.32, 0.3, 0.04, c, 0.01)
        box(self.acc, "cloth", M @ T(0, -0.02, 0.5), 0.27, 0.25, 0.05, (0.45, 0.14, 0.12), 0.04)
        box(self.acc, "wood", M @ T(0, 0.28, 0.85), 0.32, 0.04, 0.45, c, 0.012)
        box(self.acc, "cloth", M @ T(0, 0.23, 0.85), 0.25, 0.02, 0.33, (0.45, 0.14, 0.12), 0.015)
        for sx in (-0.3, 0.3):
            box(self.acc, "wood", M @ T(sx, 0, 0.62), 0.03, 0.3, 0.03, c, 0.008)
            for sy in (-0.27, 0.27):
                box(self.acc, "wood", M @ T(sx, sy, 0.31 if sy < 0 else 0.42), 0.03, 0.03, 0.31 if sy < 0 else 0.42, c * 0.9, 0.006)

    def herb_bunch(self, M, col):
        k, rng = self.k, self.rng
        tube(self.acc, "cloth", M, [(0, 0, 0.0), (0, 0, -0.12)], 0.004, 3, (0.7, 0.62, 0.45))
        for i in range(9):
            a = rng.uniform(0, 6.28)
            l = rng.uniform(0.18, 0.3)
            pts = [(0, 0, -0.1), (0.03 * math.cos(a), 0.03 * math.sin(a), -0.1 - l * 0.5), (0.06 * math.cos(a), 0.06 * math.sin(a), -0.1 - l)]
            tube(self.acc, "produce", M, pts, 0.005, 3, k.J((0.35, 0.36, 0.2), 0.1), cap=False)
            blob(self.acc, "produce", M @ T(*pts[2]), rng.uniform(0.018, 0.03), k.J(col, 0.12), seg=5, rings=3, squash=1.4, rng=rng)

    def jar(self, M, h=0.16, col=(0.55, 0.4, 0.25), glass=False):
        prof = [(0.0, 0), (0.05, 0.0), (0.06, h * 0.4), (0.05, h * 0.85), (0.035, h * 0.9), (0.04, h), (0.0, h)]
        lathe(self.acc, "glass" if glass else "stone", M, prof, 10, col)
        if not glass:
            lathe(self.acc, "cloth", M @ T(0, 0, h), [(0.0, 0.01), (0.048, 0.0), (0.05, -0.02), (0.0, -0.02)], 8, (0.8, 0.75, 0.62))

    def jar_shelf(self, Mw, u0, u1, zs):
        k, rng = self.k, self.rng
        cols = [(0.55, 0.4, 0.25), (0.35, 0.42, 0.3), (0.6, 0.55, 0.45), (0.45, 0.3, 0.2)]
        for z in zs:
            self.shelf(Mw, u0, u1, z)
            u = u0 + 0.12
            while u < u1 - 0.1:
                if rng.random() < 0.35:
                    self.bottle(Mw @ T(u, -0.13, z + 0.02) @ np.diag([0.8, 0.8, 0.8, 1.0]), k.J([(0.1, 0.25, 0.12), (0.3, 0.12, 0.08), (0.12, 0.16, 0.3)][rng.integers(3)], 0.1))
                else:
                    self.jar(Mw @ T(u, -0.13, z + 0.02), rng.uniform(0.12, 0.2), k.J(cols[rng.integers(4)], 0.08))
                u += rng.uniform(0.13, 0.18)

    def cauldron(self, M, r=0.32):
        lathe(self.acc, "iron", M, [(0.0, 0.05), (r * 0.6, 0.06), (r, 0.25), (r * 1.02, 0.42), (r * 0.9, 0.5), (r * 0.92, 0.52), (r * 0.86, 0.52), (r * 0.84, 0.45),
                                    (0.0, 0.45)], 18, (0.09, 0.09, 0.09))
        lathe(self.acc, "produce", M @ T(0, 0, 0.44), [(0.0, 0), (r * 0.84, 0.0)], 18, (0.25, 0.4, 0.2))
        for a in (0.3, 2.4, 4.5):
            tube(self.acc, "iron", M, [(r * 0.6 * math.cos(a), r * 0.6 * math.sin(a), 0.1), (r * 0.7 * math.cos(a), r * 0.7 * math.sin(a), 0.0)], 0.02, 5, (0.09, 0.09, 0.09))

    def mortar(self, M):
        lathe(self.acc, "stone", M, [(0.0, 0), (0.07, 0), (0.09, 0.07), (0.085, 0.09), (0.06, 0.03), (0.0, 0.03)], 12, (0.55, 0.53, 0.5))
        tube(self.acc, "wood", M, [(0.0, 0.0, 0.04), (0.05, 0.02, 0.17)], [0.022, 0.012], 6, self.k.J(WOOD_RAW, 0.05))

    def armour_stand(self, M):
        """A mail shirt on a post, a helmet on top, a sword belt hung from the cross-arm."""
        k = self.k
        c = k.J(OAK_DARK, 0.05)
        box(self.acc, "wood", M @ T(0, 0, 0.03), 0.25, 0.25, 0.03, c, 0.008)
        tube(self.acc, "wood", M, [(0, 0, 0.05), (0, 0, 1.55)], 0.03, 6, c)
        tube(self.acc, "wood", M, [(-0.25, 0, 1.4), (0.25, 0, 1.4)], 0.025, 6, c)
        mail = lambda U, V: np.clip(np.array([[0.42, 0.43, 0.45]]) * (0.8 + 0.2 * (np.sin(U * 160) * np.sin(V * 160) > 0))[:, None], 0, 1)
        lathe(self.acc, "iron", M @ T(0, 0, 0.75), [(0.0, 0.0), (0.24, 0.0), (0.22, 0.2), (0.2, 0.45), (0.25, 0.62), (0.12, 0.7), (0.0, 0.72)], 14, (0.42, 0.43, 0.45))
        for s in (-1, 1):
            tube(self.acc, "iron", M, [(s * 0.2, 0, 1.36), (s * 0.3, 0, 1.2), (s * 0.32, 0, 1.0)], [0.08, 0.07, 0.06], 8, (0.42, 0.43, 0.45))
        lathe(self.acc, "iron", M @ T(0, 0, 1.55), [(0.0, 0.25), (0.08, 0.24), (0.13, 0.17), (0.14, 0.05), (0.15, 0.0), (0.0, 0.0)], 14, (0.5, 0.5, 0.52))
        box(self.acc, "iron", M @ T(0, -0.13, 1.63), 0.015, 0.015, 0.1, (0.5, 0.5, 0.52), 0.004)
        tube(self.acc, "cloth", M, [(-0.24, -0.02, 0.95), (0.0, -0.25, 0.92), (0.24, -0.02, 0.95)], 0.025, 6, (0.3, 0.18, 0.1))
        self.sword(M @ T(0.28, -0.1, 0.25) @ R("y", 0.12))

    def cot(self, M, L=1.9, Wd=0.85, col=(0.3, 0.38, 0.24)):
        k = self.k
        c = k.J(OAK, 0.05)
        for sx in (-L / 2, L / 2):
            for sy in (-Wd / 2, Wd / 2):
                box(self.acc, "wood", M @ T(sx, sy, 0.22), 0.035, 0.035, 0.22, c, 0.008)
        for sy in (-Wd / 2, Wd / 2):
            box(self.acc, "wood", M @ T(0, sy, 0.3), L / 2, 0.025, 0.06, c, 0.006)
        box(self.acc, "cloth", M @ T(0, 0, 0.4), L / 2 - 0.03, Wd / 2 - 0.03, 0.07, (0.7, 0.64, 0.5), 0.05)
        blob(self.acc, "cloth", M @ T(-L / 2 + 0.25, 0, 0.5), 0.2, LINEN, seg=10, rings=6, squash=0.35, rng=k.rng, rough=0.04)
        patch(self.acc, "cloth", M @ frame((-L / 2 + 0.5, Wd / 2 + 0.04, 0.48), (1, 0, 0), (0, 0, 1), (0, -1, 0)), L - 0.55, Wd + 0.08, 8, 6, col,
              lambda U, V: 0.025 * np.sin(U * 7) * np.sin(V * 6) - 0.14 * np.clip(np.abs(V - (Wd + 0.08) / 2) - Wd / 2 + 0.03, 0, 1) * 8)

    def cradle(self, M):
        k = self.k
        c = k.J(OAK, 0.05)
        box(self.acc, "wood", M @ T(0, 0, 0.35), 0.45, 0.25, 0.15, c, 0.02)
        box(self.acc, "cloth", M @ T(0, 0, 0.47), 0.4, 0.2, 0.04, LINEN, 0.03)
        for sx in (-0.45, 0.45):
            prism(self.acc, "wood", M @ frame((sx, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)), [(-0.3, 0.12)] + [(0.3 * math.cos(t), 0.12 + 0.1 * math.sin(t) * -1 + 0.1) for t in np.linspace(math.pi, 0, 7)][1:-1] +
                  [(0.3, 0.12), (0.25, 0.55), (-0.25, 0.55)], 0.03, c)

    def toy_horse(self, M):
        k = self.k
        c = k.J((0.6, 0.45, 0.3), 0.05)
        blob(self.acc, "wood", M @ T(0, 0, 0.32) @ np.diag([1.6, 1.0, 1.0, 1.0]), 0.08, c, seg=8, rings=5, rng=k.rng, rough=0.02)
        tube(self.acc, "wood", M, [(0.1, 0, 0.36), (0.16, 0, 0.48)], 0.04, 6, c)
        blob(self.acc, "wood", M @ T(0.2, 0, 0.5) @ np.diag([1.4, 1.0, 1.0, 1.0]), 0.045, c, seg=6, rings=4, rng=k.rng, rough=0.02)
        for sx in (-0.08, 0.08):
            for sy in (-0.04, 0.04):
                tube(self.acc, "wood", M, [(sx, sy, 0.28), (sx * 1.2, sy, 0.1)], 0.015, 4, c)
        prism(self.acc, "wood", M @ frame((0, 0, 0.0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), [(-0.2, 0.1), (0.2, 0.1), (0.25, 0.14), (0.0, 0.08), (-0.25, 0.14)], 0.14, c * 0.9)

    def loom(self, M, L=1.4):
        """A floor loom: frame, warp threads, cloth wound on the front beam, a seat."""
        k = self.k
        c = k.J(OAK, 0.05)
        for sx in (-L / 2, L / 2):
            box(self.acc, "wood", M @ T(sx, 0.5, 0.9), 0.04, 0.04, 0.9, c, 0.008)
            box(self.acc, "wood", M @ T(sx, -0.5, 0.45), 0.04, 0.04, 0.45, c, 0.008)
            box(self.acc, "wood", M @ T(sx, 0, 0.12), 0.035, 0.55, 0.035, c, 0.008)
            box(self.acc, "wood", M @ T(sx, 0.25, 1.35), 0.035, 0.3, 0.03, c, 0.008)
        tube(self.acc, "wood", M, [(-L / 2, 0.5, 1.7), (L / 2, 0.5, 1.7)], 0.04, 8, c)
        tube(self.acc, "wood", M, [(-L / 2, -0.5, 0.85), (L / 2, -0.5, 0.85)], 0.06, 10, c)
        lathe(self.acc, "cloth", M @ T(-L / 2 + 0.08, -0.5, 0.85) @ R("y", math.pi / 2), [(0.075, 0), (0.075, L - 0.16)], 14, (0.6, 0.2, 0.16))
        for i, x in enumerate(np.linspace(-L / 2 + 0.1, L / 2 - 0.1, 40)):
            tube(self.acc, "cloth", M, [(x, -0.45, 0.9), (x, 0.0, 1.05), (x, 0.5, 1.68)], 0.003, 3, (0.85, 0.8, 0.68), cap=False)
        box(self.acc, "wood", M @ T(0, 0.05, 1.03), L / 2, 0.03, 0.06, c * 0.9, 0.006)
        box(self.acc, "wood", M @ T(0, -0.95, 0.5), L / 2 - 0.1, 0.15, 0.03, c, 0.008)
        for sx in (-L / 2 + 0.2, L / 2 - 0.2):
            box(self.acc, "wood", M @ T(sx, -0.95, 0.25), 0.03, 0.12, 0.25, c * 0.9, 0.006)

    def spinning_wheel(self, M):
        k = self.k
        c = k.J(OAK, 0.05)
        box(self.acc, "wood", M @ T(0, 0, 0.35), 0.4, 0.08, 0.04, c, 0.008)
        for (x, y) in ((-0.32, -0.12), (-0.32, 0.12), (0.32, 0.0)):
            tube(self.acc, "wood", M, [(x * 0.9, y * 0.6, 0.33), (x, y, 0.0)], 0.02, 5, c * 0.9)
        Mw_ = M @ T(-0.12, 0, 0.75) @ R("x", math.pi / 2)
        tube(self.acc, "wood", Mw_, [(0.32 * math.cos(a), 0.32 * math.sin(a), 0) for a in np.linspace(0, 2 * math.pi, 25)], 0.02, 5, c, cap=False)
        for a in np.linspace(0, 2 * math.pi, 10, endpoint=False):
            tube(self.acc, "wood", Mw_, [(0, 0, 0), (0.3 * math.cos(a), 0.3 * math.sin(a), 0)], 0.008, 4, c)
        for s in (-0.06, 0.06):
            tube(self.acc, "wood", M, [(-0.12, s, 0.37), (-0.12, s, 0.78)], 0.018, 5, c)
        tube(self.acc, "wood", M, [(0.28, 0, 0.37), (0.3, 0, 0.72)], 0.018, 5, c)
        lathe(self.acc, "cloth", M @ T(0.3, 0, 0.72) @ R("y", math.pi / 2), [(0.0, -0.08), (0.04, -0.05), (0.045, 0.05), (0.0, 0.08)], 10, (0.88, 0.85, 0.78))

    def cloth_bolts(self, Mw, u0, u1, zs):
        k, rng = self.k, self.rng
        cols = [(0.55, 0.15, 0.12), (0.18, 0.28, 0.45), (0.3, 0.42, 0.25), (0.75, 0.62, 0.3), (0.85, 0.82, 0.74), (0.4, 0.25, 0.4)]
        for z in zs:
            self.shelf(Mw, u0, u1, z, d=0.42)
            u = u0 + 0.12
            while u < u1 - 0.1:
                r_ = rng.uniform(0.06, 0.09)
                lathe(self.acc, "cloth", Mw @ T(u, -0.21, z + 0.02 + r_) @ R("x", math.pi / 2) @ T(0, 0, -0.19), [(r_, 0), (r_, 0.38)], 12, k.J(cols[rng.integers(len(cols))], 0.06))
                u += r_ * 2 + 0.02

    def planks(self, M, n=8, L=2.4):
        k, rng = self.k, self.rng
        for i in range(n):
            box(self.acc, "wood", M @ T(rng.normal(0, 0.04), 0, 0.025 + i * 0.052), L / 2, 0.14, 0.024, k.J(WOOD_RAW, 0.08) * rng.uniform(0.9, 1.08), 0.005)
        for sx in (-L / 2 + 0.3, L / 2 - 0.3):
            box(self.acc, "wood", M @ T(sx, 0, -0.03), 0.05, 0.16, 0.03, k.J(OAK_DARK, 0.05), 0.005)

    def onions(self, M, n=9):
        k, rng = self.k, self.rng
        tube(self.acc, "cloth", M, [(0, 0, 0), (0, 0, -0.75)], 0.008, 4, (0.7, 0.62, 0.45))
        for i in range(n):
            a = rng.uniform(0, 6.28)
            blob(self.acc, "produce", M @ T(0.04 * math.cos(a), 0.04 * math.sin(a), -0.1 - i * 0.07), rng.uniform(0.035, 0.045), k.J((0.72, 0.5, 0.25), 0.08),
                 seg=7, rings=4, squash=0.9, rng=rng, rough=0.04)

    def dresser(self, M, L=1.4):
        k = self.k
        c = k.J(OAK, 0.05)
        box(self.acc, "wood", M @ T(0, 0, 0.42), L / 2, 0.25, 0.42, c, 0.012)
        box(self.acc, "wood", M @ T(0, 0, 0.86), L / 2 + 0.03, 0.28, 0.025, c * 1.05, 0.008)
        for i in range(2):
            box(self.acc, "wood", M @ T(-L / 4 + i * L / 2, -0.253, 0.42), L / 4 - 0.03, 0.006, 0.35, c * 0.9, 0.006)
            lathe(self.acc, "iron", M @ T(-L / 4 + i * L / 2 + (0.15 if i == 0 else -0.15), -0.26, 0.5) @ R("x", math.pi / 2), [(0.0, 0), (0.015, 0), (0.0, 0.03)], 6, BRASS)
        box(self.acc, "wood", M @ T(0, 0.2, 1.45), L / 2, 0.04, 0.58, c * 0.9, 0.008)
        for z in (1.25, 1.65, 2.0):
            box(self.acc, "wood", M @ T(0, 0.08, z), L / 2, 0.13, 0.015, c, 0.004)
        for sx in (-L / 2, L / 2):
            box(self.acc, "wood", M @ T(sx, 0.08, 1.45), 0.02, 0.15, 0.58, c, 0.004)
        for z in (1.265, 1.665):
            for i in range(4):
                x = -L / 2 + 0.2 + i * (L - 0.4) / 3
                Mp = M @ T(x, 0.12, z) @ R("x", -1.35)
                self.plate(Mp, k.J((0.82, 0.78, 0.7), 0.04) if i % 2 else k.J((0.35, 0.45, 0.6), 0.05))
        for i in range(5):
            self.tankard(M @ T(-L / 2 + 0.15 + i * 0.27, 0.05, 2.015), PEWTER)
