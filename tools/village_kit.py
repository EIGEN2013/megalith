"""Building parts for the village: timber-framed and stone walls with their openings, jetties, slate and thatch
roofs, chimneys, dormers and the small things around a house. Everything is built in building-local Blender
axes (x along the front, front facing -y, z up from the ground-floor level).

Slates are not geometry here: they are recorded as instances of the village's own slate pieces (`Roof.slates`),
so the new roofs are laid exactly like the old ones.
"""
import math

import numpy as np

from geo_kit import T, R, chain, frame, box, fbox, prism, quad, lathe, tube, patch, blob

# ---------------------------------------------------------------- palette (village convention: sRGB-ish)
WOOD = (0.115, 0.088, 0.066)
WOOD_GREY = (0.20, 0.18, 0.155)
WOOD_RAW = (0.38, 0.29, 0.19)
IRON = (0.066, 0.066, 0.07)
GLASS = (0.047, 0.056, 0.064)
STONE = (0.48, 0.44, 0.38)
STONE_GREY = (0.45, 0.45, 0.43)
MORTAR = (0.36, 0.34, 0.30)
TERRA = (0.56, 0.29, 0.19)
THATCH = (0.47, 0.40, 0.26)
LEAF = (0.17, 0.26, 0.10)
PLASTER = {"cream": (0.83, 0.78, 0.65), "ochre": (0.80, 0.66, 0.45), "white": (0.87, 0.85, 0.79),
           "pink": (0.84, 0.71, 0.64), "sage": (0.75, 0.77, 0.67), "lime": (0.80, 0.80, 0.74)}
PAINT = {"red": (0.50, 0.14, 0.11), "green": (0.13, 0.28, 0.22), "blue": (0.16, 0.24, 0.35), "ochre": (0.62, 0.44, 0.17),
         "black": (0.07, 0.07, 0.07)}
FLOWERS = [(0.80, 0.20, 0.28), (0.88, 0.62, 0.16), (0.78, 0.48, 0.70), (0.93, 0.90, 0.82), (0.55, 0.30, 0.72), (0.86, 0.36, 0.20)]

TW = 0.17          # timber width (face)
TD = 0.17          # timber depth
PROUD = 0.045      # how far timbers stand out of the plaster


class Kit:
    def __init__(self, acc, rng, instanced=False):
        self.acc, self.rng = acc, rng
        self.blocks = [] if instanced else None      # stones as (4x4 local matrix, sRGB colour, front normal)

    def tone(self, col, v=0.11, light=False):
        """Stone colour: varies in brightness, with only a faint warm/cool cast."""
        c = np.asarray(col, float) * self.rng.normal(1.0, v)
        w = self.rng.normal(0, 0.022)
        c = c + np.array([w, w * 0.35, -w * 0.8])
        return np.clip(c * (1.07 if light else 1.0), 0, 1)

    def J(self, c, a=0.06):
        c = np.asarray(c, float)
        return np.clip(c * (1 + self.rng.normal(0, a)) + self.rng.normal(0, a * 0.3, 3) * c, 0, 1)

    # ------------------------------------------------------------ wall frames
    @staticmethod
    def wall_frame(side, x0, x1, y0, y1, z):
        """Matrix for a wall: local x along the wall (left -> right seen from outside), y inward, z up."""
        if side == "front":
            o, u, n, L = (x0, y0, z), (1, 0, 0), (0, -1, 0), x1 - x0
        elif side == "back":
            o, u, n, L = (x1, y1, z), (-1, 0, 0), (0, 1, 0), x1 - x0
        elif side == "left":
            o, u, n, L = (x0, y1, z), (0, -1, 0), (-1, 0, 0), y1 - y0
        else:
            o, u, n, L = (x1, y0, z), (0, 1, 0), (1, 0, 0), y1 - y0
        return frame(o, u, -np.array(n, float), (0, 0, 1)), L

    # ------------------------------------------------------------ timbers in a wall
    def timber(self, Mw, a, b, w, col, depth=TD, proud=PROUD, bev=0.014):
        """A timber in the wall plane between (u, v) points a and b."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        L = np.hypot(*d)
        if L < 1e-4:
            return
        X = np.array([d[0] / L, 0, d[1] / L])
        Y = np.array([0, 1.0, 0])
        Z = np.cross(X, Y)
        c = (a + b) / 2
        M = Mw @ frame((c[0], depth / 2 - proud, c[1]), X, Y, Z)
        fbox(self.acc, "wood", M, L / 2, depth / 2, w / 2, self.J(col, 0.08), bev)

    def plaster_cells(self, Mw, L, H, holes, col, mat="plaster", bulge=0.012, cell=0.32, y=0.0, stains=True):
        """The wall face minus its openings, as gently bulging, mottled patches."""
        us = sorted(set([0.0, L] + [h[0] for h in holes] + [h[1] for h in holes]))
        vs = sorted(set([0.0, H] + [h[2] for h in holes] + [h[3] for h in holes]))
        us = [u for u in us if -1e-6 <= u <= L + 1e-6]
        vs = [v for v in vs if -1e-6 <= v <= H + 1e-6]
        base = np.asarray(col, float)
        ph = self.rng.uniform(0, 100, 4)
        for i in range(len(us) - 1):
            for j in range(len(vs) - 1):
                u0, u1, v0, v1 = us[i], us[i + 1], vs[j], vs[j + 1]
                if u1 - u0 < 1e-4 or v1 - v0 < 1e-4:
                    continue
                cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
                if any(h[0] - 1e-6 <= cu <= h[1] + 1e-6 and h[2] - 1e-6 <= cv <= h[3] + 1e-6 for h in holes):
                    continue
                nu = max(1, int(round((u1 - u0) / cell)))
                nv = max(1, int(round((v1 - v0) / cell)))
                w, h = u1 - u0, v1 - v0

                def disp(U, V, w=w, h=h):
                    s = np.sin(np.pi * np.clip(U / w, 0, 1)) * np.sin(np.pi * np.clip(V / h, 0, 1))
                    return bulge * s * min(1.0, min(w, h) / 0.8)

                def colfn(U, V, u0=u0, v0=v0):
                    gu, gv = U + u0, V + v0
                    m = 0.5 + 0.25 * np.sin(gu * 2.1 + ph[0]) * np.sin(gv * 1.7 + ph[1]) + 0.25 * np.sin(gu * 5.3 + gv * 3.1 + ph[2])
                    k = 0.93 + 0.1 * m
                    out = base[None] * k[:, None]
                    if stains:   # rain streaks under sills and from the top
                        st = np.clip(np.sin(gu * 9.0 + ph[3]) * 0.5 + 0.5, 0, 1) ** 6 * np.clip((gv - H * 0.5) / H, 0, 1) * 0.18
                        out = out * (1 - st[:, None] * np.array([0.6, 0.62, 0.7])[None])
                    return np.clip(out, 0, 1)

                patch(self.acc, mat, Mw @ T(u0, y, v0), w, h, nu, nv, base, disp, colfn)

    def timber_wall(self, Mw, L, H, openings, pcol, wcol=WOOD, style="cross", plates=(True, True), sill_h=0.95, corner_posts=True):
        """A timber-framed storey wall: plates, posts, rails, braces, plaster infill and its openings."""
        rng = self.rng
        holes = [self.hole(o) for o in openings]
        self.plaster_cells(Mw, L, H, holes, pcol)
        for o in openings:
            self.opening(Mw, o, "plaster", pcol)
        bot = TW if plates[0] else 0.0
        top = H - (TW if plates[1] else 0.0)
        if plates[0]:
            self.timber(Mw, (0, bot / 2), (L, bot / 2), TW * 1.15, wcol)
        if plates[1]:
            self.timber(Mw, (0, H - TW / 2), (L, H - TW / 2), TW, wcol)
        # posts: the corners, both sides of every opening, then infill no wider than ~1.25 m
        posts = []
        if corner_posts:
            posts += [TW / 2, L - TW / 2]
        for o in openings:
            u0, u1 = o["u"] - o["w"] / 2, o["u"] + o["w"] / 2
            posts += [u0 - TW / 2, u1 + TW / 2]
        posts = sorted(p for p in posts if 0 <= p <= L)
        filled = []
        for p in posts:
            if filled and p - filled[-1] < TW * 1.2:
                continue
            filled.append(p)
        posts = filled
        full = [TW / 2] + posts + [L - TW / 2] if not corner_posts else posts
        allp = []
        for a, b in zip(full[:-1], full[1:]):
            allp.append(a)
            gap = b - a
            n = int(gap // 1.25)
            if gap > 1.4 and not self._opening_between(openings, a, b):
                for k in range(1, n + 1):
                    allp.append(a + gap * k / (n + 1))
        allp.append(full[-1])
        allp = sorted(set(round(p, 4) for p in allp))
        for p in allp:
            self.timber(Mw, (p, bot), (p, top), TW, wcol)
        # bays
        for i, (a, b) in enumerate(zip(allp[:-1], allp[1:])):
            ua, ub = a + TW / 2, b - TW / 2
            if ub - ua < 0.12:
                continue
            o = self._opening_in(openings, a, b)
            if o is not None:
                v0, v1 = o["sill"], o["sill"] + o["h"]
                if o["kind"] == "door":
                    self.timber(Mw, (ua - TW / 2, v1 + TW / 2), (ub + TW / 2, v1 + TW / 2), TW, wcol)
                    if top - (v1 + TW) > 0.45:
                        for s in np.linspace(ua, ub, 3)[1:-1]:
                            self.timber(Mw, (s, v1 + TW), (s, top), TW * 0.7, wcol)
                    continue
                self.timber(Mw, (ua - TW / 2, v0 - TW / 2), (ub + TW / 2, v0 - TW / 2), TW, wcol)
                self.timber(Mw, (ua - TW / 2, v1 + TW / 2), (ub + TW / 2, v1 + TW / 2), TW, wcol)
                lo_top = v0 - TW
                if lo_top - bot > 0.45:          # a small saltire under the window
                    self.timber(Mw, (ua, bot), (ub, lo_top), TW * 0.75, wcol)
                    self.timber(Mw, (ua, lo_top), (ub, bot), TW * 0.75, wcol)
                if top - (v1 + TW) > 0.4:
                    for s in np.linspace(ua, ub, 3)[1:-1]:
                        self.timber(Mw, (s, v1 + TW), (s, top), TW * 0.7, wcol)
                continue
            mid = bot + (top - bot) * (0.48 if style != "tall" else 0.0)
            edge = i == 0 or i == len(allp) - 2
            if style == "tall" or (edge and style in ("cross", "curve")):
                # long corner braces, leaning in toward the corner post
                if i == 0:
                    self.timber(Mw, (ua, top - 0.05), (ub, bot + 0.05), TW, wcol)
                else:
                    self.timber(Mw, (ua, bot + 0.05), (ub, top - 0.05), TW, wcol)
                continue
            self.timber(Mw, (ua - TW / 2, mid), (ub + TW / 2, mid), TW * 0.9, wcol)
            pat = rng.choice(["X", "X", "V", "K"]) if style == "cross" else style
            if pat in ("X",):
                for (v0, v1) in ((bot, mid - TW * 0.45), (mid + TW * 0.45, top)):
                    if v1 - v0 > 0.35:
                        self.timber(Mw, (ua, v0), (ub, v1), TW * 0.8, wcol)
                        self.timber(Mw, (ua, v1), (ub, v0), TW * 0.8, wcol)
            elif pat == "V":
                c = (ua + ub) / 2
                self.timber(Mw, (ua, top), (c, mid + TW * 0.45), TW * 0.8, wcol)
                self.timber(Mw, (ub, top), (c, mid + TW * 0.45), TW * 0.8, wcol)
                self.timber(Mw, (ua, bot), (ub, mid - TW * 0.45), TW * 0.8, wcol)
            else:
                self.timber(Mw, (ua, mid - TW * 0.45), (ub, bot), TW * 0.8, wcol)
                self.timber(Mw, (ua, mid + TW * 0.45), (ub, top), TW * 0.8, wcol)

    @staticmethod
    def _opening_in(openings, a, b):
        for o in openings:
            if a - 1e-3 <= o["u"] <= b + 1e-3 and o["u"] - o["w"] / 2 >= a - 1e-3 and o["u"] + o["w"] / 2 <= b + 1e-3 + TW:
                return o
        return None

    @staticmethod
    def _opening_between(openings, a, b):
        return any(a < o["u"] < b for o in openings)

    @staticmethod
    def hole(o):
        u0, u1 = o["u"] - o["w"] / 2, o["u"] + o["w"] / 2
        v0 = o["sill"]
        v1 = v0 + o["h"] + (o["w"] / 2 if o.get("arch") else 0)
        return (u0, u1, v0, v1)

    # ------------------------------------------------------------ stone walls
    def courses(self, H, coursed=False):
        out, v = [], 0.0
        while v < H - 0.05:
            ch = 0.3 if coursed else self.rng.uniform(0.2, 0.34)
            ch = min(ch, H - v)
            out.append((v, ch))
            v += ch
        return out

    def stone_wall(self, Mw, L, H, openings, col=STONE, coursed=False, quoins=True, courses=None, clip=None, mortar=MORTAR,
                   parity=0, proud=0.05):
        """Stones laid in courses over a mortar bed, quoins at the ends, openings left clear (with dressed surrounds)."""
        rng = self.rng
        holes = [self.hole(o) for o in openings]
        if clip is None:
            self.plaster_cells(Mw, L, H, holes, mortar, mat="stone", bulge=0.0, cell=0.6, stains=False)
        else:
            self._mortar_clipped(Mw, L, H, clip, mortar)
        courses = courses or self.courses(H, coursed)
        gap = 0.022
        qlen = (0.62, 0.36)
        for k, (v, ch) in enumerate(courses):
            vc = v + ch / 2
            segs = [(0.0, L)]
            for h in holes:
                if h[2] - 0.02 < vc < h[3] + 0.02 or (h[2] - 0.02 < v + ch and v < h[3] + 0.02):
                    nsegs = []
                    for a, b in segs:
                        if b <= h[0] - 0.03 or a >= h[1] + 0.03:
                            nsegs.append((a, b))
                            continue
                        if h[0] - 0.03 - a > 0.08:
                            nsegs.append((a, h[0] - 0.03))
                        if b - (h[1] + 0.03) > 0.08:
                            nsegs.append((h[1] + 0.03, b))
                    segs = nsegs
            for a, b in segs:
                u = a
                if quoins and a < 1e-6:
                    q = qlen[(k + parity) % 2]
                    self._block(Mw, u, u + q, v, ch, col, gap, proud + 0.025, light=True, clip=clip)
                    u += q
                end = b - (qlen[(k + parity + 1) % 2] if (quoins and abs(b - L) < 1e-6) else 0.0)
                while u < end - 0.06:
                    bl = rng.uniform(0.32, 0.78) if not coursed else rng.uniform(0.45, 0.9)
                    if end - (u + bl) < 0.2:
                        bl = end - u
                    self._block(Mw, u, u + bl, v, ch, col, gap, proud * rng.uniform(0.5, 1.1), clip=clip)
                    u += bl
                if quoins and abs(b - L) < 1e-6:
                    self._block(Mw, end, L, v, ch, col, gap, proud + 0.025, light=True, clip=clip)
        for o in openings:
            self.opening(Mw, o, "stone", mortar, stone=True, scol=col)

    def _block(self, Mw, u0, u1, v, ch, col, gap, proud, light=False, clip=None):
        if u1 - u0 < 0.06:
            return
        hu, hv = (u1 - u0 - gap) / 2, (ch - gap) / 2
        cu, cv = (u0 + u1) / 2, v + ch / 2
        if clip is not None:
            top = clip(cu)
            if cv - hv > top - 0.05:
                return
            if cv + hv > top:
                hv = (top - (cv - hv)) / 2
                cv = top - hv
                if hv < 0.04:
                    return
        d = 0.09 + proud
        c = self.tone(col, light=light)
        tilt = self.rng.normal(0, 0.012)
        M = Mw @ T(cu, d / 2 - proud, cv) @ R("y", tilt)
        if self.blocks is not None:
            self.blocks.append((M @ np.diag([2 * hu, d, 2 * hv, 1.0]), c))
            return
        fbox(self.acc, "stone", M, hu, d / 2, hv, c, min(0.035, hu * 0.3, hv * 0.3))

    def _mortar_clipped(self, Mw, L, H, clip, col):
        us = np.linspace(0, L, max(2, int(L / 0.4)) + 1)
        poly = [(0.0, 0.0), (L, 0.0)] + [(u, min(H, clip(u))) for u in us[::-1]]
        M = Mw @ frame((0, 0.02, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        prism(self.acc, "stone", M, poly, 0.04, col, cap_back=False)

    # ------------------------------------------------------------ windows and doors
    def opening(self, Mw, o, wallmat, wallcol, stone=False, scol=STONE):
        if o["kind"] == "blind":     # a hole in the wall with nothing in it (something outside covers it)
            return
        if o["kind"] == "door":
            self.door(Mw, o, wallmat, wallcol, stone, scol)
        else:
            self.window(Mw, o, wallmat, wallcol, stone, scol)

    def reveals(self, Mw, u0, u1, v0, v1, depth, mat, col):
        c = np.asarray(col, float) * 0.86
        quad(self.acc, mat, Mw, [(u0, 0, v0), (u0, depth, v0), (u0, depth, v1), (u0, 0, v1)], c)
        quad(self.acc, mat, Mw, [(u1, 0, v0), (u1, 0, v1), (u1, depth, v1), (u1, depth, v0)], c)
        quad(self.acc, mat, Mw, [(u0, 0, v0), (u1, 0, v0), (u1, depth, v0), (u0, depth, v0)], c * 0.95)
        quad(self.acc, mat, Mw, [(u0, 0, v1), (u0, depth, v1), (u1, depth, v1), (u1, 0, v1)], c * 0.7)

    def window(self, Mw, o, wallmat, wallcol, stone=False, scol=STONE):
        rng = self.rng
        u, w, h, v0 = o["u"], o["w"], o["h"], o["sill"]
        u0, u1, v1 = u - w / 2, u + w / 2, v0 + h
        depth = o.get("depth", 0.24 if stone else 0.17)
        arch = o.get("arch")
        if arch == "lancet" or arch == "round":
            self._arched_window(Mw, o, depth, scol)
            return
        if o.get("inside"):     # seen from a room: leading on the inner face and an inner sill (the glass faces outward only)
            gy_ = depth * 0.62
            self.lattice(Mw, u0 + 0.065, u1 - 0.065, v0 + 0.065, v1 - 0.065, gy_ + 0.02, inward=True)
            box(self.acc, "wood", Mw @ T(u, depth + 0.04, v0 - 0.03), w / 2 + 0.06, 0.07, 0.03, self.J(WOOD_GREY, 0.05), 0.008)
        self.reveals(Mw, u0, u1, v0, v1, depth, wallmat if not stone else "stone", wallcol if not stone else scol)
        gy = depth * 0.62
        fw = 0.065
        fcol = self.J(o.get("frame", WOOD), 0.05)
        # interior dark behind the glass
        quad(self.acc, "glass", Mw, [(u0, depth, v0), (u1, depth, v0), (u1, depth, v1), (u0, depth, v1)], np.array(GLASS) * 0.5)
        # frame, mullion, transom
        Mf = Mw @ T(0, gy, 0)
        box(self.acc, "wood", Mf @ T(u0 + fw / 2, 0, (v0 + v1) / 2), fw / 2, 0.045, h / 2, fcol, 0.008)
        box(self.acc, "wood", Mf @ T(u1 - fw / 2, 0, (v0 + v1) / 2), fw / 2, 0.045, h / 2, fcol, 0.008)
        box(self.acc, "wood", Mf @ T(u, 0, v0 + fw / 2), w / 2, 0.045, fw / 2, fcol, 0.008)
        box(self.acc, "wood", Mf @ T(u, 0, v1 - fw / 2), w / 2, 0.045, fw / 2, fcol, 0.008)
        panes_u = [u0 + fw, u1 - fw]
        panes_v = [v0 + fw, v1 - fw]
        if w > 0.62:
            box(self.acc, "wood", Mf @ T(u, -0.005, (v0 + v1) / 2), fw * 0.55, 0.05, h / 2 - fw, fcol, 0.006)
            panes_u = [u0 + fw, u - fw * 0.55, u + fw * 0.55, u1 - fw]
        if h > 1.05:
            vt = v0 + h * 0.68
            box(self.acc, "wood", Mf @ T(u, -0.005, vt), w / 2 - fw, 0.05, fw * 0.55, fcol, 0.006)
            panes_v = [v0 + fw, vt - fw * 0.55, vt + fw * 0.55, v1 - fw]
        for i in range(0, len(panes_u), 2):
            for j in range(0, len(panes_v), 2):
                a, b, c, d = panes_u[i], panes_u[i + 1], panes_v[j], panes_v[j + 1]
                gc = self.J(GLASS, 0.15)
                quad(self.acc, "glass", Mw, [(a, gy + 0.01, c), (b, gy + 0.01, c), (b, gy + 0.01, d), (a, gy + 0.01, d)], gc)
                self.lattice(Mw, a, b, c, d, gy + 0.004)
        # sill
        sc = scol if stone else fcol
        smat = "stone" if stone else "wood"
        box(self.acc, smat, Mw @ T(u, -0.05, v0 - 0.035), w / 2 + 0.07, 0.1, 0.035, self.J(sc, 0.05), 0.01)
        if stone:   # a dressed lintel
            box(self.acc, "stone", Mw @ T(u, -0.035, v1 + 0.11), w / 2 + 0.16, 0.09, 0.11, np.clip(self.J(scol, 0.05) * 1.07, 0, 1), 0.02)
        if o.get("shutters"):
            self.shutters(Mw, u0, u1, v0, v1, PAINT[o["shutters"]])
        if o.get("flowers"):
            self.flower_box(Mw, u, w, v0, o.get("box_col", WOOD))

    def lattice(self, Mw, a, b, c, d, y, step=0.105, wid=0.011, inward=False):
        """Diagonal leading of the panes, as thin flat strips."""
        w, h = b - a, d - c
        for sgn in (1, -1):
            k0 = -h if sgn > 0 else 0
            ks = np.arange(-h - w, w + h, step)
            for k in ks:
                # line: (x - a) - sgn*(z - c) = k  -> clip to rect
                pts = []
                for x in (a, b):
                    z = c + ((x - a) - k) * sgn
                    if c - 1e-6 <= z <= d + 1e-6:
                        pts.append((x, z))
                for z in (c, d):
                    x = a + k + sgn * (z - c)
                    if a - 1e-6 <= x <= b + 1e-6:
                        pts.append((x, z))
                if len(pts) < 2:
                    continue
                pts = sorted(set((round(p[0], 5), round(p[1], 5)) for p in pts))
                if len(pts) < 2:
                    continue
                p0, p1 = np.array(pts[0]), np.array(pts[-1])
                L = np.linalg.norm(p1 - p0)
                if L < 0.02:
                    continue
                dvec = (p1 - p0) / L
                nrm = np.array([-dvec[1], dvec[0]]) * wid / 2
                q = [(p0[0] - nrm[0], y, p0[1] - nrm[1]), (p1[0] - nrm[0], y, p1[1] - nrm[1]), (p1[0] + nrm[0], y, p1[1] + nrm[1]), (p0[0] + nrm[0], y, p0[1] + nrm[1])]
                n = np.cross(np.subtract(q[1], q[0]), np.subtract(q[2], q[0]))
                if (n[1] > 0) != inward:
                    q = q[::-1]
                quad(self.acc, "iron", Mw, q, IRON)

    def shutters(self, Mw, u0, u1, v0, v1, col):
        rng = self.rng
        lw = (u1 - u0) / 2 + 0.025
        h = v1 - v0 + 0.02
        for side in (-1, 1):
            ang = math.radians(rng.uniform(152, 176))
            hinge = u0 if side < 0 else u1
            M = Mw @ T(hinge, -0.035, v0 - 0.01) @ R("z", -ang if side < 0 else ang)
            sgn = 1 if side < 0 else -1
            c0 = self.J(col, 0.05)
            n = 3
            for k in range(n):
                pw = lw / n
                cx = sgn * (pw * (k + 0.5))
                box(self.acc, "wood", M @ T(cx, 0, h / 2), pw / 2 - 0.004, 0.017, h / 2, self.J(c0, 0.04), 0.006)
            for z in (0.16, h - 0.16):
                box(self.acc, "wood", M @ T(sgn * lw / 2, 0.03, z), lw / 2 - 0.02, 0.014, 0.05, c0 * 0.9, 0.005)
                box(self.acc, "iron", M @ T(sgn * lw * 0.38, -0.02, z), lw * 0.38, 0.004, 0.017, IRON, 0)
            # the Z brace on the back
            d = np.array([lw - 0.06, h - 0.42])
            L = np.linalg.norm(d)
            a = math.atan2(d[1], d[0])
            box(self.acc, "wood", M @ T(sgn * lw / 2, 0.03, h / 2) @ R("y", -a * sgn), L / 2, 0.014, 0.045, c0 * 0.9, 0.005)

    def flower_box(self, Mw, u, w, v0, col):
        rng = self.rng
        bw = w / 2 + 0.04
        M = Mw @ T(u, -0.2, v0 - 0.14)
        box(self.acc, "wood", M, bw, 0.1, 0.1, self.J(col, 0.05), 0.012)
        box(self.acc, "wood", Mw @ T(u - w * 0.35, -0.12, v0 - 0.27), 0.025, 0.11, 0.025, col, 0.005)
        box(self.acc, "wood", Mw @ T(u + w * 0.35, -0.12, v0 - 0.27), 0.025, 0.11, 0.025, col, 0.005)
        n = max(4, int(w / 0.075))
        for k in range(n):
            x = u - bw + 0.05 + (2 * bw - 0.1) * k / (n - 1)
            for _ in range(2):
                p = (x + rng.normal(0, 0.02), -0.2 + rng.normal(0, 0.04), v0 - 0.02 + rng.uniform(0, 0.06))
                blob(self.acc, "produce", Mw @ T(*p), rng.uniform(0.035, 0.055), self.J(LEAF, 0.12), seg=6, rings=3, squash=0.8, rng=rng)
            fc = FLOWERS[rng.integers(len(FLOWERS))]
            for _ in range(rng.integers(1, 4)):
                p = (x + rng.normal(0, 0.03), -0.2 + rng.normal(0, 0.05), v0 + 0.03 + rng.uniform(0, 0.08))
                blob(self.acc, "produce", Mw @ T(*p), rng.uniform(0.02, 0.032), self.J(fc, 0.08), seg=5, rings=3, squash=0.7, rng=rng)
            if rng.random() < 0.35:   # trailing leaves over the front
                tube(self.acc, "produce", Mw, [(x, -0.3, v0 - 0.05), (x + rng.normal(0, 0.03), -0.32, v0 - 0.2), (x + rng.normal(0, 0.04), -0.31, v0 - 0.34)],
                     0.012, 4, self.J(LEAF, 0.1))

    def _arched_window(self, Mw, o, depth, scol):
        """Lancet (pointed) or round-headed window in stone: splayed reveals, voussoirs, leaded glass."""
        u, w, h, v0 = o["u"], o["w"], o["h"], o["sill"]
        r = w / 2
        pts = []
        if o["arch"] == "round":
            for t in np.linspace(0, math.pi, 13):
                pts.append((u + r * math.cos(t), v0 + h + r * math.sin(t)))
            top = v0 + h + r
        else:
            # two arcs of radius w centred on the opposite springers
            for t in np.linspace(0, math.radians(60), 7):
                pts.append((u - r + w * math.cos(t), v0 + h + w * math.sin(t)))
            for t in np.linspace(math.radians(120), math.radians(180), 7)[1:]:
                pts.append((u + r + w * math.cos(t), v0 + h + w * math.sin(t)))
            top = v0 + h + w * math.sin(math.radians(60))
        outline = [(u - r, v0)] + [(u - r, v0 + h)] + pts[::-1] if False else None
        # straight reveals
        self.reveals_partial(Mw, u - r, u + r, v0, v0 + h, depth, scol)
        # arched soffit
        for (a, b) in zip(pts[:-1], pts[1:]):
            quad(self.acc, "stone", Mw, [(a[0], 0, a[1]), (a[0], depth, a[1]), (b[0], depth, b[1]), (b[0], 0, b[1])][::-1], np.asarray(scol) * 0.8)
        # glass + leading, as a polygon
        poly = [(u - r, v0), (u + r, v0)] + [(p[0], p[1]) for p in pts]
        Mg = Mw @ frame((0, depth * 0.62, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        prism(self.acc, "glass", Mg, poly, 0.01, self.J(GLASS, 0.1), cap_back=False)
        self.lattice(Mw, u - r, u + r, v0, top, depth * 0.62 - 0.012)
        if o.get("inside"):   # leading seen from the room, clipped to the opening's straight part
            self.lattice(Mw, u - r + 0.02, u + r - 0.02, v0 + 0.02, v0 + h, depth * 0.62 + 0.02, inward=True)
            box(self.acc, "stone", Mw @ T(u, depth + 0.05, v0 - 0.04), r + 0.08, 0.08, 0.04, np.clip(self.J(scol, 0.05) * 1.04, 0, 1), 0.01)
        # voussoirs round the head, keystone proud
        n = 9
        for k in range(n):
            t0, t1 = k / n, (k + 1) / n
            a = pts[int(round(t0 * (len(pts) - 1)))]
            b = pts[int(round(t1 * (len(pts) - 1)))]
            m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            dirv = np.array([m[0] - u, m[1] - (v0 + h)])
            dirv /= np.linalg.norm(dirv) + 1e-9
            out = 0.24
            poly2 = [a, b, (b[0] + dirv[0] * out, b[1] + dirv[1] * out), (a[0] + dirv[0] * out, a[1] + dirv[1] * out)]
            Mv = Mw @ frame((0, -0.02 - (0.02 if k == n // 2 else 0), 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
            prism(self.acc, "stone", Mv, poly2, 0.12, np.clip(self.J(scol, 0.06) * 1.08, 0, 1))
        box(self.acc, "stone", Mw @ T(u, -0.06, v0 - 0.05), r + 0.12, 0.12, 0.05, np.clip(self.J(scol, 0.05) * 1.06, 0, 1), 0.015)

    def reveals_partial(self, Mw, u0, u1, v0, v1, depth, col):
        c = np.asarray(col, float) * 0.85
        quad(self.acc, "stone", Mw, [(u0, 0, v0), (u0, depth, v0), (u0, depth, v1), (u0, 0, v1)], c)
        quad(self.acc, "stone", Mw, [(u1, 0, v0), (u1, 0, v1), (u1, depth, v1), (u1, depth, v0)], c)
        quad(self.acc, "stone", Mw, [(u0, 0, v0), (u1, 0, v0), (u1, depth, v0), (u0, depth, v0)], c * 0.95)

    def door(self, Mw, o, wallmat, wallcol, stone=False, scol=STONE):
        rng = self.rng
        u, w, h = o["u"], o["w"], o["h"]
        u0, u1 = u - w / 2, u + w / 2
        depth = o.get("depth", 0.26 if stone else 0.18)
        arch = o.get("arch")
        dcol = self.J(o.get("col", WOOD), 0.05)
        Mw_frame = Mw
        if arch:
            r = w / 2
            pts = [(u + r * math.cos(t), h + r * math.sin(t)) for t in np.linspace(0, math.pi, 15)]
            if arch == "lancet":
                pts = [(u - r + w * math.cos(t), h + w * math.sin(t)) for t in np.linspace(0, math.radians(60), 8)] + \
                      [(u + r + w * math.cos(t), h + w * math.sin(t)) for t in np.linspace(math.radians(120), math.radians(180), 8)[1:]]
                pts = pts[::-1]
                pts = [(p[0], p[1]) for p in sorted(pts, key=lambda p: -p[0])]
            self.reveals_partial(Mw, u0, u1, 0, h, depth, scol if stone else wallcol)
            for (a, b) in zip(pts[:-1], pts[1:]):
                quad(self.acc, "stone", Mw, [(a[0], 0, a[1]), (a[0], depth, a[1]), (b[0], depth, b[1]), (b[0], 0, b[1])][::-1], np.asarray(scol) * 0.8)
            # voussoirs
            n = 11
            for k in range(n):
                a = pts[int(round(k / n * (len(pts) - 1)))]
                b = pts[int(round((k + 1) / n * (len(pts) - 1)))]
                m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                dv = np.array([m[0] - u, m[1] - h])
                dv /= np.linalg.norm(dv) + 1e-9
                out = 0.3
                poly2 = [a, b, (b[0] + dv[0] * out, b[1] + dv[1] * out), (a[0] + dv[0] * out, a[1] + dv[1] * out)]
                Mv = Mw @ frame((0, -0.03 - (0.03 if k == n // 2 else 0), 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
                prism(self.acc, "stone", Mv, poly2, 0.14, np.clip(self.J(scol, 0.06) * 1.08, 0, 1))
            for su in (u0 - 0.15, u1 + 0.15):     # jamb stones
                for k, z in enumerate(np.arange(0, h, 0.34)):
                    ww = 0.17 if k % 2 else 0.24
                    box(self.acc, "stone", Mw @ T(su + (np.sign(su - u) * (ww - 0.15)), -0.03, z + 0.165), ww, 0.07, 0.16,
                        np.clip(self.J(scol, 0.06) * 1.08, 0, 1), 0.02)
            top_of = lambda x: h + (math.sqrt(max(0, r * r - (x - u) ** 2)) if arch == "round" else
                                   (w * math.sin(math.acos(min(1, (abs(x - u) + r) / w))) if arch == "lancet" else 0))
        else:
            self.reveals(Mw, u0, u1, 0, h, depth, "stone" if stone else wallmat, scol if stone else wallcol)
            top_of = lambda x: h
            if stone:
                box(self.acc, "stone", Mw @ T(u, -0.04, h + 0.16), w / 2 + 0.2, 0.1, 0.16, np.clip(self.J(scol, 0.05) * 1.08, 0, 1), 0.02)
            else:
                Mf = Mw @ T(0, 0, 0)
                box(self.acc, "wood", Mf @ T(u0 - 0.04, 0.0, h / 2), 0.05, 0.07, h / 2, dcol * 0.9, 0.01)
                box(self.acc, "wood", Mf @ T(u1 + 0.04, 0.0, h / 2), 0.05, 0.07, h / 2, dcol * 0.9, 0.01)
                box(self.acc, "wood", Mf @ T(u, 0.0, h + 0.05), w / 2 + 0.09, 0.07, 0.05, dcol * 0.9, 0.01)
        # the leaf: vertical planks, strap hinges, nails, ring handle
        dy = depth * 0.6
        if o.get("double"):
            leaves = [(u0, u, -1), (u, u1, 1)]
        else:
            leaves = [(u0, u1, -1)]
        for a, b, hs in leaves:
            if o.get("open"):     # swung inward about its hinge
                hx_ = a if hs < 0 else b
                ang = math.radians(o["open"]) * (1 if hs < 0 else -1)
                Mw = Mw_frame @ T(hx_, depth - 0.04, 0) @ R("z", ang) @ T(-hx_, -(depth - 0.04), 0)
                dy = depth - 0.07
            n = max(3, int(round((b - a) / 0.15)))
            pw = (b - a) / n
            for k in range(n):
                x = a + pw * (k + 0.5)
                ht = min(top_of(x - pw * 0.3), top_of(x + pw * 0.3)) - 0.005
                box(self.acc, "wood", Mw @ T(x, dy, ht / 2), pw / 2 - 0.005, 0.03, ht / 2, self.J(dcol, 0.07), 0.008)
            hx = a if hs < 0 else b
            for z in (0.35, h - 0.4) if h > 1.4 else (h * 0.5,):
                L = (b - a) * 0.8
                cx = hx + (L / 2 if hs < 0 else -L / 2)
                box(self.acc, "iron", Mw @ T(cx, dy - 0.035, z), L / 2, 0.006, 0.026, IRON, 0.002)
                for t in np.linspace(-L / 2 + 0.05, L / 2 - 0.05, 5):
                    box(self.acc, "iron", Mw @ T(cx + t, dy - 0.043, z), 0.011, 0.006, 0.011, IRON, 0.003)
                tube(self.acc, "iron", Mw, [(hx + (0.03 if hs < 0 else -0.03), dy - 0.035, z - 0.05), (hx + (0.03 if hs < 0 else -0.03), dy - 0.035, z + 0.05)], 0.018, 6, IRON)
            # studs
            for zz in np.arange(0.25, h - 0.2, 0.34):
                for xx in np.linspace(a + 0.08, b - 0.08, max(2, int((b - a) / 0.22))):
                    box(self.acc, "iron", Mw @ T(xx, dy - 0.034, zz), 0.009, 0.005, 0.009, IRON, 0.003)
        if not o.get("open"):
            hu = u1 - 0.14 if not o.get("double") else u + 0.09
            a = np.linspace(0, 2 * math.pi, 13)
            tube(self.acc, "iron", Mw, [(hu + 0.06 * math.sin(t), dy - 0.05, 1.0 - 0.06 * math.cos(t)) for t in a], 0.008, 5, IRON, cap=False)
            box(self.acc, "iron", Mw @ T(hu, dy - 0.035, 1.05), 0.03, 0.006, 0.045, IRON, 0.004)
        Mw = Mw_frame
        # threshold (right through an open doorway) and step
        box(self.acc, "stone", Mw @ T(u, -0.12 + (depth * 0.5 if o.get("open") else 0), -0.06), w / 2 + 0.18, 0.2 + depth * (0.55 if o.get("open") else 0.4), 0.08,
            self.J(STONE_GREY, 0.05), 0.02)
        if o.get("step", True) and not o.get("open"):
            box(self.acc, "stone", Mw @ T(u, -0.42, -0.14), w / 2 + 0.3, 0.2, 0.09, self.J(STONE_GREY, 0.06), 0.025)

    # ------------------------------------------------------------ jetty
    def jetty(self, Mw, L, j, z, wcol, brackets):
        """The floor above overhangs by j: joist ends, soffit boards, a bressumer and curved brackets."""
        for u in np.arange(0.2, L - 0.1, 0.42):
            box(self.acc, "wood", Mw @ T(u, -j / 2 + 0.02, z - 0.08), 0.065, j / 2 + 0.06, 0.08, self.J(wcol, 0.08), 0.012)
        quad(self.acc, "wood", Mw, [(0, 0.0, z - 0.16), (L, 0.0, z - 0.16), (L, -j, z - 0.16), (0, -j, z - 0.16)], np.asarray(wcol) * 0.8)
        box(self.acc, "wood", Mw @ T(L / 2, -j - 0.02, z - 0.11), L / 2 + 0.05, 0.07, 0.12, self.J(wcol, 0.04), 0.018)
        box(self.acc, "wood", Mw @ T(L / 2, -j - 0.085, z - 0.215), L / 2 + 0.05, 0.02, 0.025, self.J(wcol, 0.04), 0.006)
        for u in brackets:
            pts = [(0.0, -0.95), (0.08, -0.95)]
            for t in np.linspace(math.pi, math.pi / 2, 7):
                pts.append((j + (j - 0.08) * math.cos(t), -0.95 + 0.72 * math.sin(t)))
            pts += [(j, -0.16), (0.0, -0.16)]
            M = Mw @ frame((u, PROUD * -1 + 0.0, z), (0, -1, 0), (1, 0, 0), (0, 0, 1))
            prism(self.acc, "wood", M, pts, 0.13, self.J(wcol, 0.06))

    # ------------------------------------------------------------ props
    def barrel(self, M, r=0.3, h=0.85, col=WOOD_RAW, upright=True):
        prof = [(r * (0.86 + 0.14 * math.sin(math.pi * t)), h * t) for t in np.linspace(0, 1, 9)]
        lathe(self.acc, "wood", M, prof, 18, self.J(col, 0.08), smooth=False, close_top=True, close_bottom=True)
        for t in (0.12, 0.3, 0.7, 0.88):
            rr = r * (0.86 + 0.14 * math.sin(math.pi * t)) + 0.006
            lathe(self.acc, "iron", M, [(rr, h * t - 0.025), (rr + 0.004, h * t), (rr, h * t + 0.025)], 18, IRON, smooth=True)

    def crate(self, M, s=0.5, col=WOOD_RAW):
        c = self.J(col, 0.1)
        box(self.acc, "wood", M @ T(0, 0, s / 2), s / 2 - 0.02, s / 2 - 0.02, s / 2 - 0.02, c * 0.7, 0.0)
        for ax in ("x", "y"):
            for sgn in (-1, 1):
                for z in (0.07, s / 2, s - 0.07):
                    if ax == "x":
                        box(self.acc, "wood", M @ T(0, sgn * (s / 2 - 0.01), z), s / 2, 0.015, 0.055, self.J(c, 0.06), 0.006)
                    else:
                        box(self.acc, "wood", M @ T(sgn * (s / 2 - 0.01), 0, z), 0.015, s / 2, 0.055, self.J(c, 0.06), 0.006)
        box(self.acc, "wood", M @ T(0, 0, s - 0.01), s / 2, s / 2, 0.012, self.J(c, 0.05), 0.004)

    def sack(self, M, r=0.24):
        blob(self.acc, "cloth", M @ T(0, 0, r * 0.8), r, self.J((0.62, 0.55, 0.40), 0.06), seg=9, rings=6, squash=1.1, rng=self.rng, rough=0.08)
        tube(self.acc, "cloth", M, [(0, 0, r * 1.6), (0.02, 0, r * 2.0)], r * 0.18, 6, self.J((0.58, 0.5, 0.36), 0.05))

    def log_pile(self, M, length=1.6, rows=4):
        rng = self.rng
        for row in range(rows):
            n = rows - row + 2
            for k in range(n):
                y = (k - (n - 1) / 2) * 0.2
                z = 0.1 + row * 0.17
                rr = rng.uniform(0.07, 0.1)
                Ml = M @ T(0, y + rng.normal(0, 0.01), z) @ R("y", math.pi / 2)
                lathe(self.acc, "wood", Ml @ T(0, 0, -length / 2 + rng.normal(0, 0.05)), [(rr, 0), (rr, length)], 9, self.J(WOOD_GREY, 0.1), smooth=True)
                lathe(self.acc, "wood", Ml @ T(0, 0, -length / 2), [(0.0, -0.001), (rr * 0.98, 0.0)], 9, self.J(WOOD_RAW, 0.08), smooth=False)
                lathe(self.acc, "wood", Ml @ T(0, 0, length / 2), [(rr * 0.98, 0.0), (0.0, 0.001)], 9, self.J(WOOD_RAW, 0.08), smooth=False)

    def lantern(self, Mw, u, v, out=0.35):
        """Wall lantern on an iron bracket."""
        tube(self.acc, "iron", Mw, [(u, 0.0, v), (u, -out * 0.5, v + 0.05), (u, -out, v)], 0.015, 6, IRON)
        tube(self.acc, "iron", Mw, [(u, 0.0, v - 0.25), (u, -out * 0.6, v - 0.02)], 0.012, 5, IRON)
        M = Mw @ T(u, -out, v - 0.32)
        box(self.acc, "iron", M @ T(0, 0, 0.3), 0.1, 0.1, 0.02, IRON, 0.005)
        lathe(self.acc, "iron", M @ T(0, 0, 0.32), [(0.11, 0), (0.02, 0.09), (0.0, 0.1)], 4, IRON, smooth=False)
        box(self.acc, "iron", M @ T(0, 0, 0.0), 0.09, 0.09, 0.015, IRON, 0.004)
        for sx in (-1, 1):
            for sy in (-1, 1):
                box(self.acc, "iron", M @ T(sx * 0.08, sy * 0.08, 0.15), 0.01, 0.01, 0.15, IRON, 0)
        box(self.acc, "glass", M @ T(0, 0, 0.15), 0.075, 0.075, 0.14, (0.55, 0.45, 0.25), 0)

    def sign(self, Mw, u, v, emblem, board_col=WOOD, paint=(0.72, 0.58, 0.22)):
        """A hanging sign on a scrolled iron bracket, with a carved emblem."""
        L = 0.95
        tube(self.acc, "iron", Mw, [(u, 0.0, v), (u, -L, v)], 0.018, 6, IRON)
        a = np.linspace(0, 1, 12)
        tube(self.acc, "iron", Mw, [(u, -0.05 - 0.55 * t, v - 0.45 + 0.45 * t ** 0.6) for t in a], 0.013, 5, IRON)
        s = np.linspace(0, 2.4 * math.pi, 20)
        tube(self.acc, "iron", Mw, [(u, -0.35 - 0.05 * t * math.cos(t), v - 0.12 + 0.05 * t * math.sin(t) * 0.5) for t in s], 0.007, 4, IRON, cap=False)
        for k, y in enumerate((-0.28, -0.82)):
            tube(self.acc, "iron", Mw, [(u, y, v - 0.01), (u, y, v - 0.12)], 0.006, 4, IRON)
        Mb = Mw @ T(u, -0.55, v - 0.5)
        box(self.acc, "wood", Mb, 0.03, 0.36, 0.34, self.J(board_col, 0.05), 0.012)
        box(self.acc, "wood", Mb @ T(0, 0, 0), 0.036, 0.37, 0.025, np.asarray(board_col) * 0.8, 0.006) if False else None
        for sx in (-1, 1):
            Me = Mb @ T(sx * 0.035, 0, 0) @ R("z", math.pi / 2 if sx > 0 else -math.pi / 2)
            self.emblem(Me, emblem, paint)

    def emblem(self, M, kind, col):
        """Carved, painted shapes for signs (x across the board, z up, relief toward -y)."""
        def p(poly, d=0.02, c=col, y=-0.012):
            prism(self.acc, "wood", M @ frame((0, y, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), poly, d, self.J(c, 0.03))
        if kind == "tankard":
            p([(-0.12, -0.18), (0.1, -0.18), (0.11, 0.14), (-0.13, 0.14)])
            p([(0.1, 0.08), (0.2, 0.07), (0.22, -0.08), (0.11, -0.12), (0.11, -0.06), (0.16, -0.04), (0.15, 0.03), (0.1, 0.03)])
            p([(-0.15, 0.14), (0.14, 0.14), (0.12, 0.22), (0.02, 0.25), (-0.1, 0.22)], c=(0.9, 0.88, 0.82))
        elif kind == "anvil":
            p([(-0.22, 0.08), (0.2, 0.08), (0.26, 0.16), (-0.12, 0.16), (-0.24, 0.12)])
            p([(-0.08, 0.08), (0.08, 0.08), (0.05, -0.08), (-0.05, -0.08)])
            p([(-0.14, -0.08), (0.14, -0.08), (0.16, -0.16), (-0.16, -0.16)])
        elif kind == "pretzel":
            t = np.linspace(0, 2 * math.pi, 40)
            for k in range(len(t) - 1):
                a, b = t[k], t[k + 1]
                fa = (0.14 * math.sin(2 * a) * 1.1, 0.16 * math.sin(a) - 0.02)
                fb = (0.14 * math.sin(2 * b) * 1.1, 0.16 * math.sin(b) - 0.02)
                tube(self.acc, "wood", M, [(fa[0], -0.02, fa[1]), (fb[0], -0.02, fb[1])], 0.028, 6, self.J((0.62, 0.40, 0.18), 0.04), cap=False)
        elif kind == "gull":
            p([(-0.24, 0.06), (-0.1, 0.02), (0.0, -0.04), (0.1, 0.02), (0.24, 0.06), (0.12, 0.0), (0.0, -0.1), (-0.12, 0.0)], c=(0.92, 0.9, 0.86))
            p([(-0.16, -0.12), (0.16, -0.12), (0.1, -0.2), (-0.1, -0.2)], c=(0.25, 0.32, 0.42))
        elif kind == "fish":
            p([(-0.2, 0.0), (-0.08, 0.08), (0.08, 0.06), (0.16, 0.0), (0.24, 0.07), (0.24, -0.07), (0.16, 0.0), (0.08, -0.06), (-0.08, -0.08)], c=(0.62, 0.66, 0.7))

    def bench(self, M, L=1.6):
        c = self.J(WOOD_GREY, 0.08)
        box(self.acc, "wood", M @ T(0, 0, 0.44), L / 2, 0.17, 0.035, c, 0.01)
        for x in (-L / 2 + 0.15, L / 2 - 0.15):
            box(self.acc, "wood", M @ T(x, 0, 0.21), 0.04, 0.15, 0.21, c * 0.9, 0.01)

    def table(self, M, L=2.0):
        c = self.J(WOOD_GREY, 0.08)
        for k in range(4):
            box(self.acc, "wood", M @ T(0, -0.3 + k * 0.2, 0.76), L / 2, 0.095, 0.03, self.J(c, 0.05), 0.008)
        for x in (-L / 2 + 0.2, L / 2 - 0.2):
            for sgn in (-1, 1):
                Mt = M @ T(x, 0, 0.38) @ R("x", sgn * 0.35)
                box(self.acc, "wood", Mt, 0.04, 0.035, 0.42, c * 0.9, 0.008)
        box(self.acc, "wood", M @ T(0, 0, 0.32), L / 2 - 0.2, 0.03, 0.03, c * 0.9, 0.006)
        self.bench(M @ T(0, -0.72, 0), L)
        self.bench(M @ T(0, 0.72, 0), L)
