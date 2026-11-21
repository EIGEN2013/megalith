"""Roofs, chimneys and dormers.

A roof is built in its own 'canonical' frame: a along the ridge, s across it (s < 0 is the front slope), z up.
`Mr` maps that frame into the building; `Mb` maps the building into the world (Blender axes). Slates and
ridge tiles are written as instances of the village's proto_slate_* / proto_ridge_* pieces in game axes.
"""
import math

import numpy as np

from geo_kit import T, R, frame, box, beam, prism, quad, lathe, tube, patch
from village_kit import WOOD, STONE, MORTAR, TERRA, THATCH, IRON

C_GAME = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])     # Blender axes -> game axes


class Instances:
    def __init__(self):
        self.d = {}

    def add(self, proto, M3, t, rgb):
        """M3: 3x3 (columns = the piece's x, y, z axes, scaled) and t, both in Blender world axes."""
        G = C_GAME @ M3
        g = C_GAME @ t
        row = [G[0, 0], G[0, 1], G[0, 2], g[0], G[1, 0], G[1, 1], G[1, 2], g[1], G[2, 0], G[2, 1], G[2, 2], g[2], rgb[0], rgb[1], rgb[2]]
        self.d.setdefault(proto, []).append(row)


def slate_colour(rng):
    v = math.exp(rng.normal(math.log(0.0145), 0.33))
    t = rng.normal(0, 0.08)
    return (v * (1 - t * 0.3), v * 1.04, v * (1.16 + t * 0.3))


class Roof:
    def __init__(self, kit, inst, Mb):
        self.k, self.inst, self.Mb = kit, inst, Mb
        self.acc, self.rng = kit.acc, kit.rng

    def _put_slate(self, Mfull, p, e, n, L, w, rng, proto=None):
        Mw = self.Mb @ Mfull
        pw = Mw[:3, :3] @ p + Mw[:3, 3]
        ew = Mw[:3, :3] @ e
        nw = Mw[:3, :3] @ n
        zw = np.cross(ew, nw)
        M3 = np.stack([ew * w, nw, zw * L], 1)
        self.inst.add(proto or f"proto_slate_{rng.integers(4)}", M3, pw, slate_colour(rng))

    def gable(self, Mr, Lr, Sp, zt, pitch, ov_e=0.45, ov_g=0.35, kind="slate", wcol=WOOD, excl=(), barge=True, ridge_tiles=True,
              fascia=True, finial=False, thick=0.32):
        """A pitched roof over a Lr x Sp rectangle whose wall plates are at height zt."""
        k, acc, rng = self.k, self.acc, self.rng
        tp = math.tan(pitch)
        sh = Sp / 2
        lift = 0.2
        zr = zt + lift + sh * tp
        info = {"zr": zr, "sides": {}}
        for sgn in (-1, 1):
            U = np.array([0.0, -sgn * math.cos(pitch), math.sin(pitch)])
            n = np.array([0.0, sgn * math.sin(pitch), math.cos(pitch)])
            A = np.array([1.0, 0, 0]) * (1 if sgn < 0 else -1)
            eave = np.array([0.0, sgn * (sh + ov_e), zt + lift - ov_e * tp])
            Ls = (sh + ov_e) / math.cos(pitch)
            a0 = -(Lr / 2 + ov_g)
            info["sides"][sgn] = (eave, U, n, A, Ls)
            if kind == "slate":
                # boarded deck under the slates, rafter feet, fascia
                ctr = eave + U * (Ls / 2) - n * 0.06
                box(acc, "wood", Mr @ frame(ctr, A, U, n), Lr / 2 + ov_g, Ls / 2, 0.05, k.J(wcol, 0.05) * 0.85, 0.01)
                if fascia:
                    for a in np.arange(-Lr / 2 + 0.15, Lr / 2, 0.55):
                        p = eave + U * 0.45 - n * 0.17
                        p = p.copy()
                        p[0] = a
                        box(acc, "wood", Mr @ frame(p, np.array([1.0, 0, 0]) if sgn < 0 else np.array([-1.0, 0, 0]), U, n), 0.045, 0.5, 0.06,
                            k.J(wcol, 0.06), 0.008)
                    box(acc, "wood", Mr @ frame(eave - n * 0.07 + U * 0.02, A, U, n), Lr / 2 + ov_g, 0.025, 0.1, k.J(wcol, 0.04), 0.006)
                self._slates(Mr, eave, U, n, A, Ls, Lr, ov_g, sgn, excl)
            else:
                self._thatch_side(Mr, eave, U, n, A, Ls, Lr, ov_g, sgn, thick)
        # ridge
        if kind == "slate":
            beam(acc, "wood", (-(Lr / 2 + ov_g), 0, zr - 0.08), ((Lr / 2 + ov_g), 0, zr - 0.08), 0.1, 0.2, k.J(wcol, 0.04), up=(0, 0, 1)) if False else None
            if ridge_tiles:
                for a in np.arange(-(Lr / 2 + ov_g) + 0.21, Lr / 2 + ov_g - 0.1, 0.42):
                    p = np.array([a, 0.0, zr + 0.005])
                    Mw = self.Mb @ Mr
                    pw = Mw[:3, :3] @ p + Mw[:3, 3]
                    xw = Mw[:3, :3] @ np.array([1.0, 0, 0])
                    yw = Mw[:3, :3] @ np.array([0, 0, 1.0])
                    zw = np.cross(xw, yw)
                    self.inst.add(f"proto_ridge_{rng.integers(2)}", np.stack([xw * 0.46, yw, zw], 1), pw, slate_colour(rng))
        else:
            self._thatch_ridge(Mr, Lr, ov_g, zr + thick * 0.75, pitch)
        # barge boards along the gables, with a carved finial at the apex
        if barge and kind == "slate":
            for side in (-1, 1):
                a = side * (Lr / 2 + ov_g + 0.025)
                for sgn in (-1, 1):
                    eave, U, n, A, Ls = info["sides"][sgn]
                    p0 = eave + n * 0.02
                    p0 = p0.copy()
                    p0[0] = a
                    p1 = np.array([a, 0.0, zr + 0.06])
                    q0, q1 = Mr[:3, :3] @ p0 + Mr[:3, 3], Mr[:3, :3] @ p1 + Mr[:3, 3]
                    beam(acc, "wood", q0, q1, 0.05, 0.26, k.J(wcol, 0.05), up=Mr[:3, :3] @ np.array([1.0, 0, 0]), b=0.01)
                if finial:
                    pf = Mr @ T(a, 0, zr + 0.1)
                    lathe(acc, "wood", pf, [(0.06, 0), (0.06, 0.25), (0.09, 0.32), (0.05, 0.42), (0.0, 0.5)], 8, k.J(wcol, 0.05), smooth=False, close_bottom=True)
        return info

    def _slates(self, Mr, eave, U, n, A, Ls, Lr, ov_g, sgn, excl):
        rng = self.rng
        gauge = 0.2
        L = 0.42
        tilt = math.radians(2.3)
        Ut = U * math.cos(tilt) - n * math.sin(tilt)
        nt = n * math.cos(tilt) + U * math.sin(tilt)
        width = Lr + 2 * ov_g
        rows = [-0.02] + list(np.arange(0.0, Ls - 0.12, gauge))
        for r_i, b in enumerate(rows):
            x = -width / 2 + (0.0 if r_i % 2 == 0 else -0.14) + rng.uniform(-0.03, 0.0)
            while x < width / 2 - 0.05:
                w = rng.uniform(0.25, 0.31)
                cx = x + w / 2
                x += w + 0.006
                if cx < -width / 2 + 0.08 or cx > width / 2 - 0.08:
                    continue
                aa = cx if sgn < 0 else -cx      # canonical a of this slate
                if any(e[0] <= aa <= e[1] and e[2] <= b <= e[3] for e in excl):
                    continue
                p = eave + A * cx + U * b + n * (0.022 + 0.0025 * (r_i % 3))
                yaw = rng.normal(0, 0.018)
                e = A * math.cos(yaw) + Ut * math.sin(yaw)
                # flip half of them end over end for variety in the stone grain
                self._put_slate(Mr, p, e, nt, L, w * rng.uniform(0.97, 1.03), rng)

    # ------------------------------------------------------------ thatch
    def _thatch_side(self, Mr, eave, U, n, A, Ls, Lr, ov_g, sgn, thick):
        k, acc, rng = self.k, self.acc, self.rng
        width = Lr + 2 * ov_g
        ph = rng.uniform(0, 100, 3)
        o = eave + A * (-width / 2) + n * thick
        M = Mr @ frame(o, A, -n, U)

        course = 0.32           # thatch is laid in courses: each ends in a slight step down-slope

        def grain(Ua, Vb):
            g1 = np.abs(np.sin(Ua * 61 + np.sin(Vb * 1.3 + Ua * 0.7) * 1.5 + ph[0]))
            g2 = np.abs(np.sin(Ua * 157 + Vb * 2.1 + ph[1]))
            return 0.6 * g1 + 0.4 * g2

        def disp(Ua, Vb):
            straw = 0.011 * grain(Ua, Vb)
            step = 0.022 * (1 - np.mod(Vb, course) / course) ** 2
            clump = 0.028 * np.sin(Ua * 2.7 + ph[1]) * np.sin(Vb * 1.9 + ph[2]) + 0.012 * np.sin(Ua * 7.3 + Vb * 4.1)
            edge = -thick * 0.12 * np.clip(1 - Vb / 0.25, 0, 1) ** 2
            return straw + step + clump + edge

        base = np.asarray(THATCH, float)

        def colfn(Ua, Vb):
            t = Vb / Ls
            g = grain(Ua, Vb)
            streak = 0.86 + 0.18 * g + 0.05 * np.sin(Ua * 13 + ph[2])
            stepdark = 1 - 0.12 * (np.mod(Vb, course) / course) ** 3
            c = base[None] * (streak * stepdark * (0.84 + 0.22 * t))[:, None]
            moss = (np.clip(0.4 - t, 0, 1) * (0.5 + 0.5 * np.sin(Ua * 1.7 + ph[0])))[:, None]
            c = c * (1 - moss * 0.5) + c * moss * 0.5 * np.array([0.72, 0.98, 0.62])[None]
            return np.clip(c, 0, 1)

        nu = max(8, int(width / 0.05))
        nv = max(6, int(Ls / 0.09))
        patch(acc, "thatch", M, width, Ls, nu, nv, base, disp, colfn)
        # the cut eave: a face of straw ends, darker and ragged
        Ap = A if sgn < 0 else A
        Me = Mr @ frame(eave + A * (-width / 2) - U * 0.0, A, U, n)
        ends = lambda Ua, Vb: np.clip(base[None] * (0.55 + 0.25 * grain(Ua * 1.0, Vb * 0 + 0.3))[:, None], 0, 1)
        patch(acc, "thatch", Me, width, thick * 0.96, max(8, int(width / 0.05)), 2, base * 0.6,
              lambda Ua, Vb: 0.03 * np.abs(np.sin(Ua * 23 + ph[0])) + 0.015 * np.sin(Ua * 71), ends)
        # underside from the wall out to the eave
        quad(acc, "thatch", Mr, [eave + A * (-width / 2), eave + A * (width / 2), eave + A * (width / 2) + U * (Ls * 0.32),
                                 eave + A * (-width / 2) + U * (Ls * 0.32)][::-1], base * 0.45)
        # the verges along the gables: straw ends again, facing out
        for side in (-1, 1):
            if side > 0:
                Mv = Mr @ frame(eave + A * (width / 2), U, -A, n)
            else:
                Mv = Mr @ frame(eave + A * (-width / 2) + U * Ls, -U, A, n)
            patch(acc, "thatch", Mv, Ls, thick * 0.96, max(6, int(Ls / 0.06)), 2, base * 0.6,
                  lambda Ua, Vb: 0.025 * np.abs(np.sin(Ua * 19 + ph[2])), ends)

    def _thatch_ridge(self, Mr, Lr, ov_g, z, pitch):
        k, acc, rng = self.k, self.acc, self.rng
        width = Lr + 2 * ov_g - 0.2
        col = np.clip(np.asarray(THATCH) * 1.08, 0, 1)
        prof = [(-0.55, -0.42), (0.55, -0.42), (0.42, 0.0), (0.0, 0.14), (-0.42, 0.0)]
        M = Mr @ frame((0, 0, z), (0, 1.0, 0), (1.0, 0, 0), (0, 0, 1.0))
        prism(acc, "thatch", Mr @ frame((0, 0, z), (1, 0, 0), (0, 0, -1), (0, 1, 0)) if False else Mr @ frame((0, 0, z), (0, 1, 0), (-1, 0, 0), (0, 0, 1)),
              [(-0.0, 0)], 0.01, col) if False else None
        # ridge as a sequence of prisms along a (profile in s-z)
        Mp = Mr @ frame((0, 0, z), (0, -1, 0), (1, 0, 0), (0, 0, 1))
        prism(acc, "thatch", Mp, [(s, t) for s, t in prof], width, k.J(col, 0.04))
        # scalloped lower edge, both sides
        for sgn in (-1, 1):
            for a in np.arange(-width / 2 + 0.15, width / 2 - 0.1, 0.3):
                tri = [(0.0, 0.0), (0.3, 0.0), (0.15, -0.16)]
                Ms = Mr @ frame((a - 0.15, sgn * 0.58, z - 0.42 + 0.01), (1, 0, 0), (0, 1, 0), (0, 0, 1)) @ R("x", -sgn * (math.pi / 2 - pitch) * 0.0)
                prism(acc, "thatch", Ms, tri, 0.1, k.J(col, 0.05))
            # liggers: hazel rods along the ridge, and crossed spars between them
            for off in (0.2, 0.42):
                pts = [(a, sgn * off, z - 0.42 + (0.42 - off) * 0.8 + 0.03) for a in np.linspace(-width / 2, width / 2, 10)]
                tube(acc, "wood", Mr, pts, 0.013, 5, k.J((0.30, 0.22, 0.14), 0.08))
            for a in np.arange(-width / 2 + 0.2, width / 2 - 0.2, 0.36):
                p0 = (a, sgn * 0.2, z - 0.42 + 0.22 * 0.8 + 0.03)
                p1 = (a + 0.18, sgn * 0.42, z - 0.42 + 0.03)
                tube(acc, "wood", Mr, [p0, p1], 0.01, 4, k.J((0.30, 0.22, 0.14), 0.08), cap=False)
                tube(acc, "wood", Mr, [(a + 0.18, sgn * 0.2, p0[2]), (a, sgn * 0.42, p1[2])], 0.01, 4, k.J((0.30, 0.22, 0.14), 0.08), cap=False)

    def shed(self, Mr, Lr, Sp, z_high, pitch, ov_e=0.4, ov_g=0.25, wcol=WOOD):
        """A lean-to: one slope from a wall (s = 0, height z_high) down and out to s = Sp."""
        k, acc = self.k, self.acc
        tp = math.tan(pitch)
        U = np.array([0.0, -math.cos(pitch), math.sin(pitch)])
        n = np.array([0.0, math.sin(pitch), math.cos(pitch)])
        A = np.array([-1.0, 0, 0])
        eave = np.array([0.0, Sp + ov_e, z_high - (Sp + ov_e) * tp])
        Ls = (Sp + ov_e) / math.cos(pitch) - 0.05
        ctr = eave + U * (Ls / 2) - n * 0.06
        box(acc, "wood", Mr @ frame(ctr, A, U, n), Lr / 2 + ov_g, Ls / 2, 0.05, k.J(wcol, 0.05) * 0.85, 0.01)
        for a in np.arange(-Lr / 2 + 0.15, Lr / 2, 0.6):
            p = eave + U * (Ls / 2) - n * 0.17
            p = p.copy()
            p[0] = a
            box(acc, "wood", Mr @ frame(p, np.array([-1.0, 0, 0]), U, n), 0.05, Ls / 2, 0.07, k.J(wcol, 0.06), 0.008)
        box(acc, "wood", Mr @ frame(eave - n * 0.07 + U * 0.02, A, U, n), Lr / 2 + ov_g, 0.025, 0.1, k.J(wcol, 0.04), 0.006)
        box(acc, "wood", Mr @ T(0, 0.04, z_high + 0.08), Lr / 2 + ov_g, 0.05, 0.1, k.J(wcol, 0.04), 0.008)
        self._slates(Mr, eave, U, n, A, Ls, Lr, ov_g, 1, ())
        return eave

    # ------------------------------------------------------------ chimney
    def chimney(self, x, y, z0, ztop, w=0.78, d=0.62, col=STONE, pots=2):
        k, acc, rng = self.k, self.acc, self.rng
        x0, x1, y0, y1 = x - w / 2, x + w / 2, y - d / 2, y + d / 2
        courses = k.courses(ztop - z0, coursed=True)
        for i, side in enumerate(("front", "right", "back", "left")):
            Mw, L = k.wall_frame(side, x0, x1, y0, y1, z0)
            k.stone_wall(Mw, L, ztop - z0, [], col=col, coursed=True, quoins=True, courses=courses, parity=i % 2, proud=0.02)
        box(acc, "stone", T(x, y, ztop + 0.07), w / 2 + 0.08, d / 2 + 0.08, 0.07, k.J(col, 0.04) * 1.05, 0.02)
        box(acc, "stone", T(x, y, ztop + 0.17), w / 2 - 0.02, d / 2 - 0.02, 0.05, k.J(col, 0.04) * 0.95, 0.015)
        for i in range(pots):
            px = x + (i - (pots - 1) / 2) * 0.32
            prof = [(0.11, 0), (0.12, 0.05), (0.1, 0.12), (0.085, 0.36), (0.1, 0.4), (0.1, 0.44), (0.07, 0.44)]
            lathe(acc, "stone", T(px, y, ztop + 0.22), prof, 12, k.J(TERRA, 0.08), smooth=True)
            lathe(acc, "stone", T(px, y, ztop + 0.22), [(0.068, 0.44), (0.068, 0.3)], 12, (0.03, 0.025, 0.02), smooth=True)

    # ------------------------------------------------------------ dormer
    def dormer(self, Mr, info, a_c, b_d, wd, hd, pitch_main, pcol, wcol, shutters=None):
        """A small gabled dormer standing on the front slope (sgn -1) of a slate roof."""
        k, acc = self.k, self.acc
        eave, U, n, A, Ls = info["sides"][-1]
        tp = math.tan(pitch_main)
        p_front = eave + U * b_d
        s_f, z_f = p_front[1], p_front[2] - 0.02
        Mw = Mr @ frame((a_c - wd / 2, s_f, z_f), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        k.timber_wall(Mw, wd, hd, [{"kind": "win", "u": wd / 2, "w": wd * 0.52, "h": hd * 0.58, "sill": hd * 0.2, "shutters": shutters}],
                      pcol, wcol, style="tall")
        zr_main = info["zr"]
        z_top = z_f + hd
        s_back = -(zr_main - z_top) / tp
        # cheeks
        for side in (-1, 1):
            a = a_c + side * wd / 2
            tri = [(s_f, z_f), (s_f, z_top), (s_back, z_top)]
            Mc = Mr @ frame((a, 0, 0), (0, 1, 0), (1, 0, 0) if side > 0 else (-1, 0, 0), (0, 0, 1)) if False else None
            pts3 = [np.array([a, s, z]) for s, z in tri]
            P = np.array(pts3)
            quad(acc, "plaster", Mr, [P[0], P[1], P[2], P[2]] if side < 0 else [P[0], P[2], P[1], P[1]], np.asarray(pcol) * 0.92)
        # its own roof, ridge running back into the main roof
        pd = math.radians(48)
        span = wd + 0.16
        length = (s_back - s_f) * -1 + 0.3
        Md = Mr @ frame((a_c, (s_f - 0.3 + s_back) / 2, 0), (0, 1, 0), (-1, 0, 0), (0, 0, 1))
        self.gable(Md, abs(s_back - s_f) + 0.3, span, z_top, pd, ov_e=0.18, ov_g=0.0, kind="slate", wcol=wcol, fascia=False, finial=False,
                   barge=True)
        # the gable triangle above the dormer's window
        zr_d = z_top + 0.2 + (span / 2) * math.tan(pd)
        tri = [(a_c - wd / 2, s_f - 0.01, z_top), (a_c + wd / 2, s_f - 0.01, z_top), (a_c, s_f - 0.01, zr_d - 0.15)]
        quad(acc, "plaster", Mr, [tri[0], tri[1], tri[2], tri[2]], np.asarray(pcol) * 0.95)
        k.timber(Mr @ frame((0, s_f - 0.01, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), (a_c, z_top), (a_c, zr_d - 0.2), 0.12, wcol)
        b_back = (s_back - eave[1]) / math.cos(pitch_main)
        return (a_c - wd / 2 - 0.2, a_c + wd / 2 + 0.2, b_d - 0.25, b_back + 0.1)
