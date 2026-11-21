"""Geometry kit for the village buildings: numpy primitives with vertex colours, baked ambient occlusion,
and a compact GLB writer (KHR_mesh_quantization: int16 positions, int8 normals, uint8 colours).

Coordinates are Blender-style while building (x right, y back, z up); the writer converts to glTF (y up).
Vertex colours are the village's convention: sRGB-ish values that the game raises to 2.2.
"""
import base64
import json
import math
import struct

import numpy as np


# ---------------------------------------------------------------- transforms
def T(x=0.0, y=0.0, z=0.0):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def R(axis, a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    if axis == "x":
        m[1:3, 1:3] = [[c, -s], [s, c]]
    elif axis == "y":
        m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    else:
        m[:2, :2] = [[c, -s], [s, c]]
    return m


def S(x, y=None, z=None):
    return np.diag([x, x if y is None else y, x if z is None else z, 1.0])


def frame(origin, xdir, ydir, zdir):
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = xdir, ydir, zdir, origin
    return m


def chain(*ms):
    out = np.eye(4)
    for m in ms:
        out = out @ m
    return out


def apply(M, P, N=None):
    P2 = P @ M[:3, :3].T + M[:3, 3]
    if N is None:
        return P2
    nm = np.linalg.inv(M[:3, :3]).T
    N2 = N @ nm.T
    N2 /= np.linalg.norm(N2, axis=1, keepdims=True) + 1e-12
    return P2, N2


# ---------------------------------------------------------------- the accumulator
class Mesh:
    def __init__(self):
        self.P, self.N, self.C, self.I, self.n = [], [], [], [], 0

    def add(self, P, N, C, I):
        self.P.append(P)
        self.N.append(N)
        self.C.append(C)
        self.I.append(I + self.n)
        self.n += len(P)

    def arrays(self):
        if not self.P:
            return None
        return np.concatenate(self.P), np.concatenate(self.N), np.concatenate(self.C), np.concatenate(self.I)


class Acc:
    """Geometry of one building (or kit piece), per material."""

    def __init__(self, rng=None):
        self.m = {}
        self.rng = rng or np.random.default_rng(1)

    def mesh(self, mat):
        return self.m.setdefault(mat, Mesh())

    def put(self, mat, M, P, N, C, I):
        P, N = apply(M, P, N)
        C = np.broadcast_to(np.asarray(C, float), P.shape).copy() if np.ndim(C) == 1 else C
        self.mesh(mat).add(P, N, C, I)

    def merge(self, other, M=np.eye(4)):
        for mat, mh in other.m.items():
            a = mh.arrays()
            if a is not None:
                self.put(mat, M, a[0], a[1], a[2], a[3])


def jitter(rng, col, amt=0.06):
    c = np.asarray(col, float)
    return np.clip(c * (1 + rng.normal(0, amt)) + rng.normal(0, amt * 0.25, 3) * c, 0, 1)


# ---------------------------------------------------------------- bevelled box (template, normals fixed)
def _cbox_template():
    A, B, polys = [], [], []

    def vert(a, b):
        A.append(a)
        B.append(b)
        return len(A) - 1

    for ax in range(3):
        u, v = [a for a in range(3) if a != ax]
        for s in (-1, 1):
            q = []
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                a, b = np.zeros(3), np.zeros(3)
                a[ax] = s
                a[u], b[u] = su, -su
                a[v], b[v] = sv, -sv
                q.append(vert(a, b))
            polys.append(q)
    for a1, a2 in ((0, 1), (0, 2), (1, 2)):
        a3 = 3 - a1 - a2
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                q = []
                for s3 in (-1, 1):
                    a, b = np.zeros(3), np.zeros(3)
                    a[a1] = s1
                    a[a2], b[a2] = s2, -s2
                    a[a3], b[a3] = s3, -s3
                    q.append(vert(a, b))
                for s3 in (1, -1):
                    a, b = np.zeros(3), np.zeros(3)
                    a[a1], b[a1] = s1, -s1
                    a[a2] = s2
                    a[a3], b[a3] = s3, -s3
                    q.append(vert(a, b))
                polys.append(q)
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                Sg = np.array([sx, sy, sz], float)
                q = []
                for ax in range(3):
                    a, b = Sg.copy(), -Sg.copy()
                    b[ax] = 0
                    q.append(vert(a, b))
                polys.append(q)
    A, B = np.array(A), np.array(B)
    # orient outward, flatten to per-face vertices with fixed normals
    PA, PB, NN, II = [], [], [], []
    test = A * np.array([1.0, 1.3, 1.7]) + B * 0.2
    for q in polys:
        p = test[q]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.dot(n, p.mean(0)) < 0:
            q = q[::-1]
            p = test[q]
            n = -n
        n = n / np.linalg.norm(n)
        base = len(PA)
        for k in q:
            PA.append(A[k])
            PB.append(B[k])
            NN.append(n)
        for k in range(1, len(q) - 1):
            II.append((base, base + k, base + k + 1))
    return np.array(PA), np.array(PB), np.array(NN), np.array(II)


_CB = _cbox_template()
_BOX = (_CB[0][:24], _CB[2][:24], _CB[3][:12])   # the 6 main faces alone: a plain box when b = 0


def box(acc, mat, M, hx, hy, hz, col, b=0.0):
    """A box of half-extents (hx, hy, hz) centred at M's origin; b > 0 chamfers every edge."""
    H = np.array([hx, hy, hz])
    if b <= 0:
        P = _BOX[0] * H
        acc.put(mat, M, P, _BOX[1], col, _BOX[2])
        return
    b = min(b, 0.45 * H.min())
    P = _CB[0] * H + _CB[1] * b
    acc.put(mat, M, P, _CB[2], col, _CB[3])


def _fbox_template():
    """Polygons of a box bevelled only round its front (-y) face; no back face. Vertices as (A, B): p = A*H + B*b."""
    polys = []
    for sx, sz in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        pass
    F = lambda sx, sz: ((sx, -1, sz), (-sx, 0, -sz))                  # front face corner
    polys.append([F(-1, -1), F(1, -1), F(1, 1), F(-1, 1)])
    # chamfers round the front face and the sides behind them
    for ax, (s1, s2) in (("x", (1, 0)), ("x", (-1, 0)), ("z", (0, 1)), ("z", (0, -1))):
        pass
    for sx in (-1, 1):   # +-x chamfer and side
        polys.append([((sx, -1, -1), (-sx, 0, 1)), ((sx, -1, 1), (-sx, 0, -1)), ((sx, -1, 1), (0, 1, -1)), ((sx, -1, -1), (0, 1, 1))])
        polys.append([((sx, -1, -1), (0, 1, 1)), ((sx, -1, 1), (0, 1, -1)), ((sx, 1, 1), (0, 0, 0)), ((sx, 1, -1), (0, 0, 0))])
    for sz in (-1, 1):   # +-z chamfer and side
        polys.append([((-1, -1, sz), (1, 0, -sz)), ((1, -1, sz), (-1, 0, -sz)), ((1, -1, sz), (-1, 1, 0)), ((-1, -1, sz), (1, 1, 0))])
        polys.append([((-1, -1, sz), (1, 1, 0)), ((1, -1, sz), (-1, 1, 0)), ((1, 1, sz), (0, 0, 0)), ((-1, 1, sz), (0, 0, 0))])
    for sx in (-1, 1):   # corner triangles at the front, and the slivers running back along the long edges
        for sz in (-1, 1):
            polys.append([((sx, -1, sz), (-sx, 0, -sz)), ((sx, -1, sz), (0, 1, -sz)), ((sx, -1, sz), (-sx, 1, 0))])
            polys.append([((sx, -1, sz), (0, 1, -sz)), ((sx, 1, sz), (0, 0, 0)), ((sx, -1, sz), (-sx, 1, 0))])
    A, B, I, faces = [], [], [], []
    for q in polys:
        base = len(A)
        for a, b in q:
            A.append(a)
            B.append(b)
        for k in range(1, len(q) - 1):
            I.append((base, base + k, base + k + 1))
        faces.append(len(q))
    A, B, I = np.array(A, float), np.array(B, float), np.array(I)
    # orient outward (convex about the origin)
    P = A * np.array([1.0, 1.3, 1.7]) + B * 0.2
    n = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    c = (P[I[:, 0]] + P[I[:, 1]] + P[I[:, 2]]) / 3
    flip = np.einsum("ij,ij->i", n, c) < 0
    I[flip] = I[flip][:, ::-1]
    return A, B, I


_FB = _fbox_template()


def fbox_geom(hx, hy, hz, b):
    H = np.array([hx, hy, hz])
    b = min(b, 0.45 * min(hx, hz), 0.9 * hy)
    P = _FB[0] * H + _FB[1] * b
    I = _FB[2]
    n = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    N = np.zeros_like(P)
    for k in range(3):          # every polygon is planar: its vertices take its triangles' normal
        N[I[:, k]] = n
    return P, N, I


def fbox(acc, mat, M, hx, hy, hz, col, b=0.02):
    """A box bevelled round its front (-y) face only, open at the back: for stones and timbers set in a wall."""
    P, N, I = fbox_geom(hx, hy, hz, b)
    acc.put(mat, M, P, N, col, I)


def beam(acc, mat, p0, p1, w, d, col, up=(0, 0, 1), b=0.012, extend=0.0):
    """A squared timber from p0 to p1, w wide (across, in the plane of `up`), d deep."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = p1 - p0
    L = np.linalg.norm(ax)
    if L < 1e-6:
        return
    x = ax / L
    up = np.asarray(up, float)
    y = np.cross(up, x)
    if np.linalg.norm(y) < 1e-6:
        y = np.cross((1, 0, 0), x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    M = frame((p0 + p1) / 2, x, y, z)
    box(acc, mat, M, L / 2 + extend, d / 2, w / 2, col, b)


# ---------------------------------------------------------------- polygons, prisms
def _tri_fan(n):
    return np.array([(0, k, k + 1) for k in range(1, n - 1)])


def ear_clip(pts):
    """Triangulate a simple 2D polygon (CCW or CW)."""
    pts = np.asarray(pts, float)
    n = len(pts)
    area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    idx = list(range(n)) if area > 0 else list(range(n))[::-1]
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for i in range(len(idx)):
            a, b, c = idx[i - 1], idx[i], idx[(i + 1) % len(idx)]
            pa, pb, pc = pts[a], pts[b], pts[c]
            if (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]) <= 1e-12:
                continue
            ok = True
            for k in idx:
                if k in (a, b, c):
                    continue
                p = pts[k]
                d1 = (pb[0] - pa[0]) * (p[1] - pa[1]) - (pb[1] - pa[1]) * (p[0] - pa[0])
                d2 = (pc[0] - pb[0]) * (p[1] - pb[1]) - (pc[1] - pb[1]) * (p[0] - pb[0])
                d3 = (pa[0] - pc[0]) * (p[1] - pc[1]) - (pa[1] - pc[1]) * (p[0] - pc[0])
                if d1 >= 0 and d2 >= 0 and d3 >= 0:
                    ok = False
                    break
            if ok:
                tris.append((a, b, c))
                idx.pop(i)
                break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return np.array(tris)


def prism(acc, mat, M, poly, depth, col, cap_front=True, cap_back=True):
    """A 2D polygon (x, z) extruded along y from -depth/2 (front) to +depth/2."""
    poly = np.asarray(poly, float)
    n = len(poly)
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    if area < 0:
        poly = poly[::-1]
    tris = ear_clip(poly)
    P, N, I = [], [], []
    h = depth / 2
    if cap_front:   # faces -y
        base = 0
        P += [(x, -h, z) for x, z in poly]
        N += [(0, -1, 0)] * n
        I += [(base + a, base + c, base + b) for a, b, c in tris]
    if cap_back:
        base = len(P)
        P += [(x, h, z) for x, z in poly]
        N += [(0, 1, 0)] * n
        I += [(base + a, base + b, base + c) for a, b, c in tris]
    for k in range(n):
        (x0, z0), (x1, z1) = poly[k], poly[(k + 1) % n]
        e = np.array([x1 - x0, 0, z1 - z0])
        nn = np.cross(e, (0, 1, 0))
        ln = np.linalg.norm(nn)
        if ln < 1e-9:
            continue
        nn /= ln
        base = len(P)
        P += [(x0, -h, z0), (x1, -h, z1), (x1, h, z1), (x0, h, z0)]
        N += [nn] * 4
        I += [(base, base + 2, base + 1), (base, base + 3, base + 2)]
    P, N, I = np.array(P, float), np.array(N, float), np.array(I)
    # make every side face point outward (the polygon is CCW in x-z seen from -y)
    tri_n = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    flip = np.einsum("ij,ij->i", tri_n, N[I[:, 0]]) < 0
    I[flip] = I[flip][:, ::-1]
    acc.put(mat, M, P, N, col, I)


def quad(acc, mat, M, corners, col):
    """One flat quad (4 points, any winding: oriented by the right-hand rule)."""
    c = np.asarray(corners, float)
    n = np.cross(c[1] - c[0], c[2] - c[0])
    n /= np.linalg.norm(n) + 1e-12
    acc.put(mat, M, c, np.tile(n, (4, 1)), col, np.array([(0, 1, 2), (0, 2, 3)]))


# ---------------------------------------------------------------- curved things
def lathe(acc, mat, M, prof, seg, col, smooth=True, close_top=False, close_bottom=False):
    """Revolve a profile [(r, z)] about z."""
    prof = np.asarray(prof, float)
    a = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    k = len(prof)
    P = np.zeros((k, seg, 3))
    P[..., 0] = prof[:, None, 0] * np.cos(a)[None]
    P[..., 1] = prof[:, None, 0] * np.sin(a)[None]
    P[..., 2] = prof[:, None, 1]
    dr = np.gradient(prof[:, 0])
    dz = np.gradient(prof[:, 1])
    nr, nz = dz, -dr
    ln = np.hypot(nr, nz) + 1e-12
    nr, nz = nr / ln, nz / ln
    N = np.zeros_like(P)
    N[..., 0] = nr[:, None] * np.cos(a)[None]
    N[..., 1] = nr[:, None] * np.sin(a)[None]
    N[..., 2] = nz[:, None]
    I = []
    for i in range(k - 1):
        for j in range(seg):
            a0, a1 = i * seg + j, i * seg + (j + 1) % seg
            b0, b1 = a0 + seg, a1 + seg
            I += [(a0, a1, b1), (a0, b1, b0)]
    P, N, I = P.reshape(-1, 3), N.reshape(-1, 3), np.array(I)
    # outward: flip when the profile runs the other way
    tri_n = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    if np.mean(np.einsum("ij,ij->i", tri_n, N[I[:, 0]])) < 0:
        I = I[:, ::-1]
    if not smooth:
        P, N, I = flatten(P, I)
    acc.put(mat, M, P, N, col, I)
    for close, row, sgn in ((close_top, k - 1, 1), (close_bottom, 0, -1)):
        if close and prof[row, 0] > 1e-6:
            ring = P[row * seg:(row + 1) * seg] if smooth else (np.c_[prof[row, 0] * np.cos(a), prof[row, 0] * np.sin(a), np.full(seg, prof[row, 1])])
            cP = np.vstack([ring, [(0, 0, prof[row, 1])]])
            cN = np.tile((0, 0, sgn), (seg + 1, 1))
            cI = np.array([(seg, j, (j + 1) % seg) if sgn > 0 else (seg, (j + 1) % seg, j) for j in range(seg)])
            acc.put(mat, M, cP, cN, col, cI)


def flatten(P, I):
    P2 = P[I.reshape(-1)]
    I2 = np.arange(len(P2)).reshape(-1, 3)
    n = np.cross(P2[1::3] - P2[0::3], P2[2::3] - P2[0::3])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return P2, np.repeat(n, 3, axis=0), I2


def tube(acc, mat, M, pts, radius, seg, col, cap=True):
    """A round tube along a 3D polyline (parallel-transport frames)."""
    pts = np.asarray(pts, float)
    k = len(pts)
    r = np.broadcast_to(np.asarray(radius, float), (k,))
    tang = np.gradient(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
    ref = np.array([0, 0, 1.0]) if abs(tang[0, 2]) < 0.9 else np.array([1.0, 0, 0])
    nrm = np.cross(tang[0], ref)
    nrm /= np.linalg.norm(nrm)
    frames = []
    for i in range(k):
        if i:
            nrm = nrm - tang[i] * np.dot(nrm, tang[i])
            nrm /= np.linalg.norm(nrm) + 1e-12
        frames.append((nrm.copy(), np.cross(tang[i], nrm)))
    a = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    P, N, I = [], [], []
    for i in range(k):
        n1, n2 = frames[i]
        for t in a:
            d = n1 * math.cos(t) + n2 * math.sin(t)
            P.append(pts[i] + d * r[i])
            N.append(d)
    for i in range(k - 1):
        for j in range(seg):
            a0, a1 = i * seg + j, i * seg + (j + 1) % seg
            I += [(a0, a1 + seg, a1), (a0, a0 + seg, a1 + seg)]
    P, N, I = np.array(P), np.array(N), np.array(I)
    tri_n = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    if np.mean(np.einsum("ij,ij->i", tri_n, N[I[:, 0]])) < 0:
        I = I[:, ::-1]
    acc.put(mat, M, P, N, col, I)
    if cap:
        for i, sgn in ((0, -1), (k - 1, 1)):
            ring = P[i * seg:(i + 1) * seg]
            cP = np.vstack([ring, pts[i]])
            cN = np.tile(tang[i] * sgn, (seg + 1, 1))
            cI = np.array([(seg, j, (j + 1) % seg) for j in range(seg)])
            tn = np.cross(cP[cI[0, 1]] - cP[cI[0, 0]], cP[cI[0, 2]] - cP[cI[0, 0]])
            if np.dot(tn, cN[0]) < 0:
                cI = cI[:, ::-1]
            acc.put(mat, M, cP, cN, col, cI)


def patch(acc, mat, M, w, h, nu, nv, col, disp=None, colfn=None):
    """A subdivided rectangle in the x-z plane (x 0..w, z 0..h) facing -y; disp(u, v) pushes it out (-y)."""
    u = np.linspace(0, w, nu + 1)
    v = np.linspace(0, h, nv + 1)
    U, V = np.meshgrid(u, v)
    D = disp(U, V) if disp is not None else np.zeros_like(U)
    P = np.stack([U, -D, V], -1)
    gz, gx = np.gradient(-D, v, u)   # d(y)/dv, d(y)/du
    N = np.stack([-gx, -np.ones_like(U), -gz], -1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    I = []
    W = nu + 1
    for j in range(nv):
        for i in range(nu):
            a = j * W + i
            I += [(a, a + 1, a + W + 1), (a, a + W + 1, a + W)]
    I = np.array(I)
    P, N = P.reshape(-1, 3), N.reshape(-1, 3)
    tri_n = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    if np.mean(np.einsum("ij,ij->i", tri_n, N[I[:, 0]])) < 0:
        I = I[:, ::-1]
    C = colfn(U.reshape(-1), V.reshape(-1)) if colfn is not None else col
    acc.put(mat, M, P, N, C, I)


def surface(acc, mat, M, f, nu, nv, col, colfn=None, flip=False):
    """A parametric surface: f(U, V) -> (X, Y, Z) over U, V in [0, 1]; smooth normals."""
    u = np.linspace(0, 1, nu + 1)
    v = np.linspace(0, 1, nv + 1)
    U, V = np.meshgrid(u, v)
    X, Y, Z = f(U, V)
    P = np.stack([X, Y, Z], -1)
    du = np.gradient(P, axis=1)
    dv = np.gradient(P, axis=0)
    N = np.cross(du, dv)
    N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12
    if flip:
        N = -N
    W = nu + 1
    I = []
    for j in range(nv):
        for i in range(nu):
            a = j * W + i
            I += [(a, a + 1, a + W + 1), (a, a + W + 1, a + W)]
    I = np.array(I)
    P, N = P.reshape(-1, 3), N.reshape(-1, 3)
    tri_n = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    if np.mean(np.einsum("ij,ij->i", tri_n, N[I[:, 0]])) < 0:
        I = I[:, ::-1]
    C = colfn(U.reshape(-1), V.reshape(-1)) if colfn is not None else col
    acc.put(mat, M, P, N, C, I)


def blob(acc, mat, M, r, col, seg=6, rings=4, squash=1.0, rng=None, rough=0.15):
    """A lumpy low sphere (flowers, sacks, loaves, stones)."""
    rng = rng or np.random.default_rng(0)
    prof = []
    for i in range(rings + 1):
        t = math.pi * i / rings
        prof.append((max(1e-4, math.sin(t)) * r, -math.cos(t) * r * squash))
    tmp = Acc()
    lathe(tmp, mat, np.eye(4), prof, seg, col, smooth=True)
    P, N, C, I = tmp.m[mat].arrays()
    P = P * (1 + rng.normal(0, rough, (len(P), 1)) * 0.5)
    acc.put(mat, M, P, N, col, I)


# ---------------------------------------------------------------- ambient occlusion baked into vertex colours
def bake_ao(acc, voxel=0.12, ground=None, strength=0.75, extra=None, grid=None):
    """Hemisphere occlusion against a voxelised copy of everything in `acc` (plus extra points and the ground)."""
    grid = grid or occupancy(acc, extra, voxel)
    for mh in acc.m.values():
        arr = mh.arrays()
        if arr is None:
            continue
        Pm, Nm, Cm, Im = arr
        ao = ao_at(grid, Pm, Nm, ground=ground)
        f = (1 - strength) + strength * ao
        mh.P, mh.N, mh.C, mh.I = [Pm], [Nm], [Cm * f[:, None]], [Im]
        mh.n = len(Pm)
    return grid


def _bake_ao_old(acc, voxel=0.12, dirs=14, reach=(0.14, 0.28, 0.5, 0.8, 1.25, 1.9), ground=None, strength=0.75, seed=3):
    """(superseded by bake_ao)"""
    rng = np.random.default_rng(seed)
    allP, allI = [], []
    off = 0
    for mh in acc.m.values():
        a = mh.arrays()
        if a is None:
            continue
        allP.append(a[0])
        allI.append(a[3] + off)
        off += len(a[0])
    if not allP:
        return
    P = np.concatenate(allP)
    I = np.concatenate(allI)
    lo = P.min(0) - 2.0
    hi = P.max(0) + 2.0
    dims = np.ceil((hi - lo) / voxel).astype(int) + 1
    occ = np.zeros(dims, bool)
    # rasterise triangles by sampling their area
    a, b, c = P[I[:, 0]], P[I[:, 1]], P[I[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n_s = np.clip((area / (voxel * voxel) * 3).astype(int), 1, 400)
    tri_idx = np.repeat(np.arange(len(I)), n_s)
    r1 = np.sqrt(rng.random(len(tri_idx)))
    r2 = rng.random(len(tri_idx))
    pts = a[tri_idx] * (1 - r1)[:, None] + b[tri_idx] * (r1 * (1 - r2))[:, None] + c[tri_idx] * (r1 * r2)[:, None]
    vi = np.clip(((pts - lo) / voxel).astype(int), 0, dims - 1)
    occ[vi[:, 0], vi[:, 1], vi[:, 2]] = True
    # directions: cosine-ish hemisphere set
    g = rng.normal(size=(dirs * 4, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    for mh in acc.m.values():
        arr = mh.arrays()
        if arr is None:
            continue
        Pm, Nm, Cm, Im = arr
        ao = np.zeros(len(Pm))
        for s in range(0, len(Pm), 15000):
            p, n = Pm[s:s + 15000], Nm[s:s + 15000]
            hits = np.zeros(len(p))
            wsum = 0.0
            for d in g[:dirs * 2]:
                cosv = n @ d
                dd = np.where(cosv[:, None] < 0, d - 2 * cosv[:, None] * n, d)   # mirror into the hemisphere
                cw = np.abs(cosv) * 0.7 + 0.3
                blocked = np.zeros(len(p), bool)
                for rr in reach:
                    q = p + n * (voxel * 0.9) + dd * rr
                    qi = ((q - lo) / voxel).astype(int)
                    inb = np.all((qi >= 0) & (qi < dims), axis=1)
                    qi = np.clip(qi, 0, dims - 1)
                    hit = occ[qi[:, 0], qi[:, 1], qi[:, 2]] & inb
                    if ground is not None:
                        hit |= q[:, 2] < ground
                    blocked |= hit
                hits += blocked * cw
                wsum += 1
            ao[s:s + 15000] = 1 - hits / (wsum * 0.85)
        ao = np.clip(ao, 0, 1)
        f = (1 - strength) + strength * ao
        new = Cm * f[:, None]
        # write back as one block
        mh.P, mh.N, mh.C, mh.I = [Pm], [Nm], [new], [Im]
        mh.n = len(Pm)


def occupancy(acc, extra=None, voxel=0.12, seed=3):
    """Voxel grid of everything in `acc` (and extra points)."""
    rng = np.random.default_rng(seed)
    allP, allI, off = [], [], 0
    for mh in acc.m.values():
        a = mh.arrays()
        if a is None:
            continue
        allP.append(a[0])
        allI.append(a[3] + off)
        off += len(a[0])
    P = np.concatenate(allP)
    I = np.concatenate(allI)
    pts_all = [P]
    if extra is not None and len(extra):
        pts_all.append(extra)
    allpts = np.concatenate(pts_all)
    lo, hi = allpts.min(0) - 2.0, allpts.max(0) + 2.0
    dims = np.ceil((hi - lo) / voxel).astype(int) + 1
    occ = np.zeros(dims, bool)
    a, b, c = P[I[:, 0]], P[I[:, 1]], P[I[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n_s = np.clip((area / (voxel * voxel) * 3).astype(int), 1, 400)
    ti = np.repeat(np.arange(len(I)), n_s)
    r1, r2 = np.sqrt(rng.random(len(ti))), rng.random(len(ti))
    pts = a[ti] * (1 - r1)[:, None] + b[ti] * (r1 * (1 - r2))[:, None] + c[ti] * (r1 * r2)[:, None]
    if extra is not None and len(extra):
        pts = np.concatenate([pts, extra])
    vi = np.clip(((pts - lo) / voxel).astype(int), 0, dims - 1)
    occ[vi[:, 0], vi[:, 1], vi[:, 2]] = True
    return occ, lo, dims, voxel


def ao_at(grid, P, N, dirs=14, reach=(0.14, 0.28, 0.5, 0.8, 1.25, 1.9), ground=None, seed=3):
    occ, lo, dims, voxel = grid
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(dirs * 2, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    ao = np.zeros(len(P))
    for s in range(0, len(P), 15000):
        p, n = P[s:s + 15000], N[s:s + 15000]
        hits = np.zeros(len(p))
        for d in g:
            cosv = n @ d
            dd = np.where(cosv[:, None] < 0, d - 2 * cosv[:, None] * n, d)
            cw = np.abs(cosv) * 0.7 + 0.3
            blocked = np.zeros(len(p), bool)
            for rr in reach:
                q = p + n * (voxel * 0.9) + dd * rr
                qi = ((q - lo) / voxel).astype(int)
                inb = np.all((qi >= 0) & (qi < dims), axis=1)
                qi = np.clip(qi, 0, dims - 1)
                hit = occ[qi[:, 0], qi[:, 1], qi[:, 2]] & inb
                if ground is not None:
                    hit |= q[:, 2] < ground
                blocked |= hit
            hits += blocked * cw
        ao[s:s + 15000] = 1 - hits / (len(g) * 0.85)
    return np.clip(ao, 0, 1)


def grime(acc, mats, z0, height=0.9, amount=0.35, tint=(0.55, 0.5, 0.42)):
    """Darken and dirty the bottom of walls (rain splash, soil)."""
    for mat in mats:
        mh = acc.m.get(mat)
        if not mh:
            continue
        Pm, Nm, Cm, Im = mh.arrays()
        t = np.clip(1 - (Pm[:, 2] - z0) / height, 0, 1) ** 1.6 * amount
        Cm = Cm * (1 - t[:, None]) + Cm * np.array(tint) * t[:, None]
        mh.P, mh.N, mh.C, mh.I = [Pm], [Nm], [Cm], [Im]


# ---------------------------------------------------------------- GLB writer
class GlbOut:
    """Collects quantised meshes as nodes and writes one .glb."""

    def __init__(self):
        self.js = {"asset": {"version": "2.0", "generator": "megalith geo_kit"}, "scene": 0, "scenes": [{"nodes": []}],
                   "nodes": [], "meshes": [], "materials": [], "accessors": [], "bufferViews": [],
                   "extensionsUsed": ["KHR_mesh_quantization"], "extensionsRequired": ["KHR_mesh_quantization"]}
        self.bin = bytearray()
        self.mats = {}

    def _mat(self, name):
        if name not in self.mats:
            self.mats[name] = len(self.js["materials"])
            self.js["materials"].append({"name": name, "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 0.85}})
        return self.mats[name]

    def _view(self, raw, stride=None, target=34962):
        while len(self.bin) % 4:
            self.bin.append(0)
        v = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(raw), "target": target}
        if stride:
            v["byteStride"] = stride
        self.bin.extend(raw)
        self.js["bufferViews"].append(v)
        return len(self.js["bufferViews"]) - 1

    def _acc(self, view, ctype, count, kind, normalized=False, mn=None, mx=None):
        a = {"bufferView": view, "componentType": ctype, "count": int(count), "type": kind}
        if normalized:
            a["normalized"] = True
        if mn is not None:
            a["min"], a["max"] = mn, mx
        self.js["accessors"].append(a)
        return len(self.js["accessors"]) - 1

    def add(self, name, mat, P, N, C, I):
        """P, N in Blender axes (z up); stored glTF y-up, quantised about the mesh's own bounds."""
        Pg = np.stack([P[:, 0], P[:, 2], -P[:, 1]], 1)
        Ng = np.stack([N[:, 0], N[:, 2], -N[:, 1]], 1)
        c = (Pg.min(0) + Pg.max(0)) / 2
        e = max(float(np.abs(Pg - c).max()), 1e-4)
        q = np.round((Pg - c) / e * 32767).astype(np.int16)
        q4 = np.zeros((len(q), 4), np.int16)
        q4[:, :3] = q
        pv = self._view(q4.tobytes(), 8)
        pa = self._acc(pv, 5122, len(q), "VEC3", True, [int(v) for v in q.min(0)], [int(v) for v in q.max(0)])
        n4 = np.zeros((len(Ng), 4), np.int8)
        n4[:, :3] = np.round(np.clip(Ng, -1, 1) * 127).astype(np.int8)
        na = self._acc(self._view(n4.tobytes(), 4), 5120, len(Ng), "VEC3", True)
        c4 = np.full((len(C), 4), 255, np.uint8)
        c4[:, :3] = np.round(np.clip(C, 0, 1) * 255).astype(np.uint8)
        ca = self._acc(self._view(c4.tobytes(), 4), 5121, len(C), "VEC4", True)
        if len(P) < 65536:
            ib, ct = I.astype(np.uint16).tobytes(), 5123
        else:
            ib, ct = I.astype(np.uint32).tobytes(), 5125
        ia = self._acc(self._view(ib, None, 34963), ct, I.size, "SCALAR")
        self.js["meshes"].append({"name": name, "primitives": [{"attributes": {"POSITION": pa, "NORMAL": na, "COLOR_0": ca}, "indices": ia, "material": self._mat(mat)}]})
        self.js["nodes"].append({"name": name, "mesh": len(self.js["meshes"]) - 1, "translation": [float(v) for v in c], "scale": [e, e, e]})
        self.js["scenes"][0]["nodes"].append(len(self.js["nodes"]) - 1)

    def add_acc(self, prefix, acc):
        tris = 0
        for mat, mh in sorted(acc.m.items()):
            a = mh.arrays()
            if a is None:
                continue
            P, N, C, I = a
            # split very large meshes so each stays within 16-bit indices
            if len(P) >= 65536:
                order = np.argsort(P[I[:, 0], 0])
                chunks = np.array_split(order, int(np.ceil(len(P) / 60000)) + 1)
                for k, ch in enumerate(chunks):
                    used, inv = np.unique(I[ch].reshape(-1), return_inverse=True)
                    self.add(f"{prefix}_{mat}_{k}", mat, P[used], N[used], C[used], inv.reshape(-1, 3))
            else:
                self.add(f"{prefix}_{mat}", mat, P, N, C, I)
            tris += len(I)
        return tris

    def save(self, path):
        while len(self.bin) % 4:
            self.bin.append(0)
        self.js["buffers"] = [{"byteLength": len(self.bin)}]
        js = json.dumps(self.js, separators=(",", ":")).encode("utf-8")
        js += b" " * ((4 - len(js) % 4) % 4)
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(self.bin)))
            f.write(struct.pack("<II", len(js), 0x4E4F534A))
            f.write(js)
            f.write(struct.pack("<II", len(self.bin), 0x004E4942))
            f.write(bytes(self.bin))
