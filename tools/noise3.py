"""Vectorised 3-D noise for sculpting meshes with numpy: value noise, fbm and Worley (F1, F2)."""
import numpy as np

_OFFS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=np.int64)


def _hash(ix, iy, iz, seed, salt=0):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ ((seed * 2654435761 + salt * 97531) & 0x7fffffff)
    h = h & 0xffffffff
    h = ((h ^ (h >> 15)) * 0x2c1b3c6d) & 0xffffffff
    h = ((h ^ (h >> 12)) * 0x297a2d39) & 0xffffffff
    h = h ^ (h >> 15)
    return (h & 0xffffff).astype(np.float64) / float(0xffffff)


def value(p, seed=0):
    p = np.asarray(p, dtype=np.float64)
    i = np.floor(p).astype(np.int64)
    f = p - i
    u = f * f * (3 - 2 * f)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (u[:, 0] if dx else 1 - u[:, 0]) * (u[:, 1] if dy else 1 - u[:, 1]) * (u[:, 2] if dz else 1 - u[:, 2])
                out = out + w * _hash(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz, seed)
    return out * 2 - 1


def fbm(p, octaves=5, seed=0, lac=2.03, gain=0.5):
    p = np.asarray(p, dtype=np.float64)
    amp, tot, norm = 1.0, 0.0, 0.0
    for o in range(octaves):
        tot = tot + amp * value(p, seed + o * 17)
        norm += amp
        amp *= gain
        p = p * lac + 3.17
    return tot / norm


def ridged(p, octaves=4, seed=0):
    p = np.asarray(p, dtype=np.float64)
    amp, tot, norm = 1.0, 0.0, 0.0
    for o in range(octaves):
        tot = tot + amp * (1 - np.abs(value(p, seed + o * 31)))
        norm += amp
        amp *= 0.5
        p = p * 2.1 + 1.3
    return tot / norm


def worley(p, seed=0):
    """F1, F2 distances (cell size 1) and the id of the nearest cell."""
    p = np.asarray(p, dtype=np.float64)
    c = np.floor(p).astype(np.int64)
    f1 = np.full(len(p), 9.0)
    f2 = np.full(len(p), 9.0)
    cid = np.zeros(len(p))
    for o in _OFFS:
        cc = c + o
        jit = np.stack([_hash(cc[:, 0], cc[:, 1], cc[:, 2], seed, s) for s in (1, 2, 3)], axis=1)
        d = np.linalg.norm(cc + jit - p, axis=1)
        closer = d < f1
        f2 = np.where(closer, f1, np.minimum(f2, d))
        cid = np.where(closer, _hash(cc[:, 0], cc[:, 1], cc[:, 2], seed, 9), cid)
        f1 = np.where(closer, d, f1)
    return f1, f2, cid


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)
