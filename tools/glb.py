"""Minimal GLB reader/writer plus quaternion helpers (numpy only)."""
import json
import struct

import numpy as np

COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
CTYPE = {np.dtype(v): k for k, v in COMPONENT.items()}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class Glb:
    def __init__(self, path=None):
        self.json = {"asset": {"version": "2.0"}, "bufferViews": [], "accessors": []}
        self.bin = bytearray()
        if path is None:
            return
        with open(path, "rb") as f:
            data = f.read()
        magic, _version, length = struct.unpack_from("<III", data, 0)
        assert magic == 0x46546C67, "not a GLB"
        off = 12
        while off < length:
            clen, ctype = struct.unpack_from("<II", data, off)
            chunk = data[off + 8: off + 8 + clen]
            if ctype == 0x4E4F534A:
                self.json = json.loads(chunk.decode("utf-8"))
            elif ctype == 0x004E4942:
                self.bin = bytearray(chunk)
            off += 8 + clen

    def view_bytes(self, view_index):
        bv = self.json["bufferViews"][view_index]
        start = bv.get("byteOffset", 0)
        return bytes(self.bin[start: start + bv["byteLength"]])

    def read(self, accessor_index):
        acc = self.json["accessors"][accessor_index]
        bv = self.json["bufferViews"][acc["bufferView"]]
        dtype = np.dtype(COMPONENT[acc["componentType"]])
        n = NCOMP[acc["type"]]
        start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = bv.get("byteStride", 0)
        count = acc["count"]
        if stride and stride != dtype.itemsize * n:
            out = np.zeros((count, n), dtype=dtype)
            for i in range(count):
                out[i] = np.frombuffer(self.bin, dtype=dtype, count=n, offset=start + i * stride)
            return out
        arr = np.frombuffer(self.bin, dtype=dtype, count=count * n, offset=start)
        return arr.reshape(count, n).copy()

    def append_view(self, raw, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        view = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(raw)}
        if target is not None:
            view["target"] = target
        self.bin.extend(raw)
        self.json["bufferViews"].append(view)
        return len(self.json["bufferViews"]) - 1

    def add_accessor(self, array, kind, minmax=False, target=None, normalized=False):
        array = np.ascontiguousarray(array)
        view = self.append_view(array.tobytes(), target)
        acc = {"bufferView": view, "componentType": CTYPE[array.dtype], "count": int(array.shape[0]), "type": kind}
        if normalized:
            acc["normalized"] = True
        if minmax:
            flat = array.reshape(array.shape[0], -1)
            acc["min"] = [float(v) for v in flat.min(axis=0)]
            acc["max"] = [float(v) for v in flat.max(axis=0)]
        self.json["accessors"].append(acc)
        return len(self.json["accessors"]) - 1

    def save(self, path):
        while len(self.bin) % 4:
            self.bin.append(0)
        self.json["buffers"] = [{"byteLength": len(self.bin)}]
        js = json.dumps(self.json, separators=(",", ":")).encode("utf-8")
        js += b" " * ((4 - len(js) % 4) % 4)
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, total))
            f.write(struct.pack("<II", len(js), 0x4E4F534A))
            f.write(js)
            f.write(struct.pack("<II", len(self.bin), 0x004E4942))
            f.write(bytes(self.bin))


# Quaternions are stored glTF-style: [x, y, z, w].
def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def qinv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def qaxis(axis, degrees):
    half = np.radians(degrees) / 2.0
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.array([*(axis * np.sin(half)), np.cos(half)])


def qnorm(q):
    return q / np.linalg.norm(q)


def qfrommat(m):
    """Rotation matrix (3x3, orthonormal) -> quaternion [x, y, z, w]."""
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    return qnorm(np.array(q))
