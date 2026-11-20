"""Diagnostic: texture colour of arm-bound vertices, split by distance from the arm axis."""
import io
import sys

import numpy as np
from PIL import Image

from glb import Glb

g = Glb(sys.argv[1])
J = g.json
prim = J["meshes"][0]["primitives"][0]
pos = g.read(prim["attributes"]["POSITION"]).astype(np.float64)
uv = g.read(prim["attributes"]["TEXCOORD_0"]).astype(np.float64)
jnt = g.read(prim["attributes"]["JOINTS_0"]).astype(np.int64)
wgt = g.read(prim["attributes"]["WEIGHTS_0"]).astype(np.float64)
names = [J["nodes"][j]["name"] for j in J["skins"][0]["joints"]]
ibm = g.read(J["skins"][0]["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1).astype(np.float64)
jp = {n: np.linalg.inv(ibm[i])[:3, 3] for i, n in enumerate(names)}
img = np.asarray(Image.open(io.BytesIO(g.view_bytes(J["images"][0]["bufferView"]))).convert("RGB")).astype(np.float64)
h, w, _ = img.shape
px = np.clip((uv[:, 0] % 1.0) * w, 0, w - 1).astype(int)
py = np.clip((uv[:, 1] % 1.0) * h, 0, h - 1).astype(int)
rgb = img[py, px]


def seg(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


for side in ("Left", "Right"):
    sel_w = np.zeros(len(pos))
    for b in ("Arm", "ForeArm", "Hand"):
        k = names.index(side + b)
        sel_w += (wgt * (jnt == k)).sum(axis=1)
    s, e, wr = jp[side + "Arm"], jp[side + "ForeArm"], jp[side + "Hand"]
    tip = wr + (wr - e) / np.linalg.norm(wr - e) * 0.12
    d = np.minimum(np.minimum(seg(pos, s, e), seg(pos, e, wr)), seg(pos, wr, tip))
    print(side, "shoulder", s.round(3), "elbow", e.round(3), "wrist", wr.round(3))
    for lo, hi in ((0, 0.03), (0.03, 0.045), (0.045, 0.06), (0.06, 0.085), (0.085, 0.12), (0.12, 0.2), (0.2, 9)):
        m = (sel_w > 0.5) & (d >= lo) & (d < hi)
        if not m.any():
            continue
        c = rgb[m]
        r, gg, b = c[:, 0], c[:, 1], c[:, 2]
        mail = (b >= r - 12)
        skin = (r > 140) & (r - b > 30)
        green = (gg > r + 6) & (gg > b + 6)
        other = ~(mail | skin | green)
        behind = (pos[m][:, 2] < np.interp(pos[m][:, 1], [wr[1], s[1]], [wr[2], s[2]]) - 0.03).mean()
        print(f"  d {lo:.3f}-{hi:.3f}: n={m.sum():5d} mail={mail.mean():.2f} skin={skin.mean():.2f} "
              f"green={green.mean():.2f} other={other.mean():.2f} meanRGB={c.mean(axis=0).round(0)} behind={behind:.2f}")
