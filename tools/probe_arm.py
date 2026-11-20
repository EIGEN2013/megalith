"""Diagnostic: are the hands fused to the belt gear, and can surface distance tell them apart?"""
import heapq
import sys

import numpy as np

from glb import Glb

g = Glb(sys.argv[1])
J = g.json
prim = J["meshes"][0]["primitives"][0]
pos = g.read(prim["attributes"]["POSITION"]).astype(np.float64)
jnt = g.read(prim["attributes"]["JOINTS_0"]).astype(np.int64)
wgt = g.read(prim["attributes"]["WEIGHTS_0"]).astype(np.float64)
idx = g.read(prim["indices"]).reshape(-1, 3).astype(np.int64)
names = [J["nodes"][j]["name"] for j in J["skins"][0]["joints"]]

_, pid = np.unique(np.round(pos, 5), axis=0, return_inverse=True)
pid = pid.reshape(-1)
n = pid.max() + 1
ppos = np.zeros((n, 3))
ppos[pid] = pos
dense = np.zeros((n, len(names)))
for k in range(4):
    dense[pid, jnt[:, k]] = np.maximum(dense[pid, jnt[:, k]], wgt[:, k])

adj = [[] for _ in range(n)]
t = pid[idx]
for a, b in ((0, 1), (1, 2), (2, 0)):
    for u, v in zip(t[:, a], t[:, b]):
        if u != v:
            d = float(np.linalg.norm(ppos[u] - ppos[v]))
            adj[u].append((v, d))
            adj[v].append((u, d))

# connected components
comp = -np.ones(n, dtype=int)
c = 0
for s in range(n):
    if comp[s] >= 0:
        continue
    stack = [s]
    comp[s] = c
    while stack:
        u = stack.pop()
        for v, _ in adj[u]:
            if comp[v] < 0:
                comp[v] = c
                stack.append(v)
    c += 1
sizes = np.bincount(comp)
print("components:", c, "largest:", sorted(sizes.tolist(), reverse=True)[:8])

for side in ("Left", "Right"):
    arm = dense[:, names.index(side + "Arm")]
    fore = dense[:, names.index(side + "ForeArm")]
    hand = dense[:, names.index(side + "Hand")]
    seeds = np.where(arm > 0.8)[0]
    dist = np.full(n, np.inf)
    heap = [(0.0, int(s)) for s in seeds]
    for s in seeds:
        dist[s] = 0.0
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    sel = np.where(fore + hand > 0.3)[0]
    print(side, "seeds", len(seeds), "forearm/hand verts", len(sel), "components touched", np.unique(comp[sel]).tolist())
    finite = dist[sel][np.isfinite(dist[sel])]
    print("  unreachable:", int((~np.isfinite(dist[sel])).sum()))
    hist, edges = np.histogram(finite, bins=np.arange(0, 1.3, 0.05))
    print("  geodesic histogram (5 cm bins):", hist.tolist())
