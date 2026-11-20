"""Add weapon clips to Robin (live_src/hero.glb -> live_src/hero_plus.glb):

  Slash   a diagonal two-step sword cut: wind up over the right shoulder, cut down across the
          body to the left hip, follow through, recover
  Shake   both hands planted on a tree trunk, shoving it back and forth (for shaking fruit loose)

Poses are written as rotations about the model's axes (+X Robin's left, +Y up, +Z forward), the
same convention as tools/build_hero.py, so they do not depend on how each bone is oriented.
Usage: python add_hero_clips.py
"""
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from glb import Glb, qaxis, qinv, qmul, qnorm  # noqa: E402

ROOT = HERE.parent
src = Glb(str(ROOT / "live_src" / "hero.glb"))
J = src.json
nodes = J["nodes"]
joints = J["skins"][0]["joints"]
parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
BONE = {nodes[j]["name"]: j for j in joints}


def node_q(i):
    return np.array(nodes[i].get("rotation", [0, 0, 0, 1.0]), dtype=np.float64)


world_q = {}
for j in joints:
    q, k = node_q(j), parent.get(j)
    while k is not None:
        q = qmul(node_q(k), q)
        k = parent.get(k)
    world_q[j] = qnorm(q)


def local_from_model_delta(j, euler):
    rx, ry, rz = euler
    d = qmul(qmul(qaxis([0, 1, 0], ry), qaxis([1, 0, 0], rx)), qaxis([0, 0, 1], rz))
    w = world_q[j]
    return qnorm(qmul(node_q(j), qmul(qinv(w), qmul(d, w))))


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def keyed(keys):
    end = keys[-1][0]

    def pose(t):
        for (t0, p0), (t1, p1) in zip(keys, keys[1:]):
            if t <= t1:
                k = smooth((t - t0) / (t1 - t0))
                return {b: tuple(np.array(p0.get(b, (0, 0, 0))) * (1 - k) + np.array(p1.get(b, (0, 0, 0))) * k) for b in set(p0) | set(p1)}
        return keys[-1][1]
    return pose, end


READY = {"RightArm": (-35, 0, -20), "RightForeArm": (-35, 0, 0), "Spine02": (4, 0, 0)}
WINDUP = {"RightArm": (-150, 25, -55), "RightForeArm": (-70, 0, 0), "RightHand": (-20, 0, 0), "Spine02": (-6, -28, 0), "Spine01": (-3, -14, 0),
          "LeftArm": (-25, 0, 25), "LeftForeArm": (-40, 0, 0), "Head": (0, 12, 0),
          "RightUpLeg": (-10, 0, 0), "LeftUpLeg": (8, 0, 0)}
CUT = {"RightArm": (-55, 55, 25), "RightForeArm": (-15, 0, 0), "RightHand": (25, 0, 0), "Spine02": (14, 32, 0), "Spine01": (8, 16, 0),
       "LeftArm": (15, 0, 25), "LeftForeArm": (-30, 0, 0), "Head": (6, -14, 0),
       "LeftUpLeg": (-28, 0, 0), "LeftLeg": (30, 0, 0), "RightUpLeg": (18, 0, 0), "RightLeg": (12, 0, 0)}
FOLLOW = {"RightArm": (-20, 70, 40), "RightForeArm": (-25, 0, 0), "RightHand": (20, 0, 0), "Spine02": (16, 40, 0), "Spine01": (8, 18, 0),
          "LeftArm": (20, 0, 20), "Head": (6, -18, 0), "LeftUpLeg": (-25, 0, 0), "LeftLeg": (28, 0, 0), "RightUpLeg": (15, 0, 0)}
slash, slash_end = keyed([(0.0, READY), (0.16, WINDUP), (0.30, CUT), (0.44, FOLLOW), (0.7, READY)])

PUSH_A = {"RightArm": (-80, 0, 10), "LeftArm": (-80, 0, -10), "RightForeArm": (-35, 0, 0), "LeftForeArm": (-35, 0, 0), "Spine02": (14, 0, 0), "Spine01": (6, 0, 0),
          "LeftUpLeg": (-25, 0, 0), "LeftLeg": (20, 0, 0), "RightUpLeg": (15, 0, 0)}
PUSH_B = {"RightArm": (-85, 0, 10), "LeftArm": (-85, 0, -10), "RightForeArm": (-10, 0, 0), "LeftForeArm": (-10, 0, 0), "Spine02": (22, 0, 0), "Spine01": (10, 0, 0),
          "LeftUpLeg": (-32, 0, 0), "LeftLeg": (24, 0, 0), "RightUpLeg": (22, 0, 0)}
shake, shake_end = keyed([(0.0, {}), (0.15, PUSH_A), (0.3, PUSH_B), (0.45, PUSH_A), (0.6, PUSH_B), (0.75, PUSH_A), (0.95, {})])

FPS = 30
J.setdefault("animations", [])


def add_clip(name, pose_fn, end):
    frames = int(round(end * FPS))
    times = np.linspace(0, end, frames + 1).astype(np.float32)
    t_acc = src.add_accessor(times, "SCALAR", minmax=True)
    t2 = src.add_accessor(np.array([0, end], dtype=np.float32), "SCALAR", minmax=True)
    anim = {"name": name, "samplers": [], "channels": []}
    for j in joints:
        rots = []
        for t in times:
            rots.append(local_from_model_delta(j, pose_fn(float(t)).get(nodes[j]["name"], (0, 0, 0))))
        rots = np.array(rots)
        for i in range(1, len(rots)):
            if np.dot(rots[i], rots[i - 1]) < 0:
                rots[i] = -rots[i]
        anim["samplers"].append({"input": t_acc, "output": src.add_accessor(rots.astype(np.float32), "VEC4"), "interpolation": "LINEAR"})
        anim["channels"].append({"sampler": len(anim["samplers"]) - 1, "target": {"node": j, "path": "rotation"}})
        for path, default in (("translation", [0, 0, 0]), ("scale", [1, 1, 1])):
            v = np.array(nodes[j].get(path, default), dtype=np.float32)
            anim["samplers"].append({"input": t2, "output": src.add_accessor(np.stack([v, v]), "VEC3"), "interpolation": "LINEAR"})
            anim["channels"].append({"sampler": len(anim["samplers"]) - 1, "target": {"node": j, "path": path}})
    J["animations"].append(anim)
    print(f"clip {name}: {end:.2f}s, {len(anim['channels'])} channels")


add_clip("Slash", slash, slash_end)
add_clip("Shake", shake, shake_end)
src.save(str(ROOT / "live_src" / "hero_plus.glb"))
print("saved hero_plus.glb", len(src.bin))
