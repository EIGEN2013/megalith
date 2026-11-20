"""Build hero.glb from the Higgsfield (Meshy) rigged character.

Input : the rigged GLB with its one library clip (Punch_Combo).
Output: a compact GLB with welded mesh, rest-pose skeleton and these clips:
        Idle, Walk, Run, Jump, Fall, DoubleJump (authored here) + Attack (library clip).

Usage: python build_hero.py <rigged.glb> <out.glb>

Authored poses are written as rotations around the MODEL axes (degrees), so they
do not depend on how the rig's bones happen to be oriented:
    +X = character's left, +Y = up, +Z = forward
    rx > 0 pitches forward/down-and-back (leg swings back, torso leans forward)
    ry > 0 turns to the character's left
    rz > 0 raises the left side (left arm goes up; right arm goes down)
"""
import io
import math
import sys

import numpy as np
from PIL import Image

from glb import Glb, qaxis, qfrommat, qinv, qmul, qnorm

FPS = 30
src_path, dst_path = sys.argv[1], sys.argv[2]
src = Glb(src_path)
J = src.json
nodes = J["nodes"]
name_to_node = {n.get("name"): i for i, n in enumerate(nodes)}
skin = J["skins"][0]
joints = skin["joints"]
parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}

# ---------------------------------------------------------------- rest pose
ibm = src.read(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1).astype(np.float64)
G = {j: np.linalg.inv(ibm[i]) for i, j in enumerate(joints)}  # joint -> model space
armature = name_to_node["Armature"]
arm_scale = nodes[armature].get("scale", [1, 1, 1])
A = np.diag([*arm_scale, 1.0])

rest_t, rest_q, world_q = {}, {}, {}
for j in joints:
    p = parent[j]
    local = np.linalg.inv(G[p] if p in G else A) @ G[j]
    r = local[:3, :3] / np.linalg.norm(local[:3, :3], axis=0)
    rest_t[j] = local[:3, 3].copy()
    rest_q[j] = qfrommat(r)
    gr = G[j][:3, :3] / np.linalg.norm(G[j][:3, :3], axis=0)
    world_q[j] = qfrommat(gr)

# ---------------------------------------------------------------- output scaffold
out = Glb()
O = out.json
O["asset"]["generator"] = "build_hero.py (Higgsfield mesh + rig)"
O["scene"] = 0
O["scenes"] = [{"name": "Scene", "nodes": J["scenes"][J.get("scene", 0)]["nodes"]}]
O["nodes"] = []
for i, n in enumerate(nodes):
    m = {k: v for k, v in n.items() if k in ("name", "children", "mesh", "skin", "scale")}
    if i in rest_q:
        m.pop("scale", None)
        m["translation"] = [float(v) for v in rest_t[i]]
        m["rotation"] = [float(v) for v in rest_q[i]]
    O["nodes"].append(m)

# ---------------------------------------------------------------- mesh: weld + smooth normals
prim = J["meshes"][0]["primitives"][0]
attrs = prim["attributes"]
pos = src.read(attrs["POSITION"]).astype(np.float32)
uv = src.read(attrs["TEXCOORD_0"]).astype(np.float32)
jnt = src.read(attrs["JOINTS_0"])
wgt = src.read(attrs["WEIGHTS_0"]).astype(np.float32)
idx = src.read(prim["indices"]).reshape(-1).astype(np.int64)

_, pid = np.unique(np.round(pos, 5), axis=0, return_inverse=True)
pid = pid.reshape(-1)
tri = idx.reshape(-1, 3)
face_n = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])  # area weighted
acc = np.zeros((pid.max() + 1, 3))
for k in range(3):
    np.add.at(acc, pid[tri[:, k]], face_n)
acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
nrm = acc[pid].astype(np.float32)

key = np.concatenate([pid[:, None].astype(np.float64), np.round(uv, 5)], axis=1)
_, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
inverse = inverse.reshape(-1)
new_idx = inverse[idx]
pos, uv, nrm, jnt, wgt = pos[first], uv[first], nrm[first], jnt[first], wgt[first]
wgt = wgt / np.maximum(wgt.sum(axis=1, keepdims=True), 1e-8)
print(f"mesh: {len(pid)} -> {len(pos)} verts, {len(tri)} tris")

# The auto-rig binds belt gear (pouch, sheath) that rests next to the hands to the forearm/hand
# bones, so it swings with the arms. Hand those weights to the hips for anything that is not
# actually within arm's thickness of the forearm/hand.
joint_slot = {nodes[j]["name"]: s for s, j in enumerate(joints)}
joint_pos = {nodes[j]["name"]: G[j][:3, 3] for j in joints}
dense = np.zeros((len(pos), len(joints)))
np.add.at(dense, (np.repeat(np.arange(len(pos)), 4), jnt.reshape(-1).astype(np.int64)), wgt.reshape(-1))


def segment_distance(p, a, b):
    """Distance to segment ab, and how squarely the vertex normal faces away from it (-1..1)."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0)
    radial = p - (a + t[:, None] * ab)
    dist = np.linalg.norm(radial, axis=1)
    facing = np.einsum("ij,ij->i", radial / np.maximum(dist, 1e-6)[:, None], nrm.astype(np.float64))
    return dist, facing


def nearest(*pairs):
    dist = np.stack([p[0] for p in pairs])
    facing = np.stack([p[1] for p in pairs])
    pick = dist.argmin(axis=0)
    cols = np.arange(dist.shape[1])
    return dist[pick, cols], facing[pick, cols]


def ramp(x, lo, hi):
    f = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return f * f * (3 - 2 * f)


# The whole character is one fused shell, so where the cape and the belt gear touch the arms
# they are welded to them and bound to the arm bones. Find the real arm surface (close to the
# arm axis, facing away from it, connected to the elbow), release everything else to the trunk,
# and cut the few triangles that bridge the two so nothing stretches between arm and body.
pid_w = pid[first]                       # welded vertex -> position id
n_pid = int(pid.max()) + 1
_tri = pid_w[new_idx.reshape(-1, 3)]
_edges = np.concatenate([_tri[:, [0, 1]], _tri[:, [1, 2]], _tri[:, [2, 0]]])
_edges = _edges[_edges[:, 0] != _edges[:, 1]]
_edges = np.unique(np.concatenate([_edges, _edges[:, ::-1]]), axis=0)
_degree = np.maximum(np.bincount(_edges[:, 0], minlength=n_pid), 1)
_starts = np.concatenate([[0], np.cumsum(np.bincount(_edges[:, 0], minlength=n_pid))])


def neighbour_fraction(mask):
    total = np.zeros(n_pid)
    np.add.at(total, _edges[:, 0], mask[_edges[:, 1]].astype(float))
    return total / _degree


def per_position(values):
    out_v = np.zeros(n_pid, dtype=values.dtype)
    out_v[pid_w] = values
    return out_v


trunk = ["Hips", "Spine02", "Spine01", "Spine"]
trunk_y = np.array([joint_pos[b][1] for b in trunk])
nearest_trunk = np.array([joint_slot[b] for b in trunk])[
    np.abs(pos[:, 1:2].astype(np.float64) - trunk_y[None, :]).argmin(axis=1)]
rows = np.arange(len(pos))
P = pos.astype(np.float64)
tri_w = new_idx.reshape(-1, 3)
cut = np.zeros(len(tri_w), dtype=bool)
for side in ("Left", "Right"):
    shoulder, elbow, wrist = (joint_pos[side + b] for b in ("Arm", "ForeArm", "Hand"))
    tip = wrist + (wrist - elbow) / np.linalg.norm(wrist - elbow) * 0.12
    d, facing = nearest(segment_distance(P, shoulder, elbow), segment_distance(P, elbow, wrist),
                        segment_distance(P, wrist, tip))
    arm_slots = [joint_slot[side + b] for b in ("Arm", "ForeArm", "Hand")]
    bound = per_position(dense[:, arm_slots].sum(axis=1) > 0.15)
    off_arm = np.maximum(ramp(d, 0.085, 0.12), ramp(0.3 - facing, 0.0, 0.4) * ramp(d, 0.035, 0.05))
    arm = bound & (per_position(off_arm) < 0.5)
    for _ in range(3):                    # close pinholes, drop specks
        frac = neighbour_fraction(arm)
        arm = bound & ((arm & (frac > 0.3)) | (frac > 0.6))
    ppos = per_position(P[:, 0]), per_position(P[:, 1]), per_position(P[:, 2])
    to_elbow = np.sqrt(sum((c - e) ** 2 for c, e in zip(ppos, elbow)))
    # Keep every sizeable connected patch of arm surface; specks are noise.
    reached = np.zeros(n_pid, dtype=bool)
    seen = np.zeros(n_pid, dtype=bool)
    sizes = []
    for seed in np.where(arm)[0]:
        if seen[seed]:
            continue
        seen[seed] = True
        stack, patch = [int(seed)], [int(seed)]
        while stack:
            u = stack.pop()
            for v in _edges[_starts[u]:_starts[u + 1], 1]:
                if arm[v] and not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
                    patch.append(int(v))
        sizes.append(len(patch))
        if len(patch) >= 20:
            reached[patch] = True
    print("  arm patches:", sorted(sizes, reverse=True)[:10])
    arm_v = reached[pid_w]
    release = bound[pid_w] & ~arm_v
    for slot in arm_slots:
        take = np.where(release, dense[:, slot], 0.0)
        dense[:, slot] -= take
        dense[rows, nearest_trunk] += take
    # Where a hand rested on the belt the rig blended hip and leg weights into the hand itself,
    # which pulls spikes out of it whenever the hips move away (crouching, lying down). Below
    # the shoulder the arm surface belongs to the arm chain only.
    chain = [joint_slot[side + b] for b in ("Shoulder", "Arm", "ForeArm", "Hand")]
    lower_arm = arm_v & (np.linalg.norm(P - shoulder, axis=1) > 0.16)
    foreign = np.ones(len(joints), dtype=bool)
    foreign[chain] = False
    dense[np.ix_(lower_arm, foreign)] = 0.0
    orphan = lower_arm & (dense.sum(axis=1) < 1e-6)
    dense[orphan, joint_slot[side + "ForeArm"]] = 1.0
    # A bridge triangle joins arm surface to released surface; the shoulder is the one place
    # where the arm legitimately joins the body, so nothing is cut there.
    tri_arm = arm_v[tri_w]
    near_shoulder = (np.linalg.norm(P - shoulder, axis=1) < 0.14)[tri_w].all(axis=1)
    bridge = tri_arm.any(axis=1) & ~tri_arm.all(axis=1) & ~near_shoulder
    cut |= bridge
    print(f"{side} arm: {int(arm_v.sum())} arm verts, {int(release.sum())} released, {int(bridge.sum())} bridge tris cut")
new_idx = tri_w[~cut].reshape(-1)
top = np.argsort(-dense, axis=1)[:, :4]
wgt = np.take_along_axis(dense, top, axis=1).astype(np.float32)
jnt = np.where(wgt > 0, top, 0)
wgt = wgt / np.maximum(wgt.sum(axis=1, keepdims=True), 1e-8)

index_array = new_idx.astype(np.uint16 if len(pos) < 65536 else np.uint32)
O["meshes"] = [{"name": "hero", "primitives": [{
    "attributes": {
        "POSITION": out.add_accessor(pos, "VEC3", minmax=True, target=34962),
        "NORMAL": out.add_accessor(nrm, "VEC3", target=34962),
        "TEXCOORD_0": out.add_accessor(uv, "VEC2", target=34962),
        "JOINTS_0": out.add_accessor(jnt.astype(np.uint8), "VEC4", target=34962),
        "WEIGHTS_0": out.add_accessor(wgt, "VEC4", target=34962),
    },
    "indices": out.add_accessor(index_array, "SCALAR", target=34963),
    "material": 0,
}]}]

# ---------------------------------------------------------------- texture + material
img_view = J["images"][0]["bufferView"]
image = Image.open(io.BytesIO(src.view_bytes(img_view))).convert("RGB")
print("texture:", image.size)
buf = io.BytesIO()
image.save(buf, "JPEG", quality=90)
O["images"] = [{"bufferView": out.append_view(buf.getvalue()), "mimeType": "image/jpeg"}]
O["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
O["textures"] = [{"source": 0, "sampler": 0}]
O["materials"] = [{
    "name": "hero",
    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 0.85},
    "doubleSided": True,
}]
O["skins"] = [{
    "joints": joints,
    "inverseBindMatrices": out.add_accessor(src.read(skin["inverseBindMatrices"]).astype(np.float32), "MAT4"),
}]

# ---------------------------------------------------------------- animation helpers
O["animations"] = []
BONE = {nodes[j]["name"]: j for j in joints}
HIPS = BONE["Hips"]


def emit_clip(name, tracks):
    """tracks: list of (node, path, times[T], values[T, k])."""
    anim = {"name": name, "samplers": [], "channels": []}
    time_cache = {}
    for node, path, times, values in tracks:
        times = np.asarray(times, dtype=np.float32)
        tkey = times.tobytes()
        if tkey not in time_cache:
            time_cache[tkey] = out.add_accessor(times, "SCALAR", minmax=True)
        values = np.asarray(values, dtype=np.float32)
        anim["samplers"].append({
            "input": time_cache[tkey],
            "output": out.add_accessor(values, "VEC4" if path == "rotation" else "VEC3"),
            "interpolation": "LINEAR",
        })
        anim["channels"].append({"sampler": len(anim["samplers"]) - 1, "target": {"node": node, "path": path}})
    O["animations"].append(anim)
    print(f"clip {name}: {len(tracks)} tracks, {max(float(t[2][-1]) for t in tracks):.2f}s")


def keep_hemisphere(quats):
    quats = np.array(quats, dtype=np.float64)
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    return quats


def local_from_model_delta(j, euler):
    rx, ry, rz = euler
    d = qmul(qmul(qaxis([0, 1, 0], ry), qaxis([1, 0, 0], rx)), qaxis([0, 0, 1], rz))
    w = world_q[j]
    return qnorm(qmul(rest_q[j], qmul(qinv(w), qmul(d, w))))


def authored_clip(name, duration, pose_fn):
    """pose_fn(u in 0..1) -> ({bone: (rx, ry, rz)}, (dx, dy, dz) hips offset in metres)."""
    frames = max(2, int(round(duration * FPS)))
    times = np.arange(frames + 1) / frames * duration
    rot = {j: [] for j in joints}
    hip = []
    for t in times:
        pose, offset = pose_fn(t / duration)
        unknown = set(pose) - set(BONE)
        assert not unknown, unknown
        for j in joints:
            rot[j].append(local_from_model_delta(j, pose.get(nodes[j]["name"], (0, 0, 0))))
        hip.append(rest_t[HIPS] + np.asarray(offset) / arm_scale[0])
    tracks = [(j, "rotation", times, keep_hemisphere(rot[j])) for j in joints]
    tracks.append((HIPS, "translation", times, np.array(hip)))
    # Keep every clip on the same track set, so switching clips never leaves a stale value behind.
    for j in extra_translation:
        tracks.append((j, "translation", times[[0, -1]], np.array([rest_t[j], rest_t[j]])))
    emit_clip(name, tracks)


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------- library clip (read first)
lib = J["animations"][0]
attack_tracks = []
extra_translation = []
for ch in lib["channels"]:
    node, path = ch["target"]["node"], ch["target"]["path"]
    sampler = lib["samplers"][ch["sampler"]]
    assert sampler.get("interpolation", "LINEAR") == "LINEAR", sampler
    times = src.read(sampler["input"]).reshape(-1)
    values = src.read(sampler["output"]).astype(np.float64)
    if path == "rotation" and node in rest_q:
        attack_tracks.append((node, path, times, keep_hemisphere(values)))
    elif path == "translation" and node == HIPS:
        drift = (values - rest_t[HIPS]) * arm_scale[0]
        print("attack hips offset (m) min", drift.min(axis=0).round(3), "max", drift.max(axis=0).round(3))
        attack_tracks.append((node, path, times, values))
    elif path == "translation" and node in rest_t:
        dev = np.abs(values - rest_t[node]).max()
        if dev > 0.05:  # the library clip really moves this joint, so keep its track
            print(f"  attack also translates {nodes[node]['name']} (up to {dev:.2f} from rest)")
            attack_tracks.append((node, path, times, values))
            extra_translation.append(node)
_have = {t[0] for t in attack_tracks if t[1] == "rotation"}
for j in joints:
    if j not in _have:
        t_all = attack_tracks[0][2]
        attack_tracks.append((j, "rotation", t_all[[0, -1]], np.array([rest_q[j], rest_q[j]])))


# ---------------------------------------------------------------- authored clips
def idle(u):
    b = math.sin(2 * math.pi * u)
    b2 = math.sin(4 * math.pi * u)
    return {
        "Spine02": (1.0 * b, 0, 0), "Spine01": (1.2 * b, 0, 0), "Spine": (0.8 * b, 0, 0),
        "neck": (-1.0 * b, 0, 0), "Head": (-1.0 * b, 2.5 * b, 0),
        "LeftArm": (1.5 * b, 0, -5 + 1.0 * b), "RightArm": (1.5 * b, 0, 5 - 1.0 * b),
        "LeftForeArm": (-9 - 2 * b, 0, 0), "RightForeArm": (-9 - 2 * b, 0, 0),
        "LeftHand": (0, 0, 1.5 * b2), "RightHand": (0, 0, -1.5 * b2),
    }, (0, 0, 0)


def gait(u, swing, knee, knee_base, arm, elbow, elbow_swing, lean, twist, bob_base, bob_amp, bob_sign, thigh_bias):
    phi = 2 * math.pi * u
    s, c = math.sin(phi), math.cos(phi)
    pose = {}
    for side, sg in (("Left", 1.0), ("Right", -1.0)):
        thigh = thigh_bias - swing * s * sg
        k = knee_base + knee * max(0.0, c * sg) ** 1.3
        pose[side + "UpLeg"] = (thigh, 0, 0)
        pose[side + "Leg"] = (k, 0, 0)
        pose[side + "Foot"] = (-(thigh + k) * 0.5, 0, 0)
        pose[side + "Arm"] = (arm * s * sg, 0, -6 * sg)
        pose[side + "ForeArm"] = (-elbow + elbow_swing * s * sg, 0, 0)
    pose["Hips"] = (0, -twist * s, 0)
    pose["Spine02"] = (lean * 0.55, twist * 1.6 * s, 0)
    pose["Spine01"] = (lean * 0.45, twist * 0.6 * s, 0)
    pose["neck"] = (-lean * 0.35, -twist * 0.6 * s, 0)
    pose["Head"] = (-lean * 0.35, -twist * 0.6 * s, 0)
    return pose, (0, bob_base + bob_sign * bob_amp * math.cos(2 * phi), 0)


def walk(u):
    return gait(u, swing=26, knee=50, knee_base=6, arm=20, elbow=14, elbow_swing=6, lean=4, twist=4,
                bob_base=-0.018, bob_amp=0.012, bob_sign=1, thigh_bias=-3)


def run(u):
    return gait(u, swing=42, knee=92, knee_base=14, arm=38, elbow=66, elbow_swing=10, lean=16, twist=7,
                bob_base=-0.035, bob_amp=0.03, bob_sign=-1, thigh_bias=-12)


def jump(u):
    k = smooth(u * 2.2)
    return {
        "Spine02": (-6 * k, 0, 0), "Head": (-6 * k, 0, 0),
        "LeftUpLeg": (-52 * k, 0, 0), "LeftLeg": (78 * k, 0, 0), "LeftFoot": (18 * k, 0, 0),
        "RightUpLeg": (12 * k, 0, 0), "RightLeg": (18 * k, 0, 0), "RightFoot": (28 * k, 0, 0),
        "LeftArm": (-35 * k, 0, 30 * k), "RightArm": (-35 * k, 0, -30 * k),
        "LeftForeArm": (-35 * k, 0, 0), "RightForeArm": (-35 * k, 0, 0),
    }, (0, 0, 0)


def fall(u):
    w = math.sin(2 * math.pi * u)
    return {
        "Spine02": (6, 0, 0), "Head": (6, 0, 0),
        "LeftUpLeg": (-22 + 4 * w, 0, 0), "LeftLeg": (32 - 4 * w, 0, 0), "LeftFoot": (20, 0, 0),
        "RightUpLeg": (-8 - 4 * w, 0, 0), "RightLeg": (24 + 4 * w, 0, 0), "RightFoot": (20, 0, 0),
        "LeftArm": (5 * w, 0, 52 + 6 * w), "RightArm": (-5 * w, 0, -52 - 6 * w),
        "LeftForeArm": (-28, 0, 0), "RightForeArm": (-28, 0, 0),
    }, (0, 0, 0)


def double_jump(u):
    spin = 360 * (u * u * u * (u * (6 * u - 15) + 10))  # smootherstep
    tuck = math.sin(math.pi * u) ** 0.7
    pose = {
        "Hips": (spin, 0, 0),
        "Spine02": (22 * tuck, 0, 0), "Spine01": (16 * tuck, 0, 0), "neck": (10 * tuck, 0, 0), "Head": (12 * tuck, 0, 0),
    }
    for side, sg in (("Left", 1.0), ("Right", -1.0)):
        pose[side + "UpLeg"] = (-98 * tuck, 0, 0)
        pose[side + "Leg"] = (118 * tuck, 0, 0)
        pose[side + "Foot"] = (15 * tuck, 0, 0)
        pose[side + "Arm"] = (-48 * tuck, 0, -4 * sg * tuck)
        pose[side + "ForeArm"] = (-85 * tuck, 0, 0)
    return pose, (0, 0.12 * tuck, 0)


# ---- intro: asleep on the ground, then getting up with a hand to the aching head
HIP_Y = float(joint_pos["Hips"][1])


def keyed(keys):
    """keys: [(time, {bone: euler}, hips offset)] -> pose_fn(u) with eased blends between keys."""
    end = keys[-1][0]

    def pose_fn(u):
        t = u * end
        for (t0, p0, o0), (t1, p1, o1) in zip(keys, keys[1:]):
            if t <= t1:
                k = smooth((t - t0) / (t1 - t0))
                pose = {b: tuple(np.array(p0.get(b, (0, 0, 0))) * (1 - k) + np.array(p1.get(b, (0, 0, 0))) * k)
                        for b in set(p0) | set(p1)}
                return pose, tuple(np.array(o0) * (1 - k) + np.array(o1) * k)
        return keys[-1][1], keys[-1][2]
    return pose_fn


LYING = {
    "Hips": (-90, 0, 0), "Head": (0, 14, 0), "neck": (-6, 0, 0),
    "LeftUpLeg": (-8, 0, 3), "LeftLeg": (14, 0, 0), "RightUpLeg": (0, 0, -3),
    "LeftArm": (0, 0, 10), "RightArm": (0, 0, -10), "LeftForeArm": (-12, 0, 0), "RightForeArm": (-20, 0, 0),
}
LYING_OFFSET = (0, 0.17 - HIP_Y, 0)


def sleep(u):
    b = math.sin(2 * math.pi * u)
    pose = dict(LYING)
    pose["Spine01"] = (-1.5 * b, 0, 0)
    pose["Spine"] = (-1.0 * b, 0, 0)
    return pose, LYING_OFFSET


SITTING = {
    "Hips": (-14, 0, 0), "Spine02": (12, 0, 0), "Spine01": (10, 0, 0), "neck": (8, 0, 0), "Head": (14, 0, 0),
    "LeftUpLeg": (-72, 0, 4), "RightUpLeg": (-76, 0, -4), "LeftLeg": (16, 0, 0), "RightLeg": (8, 0, 0),
    "LeftArm": (28, 0, 8), "RightArm": (28, 0, -8), "LeftForeArm": (-10, 0, 0), "RightForeArm": (-10, 0, 0),
}
CROUCH = {
    "Hips": (25, 0, 0), "Spine02": (8, 0, 0), "Head": (-12, 0, 0),
    "LeftUpLeg": (-100, 0, 0), "RightUpLeg": (-100, 0, 0), "LeftLeg": (115, 0, 0), "RightLeg": (115, 0, 0),
    "LeftFoot": (-40, 0, 0), "RightFoot": (-40, 0, 0),
    "LeftArm": (-38, 0, 0), "RightArm": (-38, 0, 0), "LeftForeArm": (-30, 0, 0), "RightForeArm": (-30, 0, 0),
}
HEAD_HOLD = {
    "Spine02": (5, 0, 0), "Spine01": (5, 0, 0), "neck": (8, 0, 0), "Head": (12, -8, 0),
    "RightArm": (-62, 0, 16), "RightForeArm": (-128, 0, 0), "RightHand": (-10, 0, 0),
    "LeftArm": (0, 0, -4), "LeftForeArm": (-12, 0, 0),
}
wake_up = keyed([
    (0.0, LYING, LYING_OFFSET),
    (0.5, LYING, LYING_OFFSET),
    (1.6, SITTING, (0, 0.12 - HIP_Y, 0)),
    (2.2, SITTING, (0, 0.12 - HIP_Y, 0)),
    (3.0, CROUCH, (0, 0.54 - HIP_Y, 0.1)),
    (3.9, {}, (0, 0, 0)),
    (4.6, HEAD_HOLD, (0, 0, 0)),
    (5.6, HEAD_HOLD, (0, 0, 0)),
    (6.3, {}, (0, 0, 0)),
])

authored_clip("Sleep", 4.0, sleep)
authored_clip("WakeUp", 6.3, lambda u: wake_up(u))
authored_clip("Idle", 3.0, idle)
authored_clip("Walk", 1.0, walk)
authored_clip("Run", 0.64, run)
authored_clip("Jump", 0.4, jump)
authored_clip("Fall", 1.0, fall)
authored_clip("DoubleJump", 0.6, double_jump)

emit_clip("Attack", attack_tracks)

out.save(dst_path)
print("saved", dst_path, len(out.bin), "bytes")
