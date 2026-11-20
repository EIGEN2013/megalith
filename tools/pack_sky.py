"""Sky from the Poly Haven HDRI: a small HDR for image-based light, a 4K tonemapped JPEG for the
visible sky, and the sun's direction (brightest region) so the game's sunlight matches the photo.

Writes web/megalith/sky/sky_env.hdr, sky_bg.jpg, sky.json
Run: blender --background --factory-startup --python tools/pack_sky.py
"""
import json
import math
import pathlib

import bpy
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "megalith" / "sky"
OUT.mkdir(parents=True, exist_ok=True)
src = ROOT / "ph" / "sky.hdr"
im = bpy.data.images.load(str(src))
W, H = im.size
px = np.empty(W * H * 4, dtype=np.float32)
im.pixels.foreach_get(px)
px = px.reshape(H, W, 4)[::-1]                          # row 0 = top of the sky
lum = px[..., :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
iy, ix = np.unravel_index(np.argmax(lum), lum.shape)
u, v = (ix + 0.5) / W, (iy + 0.5) / H
phi = (u - 0.5) * 2 * math.pi                            # equirect: u=0.5 looks down -Z in three.js
theta = (0.5 - v) * math.pi                              # elevation
# (the corrected three.js convention, u = atan(z, x) / 2pi + 0.5, is applied to sky.json afterwards)
sun = [math.cos(theta) * math.sin(phi), math.sin(theta), -math.cos(theta) * math.cos(phi)]
# the sun's colour and strength (sum over the disc) and the sky's average, for balancing the lights
disc = lum > lum.max() * 0.25
sun_rgb = (px[..., :3][disc]).mean(axis=0)
sky_mean = px[..., :3][: H // 2].reshape(-1, 3).mean(axis=0)
print("sun at", ix, iy, "dir", sun, "elev deg", math.degrees(theta), "peak", lum.max())


def save(arr, path, fmt, w, h):
    img = bpy.data.images.new(path.stem, w, h, alpha=False, float_buffer=(fmt == "HDR"))
    img.pixels.foreach_set(np.ascontiguousarray(arr[::-1]).ravel())
    img.filepath_raw = str(path)
    img.file_format = fmt
    if fmt == "JPEG":
        bpy.context.scene.render.image_settings.quality = 90
    img.save()


def resize(a, w, h):
    """Box-filter downscale of an equirect image."""
    fy, fx = a.shape[0] // h, a.shape[1] // w
    return a[: h * fy, : w * fx].reshape(h, fy, w, fx, 4).mean(axis=(1, 3))


clamped = px.copy()
clamped[..., :3] = np.minimum(clamped[..., :3], 3.0)     # the sun is the game's directional light, not part of the sky probe
env = resize(clamped, 512, 256)
save(env, OUT / "sky_env.hdr", "HDR", 512, 256)
# background: clamp the sun away (the game draws its own), ACES-like curve at the game's exposure
bg = resize(px, 4096, 2048) if W >= 4096 else px.copy()
EXPOSURE = 1.15
x = np.minimum(bg[..., :3], 6.0) * EXPOSURE
a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
t = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1) ** (1 / 2.2)
bg[..., :3] = t
bg[..., 3] = 1
save(bg, OUT / "sky_bg.jpg", "JPEG", bg.shape[1], bg.shape[0])
(OUT / "sky.json").write_text(json.dumps({"sun": sun, "sun_rgb": [float(x) for x in sun_rgb], "sky_mean": [float(x) for x in sky_mean],
                                          "elevation_deg": math.degrees(theta)}, indent=1))
for f in OUT.iterdir():
    print(f.name, round(f.stat().st_size / 1e6, 2), "MB")

