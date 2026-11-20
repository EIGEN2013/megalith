"""Recompress the images inside a GLB: cap their size and re-encode (JPEG, or WebP when they carry alpha).

  python tools/glb_shrink.py in.glb out.glb --max 1024 [--keep name=2048 ...] [--quality 85]
"""
import argparse
import io
import json
import struct

from PIL import Image


def read_glb(path):
    data = open(path, "rb").read()
    magic, ver, total = struct.unpack_from("<III", data, 0)
    off = 12
    js, binc = None, b""
    while off < total:
        ln, typ = struct.unpack_from("<II", data, off)
        chunk = data[off + 8: off + 8 + ln]
        if typ == 0x4E4F534A:
            js = json.loads(chunk.decode("utf-8"))
        elif typ == 0x004E4942:
            binc = chunk
        off += 8 + ln
    return js, binc


def write_glb(path, js, binc):
    jb = json.dumps(js, separators=(",", ":")).encode("utf-8")
    jb += b" " * ((4 - len(jb) % 4) % 4)
    binc += b"\0" * ((4 - len(binc) % 4) % 4)
    total = 12 + 8 + len(jb) + 8 + len(binc)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(jb), 0x4E4F534A) + jb)
        f.write(struct.pack("<II", len(binc), 0x004E4942) + binc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--max", type=int, default=1024)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--keep", nargs="*", default=[])
    a = ap.parse_args()
    keep = dict(k.split("=") for k in a.keep)
    js, binc = read_glb(a.src)
    views = js["bufferViews"]
    img_views = {}
    for i, im in enumerate(js.get("images", [])):
        if "bufferView" not in im:
            continue
        v = views[im["bufferView"]]
        raw = binc[v.get("byteOffset", 0): v.get("byteOffset", 0) + v["byteLength"]]
        pic = Image.open(io.BytesIO(raw))
        cap = a.max
        for k, s in keep.items():
            if k in (im.get("name") or ""):
                cap = int(s)
        alpha = pic.mode in ("RGBA", "LA") and pic.getextrema()[-1][0] < 255
        if max(pic.size) > cap:
            if alpha:
                r, g, b, al = pic.convert("RGBA").split()
                rgb = Image.merge("RGB", (r, g, b)).resize((cap, cap), Image.LANCZOS)
                pic = Image.merge("RGBA", (*rgb.split(), al.resize((cap, cap), Image.LANCZOS)))
            else:
                pic = pic.convert("RGB").resize((cap, cap), Image.LANCZOS)
        out = io.BytesIO()
        if alpha:
            pic.convert("RGBA").save(out, "WEBP", quality=a.quality + 5, exact=True)
            im["mimeType"] = "image/webp"
        else:
            pic.convert("RGB").save(out, "JPEG", quality=a.quality, optimize=True)
            im["mimeType"] = "image/jpeg"
        img_views[im["bufferView"]] = out.getvalue()
    # rebuild the binary chunk with the new image bytes
    new_bin = bytearray()
    for i, v in enumerate(views):
        start = v.get("byteOffset", 0)
        data = img_views.get(i, binc[start: start + v["byteLength"]])
        new_bin += b"\0" * ((4 - len(new_bin) % 4) % 4)
        v["byteOffset"] = len(new_bin)
        v["byteLength"] = len(data)
        new_bin += data
    js["buffers"][0]["byteLength"] = len(new_bin)
    write_glb(a.dst, js, bytes(new_bin))
    print(a.dst, round(len(new_bin) / 1e6, 2), "MB")


main()
