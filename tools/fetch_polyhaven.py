"""Download CC0 assets from Poly Haven (api.polyhaven.com) into downloads/polyhaven/<id>/.

  python tools/fetch_polyhaven.py textures 2k leafy_grass rocky_trail ...
  python tools/fetch_polyhaven.py models 2k boulder_01 fern_02 ...
  python tools/fetch_polyhaven.py hdris 4k kloofendal_48d_partly_cloudy

Textures: diffuse, OpenGL normal, ARM (AO/rough/metal) and displacement as JPG/PNG.
Models: the glTF with its .bin and textures, laid out as the glTF expects.
Every file is checked against the MD5 Poly Haven publishes; existing good files are skipped.
"""
import hashlib
import json
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "ph"
UA = {"User-Agent": "megalith-asset-fetch/1.0"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def longpath(p):
    s = str(pathlib.Path(p).resolve())
    return pathlib.Path("\\\\?\\" + s) if sys.platform == "win32" and not s.startswith("\\\\?\\") else pathlib.Path(s)


def fetch(url, md5, dest):
    dest = longpath(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and hashlib.md5(dest.read_bytes()).hexdigest() == md5:
        return 0
    data = get(url)
    if md5 and hashlib.md5(data).hexdigest() != md5:
        raise RuntimeError(f"md5 mismatch {url}")
    dest.write_bytes(data)
    return len(data)


def main():
    kind, res, ids = sys.argv[1], sys.argv[2], sys.argv[3:]
    total = 0
    for aid in ids:
        files = json.loads(get(f"https://api.polyhaven.com/files/{aid}"))
        d = OUT / aid
        got = 0
        if kind == "textures":
            for key, fmt in (("Diffuse", "jpg"), ("nor_gl", "jpg"), ("arm", "jpg"), ("Displacement", "png"), ("Rough", "jpg"), ("AO", "jpg")):
                if key in files and res in files[key] and fmt in files[key][res]:
                    f = files[key][res][fmt]
                    got += fetch(f["url"], f["md5"], d / pathlib.Path(f["url"]).name)
        elif kind == "models":
            g = files["gltf"].get(res) or files["gltf"][sorted(files["gltf"])[0]]
            g = g["gltf"]
            got += fetch(g["url"], g["md5"], d / pathlib.Path(g["url"]).name)
            for rel, f in g.get("include", {}).items():
                got += fetch(f["url"], f["md5"], d / rel)
        elif kind == "hdris":
            f = files["hdri"][res]["hdr"]
            got += fetch(f["url"], f["md5"], d / pathlib.Path(f["url"]).name)
            for key in ("tonemapped",):
                if key in files:
                    f = files[key]
                    got += fetch(f["url"], f["md5"], d / pathlib.Path(f["url"]).name)
        total += got
        print(f"{aid}: {got / 1e6:.1f} MB")
    print(f"total {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
