"""Unpack the live artifact (raw/live/) into ordinary files under live_src/:

    live_src/hero.glb, live_src/world.glb, live_src/world.json, live_src/story.json
    live_src/village.glb, live_src/village_inst.f32, live_src/village.json
    live_src/main.js, live_src/page_head.html (markup and styles before the first <script>)

Usage: python unpack_live.py
"""
import base64
import gzip
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
LIVE = ROOT / "raw" / "live"
OUT = ROOT / "live_src"
OUT.mkdir(exist_ok=True)

page = (LIVE / "index.html").read_text(encoding="utf-8")


def script_body(ident):
    m = re.search(r'<script id="%s"[^>]*>(.*?)</script>' % re.escape(ident), page, flags=re.S)
    return m.group(1).strip()


(OUT / "hero.glb").write_bytes(base64.b64decode(script_body("hero-glb")))
(OUT / "world.glb").write_bytes(base64.b64decode(script_body("world-glb")))
(OUT / "world.json").write_text(script_body("world-json"), encoding="utf-8")
(OUT / "story.json").write_text(json.dumps(json.loads(script_body("story-json")), ensure_ascii=False, indent=1), encoding="utf-8")
plain = [m for m in re.finditer(r"<script>(.*?)</script>", page, flags=re.S)]
(OUT / "main.js").write_text(plain[-1].group(1), encoding="utf-8")
start = page.find("<title>")
(OUT / "page_head.html").write_text(page[start:page.find("<script", start)], encoding="utf-8")

(OUT / "village.glb").write_bytes(gzip.decompress(base64.b64decode((LIVE / "village.b64.txt").read_text().strip())))
(OUT / "village_inst.f32").write_bytes(gzip.decompress(base64.b64decode((LIVE / "village_inst.b64.txt").read_text().strip())))
(OUT / "village.json").write_bytes((LIVE / "village.json").read_bytes())
for f in sorted(OUT.iterdir()):
    print(f"{f.name:20s} {f.stat().st_size:>10d}")
