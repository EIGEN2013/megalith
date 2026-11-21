"""Assemble the playable Megalith page and the data files it loads.

Sources
    web/megalith/head.html, main.js, story.json, villagers.json   page, script, words
    live_src/hero.glb, world.json, forest.glb, forest_inst.f32, forest.json, npcs.glb
    raw/live/village.*   the village, published earlier (copied only so the local test works)

Writes web/megalith/dist/:
    megalith.html        the page as the artifact publisher wants it (no <html> skeleton)
    index.html           the same page in a full document, for a local browser
    forest.b64.txt, forest_inst.b64.txt, forest.json, npcs.b64.txt   gzip + base64, like the village files
Usage: python build_megalith.py
"""
import base64
import gzip
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "web" / "megalith"
LIVE = ROOT / "live_src"
DIST = SRC / "dist"
DIST.mkdir(exist_ok=True)


def packed(path):
    return base64.b64encode(gzip.compress(path.read_bytes(), 9)).decode("ascii")


def compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


story = json.loads((SRC / "story.json").read_text(encoding="utf-8"))
villagers = json.loads((SRC / "villagers.json").read_text(encoding="utf-8"))
story["village"]["npcs"] = villagers
for key, v in villagers.items():
    story["speakers"][key] = v["name"]
story["interact"]["altar"]["name"] = story["interact"]["altar"].get("name", "제단")

head = (SRC / "head.html").read_text(encoding="utf-8")
scripts = "\n".join([
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>',
    '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>',
    '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/libs/fflate.min.js"></script>',
    '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/utils/SkeletonUtils.js"></script>',
    '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/RGBELoader.js"></script>',
    '<script>window.MEGALITH_GROUND = %s;</script>' % compact(json.loads((SRC / "ground" / "ground.json").read_text(encoding="utf-8"))),
    '<script id="hero-glb" type="application/octet-stream">%s</script>' % base64.b64encode((LIVE / "hero_plus.glb").read_bytes()).decode("ascii"),
    '<script id="world-json" type="application/json">%s</script>' % compact(json.loads((LIVE / "world.json").read_text(encoding="utf-8"))),
    '<script id="story-json" type="application/json">%s</script>' % compact(story),
    '<script>window.MEGALITH_TEX = %s;</script>' % compact(json.loads((SRC / "tex" / "tex.json").read_text(encoding="utf-8"))),
    "<script>%s</script>" % (SRC / "main.js").read_text(encoding="utf-8"),
])
page = head.rstrip() + "\n\n" + scripts + "\n"
(DIST / "megalith.html").write_text(page, encoding="utf-8", newline="\n")
shell = ('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
         '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
         '<style>body{margin:0}[hidden]{display:none!important}</style>\n</head>\n<body>\n')
(DIST / "index.html").write_text(shell + page + "</body>\n</html>\n", encoding="utf-8", newline="\n")

(DIST / "forest.b64.txt").write_text(packed(LIVE / "forest.glb"), encoding="ascii")
(DIST / "forest_inst.b64.txt").write_text(packed(LIVE / "forest_inst.f32"), encoding="ascii")
(DIST / "npcs.b64.txt").write_text(packed(LIVE / "npcs.glb"), encoding="ascii")
shutil.copyfile(LIVE / "forest.json", DIST / "forest.json")
for name in ("village.b64.txt", "village_inst.b64.txt", "village.json"):
    if not (DIST / name).exists():
        shutil.copyfile(ROOT / "raw" / "live" / name, DIST / name)
(DIST / "tex").mkdir(exist_ok=True)
for f in (SRC / "tex").glob("*.jpg"):
    shutil.copyfile(f, DIST / "tex" / f.name)
# the land around the village (photoscanned nature)
HQ = ROOT / "assets_src" / "hq"
(DIST / "music").mkdir(exist_ok=True)
for f in (SRC / "music").glob("*.mp3"):
    shutil.copyfile(f, DIST / "music" / f.name)
for sub in ("ground", "sky"):
    (DIST / sub).mkdir(exist_ok=True)
    for f in (SRC / sub).iterdir():
        if f.suffix in (".webp", ".jpg", ".json"):
            shutil.copyfile(f, DIST / sub / f.name)
        elif f.suffix == ".hdr":           # artifacts don't serve .hdr: ship it as base64 text
            (DIST / sub / (f.name + ".txt")).write_text(base64.b64encode(f.read_bytes()).decode("ascii"), encoding="ascii")
            stale = DIST / sub / f.name
            if stale.exists():
                stale.unlink()
shutil.copyfile(HQ / "land.json", DIST / "land.json")
shutil.copyfile(HQ / "land_splat.png", DIST / "land_splat.png")
shutil.copyfile(HQ / "props.json", DIST / "props.json")
for name in ("trees_broad", "trees_fir", "props_hard", "props_soft", "castle"):
    (DIST / f"{name}.b64.txt").write_text(packed(HQ / f"{name}.glb"), encoding="ascii")
shutil.copyfile(HQ / "castle.json", DIST / "castle.json")
(DIST / "items.b64.txt").write_text(packed(ROOT / "assets_src" / "items" / "items_hq.glb"), encoding="ascii")
EXT = ROOT / "assets_src" / "village_ext"            # the enlarged village: tools/hq_village.py
(DIST / "village_ext.b64.txt").write_text(packed(EXT / "village_ext.glb"), encoding="ascii")
shutil.copyfile(EXT / "village_ext.json", DIST / "village_ext.json")
for name in ("pickups_hq", "pickups_food"):       # things to pick up: tools/hq_pickups.py
    (DIST / f"{name}.b64.txt").write_text(packed(ROOT / "assets_src" / "items" / f"{name}.glb"), encoding="ascii")
for name in ("horror", "golem"):
    (DIST / f"{name}.b64.txt").write_text(packed(ROOT / "assets_src" / "monsters" / f"{name}.glb"), encoding="ascii")
for f in sorted(DIST.iterdir()):
    print(f"{f.name:22s} {f.stat().st_size / 1e6:6.2f} MB")
