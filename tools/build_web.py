"""Pack the browser build: inline the models, the world data and the story into web/game.src.html.

Writes two files:
  web/greencloak-hero.html  page body only (the form the Claude artifact publisher expects)
  web/index.html            the same page wrapped in a normal HTML document, for any browser
"""
import base64
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
game = root / "hero_game"
page = (root / "web" / "game.src.html").read_text(encoding="utf-8")


def b64(path):
    return base64.b64encode(path.read_bytes()).decode("ascii")


def compact_json(path):
    text = json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, separators=(",", ":"))
    return text.replace("</", "<\\/")       # never close the <script> that carries it


inserts = {
    "__HERO_GLB_BASE64__": b64(game / "models" / "hero.glb"),
    "__WORLD_GLB_BASE64__": b64(game / "models" / "world.glb"),
    "__WORLD_JSON__": compact_json(game / "world.json"),
    "__STORY_JSON__": compact_json(game / "story.json"),
}
for marker, value in inserts.items():
    assert page.count(marker) == 1, marker
    page = page.replace(marker, value)
(root / "web" / "greencloak-hero.html").write_text(page, encoding="utf-8", newline="\n")
shell = ('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
         '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
         '<style>body{margin:0}[hidden]{display:none!important}</style>\n</head>\n<body>\n')
(root / "web" / "index.html").write_text(shell + page + "\n</body>\n</html>\n", encoding="utf-8", newline="\n")
print(f"web build: {len(page.encode('utf-8')) / 1e6:.2f} MB")
