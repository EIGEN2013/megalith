"""Cut each prop out of the reference painting onto a plain background, so the image-to-3D
model sees exactly one object per picture.

Outlines are traced by hand in source-image pixels (see tools/grid_crop.py). A prop may be
several polygons (the trilithon is two uprights and a lintel, with the gap left out).

Writes assets_src/inputs/<name>.png and a contact sheet, assets_src/inputs/_sheet.png.
Usage: python isolate_assets.py
"""
import pathlib

from PIL import Image, ImageDraw, ImageFilter

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "assets_src" / "reference_scene.webp"
OUT = ROOT / "assets_src" / "inputs"
BACKGROUND = (214, 214, 214)
TARGET = 900          # the prop's longer side, in output pixels

PROPS = {
    "trilithon": [
        [(383, 190), (388, 183), (455, 170), (520, 163), (575, 161), (590, 165), (598, 172), (598, 201), (580, 205),
         (537, 218), (470, 222), (400, 222), (385, 216)],
        [(362, 207), (385, 203), (387, 218), (438, 222), (441, 300), (440, 372), (440, 473), (420, 478), (370, 478),
         (355, 470), (353, 430), (357, 373), (360, 300), (362, 240)],
        [(538, 220), (598, 203), (602, 210), (608, 240), (613, 307), (618, 373), (620, 440), (622, 480), (600, 488),
         (540, 488), (510, 476), (515, 440), (517, 393), (522, 353), (527, 327), (530, 287), (540, 240)],
    ],
    "door_stone": [
        [(740, 213), (744, 217), (747, 255), (750, 272), (752, 307), (752, 340), (762, 350), (766, 380), (765, 394),
         (670, 394), (672, 365), (676, 330), (680, 297), (684, 272), (687, 262), (689, 244), (702, 232), (722, 220)],
    ],
    "altar": [
        [(655, 394), (800, 394), (800, 400), (793, 416), (775, 428), (790, 440), (803, 455), (800, 466), (650, 466),
         (645, 450), (660, 440), (667, 430), (657, 410)],
    ],
    "monolith": [
        [(1010, 201), (1018, 210), (1023, 247), (1028, 280), (1030, 320), (1037, 367), (1040, 413), (1050, 447),
         (1058, 468), (1000, 472), (968, 460), (973, 413), (975, 367), (977, 337), (982, 307), (977, 280), (978, 247),
         (980, 213), (996, 205)],
    ],
    "rune_stone": [
        [(860, 322), (877, 327), (900, 343), (910, 353), (912, 393), (917, 433), (920, 470), (927, 497), (900, 503),
         (843, 500), (838, 473), (840, 440), (842, 400), (848, 367), (853, 330)],
    ],
    "small_stone": [
        [(437, 369), (458, 380), (470, 407), (477, 457), (479, 494), (422, 494), (423, 440), (427, 387)],
    ],
    "rock_spire": [
        [(350, 505), (360, 510), (380, 532), (387, 562), (391, 592), (402, 612), (409, 628), (297, 628), (310, 587),
         (322, 562), (332, 552), (341, 517)],
    ],
    "rock_crag": [
        [(1220, 444), (1230, 453), (1240, 500), (1255, 510), (1260, 547), (1258, 575), (1200, 600), (1163, 610),
         (1165, 593), (1172, 557), (1187, 537), (1198, 503), (1200, 463)],
    ],
    "mushroom": [
        [(1283, 690), (1290, 672), (1310, 663), (1335, 665), (1346, 675), (1343, 690), (1330, 697), (1300, 700)],
        [(1312, 697), (1332, 697), (1336, 745), (1318, 750)],
    ],
    "tree_twisted": [
        [(30, 268), (35, 245), (60, 222), (95, 205), (125, 210), (150, 225), (170, 245), (190, 280), (185, 292),
         (160, 290), (140, 278), (125, 283), (110, 275), (95, 280), (85, 290), (70, 283), (50, 283)],
        [(72, 292), (100, 285), (104, 305), (126, 340), (136, 380), (136, 415), (130, 440), (102, 475), (100, 500),
         (145, 505), (145, 545), (132, 565), (140, 600), (40, 600), (40, 525), (45, 480), (55, 440), (88, 405),
         (90, 375), (84, 340), (68, 310)],
    ],
    "tree_green": [
        [(150, 345), (165, 325), (200, 315), (240, 318), (270, 312), (300, 320), (325, 335), (330, 360), (310, 380),
         (280, 392), (262, 400), (268, 425), (285, 440), (300, 445), (300, 455), (240, 455), (245, 435), (225, 415),
         (205, 400), (175, 385), (155, 370)],
    ],
    "tree_canopy": [
        [(1180, 232), (1195, 205), (1230, 185), (1232, 160), (1262, 138), (1300, 132), (1320, 110), (1360, 105),
         (1408, 125), (1408, 440), (1350, 440), (1362, 400), (1375, 360), (1360, 320), (1338, 290), (1300, 280),
         (1272, 268), (1262, 240), (1235, 228), (1200, 238)],
    ],
}


def isolate(image, polygons):
    xs = [x for poly in polygons for x, _ in poly]
    ys = [y for poly in polygons for _, y in poly]
    x0, y0, x1, y1 = max(0, min(xs) - 2), max(0, min(ys) - 2), min(image.width, max(xs) + 2), min(image.height, max(ys) + 2)
    scale = TARGET / max(x1 - x0, y1 - y0)
    size = (round((x1 - x0) * scale), round((y1 - y0) * scale))
    crop = image.crop((x0, y0, x1, y1)).resize(size, Image.LANCZOS)
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        draw.polygon([((x - x0) * scale, (y - y0) * scale) for x, y in poly], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(1.2))
    margin = round(TARGET * 0.12)
    canvas = Image.new("RGB", (size[0] + 2 * margin, size[1] + 2 * margin), BACKGROUND)
    canvas.paste(crop, (margin, margin), mask)
    return canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    image = Image.open(SOURCE).convert("RGB")
    thumbs = []
    for name, polygons in PROPS.items():
        result = isolate(image, polygons)
        result.save(OUT / f"{name}.png")
        thumb = result.copy()
        thumb.thumbnail((300, 300))
        thumbs.append((name, thumb))
        print(f"{name}: {result.size}")
    sheet = Image.new("RGB", (4 * 310, ((len(thumbs) + 3) // 4) * 330), (40, 40, 40))
    draw = ImageDraw.Draw(sheet)
    for i, (name, thumb) in enumerate(thumbs):
        x, y = (i % 4) * 310 + 5, (i // 4) * 330 + 5
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + 304), name, fill=(255, 255, 255))
    sheet.save(OUT / "_sheet.png")


if __name__ == "__main__":
    main()
