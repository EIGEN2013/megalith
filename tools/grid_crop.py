"""Helper for tracing objects in the reference painting: save an enlarged crop with a
coordinate grid (labels are source-image pixels).

Usage: python grid_crop.py <image> <out.png> x0 y0 x1 y1 [scale] [grid]
"""
import sys

from PIL import Image, ImageDraw

src, out = sys.argv[1], sys.argv[2]
x0, y0, x1, y1 = (int(v) for v in sys.argv[3:7])
scale = int(sys.argv[7]) if len(sys.argv) > 7 else 3
grid = int(sys.argv[8]) if len(sys.argv) > 8 else 20
im = Image.open(src).convert("RGB").crop((x0, y0, x1, y1))
im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
d = ImageDraw.Draw(im)
for x in range((x0 // grid + 1) * grid, x1, grid):
    major = x % (grid * 2) == 0
    d.line([((x - x0) * scale, 0), ((x - x0) * scale, im.height)], fill=(255, 255, 0) if major else (255, 255, 255), width=1)
    if major:
        d.text(((x - x0) * scale + 2, 2), str(x), fill=(255, 255, 0))
for y in range((y0 // grid + 1) * grid, y1, grid):
    major = y % (grid * 2) == 0
    d.line([(0, (y - y0) * scale), (im.width, (y - y0) * scale)], fill=(255, 255, 0) if major else (255, 255, 255), width=1)
    if major:
        d.text((2, (y - y0) * scale + 2), str(y), fill=(255, 255, 0))
im.save(out)
print(out, im.size)
