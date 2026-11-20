"""Make a large inlined HTML page readable: replace every long base64/JSON blob inside a
<script> with a short marker naming its id and size, and list what was removed.

Usage: python strip_blobs.py <in.html> <out.html>
"""
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
removed = []


def shrink(match):
    open_tag, body, close_tag = match.group(1), match.group(2), match.group(3)
    if len(body) < 20000:
        return match.group(0)
    ident = re.search(r'id="([^"]+)"', open_tag)
    removed.append((ident.group(1) if ident else "?", len(body), open_tag[:120]))
    return f"{open_tag}[[BLOB {ident.group(1) if ident else '?'} {len(body)} chars]]{close_tag}"


out = re.sub(r"(<script[^>]*>)(.*?)(</script>)", shrink, text, flags=re.S)
open(dst, "w", encoding="utf-8").write(out)
for ident, size, tag in removed:
    print(f"{ident:24s} {size:>10d}  {tag}")
print("readable size:", len(out))
