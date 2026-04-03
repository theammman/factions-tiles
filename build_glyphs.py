"""
build_glyphs.py  –  Convert a TTF into MapLibre PBF glyph files.

Generates:  glyphs/<FontName>/<start>-<end>.pbf  for every 256-char range.
MapLibre glyphs URL:  .../glyphs/{fontstack}/{range}.pbf

Usage:
    python build_glyphs.py

Requires: fonttools, Pillow
  pip install fonttools Pillow
"""

import os
import struct
import math
from pathlib import Path
from fontTools.ttLib import TTFont

# ── Config ───────────────────────────────────────────────────────────────────
TTF_PATH   = Path("font_src/PressStart2P-Regular.ttf")
FONT_NAME  = "Press Start 2P Regular"
OUT_DIR    = Path("glyphs") / FONT_NAME
BUFFER     = 3      # px SDF buffer around glyph
RADIUS     = 8      # SDF radius in pixels
CUTOFF     = 0.25
SIZE       = 24     # render size in px

# ── Protobuf helpers (no external proto dependency) ──────────────────────────
def _encode_varint(val: int) -> bytes:
    out = b""
    while True:
        bits = val & 0x7F
        val >>= 7
        if val:
            out += bytes([bits | 0x80])
        else:
            out += bytes([bits])
            break
    return out

def _field(field_num: int, wire_type: int, data: bytes) -> bytes:
    tag = (field_num << 3) | wire_type
    return _encode_varint(tag) + data

def _len_delim(field_num: int, data: bytes) -> bytes:
    return _field(field_num, 2, _encode_varint(len(data)) + data)

def _uint32_field(field_num: int, val: int) -> bytes:
    return _field(field_num, 0, _encode_varint(val))

def _sint32(val: int) -> int:
    return (val << 1) ^ (val >> 31)

def _sint32_field(field_num: int, val: int) -> bytes:
    return _field(field_num, 0, _encode_varint(_sint32(val)))

# ── SDF rasterizer ───────────────────────────────────────────────────────────
def _render_sdf(font: TTFont, glyph_name: str, size: int,
                buffer: int, radius: int, cutoff: float):
    """
    Returns (bitmap_bytes, width, height, metrics_dict) or None if no outline.
    bitmap_bytes is a bytes of uint8 SDF values (0-255).
    """
    from PIL import Image, ImageDraw
    import math

    glyphset = font.getGlyphSet()
    if glyph_name not in glyphset:
        return None

    g = glyphset[glyph_name]
    if g.width == 0:
        return None

    # Get bounding box via drawing
    try:
        bb = font['glyf'][glyph_name].boundingBox() if 'glyf' in font else None
    except Exception:
        bb = None

    # Scale factor: map em-square to SIZE px
    upem = font['head'].unitsPerEm
    scale = size / upem

    # We'll rasterize at SIZE, with BUFFER padding on each side
    pad = buffer + radius
    W = round(g.width * scale) + pad * 2
    H = size + pad * 2
    if W <= 0 or H <= 0:
        return None

    # Draw glyph on a temporary image
    tmp = Image.new("L", (W * 4, H * 4), 0)  # 4x oversample
    draw = ImageDraw.Draw(tmp)

    class SimplePen:
        def __init__(self, draw, scale, ox, oy):
            self.draw = draw
            self.scale = scale
            self.ox = ox
            self.oy = oy
            self.cur = (0, 0)
            self.contours = []
            self.cur_contour = []

        def _t(self, x, y):
            return (
                round((x * self.scale + self.ox) * 4),
                round((H * 4) - round((y * self.scale + self.oy) * 4))
            )

        def moveTo(self, pt):
            if self.cur_contour:
                self.contours.append(self.cur_contour)
            self.cur_contour = [self._t(*pt)]
            self.cur = pt

        def lineTo(self, pt):
            self.cur_contour.append(self._t(*pt))
            self.cur = pt

        def qCurveTo(self, *pts):
            # Approximate with line segments
            prev = self.cur
            for pt in pts:
                mid = ((prev[0]+pt[0])/2, (prev[1]+pt[1])/2)
                self.cur_contour.append(self._t(*mid))
                prev = pt
            self.cur_contour.append(self._t(*pts[-1]))
            self.cur = pts[-1]

        def curveTo(self, *pts):
            for pt in pts:
                self.cur_contour.append(self._t(*pt))
            self.cur = pts[-1]

        def endPath(self):
            if self.cur_contour:
                self.contours.append(self.cur_contour)
                self.cur_contour = []

        def closePath(self):
            if self.cur_contour:
                self.contours.append(self.cur_contour)
                self.cur_contour = []

    pen = SimplePen(draw, scale, pad, pad)
    try:
        g.draw(pen)
    except Exception:
        return None
    if pen.cur_contour:
        pen.contours.append(pen.cur_contour)

    for contour in pen.contours:
        if len(contour) >= 2:
            draw.polygon(contour, fill=255)

    # Downsample 4x
    tmp = tmp.resize((W, H), Image.LANCZOS)
    pixels = list(tmp.getdata())

    # Simple distance field: for each pixel, find distance to nearest edge
    # (binary approximation — fast enough for our sizes)
    mask = [1 if p > 127 else 0 for p in pixels]

    sdf = []
    for y in range(H):
        for x in range(W):
            idx = y * W + x
            inside = mask[idx]
            # Search nearby pixels for edge
            min_dist_sq = radius * radius + 1
            r = radius
            for dy in range(-r, r+1):
                ny = y + dy
                if ny < 0 or ny >= H:
                    continue
                for dx in range(-r, r+1):
                    nx = x + dx
                    if nx < 0 or nx >= W:
                        continue
                    nidx = ny * W + nx
                    if mask[nidx] != inside:
                        d2 = dx*dx + dy*dy
                        if d2 < min_dist_sq:
                            min_dist_sq = d2
            dist = math.sqrt(min_dist_sq)
            if inside:
                val = 255 - round(255 * (1 - cutoff) * dist / radius)
            else:
                val = round(255 * cutoff * (1 - dist / radius))
            sdf.append(max(0, min(255, val)))

    metrics = {
        "width":    W,
        "height":   H,
        "left":     -pad,
        "top":      size + pad,
        "advance":  round(g.width * scale),
    }
    return bytes(sdf), W, H, metrics


# ── PBF builder ──────────────────────────────────────────────────────────────
def build_range_pbf(font: TTFont, cmap: dict,
                    start: int, end: int,
                    size: int, buffer: int, radius: int, cutoff: float) -> bytes:
    """
    Build a Mapbox/MapLibre glyph PBF for codepoints [start, end].
    Returns raw PBF bytes.
    """
    glyph_pbs = []

    for cp in range(start, end + 1):
        glyph_name = cmap.get(cp)
        if glyph_name is None:
            continue

        result = _render_sdf(font, glyph_name, size, buffer, radius, cutoff)
        if result is None:
            continue

        bitmap, w, h, m = result

        # Encode one Glyph message
        g_pb = b""
        g_pb += _uint32_field(1, cp)             # id
        g_pb += _len_delim(2, bitmap)             # bitmap
        g_pb += _uint32_field(3, w)              # width
        g_pb += _uint32_field(4, h)              # height
        g_pb += _sint32_field(5, m["left"])      # left
        g_pb += _sint32_field(6, m["top"])       # top (ascender)
        g_pb += _uint32_field(7, m["advance"])   # advance

        glyph_pbs.append(_len_delim(3, g_pb))   # repeated Glyph

    # Wrap in Glyphs { Fontstack { ... } }
    fontstack_name = FONT_NAME.encode()
    fs_pb = _len_delim(1, fontstack_name)
    fs_pb += _len_delim(2, (start).to_bytes(4,'big'))  # unused range field
    for g_pb in glyph_pbs:
        fs_pb += g_pb

    glyphs_pb = _len_delim(1, fs_pb)
    return glyphs_pb


def main():
    print(f"Loading {TTF_PATH} ...")
    font = TTFont(str(TTF_PATH))
    cmap = font.getBestCmap()
    if not cmap:
        print("ERROR: No cmap in font.")
        return

    # Only generate ranges that actually have glyphs (plus basic Latin always)
    needed_starts = set()
    needed_starts.add(0)   # 0-255 always (basic Latin, digits, punctuation)
    for cp in cmap:
        needed_starts.add((cp // 256) * 256)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(needed_starts)
    print(f"Generating {total} PBF range(s) into {OUT_DIR}/")

    for i, start in enumerate(sorted(needed_starts)):
        end = start + 255
        out_path = OUT_DIR / f"{start}-{end}.pbf"
        pbf = build_range_pbf(font, cmap, start, end, SIZE, BUFFER, RADIUS, CUTOFF)
        out_path.write_bytes(pbf)
        print(f"  [{i+1}/{total}] {start}-{end}.pbf  ({len(pbf)} bytes)")

    print(f"\nDone. {total} files written.")
    print(f"\nIn your MapLibre/Maputnik style, set:")
    print(f'  "glyphs": "https://cdn.jsdelivr.net/gh/theammman/factions-tiles@main/glyphs/{{fontstack}}/{{range}}.pbf"')
    print(f'\nFont name to use in text-font: ["{FONT_NAME}"]')


if __name__ == "__main__":
    main()
