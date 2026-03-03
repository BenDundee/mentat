"""Generate frontend/favicon.ico from a pixel-art design using stdlib only."""

import struct
import zlib
from pathlib import Path

# --- colour palette -----------------------------------------------------------
BG = (26, 39, 68, 255)  # dark navy  #1a2744
STROKE = (126, 184, 247, 255)  # light blue #7eb8f7
DOT = (74, 158, 255, 255)  # accent     #4a9eff
TR = (0, 0, 0, 0)  # transparent (rounded corners)

W = H = 32


# --- pixel canvas -------------------------------------------------------------


def make_canvas() -> list[list[tuple[int, int, int, int]]]:
    return [[BG] * W for _ in range(H)]


def set_px(
    canvas: list[list[tuple[int, int, int, int]]],
    x: int,
    y: int,
    color: tuple[int, int, int, int],
) -> None:
    if 0 <= x < W and 0 <= y < H:
        canvas[y][x] = color


def fill_rect(
    canvas: list[list[tuple[int, int, int, int]]],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int, int],
) -> None:
    for y in range(y1, y2 + 1):
        for x in range(x1, x2 + 1):
            set_px(canvas, x, y, color)


def draw_line(
    canvas: list[list[tuple[int, int, int, int]]],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    half: int,
    color: tuple[int, int, int, int],
) -> None:
    """Bresenham-style line with square brush of radius `half`."""
    dx, dy = x2 - x1, y2 - y1
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        set_px(canvas, x1, y1, color)
        return
    for i in range(steps + 1):
        x = round(x1 + dx * i / steps)
        y = round(y1 + dy * i / steps)
        for tx in range(-half, half + 1):
            for ty in range(-half, half + 1):
                set_px(canvas, x + tx, y + ty, color)


def draw_dot(
    canvas: list[list[tuple[int, int, int, int]]],
    cx: int,
    cy: int,
    r: int,
    color: tuple[int, int, int, int],
) -> None:
    for tx in range(-r, r + 1):
        for ty in range(-r, r + 1):
            if tx * tx + ty * ty <= r * r:
                set_px(canvas, cx + tx, cy + ty, color)


def round_corners(canvas: list[list[tuple[int, int, int, int]]], radius: int) -> None:
    """Trim square corners to simulate a rounded-rect background."""
    for y in range(H):
        for x in range(W):
            in_tl = (
                x < radius
                and y < radius
                and (x - radius) ** 2 + (y - radius) ** 2 > radius**2
            )
            in_tr = (
                x >= W - radius
                and y < radius
                and (x - (W - 1 - radius)) ** 2 + (y - radius) ** 2 > radius**2
            )
            in_bl = (
                x < radius
                and y >= H - radius
                and (x - radius) ** 2 + (y - (H - 1 - radius)) ** 2 > radius**2
            )
            in_br = (
                x >= W - radius
                and y >= H - radius
                and (x - (W - 1 - radius)) ** 2 + (y - (H - 1 - radius)) ** 2
                > radius**2
            )
            if in_tl or in_tr or in_bl or in_br:
                canvas[y][x] = TR


# --- draw the Mentat "M" ------------------------------------------------------


def draw_m(canvas: list[list[tuple[int, int, int, int]]]) -> None:
    """Neural-node M: two verticals + two diagonals + accent dots."""
    # Left vertical bar  (x 5-8, y 7-25)
    fill_rect(canvas, 5, 7, 8, 25, STROKE)
    # Right vertical bar (x 23-26, y 7-25)
    fill_rect(canvas, 23, 7, 26, 25, STROKE)

    # Left diagonal: top-left corner → valley
    draw_line(canvas, 8, 8, 15, 18, 1, STROKE)
    # Right diagonal: valley → top-right corner
    draw_line(canvas, 17, 18, 24, 8, 1, STROKE)

    # Accent dots at the five key nodes
    for cx, cy in [(6, 7), (26, 7), (16, 18), (6, 25), (26, 25)]:
        draw_dot(canvas, cx, cy, 2, DOT)


# --- PNG encoder (stdlib only) ------------------------------------------------


def _chunk(tag: bytes, data: bytes) -> bytes:
    c = struct.pack(">I", len(data)) + tag + data
    return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def canvas_to_png(canvas: list[list[tuple[int, int, int, int]]]) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 6, 0, 0, 0))

    # Scanlines: filter byte 0 + RGBA pixels
    raw = b"".join(b"\x00" + b"".join(bytes(px) for px in row) for row in canvas)
    idat = _chunk(b"IDAT", zlib.compress(raw, 9))
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


# --- ICO wrapper --------------------------------------------------------------


def png_to_ico(png: bytes) -> bytes:
    """Wrap a single PNG image in a minimal ICO container."""
    # ICO header: reserved=0, type=1 (icon), count=1
    header = struct.pack("<HHH", 0, 1, 1)
    # Directory entry
    img_size = len(png)
    img_offset = 6 + 16  # header + one dir entry
    dir_entry = struct.pack(
        "<BBBBHHII",
        W,  # width  (0 = 256 for W≥256; 32 is fine as-is)
        H,  # height
        0,  # colour count  (0 = no palette)
        0,  # reserved
        1,  # colour planes
        32,  # bits per pixel
        img_size,
        img_offset,
    )
    return header + dir_entry + png


# --- main ---------------------------------------------------------------------


def main() -> None:
    canvas = make_canvas()
    round_corners(canvas, 6)
    draw_m(canvas)

    png = canvas_to_png(canvas)
    ico = png_to_ico(png)

    out = Path(__file__).parent.parent / "frontend" / "favicon.ico"
    out.write_bytes(ico)
    print(f"Written {len(ico)} bytes → {out}")


if __name__ == "__main__":
    main()
