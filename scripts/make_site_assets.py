#!/usr/bin/env python
"""Generate the site's icons and social-share card.

Run from the repo root:

    ./.venv/bin/python scripts/make_site_assets.py

Outputs into `docs/assets/`:
    icon.svg            scalable favicon (modern browsers)
    favicon-32.png      raster fallback
    apple-touch-icon.png  180x180, iOS home screen
    og-card.png         1200x630 link-preview card

The mark is three ascending bars — the model's whole output is a ranking, and a
bar chart reads at 16px where a basketball does not. The tallest bar is the
"capped" gold used in the UI for players priced above the maximum, which is the
one visual idea the site is actually about.

Committed outputs are what the site serves; this script only exists so they can
be regenerated rather than reverse-engineered.
"""

from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"

INK = (22, 23, 26)
PANEL = (30, 32, 36)
ACCENT = (122, 167, 232)
GOLD = (224, 179, 80)
TEXT = (236, 234, 230)
MUTED = (150, 147, 142)

SF = "/System/Library/Fonts/SFNS.ttf"
MENLO = "/System/Library/Fonts/Menlo.ttc"
ARIAL_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(path: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size, index=index)
    except OSError:
        return ImageFont.load_default()


def bars(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, gap_ratio: float = 0.22):
    """Three ascending bars in a w x h box, anchored bottom-left at (x, y+h)."""
    gap = w * gap_ratio / 2
    bw = (w - 2 * gap) / 3
    heights = [0.42, 0.68, 1.0]
    colors = [ACCENT, ACCENT, GOLD]
    r = max(2, int(bw * 0.18))
    for i, (frac, col) in enumerate(zip(heights, colors)):
        bx = x + i * (bw + gap)
        bh = h * frac
        draw.rounded_rectangle([bx, y + h - bh, bx + bw, y + h], radius=r, fill=col)


def make_icon_png(size: int, pad_ratio: float = 0.18) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=int(size * 0.22), fill=INK)
    pad = int(size * pad_ratio)
    bars(d, pad, pad, size - 2 * pad, size - 2 * pad)
    return img


def make_svg() -> str:
    a = "#%02x%02x%02x" % ACCENT
    g = "#%02x%02x%02x" % GOLD
    ink = "#%02x%02x%02x" % INK
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="14" fill="{ink}"/>
  <rect x="12" y="35.4" width="11.7" height="16.6" rx="2" fill="{a}"/>
  <rect x="26.2" y="26.2" width="11.7" height="25.8" rx="2" fill="{a}"/>
  <rect x="40.3" y="12" width="11.7" height="40" rx="2" fill="{g}"/>
</svg>
'''


def make_og_card() -> Image.Image:
    W, H = 1200, 630
    img = Image.new("RGB", (W, H), INK)
    d = ImageDraw.Draw(img)

    # subtle panel band behind the text block
    d.rectangle([0, 0, W, 8], fill=ACCENT)

    f_title = font(ARIAL_BOLD, 66)
    f_sub = font(SF, 34)
    f_mono = font(MENLO, 25, index=0)
    f_foot = font(SF, 24)

    x = 78
    d.text((x, 132), "WNBA Contract Model", font=f_title, fill=TEXT)
    d.text((x, 224), "What is a WNBA player actually worth?", font=f_sub, fill=MUTED)

    # the formula, in the same mono treatment the site uses
    box_y = 306
    d.rounded_rectangle([x, box_y, W - 78, box_y + 68], radius=8,
                        fill=PANEL, outline=(51, 54, 60))
    d.rectangle([x, box_y, x + 3, box_y + 68], fill=ACCENT)
    d.text((x + 22, box_y + 20), "value = min + WAR x $/win", font=f_mono, fill=TEXT)

    d.text((x, 432), "Possession-level RAPM, shrunk toward a box-score prior.",
           font=f_foot, fill=MUTED)
    d.text((x, 468), "Priced against the 2026 CBA.", font=f_foot, fill=MUTED)

    bars(d, W - 78 - 150, 432, 150, 96)
    d.text((x, 552), "mjwilson09.github.io/WNBA-Contract-Modeling",
           font=font(MENLO, 21, index=0), fill=(117, 114, 109))
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "icon.svg").write_text(make_svg())
    make_icon_png(32).save(OUT / "favicon-32.png")
    make_icon_png(180, pad_ratio=0.20).save(OUT / "apple-touch-icon.png")
    make_og_card().save(OUT / "og-card.png", optimize=True)
    for f in ("icon.svg", "favicon-32.png", "apple-touch-icon.png", "og-card.png"):
        p = OUT / f
        print(f"  {f:<22} {p.stat().st_size / 1024:6.1f} KB")


if __name__ == "__main__":
    main()
