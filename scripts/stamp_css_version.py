#!/usr/bin/env python
"""Stamp a content hash onto the stylesheet link in every page.

    ./.venv/bin/python scripts/stamp_css_version.py

`docs/assets/site.css` is loaded as `site.css?v=<hash>`. Without a changing
query string, browsers keep serving a cached stylesheet after a deploy — and a
stale sheet does not look like a caching problem, it looks like broken CSS:
fonts silently fall back and new rules appear to do nothing. That cost real
debugging time once already (AGENTS trap 15).

The version is the first 8 characters of the CSS's own SHA-256 rather than a
counter, which makes this idempotent: running it twice changes nothing, and
identical CSS always produces an identical stamp, so it never creates a
spurious diff. Run it after editing the stylesheet; it is safe to run any time.

Exits non-zero if a page was out of date, so it can be used as a check.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSS = ROOT / "docs" / "assets" / "site.css"
PAGES = sorted((ROOT / "docs").glob("*.html"))
PATTERN = re.compile(r'(href="assets/site\.css)(?:\?v=[^"]*)?(")')


def main() -> int:
    if not CSS.exists():
        print(f"missing {CSS}", file=sys.stderr)
        return 2

    version = hashlib.sha256(CSS.read_bytes()).hexdigest()[:8]
    changed = []
    for page in PAGES:
        text = page.read_text()
        stamped = PATTERN.sub(rf'\1?v={version}\2', text)
        if stamped != text:
            page.write_text(stamped)
            changed.append(page.name)

    print(f"site.css -> v={version}")
    for name in changed:
        print(f"  updated {name}")
    if not changed:
        print("  all pages already current")
    return 1 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
