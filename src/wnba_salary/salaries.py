"""WNBA contract data from Her Hoop Stats.

Her Hoop Stats publishes the league-wide salary table server-side, so it parses
without an account. Spotrac's equivalent is client-rendered and returns no table
to a plain fetch.

Player names come from the cell's `sorttable_customkey` ("Wilson, A'ja") rather
than the cell text, which contains a full name and an abbreviated name run
together.

Only the salary and signing-type columns are used. The table pairs 2026 salaries
with 2025 per-game stats; our stats come from elsewhere.
"""

from __future__ import annotations

import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from . import data

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)
CACHE_DIR = data.RAW_DIR / "salaries"
RATE_LIMIT_SECONDS = 4.0

_last_request = 0.0


def _url(salary_season: int, stats_season: int | None = None) -> str:
    # The stats season must match the salary season. The page inner-joins the
    # two, so pairing 2026 salaries with 2025 stats silently drops every 2026
    # rookie — 155 players instead of 217.
    stats_season = stats_season or salary_season
    return (
        "https://herhoopstats.com/salary-cap-sheet/wnba/players/"
        f"salary_{salary_season}/stats_{stats_season}/"
    )


def _money(text: str) -> float | None:
    if not text:
        return None
    cleaned = re.sub(r"[^0-9.]", "", text)
    return float(cleaned) if cleaned else None


def _flip_name(key: str) -> str:
    """'Wilson, A'ja' -> "A'ja Wilson"."""
    if "," not in key:
        return key.strip()
    last, first = key.split(",", 1)
    return f"{first.strip()} {last.strip()}"


def fetch_salaries(salary_season: int, *, refresh: bool = False) -> pd.DataFrame:
    """League-wide salaries for one season."""
    global _last_request
    path = CACHE_DIR / f"wnba_salaries_{salary_season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    wait = RATE_LIMIT_SECONDS - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.time()

    resp = requests.get(
        _url(salary_season),
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
        timeout=60,
    )
    resp.raise_for_status()
    resp.encoding = "utf-8"

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table")
    if table is None:
        raise RuntimeError("no salary table found; page layout may have changed")

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 3:
            continue
        key = cells[0].get("sorttable_customkey") or cells[0].get_text(strip=True)
        salary = _money(cells[1].get_text(strip=True))
        if salary is None:
            continue
        rows.append({
            "season": salary_season,
            "player": _flip_name(key),
            "salary": salary,
            "signing": cells[2].get_text(strip=True),
        })

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
